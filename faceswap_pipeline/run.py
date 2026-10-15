"""Labeled, causal offline evaluation of the isolated FaceSwap pipeline."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import statistics
import time
from collections import Counter
import numpy as np
from common.io import VideoReader, VideoWriter, load_reference, label_frame, comparison_frame
from common.metrics import GPUMemory, TemporalConsistency, sha256_file, environment_versions, write_json


def profile_onnx_sessions(directory):
    """Keep InsightFace's session subclassing while profiling a short probe."""
    import onnxruntime as ort
    original = ort.InferenceSession
    sessions = []
    class ProfileSession(original):
        def __init__(self, path_or_bytes, sess_options=None, *args, **kwargs):
            opts = sess_options or ort.SessionOptions()
            opts.enable_profiling = True
            opts.profile_file_prefix = str(directory / f"session_{len(sessions)}")
            super().__init__(path_or_bytes, opts, *args, **kwargs)
            sessions.append(self)
    ort.InferenceSession = ProfileSession
    return sessions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--video', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--models', type=Path, default=Path(__file__).resolve().parent / 'models')
    p.add_argument('--restore-blend', type=float, default=.75)
    p.add_argument('--no-hand-protection', action='store_true',
                   help='Disable the independent hand cascade (A/B debugging only)')
    p.add_argument('--start', type=float, default=0.)
    p.add_argument('--end', type=float)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--profile', action='store_true', help='GPU node evidence on a short probe')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new or empty output directory to preserve prior evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    profiles = args.output / 'profiles'
    profiles.mkdir(exist_ok=True)
    profiled_sessions = []
    setup_start = time.perf_counter()
    from scripts.fetch_checkpoints import verify
    project = Path(__file__).resolve().parent
    pinned_checkpoints = json.loads((project / 'checkpoints.json').read_text(encoding='utf-8-sig'))
    checkpoints = [verify(args.models / spec['file'], spec) for spec in pinned_checkpoints]
    source_provenance = {
        name: sha256_file(project / name)
        for name in ('run.py', 'faceswap/adapter.py', 'faceswap/tensors.py', 'faceswap/compositing.py',
                     'faceswap/alignment.py', 'faceswap/occlusion.py',
                     'common/io.py', 'common/metrics.py', 'scripts/fetch_checkpoints.py', 'checkpoints.json')
    }
    if args.profile:
        profiled_sessions = profile_onnx_sessions(profiles)
    from faceswap.adapter import FaceSwapBackend
    backend = FaceSwapBackend(args.models, restore_blend=args.restore_blend,
                           hand_protection=not args.no_hand_protection)
    prepared = backend.prepare(load_reference(args.reference))
    setup_ms = (time.perf_counter()-setup_start)*1000
    provenance = {'reference': {'filename': args.reference.name, 'sha256': sha256_file(args.reference)},
                  'driving': {'filename': args.video.name, 'sha256': sha256_file(args.video)}}
    reader = VideoReader(args.video, max_width=0, max_frames=args.max_frames, start_seconds=args.start, end_seconds=args.end)
    memory = GPUMemory()
    quality = TemporalConsistency()
    rows = []
    csvfile = (args.output / 'frame_metrics.csv').open('w', newline='', encoding='utf8')
    fields = ['frame_index', 'timestamp_ms', 'status', 'reason', 'total_ms', 'backend_ms', 'decode_ms', 'output_ms', 'quality_ms',
              'flow_aligned_face_bbox_temporal_l1_proxy', 'nvml_device_used_mib', 'nvml_process_used_mib', 'stages_json', 'diagnostics_json']
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    output = comparison = None
    close_info = None
    failure = None
    began = time.perf_counter()
    try:
        for frame in reader:
            start = time.perf_counter()
            result, diag = backend.process(frame.rgb, frame.timestamp_ms)
            backend_ms = (time.perf_counter()-start)*1000
            if result.shape != frame.rgb.shape or result.dtype != np.uint8:
                raise ValueError('Backend must return uint8 RGB with the input dimensions')
            status = 'animated' if diag.get('status') == 'animated' else 'original_frame_fallback'
            qstart = time.perf_counter()
            face = np.zeros(frame.rgb.shape[:2], dtype=np.float32)
            box = diag.get('face_bbox', diag.get('target_bbox'))
            if box is not None:
                x1, y1, x2, y2 = np.asarray(box, dtype=float).astype(int)
                face[max(0,y1):min(face.shape[0],y2), max(0,x1):min(face.shape[1],x2)] = 1
            hand = np.zeros_like(face)
            for hand_box in diag.get('hand_boxes') or []:
                hx1, hy1, hx2, hy2 = np.asarray(hand_box, dtype=float).astype(int)
                hand[max(0,hy1):min(hand.shape[0],hy2), max(0,hx1):min(hand.shape[1],hx2)] = 1
            q = quality.update(frame.rgb, result, face, hand)
            if status != 'animated':
                quality.previous = None
                q['flow_aligned_face_temporal_l1_proxy'] = None
            quality_ms = (time.perf_counter()-qstart)*1000
            ostart = time.perf_counter()
            labeled = label_frame(result)
            if status != 'animated':
                labeled = label_frame(labeled, 'PIPELINE FALLBACK - ORIGINAL FRAME', top=True)
            if output is None:
                output = VideoWriter(args.output / 'synthetic_avatar.mp4', reader.fps, reader.width, reader.height)
                comparison = VideoWriter(args.output / 'comparison.mp4', reader.fps, reader.width*2, reader.height)
            output.write(labeled)
            comparison.write(comparison_frame(frame.rgb, labeled, status=status))
            output_ms = (time.perf_counter()-ostart)*1000
            mem = memory.sample()
            stages = diag.get('stage_ms', diag.get('stages_ms', {}))
            row = {'frame_index': frame.index, 'timestamp_ms': frame.timestamp_ms, 'status': status,
                   'reason': diag.get('fallback_reason', diag.get('reason', '')),
                   'total_ms': frame.decode_ms+(time.perf_counter()-start)*1000, 'backend_ms': backend_ms,
                   'decode_ms': frame.decode_ms, 'output_ms': output_ms, 'quality_ms': quality_ms,
                   'flow_aligned_face_bbox_temporal_l1_proxy': q['flow_aligned_face_temporal_l1_proxy'],
                   'nvml_device_used_mib': mem.get('nvml_device_used_mib'),
                   'nvml_process_used_mib': mem.get('nvml_process_used_mib'),
                   'stages_json': json.dumps(stages), 'diagnostics_json': json.dumps(diag, default=str, allow_nan=False)}
            writer.writerow(row)
            csvfile.flush()
            rows.append({k:v for k,v in row.items() if k not in ('diagnostics_json','stages_json')} | {'stage_ms': stages})
            if len(rows) % 50 == 0:
                elapsed = time.perf_counter()-began
                progress = {'frames': len(rows), 'expected': reader.expected_selected_frames, 'elapsed_seconds': elapsed,
                            'fps': len(rows)/elapsed, 'last_pts_ms': frame.timestamp_ms,
                            'status_counts': dict(Counter(r['status'] for r in rows))}
                write_json(args.output / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        if not rows:
            raise RuntimeError('No frames selected')
        if reader.premature_end:
            raise RuntimeError('Source decoder ended before expected frame count')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        raise
    finally:
        finalization_errors = []
        for sink in (output, comparison):
            if sink is not None:
                try:
                    sink.close()
                except Exception as error:
                    finalization_errors.append(str(error))
        wall_seconds = time.perf_counter()-began
        try:
            close_info = backend.close()
        except Exception as error:
            finalization_errors.append(f'Backend close: {error}')
        node_profiles = []
        for session in profiled_sessions:
            try:
                path = Path(session.end_profiling())
                events = json.loads(path.read_text())
                counts = Counter(e.get('args', {}).get('provider') for e in events if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
                node_profiles.append({'filename': path.name, 'providers': dict(counts), 'sha256': sha256_file(path)})
            except Exception as error:
                finalization_errors.append(f'ONNX profiling close: {error}')
        reader.close()
        csvfile.close()
        def stats(vals):
            vals = [float(v) for v in vals if isinstance(v,(int,float)) and np.isfinite(v)]
            return {'mean': statistics.fmean(vals), 'p50': float(np.percentile(vals,50)), 'p95': float(np.percentile(vals,95)), 'max': max(vals)} if vals else None
        stage_names = sorted({stage for r in rows for stage in r['stage_ms']})
        summary = {
            'backend': 'faceswap',
            'variant': f'inswapper128_gfpgan_blend_{args.restore_blend}' + ('' if not args.no_hand_protection else '_no_hand_protection'),
            'complete': failure is None and not finalization_errors, 'error': failure, 'finalization_errors': finalization_errors,
            'frames': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
            'reason_counts': dict(Counter(r['reason'] for r in rows if r['reason'])),
            'wall_seconds': wall_seconds, 'end_to_end_fps': len(rows)/wall_seconds if wall_seconds else None,
            'inference_only_fps': len(rows)*1000/sum(r['backend_ms'] for r in rows) if rows else None,
            'latency_ms': {k:stats([r.get(k) for r in rows]) for k in ['total_ms','backend_ms','decode_ms','output_ms','quality_ms']},
            'stages_ms': {k:stats([r['stage_ms'].get(k,0) for r in rows]) for k in stage_names},
            'gpu_memory_peak_mib': {k:max([r[k] for r in rows if isinstance(r[k],(int,float))],default=None) for k in ['nvml_device_used_mib','nvml_process_used_mib']},
            'gpu_memory_definition': 'NVML sampled device use includes other processes. Process use may be unavailable in container PID namespaces. ONNX allocations are not Torch allocator counters.',
            'temporal_proxy': stats([r['flow_aligned_face_bbox_temporal_l1_proxy'] for r in rows]),
            'quality_limitations': 'Flow-aligned L1 inside the detected face bbox is a limited temporal proxy, not an identity, hand preservation, liveness, or fraud accuracy metric. Human inspection is required.',
            'video': {'fps': reader.fps, 'width': reader.width, 'height': reader.height, 'audio': False},
            'input_provenance': provenance, 'environment': environment_versions(), 'setup_ms': setup_ms,
            'source_provenance': source_provenance, 'checkpoint_provenance': checkpoints,
            'prepared': prepared, 'backend_close': close_info, 'node_profiles': node_profiles,
            'source_bounds': {'start_seconds': args.start, 'end_seconds': args.end, 'preroll_frames': reader.preroll_frames},
            'timing_definition': 'Completed synchronous ONNX host-return calls. Stages may cover only backend subsets; backend_ms is full process call. Full end-to-end FPS includes sequential decode, two labeled H264 exports, quality, CSV and encoder finalization; setup, model download, hash and profiling close are excluded. Playback FPS is independent.',
        }
        write_json(args.output / 'summary.json', summary)
        memory.close()
        print(json.dumps({k:summary[k] for k in ['complete','frames','end_to_end_fps','inference_only_fps','status_counts','error']}),flush=True)
        if finalization_errors and failure is None:
            raise RuntimeError('; '.join(finalization_errors))

if __name__ == '__main__':
    main()
