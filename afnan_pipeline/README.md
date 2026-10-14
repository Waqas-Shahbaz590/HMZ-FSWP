# Afnan: selected working pipeline

This is the clean project selected for further development: `D:\Hamza\Face Swap\wqs\afnan_pipeline`. It contains the working Afnan face-swap backend, a frame-based runner, verified pretrained checkpoints, inputs and reproducible setup. The old AFW application placeholders, socket server and unrelated FaceFusion/LivePortrait implementations are excluded from this runtime.

The extraction preserves the measured baseline algorithm. Aligned single-paste composition (POLISH_PLAN phase 1) and independent hand protection with confidence gates (phase 2) are implemented and tested; causal temporal smoothing, face association and the live 30 FPS path remain planned in [POLISH_PLAN.md](<D:/Hamza/Face Swap/wqs/afnan_pipeline/POLISH_PLAN.md>).

For repeated upload/test/download cycles after editing code, use [RUN_TESTS_GUIDE.md](<D:/Hamza/Face Swap/wqs/afnan_pipeline/RUN_TESTS_GUIDE.md>) and `bash scripts/test_cycle.sh change_name quick` (or `full`). Each cycle preserves its code snapshot and creates new output folders.

The extracted project was installed separately on the RTX A5000 and passed all 14 wrapper tests, verification of all eight supplied checkpoints, and a 30-frame CUDA smoke run starting at 31.24 seconds in the hand-occlusion clip. All 30 frames were synthetic, with zero processing exceptions; both silent labeled MP4s passed independent export verification. This validates extraction, not the future quality or 30 FPS changes. The remote project is `/workspace/wqs/afnan_pipeline`; local and remote selection markers identify it as the project to continue with.

## Code and data

```text
afnan_pipeline/
  run.py                       Afnan-only MP4 runner and labeled exports
  afnan/
    adapter.py                 source preparation and current-frame model stages
    alignment.py               arcface-template geometry and transform confidence gate
    occlusion.py               independent hand cascade and protection-mask math
    tensors.py                 ONNX layouts, RGB/BGR conversion and validation
    compositing.py             aligned-space masks and the single bounded ROI paste
    tests/test_adapter.py      pipeline wrapper tests with mocked sessions
    tests/test_occlusion.py    hand-cascade and confidence-gate tests
  common/
    io.py                      sequential input, frame timestamps and MP4 output
    metrics.py                 latency, NVML memory and temporal review proxy
  scripts/
    install.sh                 isolated Python environment and checkpoints
    fetch_checkpoints.py        pinned download/checksum verification
    run_gpu.sh                 CUDA-library setup and runner entry point
    verify_export.py           independent decoded export/evidence checks
  requirements.txt             Python dependencies
  environment.lock.txt         resolved environment from the deployed baseline
  checkpoints.json             pinned checkpoint locations and expected hashes
  model_manifest.json          verification receipt for supplied checkpoints
  models/
    buffalo_l/                 detector and source-identity recognition weights
    inswapper_128.onnx
    gfpgan_1.4.onnx
    xseg_1.onnx
    palm_detection_192.onnx    BlazePalm-class palm detector (Apache-2.0 conversion)
    hand_landmark_224.onnx     21-keypoint hand landmarks (Apache-2.0 conversion)
  inputs/
    reference.png
    driving_phone_720p25.mp4
  outputs/<new_run>/
    synthetic_avatar.mp4
    comparison.mp4
    frame_metrics.csv
    summary.json
```

The model operations live in [adapter.py](<D:/Hamza/Face Swap/wqs/afnan_pipeline/afnan/adapter.py>); image blending lives in [compositing.py](<D:/Hamza/Face Swap/wqs/afnan_pipeline/afnan/compositing.py>). Start new work here. Preserve `backend_comparison` as the completed comparison evidence and the original AFW archive as recovery evidence.

Existing complete Afnan videos remain at [synthetic_avatar.mp4](<D:/Hamza/Face Swap/wqs/backend_comparison/outputs/afnan_full/synthetic_avatar.mp4>) and [comparison.mp4](<D:/Hamza/Face Swap/wqs/backend_comparison/outputs/afnan_full/comparison.mp4>). Their independent export receipts and reviewed failures are in [TEAM_B_REPORT.md](<D:/Hamza/Face Swap/wqs/backend_comparison/TEAM_B_REPORT.md>). These are the baseline results, rather than results from future polish work.

## Install and run

The supported deployment is Linux with Python 3.12, an NVIDIA GPU, a compatible driver, CUDA 12/cuDNN runtime libraries, FFmpeg/FFprobe, `uv`, and a C/C++ build toolchain for InsightFace. The supplied scripts target the deployed Vast image with base Python at `/venv/main/bin/python`. A different image requires adjusting that path and the CUDA-library lookup in `scripts/run_gpu.sh`. A native Windows GPU installation has not been validated.

The isolated dependencies are NumPy 1.26.4, ONNX Runtime GPU 1.24.4, InsightFace 0.7.3, ONNX 1.17.0, OpenCV headless 4.10.0.84, SciPy 1.14.1, scikit-image 0.24.0, Albumentations 1.4.3, Cython 3.0.11 and nvidia-ml-py 13.580.82. Do not merge this environment with the comparison's NumPy 2 FaceFusion environment. The neural networks execute through ONNX Runtime; PyTorch is not the inference engine for these four components.

From the deployed project directory:

```bash
cd /workspace/wqs/afnan_pipeline
bash scripts/install.sh
.venv/bin/python scripts/fetch_checkpoints.py --verify-only
.venv/bin/python -m unittest discover -s afnan/tests -v

bash scripts/run_gpu.sh \
  --reference inputs/reference.png \
  --video inputs/driving_phone_720p25.mp4 \
  --output outputs/smoke --max-frames 30 --profile
.venv/bin/python scripts/verify_export.py outputs/smoke

bash scripts/run_gpu.sh \
  --reference inputs/reference.png \
  --video inputs/driving_phone_720p25.mp4 \
  --output outputs/full
.venv/bin/python scripts/verify_export.py outputs/full
```

Use a new output directory for each variant. `--models PATH` overrides the checkpoint directory; `--restore-blend 0.25` records a distinct restoration experiment. `--start 83 --end 93` selects the mouth/hand test window while retaining absolute source indices and timestamps. Downloads, checkpoint verification and source preparation happen before the timed frame loop; model initialization is not counted as steady-state FPS.

Every run validates checkpoint sizes, hashes and supplied publisher CRCs against `checkpoints.json`, then records source/checkpoint provenance. The export verifier independently checks decoded frame counts, silent streams, dimensions, FPS, summary/CSV agreement, timestamp ordering and file hashes. Its disclosure-band check samples pixels rather than recognizing every frame's text.

The supplied driving MP4 is silent, orientation-corrected, 720 × 1280, 25 FPS, 3,076 frames and 123.04 seconds. Reference SHA256 is `98e208d81e663d304e48f11a22fc6f8c430ba60280d54bd536779b5a530afe93`; driving SHA256 is `baf8a71c6c90091ca20e9116b419c70e65df51a393da9c2da3682e8bac78a3ae`. Keep these fixed for comparison tests.

## Frame API

The core accepts a reference once, then one RGB image and timestamp at a time:

```python
from afnan.adapter import AfnanBackend

backend = AfnanBackend("models", restore_blend=0.75, strict_cuda=True)
backend.prepare(reference_rgb)
result_rgb, diagnostics = backend.process(frame_rgb, pts_ms)
backend.close()
```

Inputs are contiguous `uint8` arrays with shape `H × W × 3`; timestamps are finite, nonnegative and strictly increasing. Use one backend instance per stream. A later camera reader can supply the same arrays/timestamps without rewriting model inference. There is no dependency on future frames or the complete video inside `AfnanBackend`.

Raw adapter arrays do not carry disclosure text. The runner visibly labels the final synthetic MP4/comparison and any original-frame fallback; a future live display must apply the same disclosure before presenting frames.

## Current behavior and limitations

The backend selects the largest current face using Buffalo L at detector size 320 and threshold 0.25. Target frames run **detection only**; the swap consumes the reference latent that is projected and cached once in `prepare`, so per-frame target recognition is no longer executed. Each frame is warped into a single landmark-aligned face space (arcface template family) shared by all stages: INSwapper 128 runs on the aligned crop, XSeg1 runs on the aligned **original** crop (never generated pixels), and GFPGAN 1.4 restores the aligned swapped face rather than a square-resized bbox. The output starts as a copy of the original frame; generated content is inverse-warped into the bounded face ROI under one alpha mask (feathered face oval × XSeg visibility, suppressed under confirmed hands) in a single paste. Missing faces, invalid masks/transforms or model errors return the original current frame with explicit diagnostics; there is no reuse of stale detections.

This aligned single-paste path replaces the upstream `INSwapper.get(paste_back=True)` full-frame paste and second crop composite. It passed the full 3,076-frame GPU regression (all five previously catalogued artifact moments visually clean) with mean backend latency **85.06 ms** (2.09x the 177.39 ms baseline); the throughput numbers below remain the measured record of the pre-polish extraction.

Hand protection (phase 2) adds evidence independent of XSeg: a BlazePalm-class detector finds palms in the full frame, a 21-keypoint landmark model confirms each palm (presence gate), and confirmed hands become conservative frame-space silhouettes (convex hull + per-joint disks, enlarged at fingertips, dilated) that suppress the generated alpha. Feathering is applied only on the face side of the protected boundary, so fingertips keep strictly original pixels. Confidence gates: an implausible alignment transform or any hand-stage error returns a labeled original frame (`transform_implausible`, `hand_*_error`); low-presence or geometrically inconsistent hand readings are discarded and counted (`hand_rejected`) without failing the frame, because XSeg still constrains the paste. `hand_protection=False` or `--no-hand-protection` reproduces the phase-1 alpha path exactly for A/B measurement.

There is currently no temporal landmark smoothing, face association or SAM segmentation; mask history and causal jitter control remain planned in [POLISH_PLAN.md](<D:/Hamza/Face Swap/wqs/afnan_pipeline/POLISH_PLAN.md>). The landmark-derived silhouette is a conservative approximation of the true hand outline, not a pixel-accurate segmentation. Largest-face selection can still switch people. Tests validate wrapper behavior, not pretrained visual quality.

On the RTX A5000 baseline, the complete backend averaged **177.39 ms / 5.637 FPS**, with **218.39 ms p95**; the full two-video export and review path achieved **4.110 FPS**. Sampled device memory peaked at **4,408.5 MiB**; process-specific NVML memory was unavailable in the container. Playback at 25 FPS does not imply 25 FPS inference. A 30 FPS camera path has a 33.3 ms per-frame budget and remains a measured development target, not a capability of this extraction.

The existing instrumented probe separates the swap call: the warm INSwapper ONNX call averaged 13.87 ms, with another 77.34 ms in the same aggregate swap stage outside that call. Fixing crop/paste overhead is a strong first performance experiment, but GFPGAN/XSeg and the other model calls still exceed a 33.3 ms serial budget. See [BASELINE_LATENCY_ANALYSIS.md](<D:/Hamza/Face Swap/wqs/afnan_pipeline/audits/BASELINE_LATENCY_ANALYSIS.md>) for the evidence and profiling limits.

Checkpoint hashes and publisher metadata must accompany reproduced runs. InsightFace code and pretrained weights have different terms; the supplied InsightFace weights carry noncommercial restrictions. The hand-cascade ONNX conversions are Apache-2.0 (MediaPipe models converted by PINTO0309, pinned commit `e2deb2e`). Retain the included GFPGAN/XSeg conversion and model notices when sharing this evaluation. Review checkpoint rights separately before commercial deployment.

## Backup and redeploy

The local project directory is the source of truth; the pod is a rebuildable runtime. If the instance is lost, redeploy on a fresh CUDA-12 Python 3.12 Vast image:

1. Upload the project (or unpack a snapshot and the verified `models/` + `inputs/` trees).
2. `bash scripts/install.sh` — builds `.venv` and re-fetches any missing checkpoints from the pinned URLs in `checkpoints.json` (hashes verified; fails closed on mismatch).
3. `bash scripts/test_cycle.sh redeploy_check quick` — must pass all automated checks before further work.

Local backup discipline: all eight pipeline checkpoints and both inputs are hash-verified against `model_manifest.json` in the local `models/` + `inputs/` trees. `D:\Hamza\Face Swap\wqs\backups\afnan-clean-project-20261015.tar.gz` is the complete pre-polish project archive; `afnan-phase1-signedoff-20261015.tar.gz` (plus its `.sha256`) is the code/config snapshot of the Phase 1 regression-passed state. Download every pod run's evidence into `test_results/` before terminating or reimaging the instance; pod `outputs/` is the only non-reproducible artifact class.
