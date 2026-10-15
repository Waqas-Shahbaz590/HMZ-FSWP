# FaceSwap baseline latency audit

The existing FaceSwap pipeline is limited by substantial work outside the face-swap model, as well as the cost of its serial restoration and occlusion models. The measured `swap` stage is **93.68 ms** across the full video; that does **not** mean the INSwapper ONNX model itself takes 93.68 ms. A separate profiled probe attributes roughly **77.34 ms** of the warm swap stage to work outside its ORT call. This is a promising optimization target, not a measured improvement or a promise of 30 FPS.

This audit only reads the completed baselines and their traces. It ran no GPU job, changed no runtime code, and did not evaluate a live camera. Machine: NVIDIA RTX A5000, driver 555.58.02, ONNX Runtime GPU 1.24.4, Python 3.12.14, AMD EPYC 7663 host with a CPU quota equivalent to about 26.88 cores.

## Full-video measurements

The unprofiled full run processed 3,076 frames from a 720 × 1280, 25 FPS input. It exported a synthetic video and comparison video with disclosure labels and no audio. Five frames used a disclosed original-frame fallback because no face was detected.

| Measurement | Mean | Practical meaning |
| --- | ---: | --- |
| Complete backend call | 177.39 ms | 5.64 processed frames/sec before offline export and quality evaluation |
| Face detection and recognition | 7.80 ms | Includes retained InsightFace models and their preprocessing/postprocessing |
| Swap | 93.68 ms | Entire `INSwapper.get(..., paste_back=True)`, including host work |
| Restoration | 32.74 ms | GFPGAN input/output preparation and synchronous model call |
| Occlusion | 27.29 ms | XSeg input/output preparation and synchronous model call |
| Composite | 15.58 ms | A second crop blend and output color conversion |
| Offline quality measurement | 54.79 ms | Flow-aligned temporal proxy; separate from avatar inference |
| Two labeled video exports | 9.07 ms | Includes both offline output sinks |
| Decode | 1.15 ms | Sequential prerecorded input |

Complete offline throughput was **4.11 FPS**; mean per-frame total was 242.94 ms, with a 291.33 ms p95. Backend p95 was 218.39 ms. The video plays at its original 25 FPS because processing happens offline. Playback metadata is not processing throughput.

The sampled device memory peak was 4,408.5 MiB. This includes other device processes; process memory was unavailable. These results do not show a VRAM-capacity bottleneck or prove that replacing the A5000 with a 3090 would improve quality.

## What the model traces actually show

The 30-frame probe enabled ORT profiling. Session mapping was checked against the exact InsightFace 0.7.3 source archive, its SHA-256 published by PyPI, the sorted model loading order, the adapter construction order, the probe loading log, and active trace tensor shapes. The archive and source hashes are recorded in the [machine-readable audit](BASELINE_LATENCY_ANALYSIS.json).

| Session ordinal | Model | Role |
| --- | --- | --- |
| 0 | `buffalo_l/1k3d68.onnx` | Constructed, then ignored; no inference calls |
| 1 | `buffalo_l/2d106det.onnx` | Constructed, then ignored; no inference calls |
| 2 | `buffalo_l/det_10g.onnx` | Detector |
| 3 | `buffalo_l/genderage.onnx` | Constructed, then ignored; no inference calls |
| 4 | `buffalo_l/w600k_r50.onnx` | Recognition embedding |
| 5 | `inswapper_128.onnx` | Face swap |
| 6 | `gfpgan_1.4.onnx` | Restoration |
| 7 | `xseg_1.onnx` | Visible-face/occlusion mask |

For the detector and recognizer, one reference-preparation call was removed before pairing the remaining 30 calls with the 30 processed frames. The table excludes the first processed frame from every model to disclose the cold-start effect. This leaves a small sample of **29 frames**, not a long production steady-state benchmark.

| Completed ORT `Session/model_run` | Warm mean | Warm p95 |
| --- | ---: | ---: |
| Detector | 6.42 ms | 6.81 ms |
| Recognition | 5.25 ms | 5.36 ms |
| INSwapper | 13.87 ms | 13.96 ms |
| GFPGAN | 27.87 ms | 28.06 ms |
| XSeg | 19.89 ms | 21.00 ms |

The warm serial model-call durations total **73.30 ms**. Removing per-target recognition would leave **68.05 ms** of measured model-call durations in this probe, before alignment, tracking, mask handling, compositing, capture, and output. This is diagnostic arithmetic on the existing serial trace, not a hardware-independent lower bound.

INSwapper's first model call took 60.81 ms. Across all 30 frames, its model-call mean was 15.43 ms and enclosing swap-stage mean was 93.11 ms. The same-frame difference averaged **77.68 ms**; across the warm 29 frames it averaged **77.34 ms**. This approximate residual is outside the ORT call. Individual operations within it have not been timed, so attributing all of it to one warp, one matrix multiplication, or GPU transfers would be unsupported.

Executed INSwapper and GFPGAN graph-node events used CUDA. Detector and XSeg traces also contained CPU-assigned graph nodes. There are no device `Kernel` events in these profiles. Provider-labeled node counts demonstrate execution assignment; their duration sums do not measure pure GPU kernel time. ORT distinguishes host node traces from optional device kernel profiling in its [profiling documentation](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html). Here `model_run` is the synchronous ORT run span, including associated runtime/transfer/synchronization work, rather than neural computation alone.

Do not subtract this short probe's model timings from the full video's stage timings. Different frame selection, cold starts, and profiling overhead affect the two runs.

## Next profiling and optimization work

1. Split the swap stage into alignment/crop, source-latent preparation, model call, output decode, inverse warp, mask morphology, and final blend. The current wrapper hides these operations behind one call. Use matched inputs, warm-up disclosure, completed device work, and the same output-quality checks.
2. Cache the fixed reference's projected, normalized latent. Avoid recognition inference on every target frame when it is not needed for swapping, while retaining a deliberate identity-lock/reacquisition policy. The exact upstream implementation uses the source embedding for its swap latent and `FaceAnalysis.get` runs all retained models on target faces; see the [0.7.3 source distribution](https://pypi.org/project/insightface/0.7.3/) and corresponding [official INSwapper implementation](https://raw.githubusercontent.com/deepinsight/insightface/v0.7/python-package/insightface/model_zoo/inswapper.py).
3. Generate the aligned face patch and perform one bounded ROI composite into the original current frame. The existing upstream paste-back does full-frame work, followed by another crop composite in the adapter. The unused `fake_diff` branch is also a candidate for removal after output-equivalence checks. No individual saving is measured yet.
4. Start the final canvas from original pixels and explicitly constrain all generated changes to the validated face ROI. The frozen compositor starts from a fully swapped frame, so its later XSeg crop does not explicitly protect pixels outside that crop.
5. Keep offline flow analysis and comparison export outside the live rendering deadline. Use bounded capture/output queues with timestamps, drop stale work when necessary, and report frame age, drops, and delivered distinct processed frames. This removes offline overhead but does not solve the 177.39 ms backend by itself.
6. Evaluate GPU ROI warping/compositing, reduced precision, TensorRT or CUDA optimizations, and device-resident I/O only against pinned models and a quality regression set. A separate real-time profile may need a cheaper or confidence-gated restoration policy. Reusing an old occlusion mask during fast hand movement can create artifacts; mask refresh shortcuts require dedicated fast-hand testing.

## Twitching and the 30 FPS target

The frozen baseline caches source identity and validates increasing timestamps. It has no temporal landmark, pose, mask, color, or restoration filter. Source-based candidates for twitching include changing detector keypoints, integer bounding boxes moving the XSeg/GFPGAN crops, independent restoration texture, changing masks, and largest-face selection switching between people. These are hypotheses to measure, not confirmed visual attribution from this latency audit.

The polish plan should introduce tracked identity selection, timestamp-aware alignment smoothing with fast-motion/scene-cut resets, a stable aligned crop, current-frame mask validation with cautious flow-assisted history, and confidence-aware restoration/color parameters. Excessive smoothing can lag gestures or smear hands; visual regression checks must cover blinks, speech, turns, and fingers crossing the face.

At 30 FPS a new frame arrives every **33.33 ms**; at 60 FPS, every **16.67 ms**. The current backend mean needs about a **5.32×** reduction to fit the 30 FPS serial deadline. The complete offline benchmark would need about **7.30×** higher throughput. The warm model calls already exceed 33.33 ms, so source-latent caching and paste-back improvements alone cannot establish 30 FPS with every current model running sequentially on every frame.

A useful live result must show sustained distinct processed frames, a bounded queue and frame age, disclosed drops/fallbacks, and quality under fast motion. The current prepared input is 25 FPS; changing its export rate to 30, repeating frames, or interpolating frames does not validate a 30 FPS camera pipeline. Record a genuine 30 FPS test stream for that acceptance test. Any budget in the forward plan remains a target until measured.

Evidence remains unchanged in the sibling `backend_comparison` tree and `../backups/comparison-source-setup-20261014.tar.gz`; these locations are relative to the project root. The JSON uses semantic evidence IDs such as `full_baseline_summary`, `profile_probe_log` and `baseline_adapter_source`; locate each original file by its recorded SHA-256 and byte count. Session `profile_file` values are basenames paired with their original SHA-256. The IDs do not claim that any comparison directory, archive or archived source file was renamed. Baseline source hashes refer to frozen archived bytes, not the current `faceswap/` source.

Pre-rename local logs, outputs and test results, together with the original tracked-source snapshot `source_before_rename.tar.gz`, are preserved intact in `../backups/pipeline_history_before_rename_20261015/`. See `BACKUP_VERIFICATION.json` in the project root for snapshot identities and checksums. The local project rename is complete; the existing remote deployment has not been changed. These documents retain baseline findings, separate from the selected `faceswap_pipeline` runtime and its next deployment to `/workspace/wqs/faceswap_pipeline`.
