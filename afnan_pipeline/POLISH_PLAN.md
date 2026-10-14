# Afnan polish and live-frame plan

Use `D:\Hamza\Face Swap\wqs\afnan_pipeline` as the selected project. Its extracted runtime is the starting baseline. Keep the completed Afnan/FaceFusion comparison, original AFW backup and existing videos intact so each change can be evaluated against the same evidence. The work below is planned; the extraction alone does not claim these improvements are implemented.

## Starting measurements and the goal

The current Afnan backend processed all 3,076 supplied frames on the RTX A5000 without a model exception: 3,071 synthetic frames and five explicit original-frame fallbacks. It uses native INSwapper 128, GFPGAN 1.4 at blend 0.75 and bbox-based XSeg1. Full measurements and sampled failed boundaries are recorded in [TEAM_B_REPORT.md](<D:/Hamza/Face Swap/wqs/backend_comparison/TEAM_B_REPORT.md>).

| Current stage | Mean completed wall time |
|---|---:|
| Detection and target recognition | 7.80 ms |
| INSwapper call, including crop/warp/paste-back | 93.68 ms |
| GFPGAN restoration | 32.74 ms |
| XSeg1 occlusion | 27.29 ms |
| Compositing and conversions | 15.58 ms |
| Complete backend call | 177.39 ms; p95 218.39 ms |

Backend throughput is 5.637 FPS. The complete decode, two-encoder export, logging and temporal-review path is 4.110 FPS. These are completed full-run measurements.

The existing 30-frame instrumented probe gives a useful decomposition: after excluding its first cold frame, the INSwapper ONNX call averaged **13.87 ms**, while the same frames spent another **77.34 ms** inside the aggregate swap stage outside that call. This points first to crop/latent preparation, conversions and full-frame warp/paste work, rather than treating the whole 93 ms stage as GPU model inference. These probe numbers may include profiling overhead and are not replacements for the full-run measurements. The detailed mapping and evidence are in [BASELINE_LATENCY_ANALYSIS.md](<D:/Hamza/Face Swap/wqs/afnan_pipeline/audits/BASELINE_LATENCY_ANALYSIS.md>).

In that warm probe, completed model calls averaged detector 6.42 ms, target recognition 5.25 ms, swap 13.87 ms, GFPGAN 27.87 ms and XSeg 19.89 ms: **73.30 ms total**, or **68.05 ms** even after hypothetically removing target recognition. Therefore removing paste-back overhead alone cannot establish 30 FPS with these serial model calls. XSeg also has neural nodes on the CPU; requesting CUDA does not make the entire graph run there.

**Use 30 genuinely processed frames per second as the first live-camera target.** It is a sensible target for natural talking and hand movement; 25 FPS can also look natural, while 60 FPS is an optional later target for smoother fast motion. Required budgets are 33.3 ms at 30 FPS and 16.7 ms at 60 FPS. The current mean backend needs about **5.32×** improvement to reach the 30 FPS frame budget. A 30 FPS MP4 or repeated display frames are not evidence that the models process 30 fresh camera frames per second.

The supplied comparison input is only 25 FPS. Use it for quality regression, then use a genuine timestamped 30 FPS capture or supplied 30 FPS test recording for rate validation. Report captured, received, inferred, fallback, dropped and displayed frame counts separately.

For quality alone, changing this A5000 to a 3090 is not expected to cure the observed finger/cheek artifacts when weights, precision, face resolution and blending stay the same. The baseline sampled about 4.31 GiB of device memory on a device reporting 23,028 MiB, so it was not forced into this variant by an observed memory shortage. This is a configuration-based inference, not a controlled cross-GPU quality experiment. Improve the masks, alignment and restoration first; compare hardware with matched software/precision only if a measured capacity or performance need remains.

## 1. Keep the working project clean and reproducible

The live algorithm starts in `afnan/adapter.py`; ONNX tensor conversion is in `afnan/tensors.py`; restoration/mask blending is in `afnan/compositing.py`. `run.py` and `common/` supply file input, timestamps, labeled output and evidence. Models and inputs are project data; virtual environments are rebuilt from the pinned installer/lock.

Preserve checkpoint/source/input hashes, actual provider lists, precision settings, detector settings and restoration strength per run. Store each experiment in its own output directory. Add new `tracking.py`, `alignment.py` and `occlusion.py` modules as their phases are implemented rather than adding camera capture or display code to the backend. Keep a small model interface so an approved alternate swapper can be tested without replacing tracking and composition.

Acceptance: the isolated project imports and runs without the old AFW application or any FaceFusion/LivePortrait checkout; its initial GPU smoke export passes decoded frame/timestamp/no-audio/disclosure checks. A baseline rerun should use the same weights/settings and preserve its five-frame fallback behavior, allowing minor platform numerical differences.

## 2. Fix composition, restoration and fingers first

1. Obtain the generated aligned crop and transform from INSwapper instead of committing its full-frame paste before occlusion. Run swap, restoration and visible-face masks in a consistent face coordinate system. Initialize the output from the original frame and perform one final paste under a validated mask. This makes the protected region explicit and avoids residual swapped pixels outside the protected ROI.
2. Replace square-resized bbox restoration with landmark-aligned restoration using a template appropriate to GFPGAN. Map the restored crop and mask through the same inverse transform. Compare blend **0, 0.25, 0.50 and 0.75**. Sharpness alone must not select a winner: inspect identity detail, beard edges, profile geometry and consistency during motion.
3. Compute XSeg from the original aligned current frame. Add independent hand landmarks/segmentation for the difficult eye/nose/mouth overlaps. Treat hand silhouettes as protection regions; landmarks alone cannot recover a full silhouette. Evaluate a lightweight hand mask first, with SAM2 as a slower quality reference or an occlusion-triggered option after timing it.
4. Make current hand protection conservative near fingertips and narrow gaps. Suppress generated alpha wherever the current hand mask requires original pixels; feather inward on the face side so blur does not spread generated beard/skin onto fingers. Preserve the original foreground pixels in the protected region.
5. Add confidence gates for masks, tracking and transforms. Large uncertainty, rejected hand segmentation or an implausible transform returns a labeled original frame; a finite XSeg tensor by itself is not proof of a correct mask. Test profiles and rapid hand entries explicitly.

Acceptance: no introduced fingertip/eye/mouth leakage on the fixed annotated keyframes, no square restoration boundary on the reviewed cheek/profile samples, and no unexplained change outside the chosen face/feather region. Review both stills and motion. Record any remaining failures and quality-versus-fallback tradeoff rather than declaring the clip artifact-free.

## 3. Address twitching at its source

Twitching can come from a shifting crop, noisy landmarks, switching target faces, an unstable occlusion boundary, restoration texture changes or source-identity drift. These require different remedies; a higher-resolution swapper does not automatically fix them.

| Symptom | Planned treatment | Guard against |
|---|---|---|
| Whole face jumps or changes scale | Associate the same face over time; filter crop center, scale and alignment landmarks with timestamp-aware adaptive smoothing | Lag during a real fast head turn |
| Mouth/eyelids feel delayed | Keep current expression pixels; smooth rigid placement more strongly than expression-sensitive geometry | Smearing speech, blinks or teeth |
| Fingers leave trails or suddenly become beard | Current-frame hand protection takes precedence; use only bounded, confidence-gated history | Reusing an old mask after a new hand enters |
| Edge/shading shimmers | Stabilize aligned mask and measured color correction; smooth color parameters rather than whole images | Keeping old lighting after a real change |
| Face looks different each frame | Cache the source embedding, control restoration strength, compare restored versus un-restored aligned output | Mistaking oversharpening for identity fidelity |

Start with a causal **One Euro filter** on normalized rigid geometry. Its cutoff adapts to signal speed, suppressing slow-motion jitter while reducing lag during faster motion. Tune against the measured timestamp spacing, beginning with the author's tuning procedure; parameters depend on units, so there is no universal useful beta. Reset after a scene cut, target change, excessive timestamp gap or lost track. [Original filter and verified implementations](https://gery.casiez.net/1euro/).

For mask continuity, evaluate a past mask warped into the current frame by optical flow, initially capped at **80 ms** of age. This cap is an experimental parameter, not a validated safe threshold. Current hand evidence always overrides history; reject history when flow confidence is weak, the pose changes sharply or the hand enters a new region. Do not average previous RGB output over eyes/mouth or reuse stale detections for 15 frames.

Acceptance: compare filtered/unfiltered versions on a stationary face, fast head turn, rapid blink/talking and hand entry/removal. Measure rigid alignment jitter and lag, mask disagreement in annotated hand regions, and limited flow-aligned temporal proxies. A lower temporal L1 score can reflect blur or reduced motion, so paired video review and expression retention remain required.

## 4. Evaluate higher-detail models without mislabeling them

The current checkpoint's native input/output is **128 × 128**. Resizing its result to 512 does not make it a native 512 model. The official InsightFace example documents this 128 resolution and its Buffalo L identity embedding requirement. [Official INSwapper example](https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md).

There is an official **INSwapper-512-live** project describing native 512 × 512 base/mini models. Its public repository currently presents model information and application links rather than a drop-in ONNX checkpoint/SDK in our project, and states academic/personal-use restrictions. Investigate legitimate checkpoint/SDK access, integration requirements and deployment rights before planning a replacement. Its publisher's performance/quality figures are not measurements of this pipeline. [Official 512-live project](https://github.com/deepinsight/inswapper-512-live).

FaceFusion's **512 pixel boost using INSwapper 128** is another technique: the pinned implementation splits the aligned 512 crop into **16 interleaved 128-sized inputs**, performs a swap for each, then reconstructs it. That is a multi-pass resolution experiment, distinct from native 512 inference; expect extra work and measure actual latency. Do not bring the entire FaceFusion application into the selected Afnan runtime just to test this idea. [Pinned pixel-boost implementation](https://github.com/facefusion/facefusion/blob/72470819a0373be3388b3929c8f8f311f418fc3c/facefusion/processors/pixel_boost.py), [pinned swap loop](https://github.com/facefusion/facefusion/blob/72470819a0373be3388b3929c8f8f311f418fc3c/facefusion/processors/modules/face_swapper/core.py).

A native 256 alternative or an available 512 model from a different family can be tested behind the same backend interface. It requires its own embedding/template/preprocessing contract and license record. Evaluate identity likeness, expressions, fingers, profiles and latency on the same inputs. High resolution is useful for fine detail, but cannot correct a wrong occlusion mask or inconsistent placement.

## 5. Build a measured path toward 30 FPS

First split timers into detector-only inference, target recognition, crop/warp, source-latent preparation, swap session, restore session, occlusion session, paste, conversions and device transfers. Profile actual ONNX node placement. The existing probe already identifies a substantial non-session portion of the swap call; benchmark one protected ROI paste rather than repeated full-frame warps, and cache the fixed source latent as well as its identity embedding. Remove redundant per-frame target recognition when the selected swapper only needs the prepared source embedding and current detection/landmarks; keep a separately measured identity/association check when needed for track acquisition, reacquisition or multiple people. Validate each change against the original output; source recognition remains necessary.

Then benchmark, individually, FP16-compatible swap weights, optimized ONNX/TensorRT execution where the required graph is supported, CUDA I/O binding, persistent device buffers and GPU crop/warp/composition. Check provider/runtime compatibility and fallback partitions for each model rather than assuming TensorRT supports the complete pipeline. Move repeated RGB/BGR conversions and large CPU/device copies out of the path. Runtime/precision changes need quality regression; neither FP16 nor TensorRT is a guaranteed acceleration or quality improvement. [ONNX Runtime I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html), [FP16/mixed precision](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html), [TensorRT provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

Keep the offline review/export path separate from the eventual live delivery path. A live stream needs one labeled output sink, while duplicate MP4 encoding and expensive review metrics can run asynchronously on sampled frames. This does not remove required tracking, hand protection or synthetic disclosure from live inference. Only reduce restoration or mask complexity after a paired quality test identifies an acceptable live variant.

After the crop/transfer changes are measured, test concurrent independent branches: current-frame occlusion can run while swap/restoration is computed. Preserve the dependency from swap to restoration and perform the final paste only when both results for that timestamp are ready. Measure contention and synchronization; overlap may improve throughput but does not make the measured 68 ms of serial model work disappear. Even removing GFPGAN and target recognition leaves approximately 40.18 ms of the current warm model calls before composition, so masks/runtime/precision or hardware still need measured improvement for that variant.

| Tentative 30 FPS engineering allocation | Target |
|---|---:|
| Current detection/causal tracking | ≤ 4 ms |
| Swap inference | ≤ 15 ms |
| Restoration, if required in the live variant | ≤ 5 ms |
| Occlusion/hand protection | ≤ 4 ms |
| Crop, transfers and final composition | ≤ 3 ms |
| Labeled frame delivery/remaining overhead | ≤ 2.3 ms |
| Total | ≤ 33.3 ms |

These are design allocations, not achieved benchmarks. If the selected swapper cannot meet its budget after profiling/optimization, compare a faster approved native model or different hardware and remeasure; piling extra models into the path will not solve the budget. More GPUs can improve independent-stream capacity but do not automatically accelerate one causally ordered frame stream.

For live input, use a bounded queue of at most one pending frame and record every skipped frame. A latest-frame policy keeps delay bounded under overload, but frame dropping must not be reported as sustained 30 FPS inference. Carry capture timestamps through the pipeline; keep one ordered state update per stream. Adaptive detector cadence is a later option only if current tracking/hand evidence remains reliable and redetection is triggered by motion, occlusion or confidence loss.

Acceptance: after warm-up, a ten-minute genuine 30 FPS test sustains 30 processed fresh frames per second, p95 required-path latency ≤ 33.3 ms, no growing queue and explicit drop/fallback statistics. Use a provisional capture-to-disclosed-display p95 target ≤ 100 ms and measure it separately, including transport if remote. Tests must include continuous talking, profiles and repeated hands crossing eyes/nose/mouth. A mean FPS result alone does not meet this gate.

## Controlled experiments and promotion

| Experiment | Keep fixed | Decide from |
|---|---|---|
| Restoration 0 / .25 / .50 / .75 | Current swapper and masks | Identity detail, profile boundaries, motion flicker, stage time |
| Aligned restoration + single protected paste | Same weights/blend | Finger leakage and restoration seams |
| Geometry filter off/on | Same masks/model | Stillness jitter versus head-turn/expression lag |
| XSeg only versus independent hand protection | Same swap/restoration | Annotated foreground preservation and fallbacks |
| Native 128 versus approved native alternative / pixel boost | Same aligned composition | Detail, likeness, expression, hands, p95 time |
| FP32 versus validated FP16/optimized execution | Same model family and inputs | Paired artifacts, actual GPU placement, latency/memory |

Start with the existing 30–40 s eye/nose occlusion, 47–57 s head-turn and 83–93 s mouth/hand windows; retain full absolute timestamps. Promote a variant only after these pass, then rerun all 3,076 frames and inspect the full video. Keep video labels, input/source/checkpoint hashes, parameter files, frame metrics and independently decoded export receipts. Download the chosen source/model/result archives and verify their checksums before considering the remote pod disposable.

The immediate order is **clean extraction → aligned composition/restoration → hand protection → causal jitter control → model/precision ablations → live-path profiling and 30 FPS validation**. Quality work starts with the observed failures, and the 30 FPS claim is made only when the complete chosen live variant passes the measured gate.
