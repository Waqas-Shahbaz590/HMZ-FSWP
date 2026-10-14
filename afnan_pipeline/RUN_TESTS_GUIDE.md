# Repeated pod testing after code changes

Edit the selected project at `D:\Hamza\Face Swap\wqs\afnan_pipeline`. Its deployed Linux directory is `/workspace/wqs/afnan_pipeline`. Use the already installed `.venv` for ordinary Python edits. `scripts/install.sh` is for a fresh environment and deliberately refuses to overwrite an existing one.

## Upload edits from Windows

For an individual edited file, use PowerShell:

```powershell
scp -i C:/Users/hd/.ssh/id_ed25519 -P 42051 "D:/Hamza/Face Swap/wqs/afnan_pipeline/afnan/adapter.py" root@167.179.138.57:/workspace/wqs/afnan_pipeline/afnan/
```

Change the source/destination to the file you edited. For example, `common/io.py` belongs under the remote `common/` directory. New source modules must be uploaded too. Keep remote models, inputs, `.venv`, and previous outputs available. A completed upload should be tested as one code version; wait for that cycle to finish before uploading the next change. File-copy uploads do not mirror local deletions, so a major restructuring should use a fresh release directory or explicitly review obsolete remote files.

Upload the repeat-test helper once, or after editing it:

```powershell
scp -i C:/Users/hd/.ssh/id_ed25519 -P 42051 "D:/Hamza/Face Swap/wqs/afnan_pipeline/scripts/test_cycle.sh" root@167.179.138.57:/workspace/wqs/afnan_pipeline/scripts/
```

Connect using the last supplied address (substitute a new SSH command if Vast changes it):

```powershell
ssh -i C:/Users/hd/.ssh/id_ed25519 -p 42051 root@167.179.138.57
```

## Run the cycle on the pod

```bash
cd /workspace/wqs/afnan_pipeline
bash scripts/test_cycle.sh hand_mask_v1 quick
```

`quick` runs the 14 focused wrapper tests, checkpoint verification, a 30-frame frontal smoke at the beginning, then the 30–40 second eye/nose clip, 47–57 second head-turn clip and 83–93 second mouth/hand clip. Each export is independently verified. Wrapper/export failures and reported processing errors stop the cycle, preserving its logs and any partial evidence. A smoke containing zero synthetic frames also fails. Missing-face fallbacks remain recorded for visual review. Every invocation creates a unique folder such as `outputs/hand_mask_v1_20261015T090000Z_1234`; previous runs are preserved. The UTC timestamp in that name identifies the run, not the local clock.

Inspect the quick comparisons, then promote that unchanged version to a full regression:

```bash
bash scripts/test_cycle.sh hand_mask_v1 full
```

`full` repeats the quick checks and processes all 3,076 supplied frames. It verifies the full result and comparison too. On the measured baseline, the full video took about 12.5 minutes of processing; the complete cycle also includes the short cases and verification. Actual time depends on the change and current machine.

For a run that continues after your SSH connection closes:

```bash
mkdir -p logs
nohup bash scripts/test_cycle.sh hand_mask_v1 full > logs/hand_mask_v1_full.log 2>&1 < /dev/null &
tail -f logs/hand_mask_v1_full.log
```

The log prints the unique output folder. `Ctrl+C` stops watching `tail`; the background job continues. Wait for its success/failure before another measured GPU cycle or code upload.

For a restoration-strength experiment:

```bash
bash scripts/test_cycle.sh restore_025 quick --restore-blend 0.25
```

When model/runtime/provider code changes, include a short CUDA node profile:

```bash
bash scripts/test_cycle.sh gpu_runtime_v1 quick --profile-smoke
```

Profiling applies only to the smoke. Other clips and the full run remain unprofiled. Instrumented smoke speed is not a performance benchmark. The old `scripts/smoke_extraction.sh` uses the already existing `outputs/extraction_smoke` directory; use the new cycle helper for repeat runs.

For another portrait/video:

```bash
bash scripts/test_cycle.sh new_input quick --reference inputs/another_reference.png --video inputs/another_driving.mp4
```

The quick windows assume that video covers the selected timestamps. The supplied fixed inputs provide comparable regression tests. Use a different purpose-built set of bounds for a shorter clip, through `scripts/run_gpu.sh --start/--end/--max-frames` and `scripts/verify_export.py`.

## Results to review

Each cycle retains a code snapshot, input/code SHA256 values, actual dependency list, GPU/driver record, parameters and logs. It writes `cycle_summary.json` only after all requested automated cases pass. Each case contains `comparison.mp4`, `synthetic_avatar.mp4`, `summary.json`, `frame_metrics.csv` and `export_verification.json`.

Review the videos at normal speed and slowly around the same source timestamps in the previous version. Look for fingertips becoming beard/skin, mask trails on hand entry/removal, cheek/jaw restoration seams, scale/position jumps, changes to teeth/eyes and delayed blinks or talking. Inspect fallbacks at 32–38 seconds explicitly. The current baseline's five original-frame fallbacks are recorded separately from synthetic frames.

Check complete backend mean/p95 latency, per-stage latency, fallback reasons, model errors and sampled VRAM alongside FPS. A change that returns more original frames can appear faster without improving synthesis. Short windows begin with fresh tracking state; a complete run is required to assess continuity through those windows. End-to-end window speed includes discarded preroll decoding, so use paired full runs for end-to-end speed comparisons. Temporal proxy improvements can reflect extra blur, and automated export checks do not judge identity or perceptual quality.

Run one measured GPU cycle at a time, keep portrait/video/settings fixed when comparing changes, and change one component per experiment. Always retain the code snapshot and full summary when accepting a version. The snapshot captures new modules beyond the runner's fixed provenance list.

## Download a cycle to Windows

Use the exact run folder printed at completion:

```powershell
New-Item -ItemType Directory -Force "D:/Hamza/Face Swap/wqs/afnan_pipeline/test_results" | Out-Null
scp -i C:/Users/hd/.ssh/id_ed25519 -P 42051 -r root@167.179.138.57:/workspace/wqs/afnan_pipeline/outputs/REPLACE_WITH_PRINTED_RUN_FOLDER "D:/Hamza/Face Swap/wqs/afnan_pipeline/test_results/"
```

This brings back videos, metrics, verification receipts, logs and the code used for that cycle. Download accepted versions before disposing of the ephemeral pod. Run-folder examples in this guide are examples, not existing results.

## Dependencies and new pods

Ordinary source edits need no environment reinstall. If dependencies change, resolve them deliberately, run `.venv/bin/python -m pip check`, retain a fresh freeze and use the short `--profile-smoke` check before comparing performance. Keep the CUDA 12/NumPy 1.26 environment separate from FaceFusion's environment. `requirements.txt` is the direct dependency specification; `environment.lock.txt` records the installed versions.

A replacement pod needs the clean project restored, its NVIDIA/framework runtime inspected, and `bash scripts/install.sh` run once in a fresh directory. The installer fetches and verifies pretrained checkpoints if absent. Inputs must be restored as well. A changed SSH address must replace the example address above.

The current input is 25 FPS. These cycles test quality and offline processing. Proving a live 30 FPS feed requires genuine fresh 30 FPS input, sustained required-path processing within 33.3 ms, measured capture-to-display delay, and separate captured/inferred/dropped/fallback counts as specified in `POLISH_PLAN.md`.
