# HMZ-FSWP

The selected face-swap pipeline: source code, tests, requirements, model download manifests, and verification tools.

Start with [`faceswap_pipeline/README.md`](faceswap_pipeline/README.md) for installation, GPU execution, and the frame API. [`ACTIVE_PROJECT.json`](ACTIVE_PROJECT.json) records the selected local project and intended remote deployment path. The existing remote deployment has not been migrated.

## Resources and setup

Model weights, virtual environments, input photos/videos, generated outputs, logs, and backup archives are excluded by `.gitignore` and remain local. Other workspace projects are also excluded.

Follow the pipeline README to install dependencies and fetch checkpoints with [`fetch_checkpoints.py`](faceswap_pipeline/scripts/fetch_checkpoints.py). Model URLs and integrity metadata are retained in [`checkpoints.json`](faceswap_pipeline/checkpoints.json) and [`model_manifest.json`](faceswap_pipeline/model_manifest.json). Supply your own input media using the paths described there.

The pipeline targets NVIDIA GPU environments. Consult its README for Python, CUDA, FFmpeg, and platform requirements. Historical references to local comparison results and backups describe evidence outside this repository.

Earlier local logs, outputs and test results are preserved unchanged in `backups/pipeline_history_before_rename_20261015/`. The active project uses the `faceswap` Python package and `FaceSwapBackend` frame API.
