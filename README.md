# HMZ-FSWP

The selected Afnan face-swap pipeline: source code, tests, requirements, model download manifests, and verification tools.

Start with [`afnan_pipeline/README.md`](afnan_pipeline/README.md) for installation, GPU execution, and the frame API. [`ACTIVE_AFNAN_PROJECT.json`](ACTIVE_AFNAN_PROJECT.json) records the selected project and its original workspace locations.

## Resources and setup

Model weights, virtual environments, input photos/videos, generated outputs, logs, and backup archives are excluded by `.gitignore` and remain local. Other workspace projects are also excluded.

Follow the pipeline README to install dependencies and fetch checkpoints with [`fetch_checkpoints.py`](afnan_pipeline/scripts/fetch_checkpoints.py). Model URLs and integrity metadata are retained in [`checkpoints.json`](afnan_pipeline/checkpoints.json) and [`model_manifest.json`](afnan_pipeline/model_manifest.json). Supply your own input media using the paths described there.

The pipeline targets NVIDIA GPU environments. Consult its README for Python, CUDA, FFmpeg, and platform requirements. Historical references to local comparison results and backups describe evidence outside this repository.
