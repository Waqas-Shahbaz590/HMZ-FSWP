#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Optional library locations are supplied by the deployment environment.
# Vast's tested image exposes CUDA 12/cuDNN libraries in its base Python env.
TASK_CUDA_PATH="${FACESWAP_CUDA_LIB_DIRS:-}"
if [[ -z "$TASK_CUDA_PATH" && -x /venv/main/bin/python ]]; then
    TASK_CUDA_PATH=$(/venv/main/bin/python -c 'import glob; print(":".join(glob.glob("/venv/main/lib/python3.12/site-packages/nvidia/*/lib")))')
fi
export LD_LIBRARY_PATH="$TASK_CUDA_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec .venv/bin/python run.py "$@"
