#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_PYTHON="${FACESWAP_PYTHON:-/venv/main/bin/python}"
if [[ -e .venv ]]; then
    echo "Existing .venv preserved. Use a fresh project directory for installation." >&2
    exit 1
fi
"$TASK_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
"$TASK_PYTHON" -m venv .venv
if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python pip setuptools wheel numpy==1.26.4 cython==3.0.11
    uv pip install --python .venv/bin/python --no-build-isolation -r requirements.txt
else
    .venv/bin/python -m pip install pip setuptools wheel numpy==1.26.4 cython==3.0.11
    .venv/bin/python -m pip install --no-build-isolation -r requirements.txt
fi
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > environment.lock.txt
.venv/bin/python scripts/fetch_checkpoints.py
