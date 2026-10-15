#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m unittest discover -s faceswap/tests -v
.venv/bin/python scripts/fetch_checkpoints.py --verify-only
bash scripts/run_gpu.sh --reference inputs/reference.png --video inputs/driving_phone_720p25.mp4 --output outputs/extraction_smoke --start 31.24 --max-frames 30 --profile
.venv/bin/python scripts/verify_export.py outputs/extraction_smoke
