#!/usr/bin/env bash
# Repeatable quality regression; no GPU benchmark runs concurrently here.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/test_cycle.sh TAG quick|full [options]
  --restore-blend VALUE   Restoration blend (default 0.75)
  --profile-smoke         Profile only the 30-frame smoke for GPU placement
  --reference PATH       Reference portrait (default inputs/reference.png)
  --video PATH           Driving video (default inputs/driving_phone_720p25.mp4)
quick: unit tests, checkpoints, smoke, three difficult windows, export checks
full:  all quick checks followed by the complete input video and export check
USAGE
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if [[ $# -lt 2 ]]; then usage >&2; exit 2; fi
TASK_TAG="$1"
TASK_MODE="$2"
shift 2
if [[ ! "$TASK_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "TAG must begin with a letter/digit and contain only letters/digits/_/-." >&2
    exit 2
fi
if [[ "$TASK_MODE" != quick && "$TASK_MODE" != full ]]; then usage >&2; exit 2; fi
TASK_BLEND=0.75
TASK_REFERENCE=inputs/reference.png
TASK_VIDEO=inputs/driving_phone_720p25.mp4
TASK_SMOKE_FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile-smoke) TASK_SMOKE_FLAGS=(--profile); shift ;;
        --restore-blend|--reference|--video)
            if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --restore-blend) TASK_BLEND="$2" ;;
                --reference) TASK_REFERENCE="$2" ;;
                --video) TASK_VIDEO="$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
    echo "Install the project environment first: bash scripts/install.sh" >&2
    exit 1
fi
.venv/bin/python - "$TASK_BLEND" <<'PY'
import math, sys
value = float(sys.argv[1])
if not math.isfinite(value) or not 0 <= value <= 1:
    raise ValueError('Restoration blend must be finite and between 0 and 1')
PY
TASK_RUN="outputs/${TASK_TAG}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p outputs
mkdir "$TASK_RUN"
printf 'Test folder: %s/%s\n' "$PWD" "$TASK_RUN"
printf 'tag=%s\nmode=%s\nrestore_blend=%s\nreference=%s\nvideo=%s\nprofile_smoke=%s\n' \
    "$TASK_TAG" "$TASK_MODE" "$TASK_BLEND" "$TASK_REFERENCE" "$TASK_VIDEO" \
    "${TASK_SMOKE_FLAGS[*]:-off}" > "$TASK_RUN/parameters.txt"
tar --exclude='./models' --exclude='./inputs' --exclude='./outputs' \
    --exclude='./.venv' --exclude='./.git' --exclude='./logs' --exclude='./test_results' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.part' \
    -czf "$TASK_RUN/code_snapshot.tar.gz" .
sha256sum "$TASK_RUN/code_snapshot.tar.gz" "$TASK_REFERENCE" "$TASK_VIDEO" \
    > "$TASK_RUN/input_and_code.sha256"
.venv/bin/python -m pip freeze > "$TASK_RUN/environment.lock.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv \
    > "$TASK_RUN/hardware.csv"
.venv/bin/python -m unittest discover -s faceswap/tests -v 2>&1 | tee "$TASK_RUN/unit_tests.log"
.venv/bin/python scripts/fetch_checkpoints.py --verify-only 2>&1 | tee "$TASK_RUN/checkpoints.log"

run_case() {
    local TASK_CASE="$1"
    shift
    printf '\nRunning %s\n' "$TASK_CASE"
    PYTHONUNBUFFERED=1 bash scripts/run_gpu.sh \
        --reference "$TASK_REFERENCE" --video "$TASK_VIDEO" \
        --restore-blend "$TASK_BLEND" --output "$TASK_RUN/$TASK_CASE" "$@" \
        2>&1 | tee "$TASK_RUN/$TASK_CASE.log"
    .venv/bin/python scripts/verify_export.py "$TASK_RUN/$TASK_CASE" \
        2>&1 | tee "$TASK_RUN/${TASK_CASE}_verification.log"
    .venv/bin/python - "$TASK_RUN/$TASK_CASE" "$TASK_CASE" <<'PY'
import json, pathlib, sys
root, name = pathlib.Path(sys.argv[1]), sys.argv[2]
s = json.loads((root / 'summary.json').read_text())
errors = {r:n for r,n in s.get('reason_counts', {}).items() if r.endswith('_error')}
if errors:
    raise RuntimeError(f'{name}: processing errors returned original frames: {errors}')
if name == 'smoke' and (s['frames'] != 30 or s['status_counts'].get('animated', 0) == 0):
    raise RuntimeError('Smoke requires 30 input frames and at least one synthetic frame')
PY
}
run_case smoke --max-frames 30 "${TASK_SMOKE_FLAGS[@]}"
run_case eyes_nose --start 30 --end 40
run_case head_turn --start 47 --end 57
run_case mouth_hand --start 83 --end 93
if [[ "$TASK_MODE" == full ]]; then run_case full; fi
.venv/bin/python - "$TASK_RUN" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cases = []
for name in ('smoke', 'eyes_nose', 'head_turn', 'mouth_hand', 'full'):
    path = root / name / 'summary.json'
    if not path.exists():
        continue
    s = json.loads(path.read_text())
    v = json.loads((path.parent / 'export_verification.json').read_text())
    if s.get('complete') is not True or v.get('complete_exports') is not True:
        raise RuntimeError(f'{name}: incomplete or unverified export')
    cases.append({'case': name, 'frames': s['frames'], 'status_counts': s['status_counts'],
                  'error': s.get('error'), 'reason_counts': s.get('reason_counts'),
                  'end_to_end_fps': s['end_to_end_fps'], 'backend_fps': s['inference_only_fps'],
                  'backend_latency_ms': s['latency_ms']['backend_ms'],
                  'stages_ms': s['stages_ms'], 'gpu_memory_peak_mib': s['gpu_memory_peak_mib']})
result = {'automated_checks_passed': True, 'visual_review_required': True, 'cases': cases,
          'limitations': 'Wrapper tests and export checks do not establish visual quality. Window throughput includes preroll/setup effects; compare full runs for end-to-end speed. Smoke profiling affects timings. Fresh-state windows cannot replace full temporal review.'}
(root / 'cycle_summary.json').write_text(json.dumps(result, indent=2) + '\n')
print('\nAll automated checks passed. Inspect comparison.mp4 in each case.')
for c in cases:
    print(f"{c['case']}: {c['frames']} frames; backend {c['backend_fps']:.2f} FPS; statuses {c['status_counts']}")
PY
printf '\nResults and code snapshot: %s/%s\n' "$PWD" "$TASK_RUN"
