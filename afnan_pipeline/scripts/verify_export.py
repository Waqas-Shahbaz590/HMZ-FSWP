"""Independently verify finalized, silent synthetic-avatar comparison exports.

Usage: python scripts/verify_export.py /path/to/output_dir
The band check samples pixels; it does not recognize disclosure text or prove
that the subject's appearance is correct. Frame counts come from ffprobe's
decoder, independently of the inference runner and its writer counters.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any


VIDEO_NAMES = ("synthetic_avatar.mp4", "comparison.mp4")


class VerificationError(ValueError):
    """A completed export does not agree with its metrics or disclosure contract."""


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise VerificationError(f"{label} must be a positive integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} must be a positive integer") from error
    if result <= 0:
        raise VerificationError(f"{label} must be positive")
    return result


def finite_number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise VerificationError(f"{label} must be finite")
    return result


def _first(mapping: dict[str, Any], names: tuple[str, ...]) -> Any:
    return next((mapping[name] for name in names if name in mapping), None)


def validate_rows(summary: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    if summary.get("complete") is False or summary.get("error") or summary.get("finalization_errors"):
        raise VerificationError("Summary records an incomplete run or processing/finalization failure")
    expected = positive_integer(summary.get("frames"), "summary.frames")
    if len(rows) != expected:
        raise VerificationError(f"CSV has {len(rows)} rows, summary reports {expected} frames")
    for name in ("output_frames", "comparison_frames"):
        if name in summary and positive_integer(summary[name], f"summary.{name}") != expected:
            raise VerificationError(f"summary.{name} does not match summary.frames")
    indices: list[int] = []
    timestamps: list[float] = []
    for position, row in enumerate(rows):
        raw_index = _first(row, ("frame_index", "index", "source_frame_index", "source_index"))
        try:
            index = int(str(raw_index))
        except (TypeError, ValueError) as error:
            raise VerificationError(f"CSV row {position} has no integer frame index") from error
        timestamp = finite_number(_first(row, ("timestamp_ms", "pts_ms")), f"CSV row {position} timestamp")
        if index < 0 or timestamp < 0:
            raise VerificationError(f"CSV row {position} has a negative index or timestamp")
        if indices and index <= indices[-1]:
            raise VerificationError(f"CSV frame indices are not strictly increasing at row {position}")
        if timestamps and timestamp <= timestamps[-1]:
            raise VerificationError(f"CSV timestamps are not strictly increasing at row {position}")
        indices.append(index)
        timestamps.append(timestamp)
    return {
        "frames": expected,
        "first_source_frame_index": indices[0],
        "last_source_frame_index": indices[-1],
        "first_timestamp_ms": timestamps[0],
        "last_timestamp_ms": timestamps[-1],
        "status_counts": dict(Counter(row.get("status", "unspecified") for row in rows)),
        "strictly_increasing_frame_indices": True,
        "strictly_increasing_timestamps": True,
    }


def _probe_video(probe: dict[str, Any], name: str, expected: int) -> dict[str, Any]:
    streams = probe.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise VerificationError(f"{name} must contain exactly one video stream and no audio")
    stream = streams[0]
    frames = positive_integer(stream.get("nb_read_frames"), f"{name} decoded frames")
    if frames != expected:
        raise VerificationError(f"{name} decoded {frames} frames, expected {expected}")
    width = positive_integer(stream.get("width"), f"{name} width")
    height = positive_integer(stream.get("height"), f"{name} height")
    if width % 2 or height % 2:
        raise VerificationError(f"{name} has odd MP4 dimensions")
    try:
        fps = float(Fraction(str(stream.get("avg_frame_rate"))))
    except (ValueError, ZeroDivisionError) as error:
        raise VerificationError(f"{name} has no usable average FPS") from error
    if not math.isfinite(fps) or fps <= 0:
        raise VerificationError(f"{name} has no usable average FPS")
    duration_value = stream.get("duration")
    if duration_value in (None, "N/A"):
        duration_value = probe.get("format", {}).get("duration")
    duration = finite_number(duration_value, f"{name} duration")
    expected_duration = expected / fps
    tolerance = 1.0 / fps + 0.02
    if duration <= 0 or abs(duration - expected_duration) > tolerance:
        raise VerificationError(f"{name} duration {duration} disagrees with {expected} frames at {fps} FPS")
    return {
        "codec": stream.get("codec_name"), "decoded_frames": frames,
        "width": width, "height": height, "fps": fps,
        "duration_seconds": duration, "audio_streams": 0,
    }


def validate_probes(summary: dict[str, Any], probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = positive_integer(summary.get("frames"), "summary.frames")
    videos = {name: _probe_video(probes[name], name, expected) for name in VIDEO_NAMES}
    avatar, comparison = (videos[name] for name in VIDEO_NAMES)
    if comparison["width"] != 2 * avatar["width"] or comparison["height"] != avatar["height"]:
        raise VerificationError("Comparison dimensions must contain two full-size side-by-side panels")
    if not math.isclose(avatar["fps"], comparison["fps"], rel_tol=1e-6, abs_tol=1e-6):
        raise VerificationError("Synthetic-avatar and comparison frame rates differ")
    metadata = summary.get("video", summary.get("output", {}))
    for key, aliases in (("width", ("output_width", "width")), ("height", ("output_height", "height"))):
        value = _first(metadata, aliases)
        if value is not None and positive_integer(value, f"summary.video.{key}") != avatar[key]:
            raise VerificationError(f"Summary output {key} disagrees with decoded video")
    fps_value = _first(metadata, ("output_fps", "fps"))
    if fps_value is not None:
        fps = finite_number(fps_value, "summary.video.fps")
        if not math.isclose(fps, avatar["fps"], rel_tol=1e-5, abs_tol=1e-5):
            raise VerificationError("Summary output FPS disagrees with decoded video")
    return videos


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_disclosure_bands(path: Path, frames: int, *, comparison: bool) -> dict[str, Any]:
    """Sample expected opaque disclosure borders; no OCR or quality claim."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VerificationError(f"OpenCV cannot open {path.name} for disclosure samples")
    records = []
    try:
        for index in sorted({0, frames // 2, frames - 1}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise VerificationError(f"Cannot decode disclosure sample {index} from {path.name}")
            height, width = bgr.shape[:2]
            if comparison:
                panels = {
                    "right_bottom": bgr[max(0, height - 3):, width // 2:],
                    "right_top": bgr[:3, width // 2:],
                    "left_top": bgr[:3, :width // 2],
                }
            else:
                panels = {"bottom": bgr[max(0, height - 3):]}
            fractions = {key: float(np.mean(np.max(band, axis=2) < 55)) for key, band in panels.items()}
            if any(fraction < 0.90 for fraction in fractions.values()):
                raise VerificationError(f"Expected opaque disclosure border is absent at frame {index} of {path.name}")
            records.append({"frame_index": index, "dark_pixel_fraction": fractions})
    finally:
        capture.release()
    return {
        "passed": True, "samples": records,
        "scope": "First, middle and last frame border pixels only. No OCR; does not verify text semantics or every frame.",
    }


def verify_output(output_dir: Path, *, ffprobe: str = "ffprobe", check_bands: bool = True) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "frame_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    with metrics_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = validate_rows(summary, rows)
    probes = {}
    for name in VIDEO_NAMES:
        completed = subprocess.run([
            ffprobe, "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,duration,nb_read_frames:format=duration",
            "-of", "json", str(output_dir / name),
        ], check=True, capture_output=True, text=True, timeout=3600)
        probes[name] = json.loads(completed.stdout)
    result["videos"] = validate_probes(summary, probes)
    result["files"] = {
        name: {"bytes": (output_dir / name).stat().st_size, "sha256": sha256_file(output_dir / name)}
        for name in (*VIDEO_NAMES, "summary.json", "frame_metrics.csv")
    }
    hashes_path = output_dir / "output_hashes.json"
    if hashes_path.is_file():
        expected_hashes = json.loads(hashes_path.read_text(encoding="utf-8-sig"))
        for name, expected_hash in expected_hashes.items():
            # Expected hashes may refer only to files inside this completed export.
            target = (output_dir / name).resolve()
            if target.parent != output_dir or not target.is_file():
                raise VerificationError(f"Invalid expected-hash file name: {name}")
            actual_hash = result["files"].get(name, {}).get("sha256") or sha256_file(target)
            if actual_hash.lower() != str(expected_hash).lower():
                raise VerificationError(f"Output SHA256 mismatch: {name}")
        result["expected_hashes_verified"] = True
    result["disclosure_band_samples"] = {
        name: sample_disclosure_bands(output_dir / name, result["frames"], comparison=name == "comparison.mp4")
        for name in VIDEO_NAMES
    } if check_bands else {"checked": False, "reason": "Explicit --skip-band-check"}
    result["complete_exports"] = True
    result["validation_scope"] = "Decoded counts, silent streams, geometry, FPS, duration, metrics order, file hashes and sampled disclosure borders. Not identity or perceptual-quality validation."
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--skip-band-check", action="store_true", help="Skip pixel samples; recorded explicitly in receipt.")
    args = parser.parse_args()
    destination = args.output_dir / "export_verification.json"
    try:
        result = verify_output(args.output_dir, ffprobe=args.ffprobe, check_bands=not args.skip_band_check)
    except Exception as error:
        result = {"complete_exports": False, "error_type": type(error).__name__, "error": str(error)}
        # Replace stale success receipts after a failed re-verification.
        if args.output_dir.is_dir():
            destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 1
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete_exports": True, "frames": result["frames"], "status_counts": result["status_counts"], "receipt": str(destination.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
