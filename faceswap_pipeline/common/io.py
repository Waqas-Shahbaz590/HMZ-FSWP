"""Bounded, sequential video I/O. The AI pipeline never sees a video file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoFrame:
    rgb: np.ndarray
    timestamp_ms: int
    index: int
    decode_ms: float


class VideoReader:
    """Decode sequentially with absolute source timestamps and bounded memory.

    Optional bounds select ``start_seconds <= PTS < end_seconds``. Frames before
    the start are decoded and discarded, preserving frame indices and VFR PTS
    without relying on inaccurate codec seeks. Only selected frames reach the AI
    pipeline; no future frame is retained or used to process a selected frame.
    ``expected_selected_frames`` is a nominal-FPS estimate, whereas
    ``selected_frames`` is the exact number yielded.
    """

    def __init__(self, path: str | Path, max_width: int = 960, max_frames: int | None = None,
                 *, start_seconds: float = 0.0, end_seconds: float | None = None):
        if not np.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("start_seconds must be finite and nonnegative.")
        if end_seconds is not None and (not np.isfinite(end_seconds) or end_seconds <= start_seconds):
            raise ValueError("end_seconds must be finite and greater than start_seconds.")
        if max_frames is not None and (not isinstance(max_frames, int) or max_frames < 0):
            raise ValueError("max_frames must be a nonnegative integer or None.")
        self.path = Path(path)
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open driving video: {self.path}")
        raw_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(raw_fps) or raw_fps <= 0:
            self.capture.release()
            raise ValueError("Video has no usable frame rate; supply a valid prerecorded MP4.")
        self.fps = raw_fps
        self.expected_frames = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.source_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.max_width = max_width
        self.max_frames = max_frames
        self.start_seconds = float(start_seconds)
        self.end_seconds = float(end_seconds) if end_seconds is not None else None
        first_nominal_index = int(np.ceil(self.start_seconds * self.fps - 1e-9))
        last_nominal_index = self.expected_frames
        if self.end_seconds is not None:
            last_nominal_index = min(last_nominal_index, int(np.ceil(self.end_seconds * self.fps - 1e-9)))
        self.expected_selected_frames = max(0, last_nominal_index - first_nominal_index)
        if max_frames is not None:
            self.expected_selected_frames = min(self.expected_selected_frames, max_frames)
        self.timestamp_fallback_count = 0
        self.decoded_frames = 0
        self.source_frames_decoded = 0
        self.preroll_frames = 0
        self.preroll_decode_ms = 0.0
        self.ended_by_time_bound = False
        self._last_timestamp_ms = -1
        self.width = 0
        self.height = 0

    def __iter__(self) -> Iterator[VideoFrame]:
        while self.max_frames is None or self.decoded_frames < self.max_frames:
            started = time.perf_counter()
            ok, bgr = self.capture.read()
            if not ok:
                break
            index = self.source_frames_decoded
            self.source_frames_decoded += 1
            presentation = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            fallback = int(round(index * 1000.0 / self.fps))
            if np.isfinite(presentation) and presentation >= 0 and int(round(presentation)) > self._last_timestamp_ms:
                timestamp = int(round(presentation))
            else:
                timestamp = max(self._last_timestamp_ms + 1, fallback)
                self.timestamp_fallback_count += 1
            self._last_timestamp_ms = timestamp
            if timestamp < self.start_seconds * 1000.0:
                self.preroll_frames += 1
                self.preroll_decode_ms += (time.perf_counter() - started) * 1000.0
                continue
            if self.end_seconds is not None and timestamp >= self.end_seconds * 1000.0:
                self.ended_by_time_bound = True
                break
            height, width = bgr.shape[:2]
            scale = min(1.0, self.max_width / width) if self.max_width > 0 else 1.0
            # MP4 encoders require even dimensions. Resize at most once before the pipeline.
            output_width = max(2, int(round(width * scale)) // 2 * 2)
            output_height = max(2, int(round(height * scale)) // 2 * 2)
            if (output_width, output_height) != (width, height):
                bgr = cv2.resize(bgr, (output_width, output_height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.width, self.height = output_width, output_height
            self.decoded_frames += 1
            yield VideoFrame(rgb, timestamp, index, (time.perf_counter() - started) * 1000.0)

    @property
    def premature_end(self) -> bool:
        if self.ended_by_time_bound or (self.max_frames is not None and self.decoded_frames >= self.max_frames):
            return False
        return self.expected_frames > 0 and self.source_frames_decoded < self.expected_frames

    @property
    def selected_frames(self) -> int:
        return self.decoded_frames

    def close(self) -> None:
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class VideoWriter:
    """Causal raw-RGB H264 output; OpenCV's lossy exporter is an explicit option."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int, encoder: str = "h264"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self.encoder = encoder
        self.writer = self.process = self._stderr_file = None
        self._stderr_path = None
        self._closed = False
        self.frames_written = 0
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("MP4 output dimensions must be positive and even.")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("MP4 output requires a finite positive frame rate.")
        if encoder == "opencv":
            self.writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"OpenCV MP4 encoder could not open output: {self.path}")
        elif encoder == "h264":
            executable = shutil.which("ffmpeg")
            if executable is None:
                raise RuntimeError("FFmpeg is required for H264 output. Install FFmpeg or explicitly select --encoder opencv.")
            self._stderr_file = tempfile.NamedTemporaryFile(prefix="avatar_ffmpeg_", suffix=".log", delete=False)
            self._stderr_path = Path(self._stderr_file.name)
            command = [
                executable, "-hide_banner", "-loglevel", "warning", "-nostdin", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
                "-framerate", f"{fps:.12g}", "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-crf", "17", "-preset", "fast", "-pix_fmt", "yuv420p", "-threads", "4",
                "-tune", "zerolatency", "-bf", "0", "-movflags", "+faststart", str(self.path),
            ]
            try:
                self.process = subprocess.Popen(
                    command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=self._stderr_file,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
            except Exception:
                self._stderr_file.close()
                self._stderr_path.unlink(missing_ok=True)
                raise
        else:
            raise ValueError("encoder must be 'h264' or 'opencv'")

    def _encoder_error(self, prefix: str) -> RuntimeError:
        if self._stderr_file is not None and not self._stderr_file.closed:
            self._stderr_file.flush()
        details = ""
        if self._stderr_path is not None and self._stderr_path.is_file():
            details = self._stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
        return RuntimeError(f"{prefix}: {self.path}. FFmpeg log: {self._stderr_path}. {details}")

    def write(self, rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a finalized video.")
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Video output must be uint8 RGB with a constant frame size.")
        if self.encoder == "opencv":
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            if self.process.poll() is not None:
                raise self._encoder_error(f"FFmpeg terminated with code {self.process.returncode}")
            try:
                self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
            except (BrokenPipeError, OSError) as error:
                raise self._encoder_error(f"FFmpeg input write failed ({type(error).__name__})") from error
        self.frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.encoder == "opencv":
            self.writer.release()
            return
        stdin_error = None
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError) as error:
            stdin_error = error
        try:
            return_code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self._stderr_file.close()
            raise self._encoder_error("FFmpeg did not finalize within 60 seconds")
        self._stderr_file.close()
        if return_code != 0 or stdin_error is not None:
            raise self._encoder_error(f"FFmpeg finalization failed with code {return_code}")
        self._stderr_path.unlink(missing_ok=True)


def label_frame(rgb: np.ndarray, text: str = "SYNTHETIC AVATAR - TEST ONLY", *, top: bool = False) -> np.ndarray:
    """Burn a visible, opaque disclosure into every exported synthetic frame."""
    result = rgb.copy()
    height, width = result.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.24, min(0.70, width / 1150.0))
    thickness = max(1, int(round(scale * 2)))
    # Fit labels even when testing very small videos.
    while cv2.getTextSize(text, font, scale, thickness)[0][0] > width - 12 and scale > 0.12:
        scale *= 0.92
    (_, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    band_height = min(height, text_height + baseline + 16)
    y0 = 0 if top else height - band_height
    cv2.rectangle(result, (0, y0), (width - 1, y0 + band_height - 1), (12, 12, 12), -1)
    cv2.putText(result, text, (6, y0 + text_height + 6), font, scale, (255, 230, 80), thickness, cv2.LINE_AA)
    return result


def comparison_frame(driving_rgb: np.ndarray, output_rgb: np.ndarray, *, status: str | None = None) -> np.ndarray:
    """Render disclosures explicitly so the right header cannot hide a fallback."""
    if driving_rgb.shape != output_rgb.shape:
        raise ValueError("Comparison panels must have identical dimensions.")
    left = label_frame(driving_rgb, "DRIVING INPUT", top=True)
    right = label_frame(output_rgb)
    header = "PIPELINE FALLBACK - ORIGINAL FRAME" if status and status.startswith("original_frame") else "SYNTHETIC AVATAR - TEST ONLY"
    right = label_frame(right, header, top=True)
    return np.concatenate((left, right), axis=1)


def load_reference(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode reference portrait: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
