"""ONNX/NVML GPU memory and explicitly limited temporal-quality proxies."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

import cv2
import numpy as np


class GPUMemory:
    """Sample device/process GPU use through NVML, including ONNX allocations.

    Device use includes other processes. Container PID namespaces can prevent
    matching the current process. These are sampled values, not allocator peaks.
    No PyTorch allocator values are reported for this ONNX pipeline.
    """

    def __init__(self, device_id: int = 0):
        self.nvml = None
        self.handle = None
        self.status = "unavailable"
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            chosen = visible[device_id] if device_id < len(visible) and visible[device_id] else str(device_id)
            if chosen.startswith(("GPU-", "MIG-")):
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(chosen)
            else:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(chosen))
            self.status = "available"
        except Exception as error:
            self.status = f"unavailable: {type(error).__name__}"

    def sample(self) -> dict[str, Any]:
        values: dict[str, Any] = {"nvml_status": self.status}
        if self.nvml is not None and self.handle is not None:
            try:
                info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                values.update(nvml_device_used_mib=info.used / 2**20, nvml_device_total_mib=info.total / 2**20)
                matches = []
                for getter_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                    try:
                        for process in getattr(self.nvml, getter_name)(self.handle):
                            used = getattr(process, "usedGpuMemory", None)
                            if process.pid == os.getpid() and isinstance(used, int) and 0 <= used < 2**60:
                                matches.append(used)
                    except Exception:
                        pass
                values["nvml_process_used_mib"] = max(matches) / 2**20 if matches else None
                values["nvml_process_status"] = "available" if matches else "PID not reported (possible container PID namespace)"
            except Exception as error:
                values["nvml_status"] = f"sample failed: {type(error).__name__}"
        return values

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


class TemporalConsistency:
    """Causal optical-flow alignment, never an identity/authenticity score."""

    def __init__(self, max_width: int = 320):
        self.max_width = max_width
        self.previous = None

    def update(self, driver_rgb: np.ndarray, output_rgb: np.ndarray, face_mask: np.ndarray,
               hand_mask: np.ndarray) -> dict[str, float | None]:
        height, width = driver_rgb.shape[:2]
        scale = min(1.0, self.max_width / width)
        size = (max(2, int(width * scale)), max(2, int(height * scale)))
        driver_gray = cv2.cvtColor(cv2.resize(driver_rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        output = cv2.resize(output_rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        face = cv2.resize(face_mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        hand = cv2.resize(hand_mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        metrics: dict[str, float | None] = {
            "flow_aligned_face_temporal_l1_proxy": None,
            "flow_aligned_hand_mask_iou_proxy": None,
            "flow_aligned_face_mask_iou_proxy": None,
        }
        current = (driver_gray, output, face, hand)
        if self.previous is not None and self.previous[0].shape == driver_gray.shape:
            prev_gray, prev_output, prev_face, prev_hand = self.previous
            flow = cv2.calcOpticalFlowFarneback(driver_gray, prev_gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            yy, xx = np.mgrid[:driver_gray.shape[0], :driver_gray.shape[1]].astype(np.float32)
            map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
            inside = (map_x >= 0) & (map_x < size[0] - 1) & (map_y >= 0) & (map_y < size[1] - 1)
            warped_output = cv2.remap(prev_output, map_x, map_y, cv2.INTER_LINEAR)
            warped_face = cv2.remap(prev_face, map_x, map_y, cv2.INTER_LINEAR)
            warped_hand = cv2.remap(prev_hand, map_x, map_y, cv2.INTER_LINEAR)
            region = (face > .5) & (warped_face > .5) & (hand < .1) & (warped_hand < .1) & inside
            if np.any(region):
                metrics["flow_aligned_face_temporal_l1_proxy"] = float(np.mean(np.abs(output - warped_output)[region]))
            for name, present, previous in (
                ("flow_aligned_hand_mask_iou_proxy", hand, warped_hand),
                ("flow_aligned_face_mask_iou_proxy", face, warped_face),
            ):
                present_binary, previous_binary = (present > .5) & inside, (previous > .5) & inside
                union = np.count_nonzero(present_binary | previous_binary)
                if union:
                    metrics[name] = float(np.count_nonzero(present_binary & previous_binary) / union)
        self.previous = current
        return metrics


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_versions() -> dict[str, Any]:
    versions = {"python": platform.python_version(), "platform": platform.platform(), "opencv": cv2.__version__}
    for name in ("numpy", "insightface", "onnx", "onnxruntime-gpu", "nvidia-ml-py", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed / package metadata unavailable"
    try:
        import onnxruntime as ort
        versions["ort_available_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    return versions


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
