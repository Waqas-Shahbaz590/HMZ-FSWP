"""Explicit model layout and color handling; no downloads or model loading."""

from dataclasses import dataclass

import cv2
import numpy as np


def validate_rgb(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Expected nonempty HxWx3 RGB image")
    return np.ascontiguousarray(image)


@dataclass(frozen=True)
class ImageInput:
    name: str
    layout: str
    size: tuple

    @classmethod
    def from_session(cls, session, default_size):
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Expected exactly one image input")
        meta = inputs[0]
        shape = tuple(meta.shape)
        if len(shape) != 4:
            raise ValueError(f"Unsupported image input shape: {shape}")
        if isinstance(shape[0], int) and shape[0] != 1:
            raise ValueError(f"Expected batch size 1: {shape}")
        if shape[1] == 3 and shape[-1] != 3:
            layout, height, width = "NCHW", shape[2], shape[3]
        elif shape[-1] == 3 and shape[1] != 3:
            layout, height, width = "NHWC", shape[1], shape[2]
        else:
            raise ValueError(f"Ambiguous image channel layout: {shape}")
        height = height if isinstance(height, int) and height > 0 else default_size
        width = width if isinstance(width, int) and width > 0 else default_size
        return cls(meta.name, layout, (width, height))

    def tensor(self, bgr, *, normalized_rgb=False):
        image = cv2.resize(bgr, self.size, interpolation=cv2.INTER_LINEAR)
        if normalized_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        else:
            image = image.astype(np.float32) / 255.0
        if self.layout == "NCHW":
            image = image.transpose(2, 0, 1)
        return np.ascontiguousarray(image[None], dtype=np.float32)


def decode_mask(output):
    mask = np.asarray(output)
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            raise ValueError(f"Expected single-channel mask, got {mask.shape}")
    if mask.ndim != 2 or min(mask.shape) < 2 or not np.isfinite(mask).all():
        raise ValueError("Invalid/nonfinite occlusion mask")
    if float(mask.min()) < -0.05 or float(mask.max()) > 1.05:
        raise ValueError("Occlusion mask outside probability range")
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def decode_restoration(output):
    restored = np.asarray(output)
    if restored.ndim == 4 and restored.shape[0] == 1:
        restored = restored[0]
    if restored.ndim != 3 or not np.isfinite(restored).all():
        raise ValueError("Invalid/nonfinite restoration output")
    if restored.shape[0] == 3 and restored.shape[-1] != 3:
        restored = restored.transpose(1, 2, 0)
    elif restored.shape[-1] != 3:
        raise ValueError(f"Unsupported restoration output layout {restored.shape}")
    rgb = np.clip((restored + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
