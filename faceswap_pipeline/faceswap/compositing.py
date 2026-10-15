"""Aligned-space mask building and the single bounded ROI paste.

The output frame starts as a copy of the original current frame. Generated
content is inverse-warped from the aligned working space into the face ROI
only, under one validated alpha mask, so pixels outside the protected region
can never carry swapped/restored content.
"""

import cv2
import numpy as np

from . import alignment


def padded_box(bbox, image_shape, padding=0.18):
    coords = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if coords.size != 4 or not np.isfinite(coords).all():
        raise ValueError("Invalid face bounding box")
    x1, y1, x2, y2 = coords.astype(int).tolist()
    h, w = image_shape[:2]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty face bounding box")
    pad = int(max(x2 - x1, y2 - y1) * padding)
    return max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)


def base_face_mask(size=128):
    """Soft oval support of the aligned face, in [0,1] float32.

    Aligned-space equivalent of the upstream paste-back mask: a white square
    eroded by ~10% of the aligned size, then Gaussian-feathered. Because the
    working space is normalized, these kernels no longer depend on the face's
    pixel size in the frame.
    """
    if not isinstance(size, (int, np.integer)) or size < 32:
        raise ValueError(f"Mask size must be an integer >= 32: {size!r}")
    mask = np.full((size, size), 255, dtype=np.uint8)
    erode = max(3, int(round(size * 0.10)))
    mask = cv2.erode(mask, np.ones((erode, erode), np.uint8), iterations=1)
    blur = 2 * max(1, int(round(size / 20.0))) + 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask.astype(np.float32) / 255.0


def compose_aligned(frame_bgr, content_bgr, alpha, forward_matrix, pad=2):
    """Paste aligned content into the original frame under one alpha mask.

    forward_matrix maps frame pixels to the aligned working space of
    content_bgr/alpha (square, identical size). Only the bounding ROI of the
    inverse-warped aligned square is touched; everything else stays original.
    """
    if not isinstance(content_bgr, np.ndarray) or content_bgr.dtype != np.uint8 \
            or content_bgr.ndim != 3 or content_bgr.shape[2] != 3:
        raise ValueError("Aligned content must be a uint8 BGR image")
    size = content_bgr.shape[0]
    if content_bgr.shape[1] != size or min(content_bgr.shape[:2]) < 32:
        raise ValueError("Aligned content must be square and >= 32 px")
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.shape != (size, size) or not np.isfinite(alpha).all():
        raise ValueError("Alpha mask must match the aligned working space and be finite")
    inverse_matrix = cv2.invertAffineTransform(np.asarray(forward_matrix, dtype=np.float64))
    x1, y1, x2, y2 = alignment.roi_from_inverse(inverse_matrix, size, frame_bgr.shape, pad=pad)
    roi_matrix = alignment.translate_to_roi(inverse_matrix, x1, y1)
    roi_size = (x2 - x1, y2 - y1)
    warped_content = cv2.warpAffine(content_bgr, roi_matrix, roi_size, borderValue=0.0)
    warped_alpha = cv2.warpAffine(alpha, roi_matrix, roi_size, borderValue=0.0)
    a = np.clip(warped_alpha, 0.0, 1.0)[..., None]
    result = frame_bgr.copy()
    region = result[y1:y2, x1:x2].astype(np.float32)
    blended = warped_content.astype(np.float32) * a + region * (1.0 - a)
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result, {
        "visible_mask_mean": float(a.mean()),
        "visible_mask_low_fraction": float((a < 0.5).mean()),
        "crop_box": [x1, y1, x2, y2],
    }
