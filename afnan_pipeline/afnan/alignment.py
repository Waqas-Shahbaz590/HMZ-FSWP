"""Landmark-similarity geometry for one consistent aligned face space.

Replicates the InsightFace arcface template family (face_align.estimate_norm)
so the swap space (128), occlusion space (256) and restoration space (512)
share a single coordinate frame per detected face. Forward matrices map frame
pixels to aligned pixels; inverse matrices map back for the single bounded
ROI paste. All masks and model inputs are derived in this aligned space, so
no stage sees a differently cropped or aspect-distorted face.
"""

import cv2
import numpy as np
from skimage import transform as sktransform


class ImplausibleTransform(ValueError):
    """Alignment geometry that cannot correspond to the detected face."""

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def face_template(size):
    """Arcface destination landmarks for a square aligned size.

    Matches InsightFace estimate_norm: multiples of 112 scale directly;
    multiples of 128 add the 8-pixel-per-128 horizontal padding offset.
    """
    if not isinstance(size, (int, np.integer)) or size <= 0:
        raise ValueError(f"Aligned size must be a positive integer: {size!r}")
    if size % 112 == 0:
        ratio = float(size) / 112.0
        diff_x = 0.0
    elif size % 128 == 0:
        ratio = float(size) / 128.0
        diff_x = 8.0 * ratio
    else:
        raise ValueError(f"Aligned size must be a multiple of 112 or 128: {size}")
    template = ARCFACE_TEMPLATE.astype(np.float64) * ratio
    template[:, 0] += diff_x
    return template


def estimate_affine(kps, size):
    """Similarity transform (2x3) mapping frame landmarks to the template."""
    kps = np.asarray(kps, dtype=np.float64)
    if kps.shape != (5, 2) or not np.isfinite(kps).all():
        raise ValueError(f"Expected five finite 2D landmarks, got shape {kps.shape}")
    tform = sktransform.SimilarityTransform()
    if not tform.estimate(kps, face_template(size)):
        raise ValueError("Degenerate landmark geometry for alignment")
    matrix = tform.params[0:2, :]
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite alignment transform")
    return np.ascontiguousarray(matrix)


def validate_affine(matrix, bbox, size, *, max_rotation_deg=80.0,
                    min_face_fraction=0.25, max_face_fraction=1.75):
    """Confidence gate: reject transforms implausible for the detected bbox.

    A similarity matrix from corrupt landmarks can silently smear generated
    pixels across the frame; the aligned projection of the detection box must
    stay a credible fraction of the aligned crop and near-planar rotations
    beyond max_rotation_deg are rejected outright. Raises
    ImplausibleTransform so the caller can return a labeled original frame.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or bbox.size != 4:
        raise ImplausibleTransform("Non-finite alignment matrix or bounding box")
    face_px = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    if face_px <= 0:
        raise ImplausibleTransform("Empty face bounding box")
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    scale = float(np.sqrt(abs(det)))
    aligned = scale * face_px
    if not (min_face_fraction * size <= aligned <= max_face_fraction * size):
        raise ImplausibleTransform(
            f"Aligned face extent {aligned:.1f}px implausible for crop {size}px")
    rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    if abs(rotation) > max_rotation_deg:
        raise ImplausibleTransform(f"Alignment rotation {rotation:.1f}deg exceeds gate")
    return {"aligned_face_px": aligned, "rotation_deg": rotation, "scale": scale}


def aligned_crop(frame_bgr, kps, size):
    """Warp the frame into the aligned space; returns (crop, forward matrix)."""
    matrix = estimate_affine(kps, size)
    crop = cv2.warpAffine(frame_bgr, matrix, (size, size), borderValue=0.0)
    return crop, matrix


def roi_from_inverse(inverse_matrix, aligned_size, frame_shape, pad=2):
    """Bounding box of the inverse-warped aligned square, clamped to the frame."""
    inverse_matrix = np.asarray(inverse_matrix, dtype=np.float64)
    if inverse_matrix.shape != (2, 3) or not np.isfinite(inverse_matrix).all():
        raise ValueError("Invalid inverse alignment transform")
    corners = np.array(
        [[0.0, 0.0], [aligned_size, 0.0], [aligned_size, aligned_size], [0.0, aligned_size]],
        dtype=np.float64,
    )
    pts = corners @ inverse_matrix[:, :2].T + inverse_matrix[:, 2]
    height, width = frame_shape[:2]
    x1 = int(max(0, np.floor(pts[:, 0].min()) - pad))
    y1 = int(max(0, np.floor(pts[:, 1].min()) - pad))
    x2 = int(min(width, np.ceil(pts[:, 0].max()) + pad))
    y2 = int(min(height, np.ceil(pts[:, 1].max()) + pad))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("Empty paste region after clamping")
    return x1, y1, x2, y2


def translate_to_roi(inverse_matrix, x1, y1):
    """Adjust an aligned->frame inverse matrix to an aligned->ROI matrix."""
    shifted = np.asarray(inverse_matrix, dtype=np.float64).copy()
    shifted[0, 2] -= x1
    shifted[1, 2] -= y1
    return shifted
