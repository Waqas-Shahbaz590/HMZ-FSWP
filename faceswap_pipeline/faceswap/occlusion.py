"""Independent per-frame hand evidence for occlusion protection.

XSeg alone missed thin fingertips, which leaked generated beard/skin onto
fingers (TEAM_B_REPORT 31.24s/35.32s/89.44s). This module adds an independent
two-stage hand cascade on the current frame and converts confirmed hands into
conservative protection masks composed with the aligned face alpha.

Model contracts (MediaPipe topology, Apache-2.0 ONNX conversions by
PINTO0309, pinned commit e2deb2e024c855b2d7d875975c4ff3f636b6e83b):
* palm detector: letterboxed square RGB/255 NCHW float32; output rows
  [score, cx, cy, size, kp0x, kp0y, kp2x, kp2y] in normalized letterbox
  coordinates, NMS already applied (kp0 wrist side, kp2 middle-MCP side).
* hand landmark: upright square palm crop resized to the model input;
  outputs 21 xyz in input pixels, a presence score and handedness.

Only the mask math lives here; ONNX sessions are supplied by the caller so
this module stays importable without onnxruntime for CPU unit tests.
"""

from math import atan2, cos, degrees, floor, pi, sin

import cv2
import numpy as np

# 21-keypoint MediaPipe hand topology; index 0 is the wrist, 4/8/12/16/20 are
# the fingertips, which receive the most conservative dilation. FINGER_CHAINS
# are the bone chains (wrist -> tip) used to draw full-width finger capsules:
# landmark disks alone leave the finger pads between/beyond joints uncovered.
FINGERTIPS = (4, 8, 12, 16, 20)
KEYPOINT_COUNT = 21
FINGER_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))


def letterbox_square(bgr, size):
    """Resize with preserved aspect into a centered square letterbox.

    Returns (letterboxed image, scale, pad_x, pad_y) where
    frame_coord = (letterbox_coord - pad) / scale maps back exactly.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected an HxWx3 image for letterboxing")
    height, width = bgr.shape[:2]
    scale = size / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interpolation)
    padded = np.zeros((size, size, 3), dtype=resized.dtype)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return padded, scale, pad_x, pad_y


def decode_palm_output(output, score_threshold):
    """Validate and threshold palm detector rows -> float64 [N, 8]."""
    boxes = np.asarray(output)
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if boxes.ndim != 2 or boxes.shape[1] != 8:
        raise ValueError(f"Expected palm rows [N, 8], got {boxes.shape}")
    if not np.isfinite(boxes).all():
        raise ValueError("Non-finite palm detection output")
    boxes = boxes[boxes[:, 0] >= float(score_threshold)]
    return np.ascontiguousarray(boxes, dtype=np.float64)


def _normalize_radians(angle):
    return angle - 2.0 * pi * floor((angle + pi) / (2.0 * pi))


def palm_squares(boxes, frame_shape, letterbox_size, scale, pad_x, pad_y):
    """Map decoded palm rows to rotated square hand crops in frame pixels.

    Follows the MediaPipe/pinned-demo convention: the square is enlarged by
    2.9x so fingers fit, rotated so the wrist->middle-MCP axis becomes the
    crop's vertical axis, and recentered half a box toward the fingers.
    Returns a list of dicts with score, center, side and rotation (radians).
    """
    height, width = frame_shape[:2]
    if min(height, width) < 2:
        raise ValueError("Invalid frame shape for palm mapping")
    hands = []
    for score, bx, by, bsize, kp0x, kp0y, kp2x, kp2y in boxes:
        if bsize <= 0:
            continue
        rotation = _normalize_radians(0.5 * pi - atan2(-(kp2y - kp0y), kp2x - kp0x))
        cx = bx + 0.5 * bsize * sin(rotation)
        cy = by - 0.5 * bsize * cos(rotation)
        side = 2.9 * bsize
        # Letterbox-normalized -> letterbox pixels -> original frame pixels.
        to_frame = letterbox_size / scale
        hands.append({
            "score": float(score),
            "cx": (cx * letterbox_size - pad_x) / scale,
            "cy": (cy * letterbox_size - pad_y) / scale,
            "side": side * to_frame,
            "rotation": float(rotation),
        })
    return hands


def crop_affine(cx, cy, side, rotation, out_size):
    """2x3 affine mapping frame pixels into the upright square hand crop.

    Rotates the frame by +rotation about the crop center (OpenCV convention,
    matching the pinned demo), then cuts the centered square of the given
    side and scales it to out_size. Out-of-frame samples read as zeros.
    """
    if not np.isfinite([cx, cy, side, rotation]).all() or side < 4:
        raise ValueError(f"Invalid hand crop geometry: {(cx, cy, side, rotation)!r}")
    rotate = cv2.getRotationMatrix2D((float(cx), float(cy)), degrees(rotation), 1.0)
    k = float(out_size) / float(side)
    shift = np.array([[k, 0.0, -k * (cx - side / 2.0)],
                      [0.0, k, -k * (cy - side / 2.0)],
                      [0.0, 0.0, 1.0]])
    matrix = shift @ np.vstack([rotate, [0.0, 0.0, 1.0]])
    return np.ascontiguousarray(matrix[:2])


def rgb01_nchw(bgr_images):
    """Stack exact-size BGR crops into an RGB/255 NCHW float32 batch."""
    batch = []
    for bgr in bgr_images:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch.append(rgb.transpose(2, 0, 1))
    return np.ascontiguousarray(np.stack(batch), dtype=np.float32)


def decode_landmark_outputs(outputs):
    """Validate landmark session outputs -> (xyz[N,21,3], presence[N], handed[N]).

    The xyz tensor is identified by its trailing dimension of 63 (21 x 3);
    the two [N, 1] tensors follow the session's output order for presence
    score and handedness. Presence gating is the caller's decision.
    """
    if len(outputs) != 3:
        raise ValueError(f"Expected three landmark outputs, got {len(outputs)}")
    xyz = None
    scalars = []
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim >= 2 and arr.shape[-1] == 63:
            xyz = arr
        else:
            scalars.append(arr.reshape(arr.shape[0], -1)[:, 0] if arr.ndim >= 1 else arr.reshape(1))
    if xyz is None or len(scalars) != 2:
        raise ValueError("Landmark outputs must contain one [N, 63] tensor and two [N, 1] tensors")
    xyz = xyz.reshape(-1, KEYPOINT_COUNT, 3).astype(np.float64)
    presence = np.asarray(scalars[0], dtype=np.float64).reshape(-1)
    handed = np.asarray(scalars[1], dtype=np.float64).reshape(-1)
    if not (np.isfinite(xyz).all() and np.isfinite(presence).all() and np.isfinite(handed).all()):
        raise ValueError("Non-finite landmark output")
    if xyz.shape[0] != presence.shape[0]:
        raise ValueError("Landmark batch mismatch between outputs")
    return xyz, presence, handed


def landmarks_to_frame(xyz, matrix, crop_size):
    """Map one hand's 21 landmarks from crop pixels back to frame pixels."""
    inverse = cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float64))
    # The model emits landmark x/y in input-image pixels (0..crop_size); the
    # crop is exactly the model input, so no letterbox compensation applies.
    points = np.asarray(xyz, dtype=np.float64)[:, :2]
    inside = ((points[:, 0] >= -0.25 * crop_size) & (points[:, 0] <= 1.25 * crop_size)
              & (points[:, 1] >= -0.25 * crop_size) & (points[:, 1] <= 1.25 * crop_size))
    if int(inside.sum()) < 11:
        raise ValueError("Landmarks fall outside the hand crop; rejecting hand evidence")
    return points @ inverse[:, :2].T + inverse[:, 2]


def hand_mask_frame(frame_shape, hands_landmarks):
    """Conservative binary hand silhouette in frame space, float32 in [0, 1].

    Each confirmed hand contributes full-width capsules along the finger
    bone chains, a filled convex hull over the palm, a disk per joint
    (larger at fingertips) and a final dilation, so narrow gaps, finger
    pads and fingertip edges stay protected. Over-coverage is intentional:
    protected pixels keep the original frame, so excess only costs
    synthesis area.
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for points, side in hands_landmarks:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if points.shape[0] != KEYPOINT_COUNT or not np.isfinite(points).all():
            raise ValueError("Expected 21 finite hand landmarks")
        single = np.zeros((height, width), dtype=np.uint8)
        radius = max(2.0, 0.055 * side)
        tip_radius = max(3.0, 0.075 * side)
        thickness = max(5, int(round(2.0 * radius)))
        for chain in FINGER_CHAINS:
            cv2.polylines(single, [points[list(chain)].astype(np.int32)],
                          isClosed=False, color=255, thickness=thickness,
                          lineType=cv2.LINE_8)
        hull = cv2.convexHull(points.astype(np.int32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(single, hull, 255)
        for index, point in enumerate(points):
            r = tip_radius if index in FINGERTIPS else radius
            cv2.circle(single, (int(round(point[0])), int(round(point[1]))), int(round(r)), 255, -1)
        dilate = max(3, int(round(0.05 * side)) | 1)
        single = cv2.dilate(single, np.ones((dilate, dilate), np.uint8), iterations=1)
        mask = cv2.max(mask, single)
    return mask.astype(np.float32) / 255.0


def protect_alpha(alpha, hand_mask, blur_ksize=5, core_dilate=3):
    """Suppress generated alpha wherever current hand evidence exists.

    The Gaussian feather of alpha * (1 - hand) is re-clamped by the dilated
    hand core, so softening spreads only into the face side and the protected
    hand region keeps strictly-original pixels. With an empty hand mask this
    reduces to the plain blurred alpha of the previous pipeline stage.
    """
    alpha = np.asarray(alpha, dtype=np.float32)
    hand = np.asarray(hand_mask, dtype=np.float32)
    if alpha.shape != hand.shape or alpha.ndim != 2:
        raise ValueError("Alpha and hand mask must share one 2D aligned shape")
    if not np.isfinite(alpha).all() or not np.isfinite(hand).all():
        raise ValueError("Non-finite alpha or hand mask")
    hand = np.clip(hand, 0.0, 1.0)
    core = (hand > 0.5).astype(np.uint8)
    if core_dilate >= 3:
        core = cv2.dilate(core, np.ones((core_dilate, core_dilate), np.uint8), iterations=1)
    softened = cv2.GaussianBlur(alpha * (1.0 - hand), (blur_ksize, blur_ksize), 0)
    protected = np.minimum(softened, 1.0 - core.astype(np.float32))
    coverage = float((core > 0).mean())
    return np.clip(protected, 0.0, 1.0).astype(np.float32), coverage
