import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from faceswap import occlusion
from faceswap.alignment import ImplausibleTransform, estimate_affine, face_template, validate_affine

from test_adapter import backend, face


class PalmSession:
    """Pinned palm-detector contract: one [1,3,192,192] input, rows [N,8] out."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.received = []

    def get_inputs(self):
        return [SimpleNamespace(name="palm_input", shape=[1, 3, 192, 192])]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        self.received.append(inputs)
        if isinstance(self.boxes, Exception):
            raise self.boxes
        return [self.boxes.copy()]


class LandmarkSession:
    """Pinned landmark contract: [N,3,224,224] in; xyz[N,63], presence, handedness out."""

    def __init__(self, xyz, presence=0.9, handed=0.0):
        self.xyz = xyz
        self.presence = presence
        self.handed = handed
        self.received = []

    def get_inputs(self):
        return [SimpleNamespace(name="hand_input", shape=["N", 3, 224, 224])]

    def get_outputs(self):
        return [SimpleNamespace(name="landmarks", shape=["N", 63]),
                SimpleNamespace(name="presence", shape=["N", 1]),
                SimpleNamespace(name="handedness", shape=["N", 1])]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, output_names, inputs):
        self.received.append(inputs)
        batch = next(iter(inputs.values())).shape[0]
        xyz = np.asarray(self.xyz, dtype=np.float32).reshape(1, 63)
        return [np.repeat(xyz, batch, axis=0),
                np.full((batch, 1), self.presence, np.float32),
                np.full((batch, 1), self.handed, np.float32)]


def palm_box_over_face(score=0.9):
    # rotation = pi/2 - atan2(-(0.6-0.8), 0) = 0 -> fingers point up;
    # center (0.5, 0.7 - 0.5*0.4) = (0.5, 0.5), side = 2.9*0.4.
    return np.array([[score, 0.5, 0.7, 0.4, 0.5, 0.8, 0.5, 0.6]], dtype=np.float32)


def landmark_blob(center=(112.0, 112.0), radius=55.0):
    angles = np.linspace(0, 2 * np.pi, 21, endpoint=False)
    points = np.stack([center[0] + radius * np.cos(angles),
                       center[1] + radius * np.sin(angles),
                       np.zeros(21)], axis=1)
    return points.reshape(1, 63)


def hand_backend(palm, landmark, **kwargs):
    adapter, app, swapper, gfp, occ = backend(hand_protection=True, palm=palm, landmark=landmark, **kwargs)
    return adapter, app, swapper, gfp, occ


class OcclusionUnitTests(unittest.TestCase):
    def test_letterbox_round_trip(self):
        import cv2
        image = np.random.default_rng(0).integers(0, 255, (1280, 720, 3), dtype=np.uint8)
        padded, scale, pad_x, pad_y = occlusion.letterbox_square(image, 192)
        self.assertEqual(padded.shape, (192, 192, 3))
        self.assertAlmostEqual(scale, 192 / 1280)
        self.assertEqual((pad_x, pad_y), (42, 0))
        expected = cv2.resize(image, (108, 192), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(padded[0:192, 42:150], expected)
        self.assertEqual(padded[:, 0].sum(), 0)
        self.assertEqual(padded[:, 191].sum(), 0)
        # frame -> letterbox -> frame is an exact affine inverse
        x_lb, y_lb = 360 * scale + pad_x, 640 * scale + pad_y
        self.assertAlmostEqual((x_lb - pad_x) / scale, 360.0)
        self.assertAlmostEqual((y_lb - pad_y) / scale, 640.0)

    def test_decode_palm_output_validation(self):
        rows = np.array([[0.7] + [0.0] * 7, [0.4] + [0.0] * 7], dtype=np.float32)
        kept = occlusion.decode_palm_output(rows, 0.5)
        self.assertEqual(kept.shape, (1, 8))
        squeezed = occlusion.decode_palm_output(rows[None], 0.5)
        self.assertEqual(squeezed.shape, (1, 8))
        with self.assertRaises(ValueError):
            occlusion.decode_palm_output(np.zeros((4, 7), np.float32), 0.5)
        with self.assertRaises(ValueError):
            occlusion.decode_palm_output(np.full((1, 8), np.nan, np.float32), 0.5)

    def test_palm_squares_geometry_and_letterbox_inverse(self):
        # Portrait 720x1280: scale 0.15, letterbox pad_x = 42, pad_y = 0.
        padded, scale, pad_x, pad_y = occlusion.letterbox_square(np.zeros((1280, 720, 3), np.uint8), 192)
        self.assertEqual((pad_x, pad_y), (42, 0))
        boxes = np.array([[0.9, 0.5, 0.5, 0.2, 0.5, 0.55, 0.5, 0.45]])
        hands = occlusion.palm_squares(boxes, (1280, 720, 3), 192, scale, pad_x, pad_y)
        self.assertEqual(len(hands), 1)
        hand = hands[0]
        self.assertAlmostEqual(hand["rotation"], 0.0, places=6)
        self.assertAlmostEqual(hand["cx"], (0.5 * 192 - 42) / 0.15, places=4)
        self.assertAlmostEqual(hand["cy"], (0.4 * 192 - 0) / 0.15, places=4)
        self.assertAlmostEqual(hand["side"], 2.9 * 0.2 * 192 / 0.15, places=4)
        # Hand pointing right -> rotation +90deg, center shifted toward +x.
        right = occlusion.palm_squares(
            np.array([[0.9, 0.5, 0.5, 0.2, 0.45, 0.5, 0.55, 0.5]]), (1280, 720, 3), 192, scale, pad_x, pad_y)[0]
        self.assertAlmostEqual(right["rotation"], np.pi / 2, places=6)
        self.assertAlmostEqual(right["cx"], (0.5 + 0.1) * 192 / 0.15 - 42 / 0.15, places=4)
        self.assertAlmostEqual(right["cy"], 0.5 * 192 / 0.15, places=4)

    def test_crop_affine_convention_and_inverse(self):
        matrix = occlusion.crop_affine(100.0, 100.0, 50.0, 0.0, 224)
        # rotation 0: frame (75,75) -> (0,0), center -> (112,112)
        np.testing.assert_allclose(matrix @ [75, 75, 1], [0, 0], atol=1e-6)
        np.testing.assert_allclose(matrix @ [100, 100, 1], [112, 112], atol=1e-6)
        rotated = occlusion.crop_affine(100.0, 100.0, 50.0, np.pi / 2, 224)
        # Pinned convention: +90deg maps a point below center to the crop right edge.
        np.testing.assert_allclose(rotated @ [100, 125, 1], [224, 112], atol=1e-4)
        import cv2
        inverse = cv2.invertAffineTransform(rotated)
        np.testing.assert_allclose(inverse[:, :2] @ np.array([224.0, 112.0]) + inverse[:, 2], [100, 125], atol=1e-4)
        with self.assertRaises(ValueError):
            occlusion.crop_affine(0, 0, 2.0, 0.0, 224)

    def test_decode_landmark_outputs_validation(self):
        xyz = np.zeros((2, 63), np.float32)
        presence = np.full((2, 1), 0.9, np.float32)
        handed = np.zeros((2, 1), np.float32)
        out_xyz, out_presence, out_handed = occlusion.decode_landmark_outputs([xyz, presence, handed])
        self.assertEqual(out_xyz.shape, (2, 21, 3))
        np.testing.assert_allclose(out_presence, [0.9, 0.9])
        with self.assertRaises(ValueError):
            occlusion.decode_landmark_outputs([xyz, presence])
        with self.assertRaises(ValueError):
            occlusion.decode_landmark_outputs([np.zeros((2, 62), np.float32), presence, handed])
        with self.assertRaises(ValueError):
            occlusion.decode_landmark_outputs([np.full((2, 63), np.nan, np.float32), presence, handed])
        with self.assertRaises(ValueError):
            occlusion.decode_landmark_outputs([xyz, np.zeros((3, 1), np.float32), handed])

    def test_landmarks_to_frame_rejects_out_of_crop(self):
        matrix = occlusion.crop_affine(100.0, 100.0, 50.0, 0.0, 224)
        with self.assertRaises(ValueError):
            occlusion.landmarks_to_frame(np.full((21, 3), 500.0), matrix, 224)
        points = occlusion.landmarks_to_frame(np.tile([112.0, 112.0, 0.0], (21, 1)), matrix, 224)
        np.testing.assert_allclose(points, [[100.0, 100.0]] * 21, atol=1e-4)

    def test_hand_mask_covers_fingertips_with_margin(self):
        side = 100.0
        points = np.array([
            [50, 80], [30, 70], [22, 60], [16, 50], [12, 40],
            [38, 55], [36, 38], [35, 28], [34, 20],
            [50, 53], [50, 35], [50, 25], [50, 16],
            [60, 55], [63, 38], [65, 28], [66, 20],
            [70, 60], [75, 48], [78, 40], [80, 32],
        ], dtype=np.float64)
        mask = occlusion.hand_mask_frame((128, 128, 3), [(points, side)])
        self.assertEqual(mask.shape, (128, 128))
        self.assertEqual(mask[80, 50], 1.0)   # wrist disk
        self.assertEqual(mask[60, 50], 1.0)   # palm hull interior
        self.assertEqual(mask[8, 50], 1.0)    # beyond middle fingertip: conservative margin
        self.assertEqual(mask[5, 5], 0.0)     # far away stays unprotected
        self.assertTrue(np.isfinite(mask).all())

    def test_protect_alpha_semantics(self):
        import cv2
        rng = np.random.default_rng(1)
        alpha = rng.random((64, 64), dtype=np.float32)
        empty = np.zeros((64, 64), np.float32)
        protected, coverage = occlusion.protect_alpha(alpha, empty)
        # No hand evidence: identical to the legacy blurred alpha.
        np.testing.assert_allclose(protected, cv2.GaussianBlur(alpha, (5, 5), 0), atol=1e-6)
        self.assertEqual(coverage, 0.0)
        hand = np.zeros((64, 64), np.float32)
        cv2.circle(hand, (32, 32), 10, 1.0, -1)
        protected, coverage = occlusion.protect_alpha(np.ones((64, 64), np.float32), hand)
        self.assertEqual(protected[32, 32], 0.0)          # core strictly original
        self.assertEqual(protected[32, 43], 0.0)          # still inside dilated core (r 10 + 1)
        self.assertGreater(protected[5, 5], 0.9)          # far face side untouched
        # Feather rises monotonically on the face side of the boundary only.
        self.assertGreater(protected[32, 44], protected[32, 43])
        self.assertGreater(protected[32, 45], protected[32, 44])
        self.assertGreater(coverage, 0.0)
        with self.assertRaises(ValueError):
            occlusion.protect_alpha(alpha, np.zeros((32, 32), np.float32))
        with self.assertRaises(ValueError):
            occlusion.protect_alpha(np.full((64, 64), np.nan, np.float32), empty)

    def test_validate_affine_gate(self):
        kps = face_template(128).astype(np.float64) * 0.5
        matrix = estimate_affine(kps, 128)
        info = validate_affine(matrix, [12, 12, 52, 52], 128)
        self.assertAlmostEqual(info["scale"], 2.0, places=5)
        self.assertAlmostEqual(info["rotation_deg"], 0.0, places=4)
        with self.assertRaises(ImplausibleTransform):
            validate_affine(estimate_affine(kps * 10.0, 128), [12, 12, 52, 52], 128)  # scale gate
        rotated = np.array([[0.0, -2.0, 32.0], [2.0, 0.0, 32.0]])  # 90 degrees
        with self.assertRaises(ImplausibleTransform):
            validate_affine(rotated, [12, 12, 52, 52], 128)
        with self.assertRaises(ImplausibleTransform):
            validate_affine(np.full((2, 3), np.nan), [12, 12, 52, 52], 128)


class OcclusionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.rgb[..., 0] = 30
        self.rgb[..., 1] = 60
        self.rgb[..., 2] = 90

    def test_hand_suppresses_generated_pixels(self):
        adapter, _, _, _, _ = hand_backend(PalmSession(palm_box_over_face()),
                                           LandmarkSession(landmark_blob()))
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["status"], "animated")
        self.assertEqual(diag["hand_count"], 1)
        self.assertEqual(diag["hand_rejected"], 0)
        np.testing.assert_array_equal(output[32, 32], self.rgb[32, 32])  # protected
        self.assertGreater(diag["hand_core_fraction"], 0.0)
        self.assertEqual(diag["hand_boxes"], [[0.0, 0.0, 64.0, 64.0]])
        legacy, _, _, _, _ = backend()
        legacy.prepare(self.rgb)
        legacy_output, _ = legacy.process(self.rgb, 0)
        self.assertNotEqual(legacy_output[32, 32].tolist(), self.rgb[32, 32].tolist())  # normally generated

    def test_no_hands_matches_legacy_path(self):
        adapter, _, _, _, _ = hand_backend(PalmSession(np.zeros((0, 8), np.float32)),
                                           LandmarkSession(landmark_blob()))
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["status"], "animated")
        self.assertEqual(diag["hand_count"], 0)
        self.assertEqual(diag["hand_core_fraction"], 0.0)
        legacy, _, _, _, _ = backend()
        legacy.prepare(self.rgb)
        legacy_output, _ = legacy.process(self.rgb, 0)
        np.testing.assert_array_equal(output, legacy_output)

    def test_low_presence_hand_rejected_without_failing_frame(self):
        adapter, _, _, _, _ = hand_backend(PalmSession(palm_box_over_face()),
                                           LandmarkSession(landmark_blob(), presence=0.1))
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["status"], "animated")
        self.assertEqual(diag["hand_count"], 0)
        self.assertEqual(diag["hand_rejected"], 1)
        self.assertEqual(diag["hand_core_fraction"], 0.0)

    def test_palm_session_error_is_labeled_fallback(self):
        adapter, _, _, _, _ = hand_backend(PalmSession(RuntimeError("palm boom")),
                                           LandmarkSession(landmark_blob()))
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["reason"], "hand_detect_session_error")
        self.assertFalse(diag["synthetic"])
        np.testing.assert_array_equal(output, self.rgb)

    def test_implausible_transform_is_labeled_fallback(self):
        adapter, app, _, _, _ = backend()
        app.faces = [face(kps=face_template(128).astype(np.float32) * 10.0)]
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["reason"], "transform_implausible")
        np.testing.assert_array_equal(output, self.rgb)

    def test_hand_protection_requires_sessions(self):
        with self.assertRaises(ValueError):
            backend(hand_protection=True)


if __name__ == "__main__":
    unittest.main()
