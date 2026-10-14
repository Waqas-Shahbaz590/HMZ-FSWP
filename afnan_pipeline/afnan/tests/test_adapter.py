import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afnan.adapter import AfnanBackend
from afnan.alignment import estimate_affine, face_template
from afnan.compositing import compose_aligned, padded_box
from afnan.tensors import ImageInput, decode_mask


# Landmark geometry: face_template(128) * 0.5 places a valid face inside the
# 64x64 test frames and yields a clean 2x frame->aligned similarity transform.
def face(box=(12, 12, 52, 52), score=0.9, kps=None):
    return SimpleNamespace(bbox=np.array(box, dtype=np.float32), det_score=score,
                           kps=(face_template(128).astype(np.float32) * 0.5 if kps is None else kps),
                           normed_embedding=np.array([1.0, 0.0], dtype=np.float32))


class Session:
    def __init__(self, shape, output, providers=None):
        self.shape, self.output = shape, output
        self.providers = providers or ["CPUExecutionProvider"]
        self.fallback_disabled = False
        self.received = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=self.shape)]

    def get_providers(self):
        return self.providers

    def disable_fallback(self):
        self.fallback_disabled = True

    def run(self, output_names, inputs):
        self.received.append(inputs)
        if isinstance(self.output, Exception):
            raise self.output
        return [self.output.copy()]


class DetModel:
    """Mirrors the SCRFD detect() contract used by the adapter."""

    def __init__(self):
        self.session = Session([1, 3, 320, 320], None)
        self.faces = []
        self.calls = 0

    def set_faces(self, faces):
        self.faces = list(faces)
        rows = [list(f.bbox) + [f.det_score] for f in self.faces]
        self.bboxes = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        self.kpss = (np.asarray([f.kps for f in self.faces], dtype=np.float32).reshape(-1, 5, 2)
                     if self.faces else np.zeros((0, 5, 2), np.float32))

    def detect(self, img, max_num=0, metric="default"):
        self.calls += 1
        return self.bboxes.copy(), self.kpss.copy()


class App:
    def __init__(self, faces):
        self.det_model = DetModel()
        self.models = {"detection": self.det_model}
        self.received = []
        self.faces = faces

    @property
    def faces(self):
        return self.det_model.faces

    @faces.setter
    def faces(self, faces):
        self.det_model.set_faces(faces)

    def get(self, bgr):
        self.received.append(bgr.copy())
        return self.det_model.faces


class Swapper:
    """Session-level INSwapper contract; get() must never be called."""

    def __init__(self, output=None):
        self.session = Session([1, 3, 128, 128],
                               output if output is not None
                               else np.full((1, 3, 128, 128), 200.5 / 255.0, np.float32))
        self.emap = np.eye(2, dtype=np.float32)
        self.input_names = ["target", "latent"]
        self.output_names = ["output"]
        self.input_size = (128, 128)

    def get(self, *args, **kwargs):
        raise AssertionError("upstream paste-back path must not be used")


def backend(mask=None, gfp_output=None, restore_blend=0.75, mask_shape=None, swap_output=None,
            hand_protection=False, palm=None, landmark=None):
    app = App([face()])
    swapper = Swapper(swap_output)
    gfp = Session([1, 3, 512, 512], gfp_output if gfp_output is not None else np.zeros((1, 3, 512, 512), np.float32))
    occ = Session(mask_shape or [1, 256, 256, 3], mask if mask is not None else np.ones((1, 256, 256, 1), np.float32))
    adapter = AfnanBackend("unused", providers=["CPUExecutionProvider"], strict_cuda=False,
                           restore_blend=restore_blend, hand_protection=hand_protection,
                           _components=(app, swapper, gfp, occ, palm, landmark))
    return adapter, app, swapper, gfp, occ


class AfnanTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.rgb[..., 0] = 30
        self.rgb[..., 1] = 60
        self.rgb[..., 2] = 90

    def test_rgb_bgr_boundary_and_reference_largest(self):
        adapter, app, _, _, _ = backend()
        app.faces = [face((2, 2, 12, 12)), face()]
        diag = adapter.prepare(self.rgb)
        np.testing.assert_array_equal(app.received[0][0, 0], [90, 60, 30])
        self.assertEqual(diag["source_bbox"], [12.0, 12.0, 52.0, 52.0])

    def test_missing_face_never_uses_stale_geometry(self):
        adapter, app, swapper, _, _ = backend()
        adapter.prepare(self.rgb)
        output, first = adapter.process(self.rgb, 0)
        self.assertEqual(first["status"], "animated")
        app.faces = []
        output, second = adapter.process(self.rgb, 40)
        self.assertEqual(second["reason"], "face_missing")
        np.testing.assert_array_equal(output, self.rgb)
        self.assertEqual(len(swapper.session.received), 1)

    def test_failed_mask_returns_entire_original(self):
        adapter, _, _, _, _ = backend(mask=RuntimeError("bad mask"))
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["reason"], "occlusion_session_error")
        self.assertFalse(diag["synthetic"])
        np.testing.assert_array_equal(output, self.rgb)

    def test_nonfinite_mask_returns_original(self):
        adapter, _, _, _, _ = backend(mask=np.full((1, 256, 256, 1), np.nan, np.float32))
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["reason"], "occlusion_session_error")
        np.testing.assert_array_equal(output, self.rgb)

    def test_original_occluded_pixels_preserved(self):
        mask = np.ones((1, 256, 256, 1), np.float32)
        mask[:, :, :128, :] = 0
        adapter, _, _, _, _ = backend(mask=mask)
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        np.testing.assert_array_equal(output[30, 12], self.rgb[30, 12])
        self.assertNotEqual(output[30, 50].tolist(), self.rgb[30, 50].tolist())
        self.assertEqual(diag["status"], "animated")

    def test_restoration_failure_is_explicit_no_mixed_variant(self):
        adapter, _, _, _, _ = backend(gfp_output=ValueError("failed restore"))
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["reason"], "restore_session_error")
        np.testing.assert_array_equal(output, self.rgb)

    def test_aligned_restoration_blend(self):
        adapter, _, _, _, _ = backend()
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        # GFPGAN zero normalized RGB ->127, then .75*127+.25*200 ->145.
        np.testing.assert_allclose(output[32, 32], [145, 145, 145], atol=1)
        self.assertEqual(set(diag["stages_ms"]),
                         {"detect", "hand_detect_prep", "hand_detect_session",
                          "hand_landmark_prep", "hand_landmark_session",
                          "swap_align", "swap_session", "swap_decode",
                          "occlusion_prep", "occlusion_session",
                          "restore_prep", "restore_session", "composite", "convert"})
        self.assertTrue(all(x >= 0 for x in diag["stages_ms"].values()))

    def test_restore_off_is_named_and_does_not_run_restoration(self):
        adapter, _, _, gfp, _ = backend(restore_blend=0)
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["restore_blend"], 0)
        self.assertEqual(len(gfp.received), 0)
        np.testing.assert_allclose(output[32, 32], [200, 200, 200], atol=1)

    def test_duplicate_backward_nan_timestamps_rejected(self):
        adapter, _, _, _, _ = backend()
        adapter.prepare(self.rgb)
        adapter.process(self.rgb, 40)
        for invalid in [40, 39, float("nan"), -1]:
            with self.assertRaises(ValueError):
                adapter.process(self.rgb, invalid)

    def test_mask_layouts_equivalent(self):
        image = np.full((32, 32, 3), 255, np.uint8)
        for shape, expected in [([1, 3, 256, 256], (1, 3, 256, 256)),
                                ([1, 256, 256, 3], (1, 256, 256, 3))]:
            inp = ImageInput.from_session(Session(shape, None), 256)
            tensor = inp.tensor(image)
            self.assertEqual(tensor.shape, expected)
            self.assertEqual(tensor.dtype, np.float32)
            self.assertTrue(np.all(tensor == 1))
        for shape in [(1, 1, 32, 32), (1, 32, 32, 1)]:
            np.testing.assert_array_equal(decode_mask(np.ones(shape)), np.ones((32, 32)))

    def test_ambiguous_model_layout_rejected(self):
        with self.assertRaises(ValueError):
            ImageInput.from_session(Session([1, 3, 3, 3], None), 256)

    def test_padded_box_matches_legacy_integer_padding(self):
        self.assertEqual(padded_box([12.9, 12.9, 52.9, 52.9], self.rgb.shape), (5, 5, 59, 59))
        self.assertEqual(padded_box([0, 0, 40, 40], self.rgb.shape), (0, 0, 47, 47))

    def test_strict_cuda_missing_provider_rejected_at_construction(self):
        adapter, app, swapper, gfp, occ = backend()
        with self.assertRaises(RuntimeError):
            AfnanBackend("unused", providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                         strict_cuda=True, hand_protection=False,
                         _components=(app, swapper, gfp, occ))

    def test_invalid_reference_and_input_rejected(self):
        adapter, app, _, _, _ = backend()
        app.faces = []
        with self.assertRaises(ValueError):
            adapter.prepare(self.rgb)
        with self.assertRaises(ValueError):
            adapter.prepare(self.rgb.astype(np.float32))

    def test_swap_uses_aligned_blob_and_cached_latent(self):
        adapter, _, swapper, _, _ = backend()
        adapter.prepare(self.rgb)
        adapter.process(self.rgb, 0)
        self.assertEqual(len(swapper.session.received), 1)
        inputs = swapper.session.received[0]
        self.assertEqual(set(inputs), {"target", "latent"})
        blob = inputs["target"]
        self.assertEqual(blob.shape, (1, 3, 128, 128))
        self.assertEqual(blob.dtype, np.float32)
        # Uniform test frame RGB (30,60,90) -> aligned crop BGR (90,60,30) ->
        # blob is RGB order scaled by 1/255, matching the upstream contract.
        np.testing.assert_allclose(blob[0, :, 64, 64], [30/255.0, 60/255.0, 90/255.0], atol=1e-4)
        np.testing.assert_allclose(inputs["latent"], [[1.0, 0.0]], atol=1e-6)

    def test_target_recognition_not_run_per_frame(self):
        adapter, app, _, _, _ = backend()
        adapter.prepare(self.rgb)
        self.assertEqual(len(app.received), 1)
        adapter.process(self.rgb, 0)
        adapter.process(self.rgb, 40)
        # app.get (detection + recognition) ran only for the reference frame.
        self.assertEqual(len(app.received), 1)
        self.assertEqual(app.det_model.calls, 2)

    def test_roi_reported_and_covers_aligned_face(self):
        adapter, _, _, _, _ = backend()
        adapter.prepare(self.rgb)
        _, diag = adapter.process(self.rgb, 0)
        # kps = template*0.5 maps the full 64x64 frame to the aligned space,
        # so the padded paste ROI clamps to the whole frame.
        self.assertEqual(diag["crop_box"], [0, 0, 64, 64])
        self.assertEqual(diag["aligned_work_size"], 512)

    def test_nan_landmarks_fall_back_to_original(self):
        adapter, app, _, _, _ = backend()
        bad = face(kps=np.full((5, 2), np.nan, np.float32))
        app.faces = [bad]
        adapter.prepare(self.rgb)
        output, diag = adapter.process(self.rgb, 0)
        self.assertEqual(diag["reason"], "swap_align_error")
        np.testing.assert_array_equal(output, self.rgb)

    def test_reference_embedding_must_match_emap(self):
        adapter, app, _, _, _ = backend()
        bad = face()
        bad.normed_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        app.faces = [bad]
        with self.assertRaises(ValueError):
            adapter.prepare(self.rgb)

    def test_face_template_scales_by_family(self):
        t128 = face_template(128)
        np.testing.assert_allclose(t128[0], [46.2946, 51.6963], atol=1e-4)
        np.testing.assert_allclose(face_template(112)[0], [38.2946, 51.6963], atol=1e-4)
        np.testing.assert_allclose(face_template(256), t128 * 2, atol=1e-4)
        np.testing.assert_allclose(face_template(512), t128 * 4, atol=1e-4)
        with self.assertRaises(ValueError):
            face_template(100)

    def test_estimate_affine_simple_scale(self):
        matrix = estimate_affine(face_template(128) * 0.5, 128)
        np.testing.assert_allclose(matrix, [[2, 0, 0], [0, 2, 0]], atol=1e-4)

    def test_compose_aligned_restricts_to_roi(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        frame[..., 0], frame[..., 1], frame[..., 2] = 10, 20, 30
        content = np.full((128, 128, 3), 250, np.uint8)
        alpha = np.ones((128, 128), np.float32)
        forward = np.array([[2.0, 0.0, 8.0], [0.0, 2.0, 8.0]])
        out, diag = compose_aligned(frame, content, alpha, forward)
        self.assertEqual(diag["crop_box"], [0, 0, 62, 62])
        np.testing.assert_array_equal(out[32, 32], [250, 250, 250])
        np.testing.assert_array_equal(out[63, 63], [10, 20, 30])
        np.testing.assert_array_equal(out[63, 0], [10, 20, 30])

    def test_compose_aligned_rejects_bad_inputs(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        forward = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            compose_aligned(frame, np.zeros((128, 127, 3), np.uint8), np.ones((128, 127), np.float32), forward)
        with self.assertRaises(ValueError):
            compose_aligned(frame, np.zeros((128, 128, 3), np.uint8),
                            np.full((128, 128), np.nan, np.float32), forward)


if __name__ == "__main__":
    unittest.main()
