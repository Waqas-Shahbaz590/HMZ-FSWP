"""Frame-based Afnan adapter: aligned single-paste face swap pipeline.

The evidence copy source_original.py is intentionally never imported. There are
no listeners, cameras, JPEG transport, automatic downloads, or future frames.

Per-frame flow (all model views share one landmark-aligned face space):
  detect (detector only; recognition is NOT run on target frames)
  -> optional independent hand cascade (palm detect -> 21-pt landmark ->
     conservative frame-space hand mask)
  -> aligned 128 crop -> INSwapper session with the cached source latent
  -> XSeg on the aligned ORIGINAL crop (never generated pixels)
  -> optional GFPGAN on the aligned swapped face
  -> one alpha mask (face oval x XSeg, suppressed under confirmed hands),
     one bounded ROI paste into a copy of the original frame.
Confidence gates: an implausible alignment transform or a failed model stage
returns a labeled original frame; low-confidence hand evidence is discarded
(counted in hand_rejected) without failing the frame, since XSeg still
constrains the paste.
This replaces the upstream INSwapper.get(paste_back=True) full-frame paste and
the second crop composite, whose full-frame host work dominated the old
backend latency and could leave generated pixels outside the protected ROI.
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from . import occlusion
from .alignment import ImplausibleTransform, aligned_crop, estimate_affine, validate_affine
from .compositing import base_face_mask, compose_aligned
from .tensors import ImageInput, decode_mask, decode_restoration, validate_rgb


STAGES = ("detect", "hand_detect_prep", "hand_detect_session",
          "hand_landmark_prep", "hand_landmark_session",
          "swap_align", "swap_session", "swap_decode",
          "occlusion_prep", "occlusion_session",
          "restore_prep", "restore_session",
          "composite", "convert")


def _provider_name(provider):
    return provider[0] if isinstance(provider, (tuple, list)) else provider


def _session_providers(session, strict_cuda, stage):
    actual = list(session.get_providers())
    if strict_cuda and "CUDAExecutionProvider" not in actual:
        raise RuntimeError(f"{stage} lost CUDA provider: {actual}")
    if hasattr(session, "disable_fallback"):
        session.disable_fallback()
    return actual


def _largest_face(faces):
    return max(faces, key=lambda face: max(0, float(face.bbox[2]-face.bbox[0])) * max(0, float(face.bbox[3]-face.bbox[1])))


def _largest_detection(bboxes):
    areas = [max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1])) for b in bboxes[:, 0:4]]
    return int(np.argmax(areas))


class AfnanBackend:
    """One fixed reference identity and causal current-frame inference.

    models_dir contains inswapper_128.onnx, gfpgan_1.4.onnx, xseg_1.onnx and
    buffalo_l/*.onnx. CUDA initialization is mandatory when strict_cuda=True.
    Session provider availability is recorded; common-runner profiling can
    establish node execution. CPU graph nodes remain explicitly permitted.

    Target frames run detection only; the swap consumes the cached reference
    latent, so per-frame recognition on the target is not needed here. A
    deliberate identity/reacquisition policy is future tracking work.
    """

    def __init__(self, models_dir, providers=None, restore_blend=0.75,
                 strict_cuda=True, *, hand_protection=True,
                 palm_score_threshold=0.5, hand_presence_threshold=0.5,
                 _components=None):
        self.models_dir = Path(models_dir)
        self.providers = providers or [
            ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}),
            "CPUExecutionProvider",
        ]
        self.strict_cuda = bool(strict_cuda)
        if self.strict_cuda and "CUDAExecutionProvider" not in [_provider_name(x) for x in self.providers]:
            raise ValueError("strict_cuda requires a requested CUDAExecutionProvider")
        self.restore_blend = float(restore_blend)
        if not 0 <= self.restore_blend <= 1:
            raise ValueError("restore_blend must be in [0,1]")
        self.hand_protection = bool(hand_protection)
        for name, value in (("palm_score_threshold", palm_score_threshold),
                            ("hand_presence_threshold", hand_presence_threshold)):
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        self.palm_score_threshold = float(palm_score_threshold)
        self.hand_presence_threshold = float(hand_presence_threshold)
        self.source_face = None
        self._source_latent = None
        self.last_pts_ms = None
        if _components is None:
            self.app, self.swapper, self.gfpgan, self.occluder = self._load_models()
            self.palm, self.hand_landmark = self._load_hand_models() if self.hand_protection else (None, None)
        else:
            parts = tuple(_components)
            if len(parts) == 4:
                parts = parts + (None, None)
            if len(parts) != 6:
                raise ValueError("_components must be (app, swapper, gfpgan, occluder[, palm, hand_landmark])")
            self.app, self.swapper, self.gfpgan, self.occluder, self.palm, self.hand_landmark = parts
        if self.hand_protection and (self.palm is None or self.hand_landmark is None):
            raise ValueError("hand_protection requires palm and hand_landmark sessions")
        self.actual_providers = {
            "swap": _session_providers(self.swapper.session, self.strict_cuda, "swap"),
            "restore": _session_providers(self.gfpgan, self.strict_cuda, "restore"),
            "occlusion": _session_providers(self.occluder, self.strict_cuda, "occlusion"),
        }
        if self.hand_protection:
            self.actual_providers["palm"] = _session_providers(self.palm, self.strict_cuda, "palm")
            self.actual_providers["hand_landmark"] = _session_providers(
                self.hand_landmark, self.strict_cuda, "hand_landmark")
        for name, model in self.app.models.items():
            self.actual_providers[f"face_{name}"] = _session_providers(model.session, self.strict_cuda, f"face_{name}")
        self.restore_input = ImageInput.from_session(self.gfpgan, 512)
        self.occlusion_input = ImageInput.from_session(self.occluder, 256)
        self._validate_swapper()
        if self.hand_protection:
            self._validate_hands()
        self._base_mask = base_face_mask(128)

    def _validate_swapper(self):
        """Pin the INSwapper contract used by the aligned swap path."""
        input_names = list(getattr(self.swapper, "input_names", []) or [])
        output_names = list(getattr(self.swapper, "output_names", []) or [])
        if len(input_names) != 2 or len(output_names) != 1:
            raise ValueError("Swapper must expose two inputs (target, latent) and one output")
        input_size = tuple(getattr(self.swapper, "input_size", ()) or ())
        if len(input_size) != 2 or input_size[0] != input_size[1]:
            raise ValueError(f"Swapper input must be square, got {input_size}")
        self.swap_size = int(input_size[0])
        if self.swap_size % 112 != 0 and self.swap_size % 128 != 0:
            raise ValueError(f"Swapper input size must align to the 112/128 template family: {self.swap_size}")
        emap = np.asarray(getattr(self.swapper, "emap", None))
        if emap.ndim != 2 or emap.shape[0] != emap.shape[1] or not np.isfinite(emap).all():
            raise ValueError("Swapper emap must be a finite square matrix")
        self._swap_emap = emap.astype(np.float32)
        self._swap_target_name, self._swap_latent_name = input_names
        self._swap_output_names = list(output_names)
        for name, image_input in (("restore", self.restore_input), ("occlusion", self.occlusion_input)):
            width, height = image_input.size
            if width != height:
                raise ValueError(f"{name} model input must be square, got {image_input.size}")

    def _validate_hands(self):
        """Pin the hand-cascade contract used by the protection path."""
        palm_inputs = self.palm.get_inputs()
        if len(palm_inputs) != 1:
            raise ValueError("Palm detector must expose exactly one input")
        pshape = tuple(palm_inputs[0].shape)
        if len(pshape) != 4 or pshape[1] != 3 or not isinstance(pshape[2], int) \
                or pshape[2] != pshape[3] or pshape[2] < 96:
            raise ValueError(f"Palm detector input must be [1, 3, S, S] with S >= 96: {pshape}")
        self.palm_input_name = palm_inputs[0].name
        self.palm_size = int(pshape[2])
        landmark_inputs = self.hand_landmark.get_inputs()
        if len(landmark_inputs) != 1:
            raise ValueError("Hand landmark model must expose exactly one input")
        lshape = tuple(landmark_inputs[0].shape)
        if len(lshape) != 4 or lshape[1] != 3 or not isinstance(lshape[2], int) \
                or lshape[2] != lshape[3] or lshape[2] < 96:
            raise ValueError(f"Hand landmark input must be [N, 3, S, S] with S >= 96: {lshape}")
        self.hand_input_name = landmark_inputs[0].name
        self.landmark_size = int(lshape[2])
        outputs = self.hand_landmark.get_outputs()
        trailing = [tuple(out.shape)[-1] for out in outputs if isinstance(tuple(out.shape)[-1], int)]
        if len(outputs) != 3 or 63 not in trailing:
            raise ValueError("Hand landmark model must emit xyz[N,63], presence and handedness")

    def _load_hand_models(self):
        import onnxruntime as ort
        palm_path = self.models_dir / "palm_detection_192.onnx"
        landmark_path = self.models_dir / "hand_landmark_224.onnx"
        missing = [str(path) for path in (palm_path, landmark_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("hand_protection requires hand models: " + ", ".join(missing))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        palm = ort.InferenceSession(str(palm_path), sess_options=options, providers=self.providers)
        landmark = ort.InferenceSession(str(landmark_path), sess_options=options, providers=self.providers)
        return palm, landmark

    def _load_models(self):
        # Importing torch lets ORT reuse compatible CUDA libraries on GPU images;
        # CPU-only unit tests do not require either torch or InsightFace.
        if self.strict_cuda:
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model

        pack = self.models_dir / "buffalo_l"
        required = [self.models_dir/x for x in ("inswapper_128.onnx", "gfpgan_1.4.onnx", "xseg_1.onnx")]
        missing = [str(path) for path in required if not path.is_file()]
        if not pack.is_dir() or not list(pack.glob("*.onnx")):
            missing.append(str(pack))
        if missing:
            raise FileNotFoundError("Supply pretrained models before construction: " + ", ".join(missing))
        # An existing absolute pack path resolves directly in InsightFace 0.7.3
        # storage.ensure_available; validation above prevents its downloader.
        app = FaceAnalysis(name=str(pack.resolve()), root=str(self.models_dir.resolve()),
                           allowed_modules=["detection", "recognition"], providers=self.providers)
        app.prepare(ctx_id=0 if self.strict_cuda else -1, det_size=(320, 320), det_thresh=0.25)
        swapper = get_model(str(required[0]), providers=self.providers)
        if swapper is None:
            raise RuntimeError("INSwapper model was not recognized")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        gfpgan = ort.InferenceSession(str(required[1]), sess_options=options, providers=self.providers)
        occluder = ort.InferenceSession(str(required[2]), sess_options=options, providers=self.providers)
        return app, swapper, gfpgan, occluder

    def prepare(self, reference_rgb):
        reference_rgb = validate_rgb(reference_rgb)
        began = perf_counter()
        faces = self.app.get(cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR))
        if not faces:
            raise ValueError("No usable reference face detected")
        source = _largest_face(faces)
        embedding = np.asarray(source.normed_embedding)
        if embedding.ndim != 1 or embedding.size < 1 or not np.isfinite(embedding).all() or np.linalg.norm(embedding) < 1e-8:
            raise ValueError("Invalid reference embedding")
        if embedding.size != self._swap_emap.shape[0]:
            raise ValueError(f"Reference embedding size {embedding.size} does not match emap {self._swap_emap.shape[0]}")
        # Project and normalize the source latent once; the per-frame swap no
        # longer recomputes it and no longer needs target-side recognition.
        latent = embedding.reshape(1, -1).astype(np.float32) @ self._swap_emap
        latent_norm = np.linalg.norm(latent)
        if not np.isfinite(latent).all() or latent_norm < 1e-8:
            raise ValueError("Invalid projected reference latent")
        self._source_latent = np.ascontiguousarray(latent / latent_norm, dtype=np.float32)
        self.source_face = source
        self.last_pts_ms = None
        return {
            "prepare_ms": (perf_counter()-began)*1000,
            "source_face_count": len(faces),
            "source_bbox": np.asarray(source.bbox).tolist(),
            "source_detection_score": float(source.det_score),
            "actual_providers": self.actual_providers,
            "restore_blend": self.restore_blend,
            "detector_size": [320, 320],
            "detector_threshold": 0.25,
        }

    def process(self, frame_rgb, pts_ms):
        frame_rgb = validate_rgb(frame_rgb)
        if self._source_latent is None:
            raise RuntimeError("Call prepare(reference_rgb) before process")
        pts_ms = float(pts_ms)
        if not np.isfinite(pts_ms) or pts_ms < 0 or (self.last_pts_ms is not None and pts_ms <= self.last_pts_ms):
            raise ValueError("Frame timestamps must be finite, nonnegative and strictly increasing")
        self.last_pts_ms = pts_ms
        began = perf_counter()
        diag = {"pts_ms": pts_ms, "status": "original_fallback", "reason": None,
                "synthetic": False, "stages_ms": {stage: 0.0 for stage in STAGES},
                "restore_blend": self.restore_blend,
                "hand_protection": self.hand_protection,
                "hand_count": 0, "hand_rejected": 0, "hand_boxes": []}
        diag["stage_ms"] = diag["stages_ms"]
        convert_begin = perf_counter()
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        diag["stages_ms"]["convert"] += (perf_counter()-convert_begin)*1000
        stage = "detect"
        stage_begin = perf_counter()
        try:
            # Detection only: FaceAnalysis.get would also run recognition on
            # every target face, which the swap does not consume.
            bboxes, kpss = self.app.det_model.detect(bgr, max_num=0, metric="default")
            bboxes = np.asarray(bboxes, dtype=np.float64).reshape(-1, 5)
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000
            diag["target_face_count"] = int(bboxes.shape[0])
            if bboxes.shape[0] == 0:
                diag["reason"] = "face_missing"
                diag["total_ms"] = (perf_counter()-began)*1000
                return frame_rgb.copy(), diag
            if kpss is None:
                raise ValueError("Detector returned no landmarks for alignment")
            index = _largest_detection(bboxes)
            bbox = bboxes[index, 0:4]
            kps = np.asarray(kpss[index], dtype=np.float64)
            diag["target_detection_score"] = float(bboxes[index, 4])
            diag["target_bbox"] = bbox.tolist()
            diag["face_bbox"] = diag["target_bbox"]
            if bbox[2]-bbox[0] < 32 or bbox[3]-bbox[1] < 32:
                raise ValueError("Detected face is too small for aligned processing")

            hand_mask = None
            hands_summary = []
            hand_rejected = 0
            if self.hand_protection:
                stage, stage_begin = "hand_detect_prep", perf_counter()
                letterboxed, lb_scale, lb_pad_x, lb_pad_y = occlusion.letterbox_square(bgr, self.palm_size)
                palm_tensor = occlusion.rgb01_nchw([letterboxed])
                diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

                stage, stage_begin = "hand_detect_session", perf_counter()
                boxes = occlusion.decode_palm_output(
                    self.palm.run(None, {self.palm_input_name: palm_tensor})[0],
                    self.palm_score_threshold)
                hands = occlusion.palm_squares(boxes, bgr.shape, self.palm_size,
                                               lb_scale, lb_pad_x, lb_pad_y)
                diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

                if hands:
                    stage, stage_begin = "hand_landmark_prep", perf_counter()
                    matrices = [occlusion.crop_affine(h["cx"], h["cy"], h["side"],
                                                      h["rotation"], self.landmark_size)
                                for h in hands]
                    crops = [cv2.warpAffine(bgr, m, (self.landmark_size, self.landmark_size),
                                            borderValue=0.0) for m in matrices]
                    landmark_tensor = occlusion.rgb01_nchw(crops)
                    diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

                    stage, stage_begin = "hand_landmark_session", perf_counter()
                    xyz, presence, handed = occlusion.decode_landmark_outputs(
                        self.hand_landmark.run(None, {self.hand_input_name: landmark_tensor}))
                    if xyz.shape[0] != len(crops):
                        raise ValueError("Landmark batch size does not match palm count")
                    confirmed = []
                    for hand, matrix, points, score, hand_flag in zip(hands, matrices, xyz, presence, handed):
                        if score < self.hand_presence_threshold:
                            hand_rejected += 1
                            continue
                        try:
                            frame_points = occlusion.landmarks_to_frame(points, matrix, self.landmark_size)
                        except ValueError:
                            hand_rejected += 1
                            continue
                        confirmed.append((frame_points, hand["side"]))
                        x1, y1 = hand["cx"] - hand["side"]/2, hand["cy"] - hand["side"]/2
                        hands_summary.append({
                            "palm_score": round(float(hand["score"]), 4),
                            "presence": round(float(score), 4),
                            "cx": round(float(hand["cx"]), 1), "cy": round(float(hand["cy"]), 1),
                            "side": round(float(hand["side"]), 1),
                            "rotation_deg": round(float(np.degrees(hand["rotation"])), 1),
                        })
                        diag["hand_boxes"].append([
                            round(float(max(0, x1)), 1), round(float(max(0, y1)), 1),
                            round(float(min(bgr.shape[1], x1 + hand["side"])), 1),
                            round(float(min(bgr.shape[0], y1 + hand["side"])), 1)])
                    if confirmed:
                        hand_mask = occlusion.hand_mask_frame(bgr.shape, confirmed)
                    diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000
                diag["hand_count"] = len(hands_summary)
                diag["hand_rejected"] = hand_rejected
                if hands_summary:
                    diag["hands"] = hands_summary

            stage, stage_begin = "swap_align", perf_counter()
            aimg, forward = aligned_crop(bgr, kps, self.swap_size)
            diag["alignment_gate"] = validate_affine(forward, bbox, self.swap_size)
            blob = cv2.dnn.blobFromImage(aimg, 1.0/255.0, (self.swap_size, self.swap_size),
                                         (0.0, 0.0, 0.0), swapRB=True)
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

            stage, stage_begin = "swap_session", perf_counter()
            pred = self.swapper.session.run(self._swap_output_names,
                                            {self._swap_target_name: blob,
                                             self._swap_latent_name: self._source_latent})[0]
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

            stage, stage_begin = "swap_decode", perf_counter()
            swapped = self._decode_swap(pred)
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

            # XSeg sees the original aligned current-frame crop, never
            # generated pixels, and shares the same face coordinate family.
            stage, stage_begin = "occlusion_prep", perf_counter()
            occ_size = self.occlusion_input.size[0]
            occ_crop, _ = aligned_crop(bgr, kps, occ_size)
            occ_tensor = self.occlusion_input.tensor(occ_crop)
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

            stage, stage_begin = "occlusion_session", perf_counter()
            mask = decode_mask(self.occluder.run(None, {self.occlusion_input.name: occ_tensor})[0])
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

            restored = None
            if self.restore_blend > 0:
                stage, stage_begin = "restore_prep", perf_counter()
                swapped_up = cv2.resize(swapped, self.restore_input.size, interpolation=cv2.INTER_LINEAR)
                restore_tensor = self.restore_input.tensor(swapped_up, normalized_rgb=True)
                diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

                stage, stage_begin = "restore_session", perf_counter()
                restored = decode_restoration(self.gfpgan.run(None, {self.restore_input.name: restore_tensor})[0])
                diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000

            stage, stage_begin = "composite", perf_counter()
            forward128 = forward
            if restored is not None:
                work_size = self.restore_input.size[0]
                swapped_work = cv2.resize(swapped, (work_size, work_size), interpolation=cv2.INTER_LINEAR)
                if restored.shape[:2] != (work_size, work_size):
                    restored = cv2.resize(restored, (work_size, work_size), interpolation=cv2.INTER_LINEAR)
                content = cv2.addWeighted(restored, self.restore_blend, swapped_work, 1-self.restore_blend, 0)
                forward = estimate_affine(kps, work_size)
            else:
                work_size = self.swap_size
                content = swapped
            xseg_mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_LINEAR)
            alpha_raw = self._base_mask * xseg_mask
            if self.hand_protection:
                if hand_mask is None:
                    hand_mask = np.zeros(bgr.shape[:2], dtype=np.float32)
                hand128 = cv2.warpAffine(hand_mask, forward128, (128, 128), borderValue=0.0)
                alpha, coverage = occlusion.protect_alpha(alpha_raw, hand128)
                diag["hand_core_fraction"] = float(coverage)
            else:
                alpha = cv2.GaussianBlur(alpha_raw, (5, 5), 0)
            if work_size != alpha.shape[0]:
                alpha = cv2.resize(alpha, (work_size, work_size), interpolation=cv2.INTER_LINEAR)
            result, mask_diag = compose_aligned(bgr, content, alpha, forward)
            diag.update(mask_diag)
            diag["aligned_work_size"] = work_size
            convert_begin = perf_counter()
            output = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            diag["stages_ms"]["convert"] += (perf_counter()-convert_begin)*1000
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000
            diag.update(status="animated", synthetic=True)
            diag["total_ms"] = (perf_counter()-began)*1000
            return output, diag
        except Exception as exc:
            diag["stages_ms"][stage] = (perf_counter()-stage_begin)*1000
            diag["reason"] = "transform_implausible" if isinstance(exc, ImplausibleTransform) else f"{stage}_error"
            diag["error"] = f"{type(exc).__name__}: {exc}"
            diag["total_ms"] = (perf_counter()-began)*1000
            return frame_rgb.copy(), diag

    def _decode_swap(self, pred):
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[None]
        if pred.ndim != 4 or pred.shape[0] != 1 or pred.shape[1] != 3:
            raise ValueError(f"Unsupported swap output shape: {pred.shape}")
        if not np.isfinite(pred).all():
            raise ValueError("Non-finite swap output")
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        if img_fake.shape[:2] != (self.swap_size, self.swap_size):
            raise ValueError(f"Swap output must match the aligned size {self.swap_size}: {img_fake.shape}")
        bgr_fake = np.clip(255.0 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        return np.ascontiguousarray(bgr_fake)

    def close(self):
        self.source_face = None
        self._source_latent = None
        self.last_pts_ms = None
