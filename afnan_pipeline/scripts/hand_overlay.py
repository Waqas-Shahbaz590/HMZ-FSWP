"""Diagnostic: run the independent hand cascade on selected source frames.

Saves one overlay PNG per frame (palm square, mapped 21-pt landmarks, mask
contour) plus the exact upright crop the landmark model consumed, and prints
per-frame JSON (scores, presence, stage timings). This is a development
instrument for validating model I/O conventions and mask placement; it is not
part of the offline evaluation or the live path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afnan import occlusion  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--frames", type=int, nargs="+", required=True)
    p.add_argument("--palm-score", type=float, default=0.5)
    p.add_argument("--presence", type=float, default=0.5)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    palm = ort.InferenceSession(str(args.models / "palm_detection_192.onnx"), providers=providers)
    landmark = ort.InferenceSession(str(args.models / "hand_landmark_224.onnx"), providers=providers)
    palm_meta = palm.get_inputs()[0]
    landmark_meta = landmark.get_inputs()[0]
    palm_size = int(palm_meta.shape[2])
    landmark_size = int(landmark_meta.shape[2])
    print(json.dumps({
        "palm": {"input": palm_meta.name, "shape": [str(x) for x in palm_meta.shape],
                 "outputs": [[o.name, [str(d) for d in o.shape]] for o in palm.get_outputs()],
                 "providers": palm.get_providers()},
        "landmark": {"input": landmark_meta.name, "shape": [str(x) for x in landmark_meta.shape],
                     "outputs": [[o.name, [str(d) for d in o.shape]] for o in landmark.get_outputs()],
                     "providers": landmark.get_providers()},
    }), flush=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    for index in args.frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            print(json.dumps({"frame": index, "error": "decode failed"}), flush=True)
            continue
        timings = {}
        began = time.perf_counter()
        letterboxed, scale, pad_x, pad_y = occlusion.letterbox_square(bgr, palm_size)
        tensor = occlusion.rgb01_nchw([letterboxed])
        timings["hand_detect_prep_ms"] = (time.perf_counter() - began) * 1000
        began = time.perf_counter()
        raw = palm.run(None, {palm_meta.name: tensor})
        boxes = occlusion.decode_palm_output(raw[0], args.palm_score)
        hands = occlusion.palm_squares(boxes, bgr.shape, palm_size, scale, pad_x, pad_y)
        timings["hand_detect_session_ms"] = (time.perf_counter() - began) * 1000
        overlay = bgr.copy()
        report = {"frame": index, "raw_palm_output_shape": [str(d) for d in np.asarray(raw[0]).shape],
                  "palms_above_threshold": len(hands), "hands": [], "timings_ms": timings}
        if hands:
            began = time.perf_counter()
            matrices = [occlusion.crop_affine(h["cx"], h["cy"], h["side"], h["rotation"], landmark_size)
                        for h in hands]
            crops = [cv2.warpAffine(bgr, m, (landmark_size, landmark_size), borderValue=0.0)
                     for m in matrices]
            batch = occlusion.rgb01_nchw(crops)
            timings["hand_landmark_prep_ms"] = (time.perf_counter() - began) * 1000
            began = time.perf_counter()
            xyz, presence, handed = occlusion.decode_landmark_outputs(
                landmark.run(None, {landmark_meta.name: batch}))
            timings["hand_landmark_session_ms"] = (time.perf_counter() - began) * 1000
            confirmed = []
            for number, (hand, matrix, points, score) in enumerate(zip(hands, matrices, xyz, presence)):
                entry = {"palm_score": round(float(hand["score"]), 4),
                         "presence": round(float(score), 4),
                         "cx": round(hand["cx"], 1), "cy": round(hand["cy"], 1),
                         "side": round(hand["side"], 1),
                         "rotation_deg": round(float(np.degrees(hand["rotation"])), 1)}
                if score < args.presence:
                    entry["rejected"] = "presence"
                    report["hands"].append(entry)
                    continue
                try:
                    frame_points = occlusion.landmarks_to_frame(points, matrix, landmark_size)
                except ValueError:
                    entry["rejected"] = "geometry"
                    report["hands"].append(entry)
                    continue
                confirmed.append((frame_points, hand["side"]))
                report["hands"].append(entry)
                rect = ((hand["cx"], hand["cy"]), (hand["side"], hand["side"]),
                        float(np.degrees(hand["rotation"])))
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(overlay, [box], 0, (0, 255, 255), 2)
                for px, py in frame_points.astype(int):
                    cv2.circle(overlay, (px, py), 4, (0, 255, 0), -1)
                cv2.imwrite(str(args.output / f"crop_{index:05d}_hand{number}.png"), crops[number])
            if confirmed:
                mask = occlusion.hand_mask_frame(bgr.shape, confirmed)
                tinted = overlay.copy()
                tinted[mask > 0.5] = (0, 0, 255)
                overlay = cv2.addWeighted(tinted, 0.35, overlay, 0.65, 0)
        out_path = args.output / f"overlay_{index:05d}.png"
        cv2.imwrite(str(out_path), overlay)
        report["overlay"] = str(out_path)
        print(json.dumps(report), flush=True)
    capture.release()


if __name__ == "__main__":
    main()
