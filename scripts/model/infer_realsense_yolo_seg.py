#!/usr/bin/env python3
"""Run real-time YOLO Seg inference from an Intel RealSense D435i RGB stream."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def parse_imgsz(values: list[int]) -> int | list[int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values
    raise argparse.ArgumentTypeError("--imgsz expects one value or two values: --imgsz 640 or --imgsz 736 1280")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense RGB YOLO Seg inference.")
    parser.add_argument(
        "--weights",
        default="runs/segment/glass_hole_yolo26n/weights/best.pt",
        help="Trained YOLO Seg weights path.",
    )
    parser.add_argument("--width", type=int, default=1280, help="RGB stream width.")
    parser.add_argument("--height", type=int, default=720, help="RGB stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RGB stream FPS.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Inference image size. Use one value for square input or two values for height width, e.g. 640 or 736 1280.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Device, for example 0, cpu, or cuda:0.")
    parser.add_argument(
        "--save-dir",
        default="runs/realsense/glass_hole",
        help="Folder used when pressing s to save raw and overlay frames.",
    )
    return parser.parse_args()


def polygon_geometry(points: np.ndarray) -> tuple[float, float, float, float]:
    if len(points) < 3:
        return 0.0, 0.0, 0.0, 0.0

    contour = points.astype(np.float32).reshape((-1, 1, 2))
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
    else:
        center_x = float(points[:, 0].mean())
        center_y = float(points[:, 1].mean())

    area = float(abs(cv2.contourArea(contour)))
    angle = float(cv2.minAreaRect(contour)[2])
    return float(center_x), float(center_y), area, angle


def draw_geometry(overlay, result) -> int:
    if result.masks is None:
        return 0

    polygons = result.masks.xy
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

    for index, points in enumerate(polygons):
        points_array = np.asarray(points, dtype=np.float32)
        center_x, center_y, area, angle = polygon_geometry(points_array)
        center = (int(round(center_x)), int(round(center_y)))
        confidence = float(confidences[index]) if index < len(confidences) else 0.0

        cv2.circle(overlay, center, 4, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            f"#{index} conf={confidence:.2f} area={area:.0f} angle={angle:.1f}",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return len(polygons)


def predict_frame(model: YOLO, image, args: argparse.Namespace):
    predict_kwargs = {
        "source": image,
        "imgsz": args.parsed_imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device
    return model.predict(**predict_kwargs)[0]


def save_frame(save_dir: Path, raw_image, overlay) -> None:
    raw_dir = save_dir / "raw"
    overlay_dir = save_dir / "overlays"
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = raw_dir / f"{timestamp}.jpg"
    overlay_path = overlay_dir / f"{timestamp}.jpg"
    cv2.imwrite(str(raw_path), raw_image)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Saved {raw_path} and {overlay_path}")


def main() -> None:
    args = parse_args()
    args.parsed_imgsz = parse_imgsz(args.imgsz)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    model = YOLO(str(weights_path))
    save_dir = Path(args.save_dir)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    window_name = "RealSense YOLO Seg - q/Esc: quit, s: save"
    pipeline.start(config)

    previous_time = perf_counter()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            result = predict_frame(model, image, args)
            overlay = result.plot()
            instance_count = draw_geometry(overlay, result)

            now = perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            cv2.putText(
                overlay,
                f"holes={instance_count} fps={fps:.1f} conf={args.conf:.2f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                save_frame(save_dir, image, overlay)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
