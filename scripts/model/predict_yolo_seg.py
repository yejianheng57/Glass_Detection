#!/usr/bin/env python3
"""Run YOLO Seg prediction on images and export mask geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_imgsz(values: list[int]) -> int | list[int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values
    raise argparse.ArgumentTypeError("--imgsz expects one value or two values: --imgsz 640 or --imgsz 736 1280")


CSV_FIELDS = [
    "image_name",
    "instance_index",
    "class_id",
    "confidence",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "center_x",
    "center_y",
    "area_px",
    "angle_deg",
    "point_count",
    "polygon_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict glass hole masks on images.")
    parser.add_argument(
        "--weights",
        default="runs/segment/glass_hole_yolo26n/weights/best.pt",
        help="Trained YOLO Seg weights path.",
    )
    parser.add_argument(
        "--source",
        default="datasets/glass_rect/images/val",
        help="Image file or image directory.",
    )
    parser.add_argument("--output-dir", default="runs/predict/glass_hole", help="Output folder.")
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


def extract_rows(result) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    image_name = Path(result.path).name

    if result.masks is None or result.boxes is None:
        return rows

    polygons = result.masks.xy
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    for index, points in enumerate(polygons):
        points_array = np.asarray(points, dtype=np.float32)
        center_x, center_y, area, angle = polygon_geometry(points_array)
        box = boxes[index]
        polygon = [[round(float(x), 2), round(float(y), 2)] for x, y in points_array]

        rows.append(
            {
                "image_name": image_name,
                "instance_index": index,
                "class_id": int(class_ids[index]),
                "confidence": round(float(confidences[index]), 4),
                "box_x1": round(float(box[0]), 2),
                "box_y1": round(float(box[1]), 2),
                "box_x2": round(float(box[2]), 2),
                "box_y2": round(float(box[3]), 2),
                "center_x": round(center_x, 2),
                "center_y": round(center_y, 2),
                "area_px": round(area, 2),
                "angle_deg": round(angle, 2),
                "point_count": len(points_array),
                "polygon_json": json.dumps(polygon, ensure_ascii=True, separators=(",", ":")),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    imgsz = parse_imgsz(args.imgsz)
    weights_path = Path(args.weights)
    source_path = Path(args.source)
    output_dir = Path(args.output_dir)
    overlays_dir = output_dir / "overlays"
    csv_path = output_dir / "predictions.csv"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    predict_kwargs = {
        "source": str(source_path),
        "imgsz": imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "stream": True,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    all_rows: list[dict[str, str | int | float]] = []
    for result in model.predict(**predict_kwargs):
        image_name = Path(result.path).name
        overlay = result.plot()
        cv2.imwrite(str(overlays_dir / image_name), overlay)

        rows = extract_rows(result)
        all_rows.extend(rows)
        print(f"{image_name}: {len(rows)} instances")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved overlays to: {overlays_dir}")
    print(f"Saved geometry csv to: {csv_path}")


if __name__ == "__main__":
    main()
