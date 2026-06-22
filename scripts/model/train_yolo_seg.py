#!/usr/bin/env python3
"""Train a YOLO segmentation model for glass hole masks."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_imgsz(values: list[int]) -> int | list[int]:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return values
    raise argparse.ArgumentTypeError("--imgsz expects one value or two values: --imgsz 640 or --imgsz 736 1280")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO Seg on the glass hole dataset.")
    parser.add_argument(
        "--data",
        default="datasets/glass_rect/data.yaml",
        help="YOLO dataset yaml path.",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-seg.pt",
        help="Pretrained YOLO Seg weight, for example yolo26n-seg.pt or yolo26s-seg.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Input image size. Use one value for square input or two values for height width, e.g. 640 or 736 1280.",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device passed to Ultralytics, for example 0, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. 0 is the safest default on Windows.",
    )
    parser.add_argument("--project", default="runs/segment", help="Training output project folder.")
    parser.add_argument("--name", default="glass_hole_yolo26n", help="Training run name.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from an existing last.pt checkpoint instead of starting a new run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imgsz = parse_imgsz(args.imgsz)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml does not exist: {data_path}")

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        train_kwargs = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
            "patience": args.patience,
            "exist_ok": True,
        }
        if args.device is not None:
            train_kwargs["device"] = args.device
        results = model.train(**train_kwargs)

    print("Training finished.")
    print(results)


if __name__ == "__main__":
    main()
