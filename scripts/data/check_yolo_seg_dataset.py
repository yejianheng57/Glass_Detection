#!/usr/bin/env python3
"""Validate YOLO segmentation images and labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO Seg dataset structure and labels.")
    parser.add_argument(
        "--dataset-dir",
        default="datasets/glass_rect",
        help="Dataset root containing images/ and labels/ folders.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="all",
        help="Dataset split to check.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Open every image with OpenCV to verify the file is readable.",
    )
    return parser.parse_args()


def selected_splits(split: str) -> list[str]:
    return ["train", "val"] if split == "all" else [split]


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_label(label_path: Path) -> list[str]:
    errors: list[str] = []
    if not label_path.exists():
        return [f"missing label file: {label_path}"]

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            errors.append(f"{label_path}:{line_number}: needs class_id and at least 3 x/y pairs")
            continue
        if len(parts[1:]) % 2 != 0:
            errors.append(f"{label_path}:{line_number}: polygon coordinate count must be even")
            continue
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: contains non-numeric values")
            continue
        if class_id != 0:
            errors.append(f"{label_path}:{line_number}: class_id must be 0")
        if any(coord < 0.0 or coord > 1.0 for coord in coords):
            errors.append(f"{label_path}:{line_number}: coordinates must be normalized to 0-1")
    return errors


def check_split(dataset_dir: Path, split: str, check_images: bool) -> tuple[int, int, list[str]]:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    errors: list[str] = []
    object_count = 0

    images = list_images(images_dir)
    if not images:
        errors.append(f"no images found in {images_dir}")
        return 0, 0, errors

    for image_path in images:
        if check_images and cv2.imread(str(image_path)) is None:
            errors.append(f"unreadable image: {image_path}")

        label_path = labels_dir / f"{image_path.stem}.txt"
        label_errors = validate_label(label_path)
        errors.extend(label_errors)

        if label_path.exists() and not label_errors:
            object_count += sum(
                1
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    image_stems = {image.stem for image in images}
    for label_path in label_files:
        if label_path.stem not in image_stems:
            errors.append(f"orphan label without image: {label_path}")

    return len(images), object_count, errors


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    all_errors: list[str] = []
    total_images = 0
    total_objects = 0

    for split in selected_splits(args.split):
        image_count, object_count, errors = check_split(dataset_dir, split, args.check_images)
        total_images += image_count
        total_objects += object_count
        all_errors.extend(errors)
        print(f"{split}: images={image_count}, objects={object_count}, errors={len(errors)}")

    if all_errors:
        print("\nErrors:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"\nOK: images={total_images}, objects={total_objects}")


if __name__ == "__main__":
    main()
