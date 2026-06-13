#!/usr/bin/env python3
"""Build a YOLO segmentation train/val dataset from raw images and labels."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare images/train, images/val, labels/train, labels/val for YOLO Seg."
    )
    parser.add_argument(
        "--raw-images-dir",
        default="datasets/glass_rect/raw_images",
        help="Directory containing captured images.",
    )
    parser.add_argument(
        "--raw-labels-dir",
        default="datasets/glass_rect/raw_labels",
        help="Directory containing YOLO Seg txt labels from the annotator.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/glass_rect",
        help="Dataset root where images/, labels/ and data.yaml will be written.",
    )
    parser.add_argument("--class-name", default="glass_hole", help="YOLO class name.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include images without label files as negative samples with empty labels.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated train/val image and label folders before copying.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
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
            errors.append(f"{label_path}:{line_number}: class_id must be 0 for glass_hole")
        if any(coord < 0.0 or coord > 1.0 for coord in coords):
            errors.append(f"{label_path}:{line_number}: coordinates must be normalized to 0-1")
    return errors


def collect_samples(
    raw_images_dir: Path,
    raw_labels_dir: Path,
    include_unlabeled: bool,
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    warnings: list[str] = []

    for image_path in list_images(raw_images_dir):
        label_path = raw_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            message = f"skip {image_path.name}: missing {label_path.name}"
            if include_unlabeled:
                warnings.append(f"include {image_path.name} as unlabeled negative sample")
                samples.append(Sample(image_path=image_path, label_path=None))
            else:
                warnings.append(message)
            continue

        label_errors = validate_label(label_path)
        if label_errors:
            warnings.extend(label_errors)
            continue

        samples.append(Sample(image_path=image_path, label_path=label_path))

    return samples, warnings


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0.0, 1.0)")

    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) < 2 or val_ratio == 0.0:
        return shuffled, []

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def reset_generated_dirs(output_dir: Path, clean: bool) -> None:
    generated_dirs = [
        output_dir / "images" / "train",
        output_dir / "images" / "val",
        output_dir / "labels" / "train",
        output_dir / "labels" / "val",
    ]
    for directory in generated_dirs:
        if clean and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def copy_split(samples: list[Sample], output_dir: Path, split: str) -> None:
    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split

    for sample in samples:
        shutil.copy2(sample.image_path, images_dir / sample.image_path.name)
        output_label_path = labels_dir / f"{sample.image_path.stem}.txt"
        if sample.label_path is None:
            output_label_path.write_text("", encoding="utf-8")
        else:
            shutil.copy2(sample.label_path, output_label_path)


def write_data_yaml(output_dir: Path, class_name: str) -> None:
    yaml_path = output_dir / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_images_dir = Path(args.raw_images_dir)
    raw_labels_dir = Path(args.raw_labels_dir)
    output_dir = Path(args.output_dir)

    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Raw images directory does not exist: {raw_images_dir}")
    if not raw_labels_dir.exists() and not args.include_unlabeled:
        raise FileNotFoundError(f"Raw labels directory does not exist: {raw_labels_dir}")

    samples, warnings = collect_samples(raw_images_dir, raw_labels_dir, args.include_unlabeled)
    for warning in warnings:
        print(f"Warning: {warning}")

    if not samples:
        raise RuntimeError("No valid samples found. Check image and label directories.")

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    reset_generated_dirs(output_dir, args.clean)
    copy_split(train_samples, output_dir, "train")
    copy_split(val_samples, output_dir, "val")
    write_data_yaml(output_dir, args.class_name)

    print(f"Prepared dataset: {output_dir}")
    print(f"Train images: {len(train_samples)}")
    print(f"Val images: {len(val_samples)}")
    print(f"Config: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
