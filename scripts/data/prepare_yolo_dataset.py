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
    scene_name: str | None = None


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
        "--scene-dirs",
        nargs="*",
        default=None,
        help=(
            "Optional scene directories to merge. Each scene should contain raw_images/raw_labels. "
            "If omitted and the default raw directories do not exist, child scene folders under "
            "--output-dir are discovered automatically."
        ),
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


def find_existing_dir(parent: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = parent / name
        if path.exists():
            return path
    return None


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
    scene_name: str | None = None,
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    warnings: list[str] = []

    for image_path in list_images(raw_images_dir):
        label_path = raw_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            message = f"skip {image_path.name}: missing {label_path.name}"
            if include_unlabeled:
                warnings.append(f"include {image_path.name} as unlabeled negative sample")
                samples.append(Sample(image_path=image_path, label_path=None, scene_name=scene_name))
            else:
                warnings.append(message)
            continue

        label_errors = validate_label(label_path)
        if label_errors:
            warnings.extend(label_errors)
            continue

        samples.append(Sample(image_path=image_path, label_path=label_path, scene_name=scene_name))

    return samples, warnings


def discover_scene_dirs(output_dir: Path) -> list[Path]:
    scene_dirs: list[Path] = []
    if not output_dir.exists():
        return scene_dirs

    for path in sorted(output_dir.iterdir()):
        if not path.is_dir():
            continue
        images_dir = find_existing_dir(path, ("raw_images", "raw_image"))
        labels_dir = find_existing_dir(path, ("raw_labels", "raw_lables"))
        if images_dir is not None and labels_dir is not None:
            scene_dirs.append(path)
    return scene_dirs


def collect_scene_samples(
    scene_dirs: list[Path],
    include_unlabeled: bool,
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    warnings: list[str] = []

    for scene_dir in scene_dirs:
        images_dir = find_existing_dir(scene_dir, ("raw_images", "raw_image"))
        labels_dir = find_existing_dir(scene_dir, ("raw_labels", "raw_lables"))
        if images_dir is None:
            warnings.append(f"skip {scene_dir}: missing raw_images/raw_image")
            continue
        if labels_dir is None:
            if include_unlabeled:
                warnings.append(f"include {scene_dir}: missing raw_labels/raw_lables, all images unlabeled")
                labels_dir = scene_dir / "raw_labels"
            else:
                warnings.append(f"skip {scene_dir}: missing raw_labels/raw_lables")
                continue

        scene_samples, scene_warnings = collect_samples(
            images_dir,
            labels_dir,
            include_unlabeled,
            scene_name=scene_dir.name,
        )
        samples.extend(scene_samples)
        warnings.extend(f"{scene_dir.name}: {warning}" for warning in scene_warnings)
        warnings.append(f"{scene_dir.name}: samples={len(scene_samples)}")

    return samples, warnings


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0.0, 1.0)")

    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        key = sample.scene_name or "__default__"
        grouped.setdefault(key, []).append(sample)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    rng = random.Random(seed)

    for group_samples in grouped.values():
        shuffled = group_samples.copy()
        rng.shuffle(shuffled)

        if len(shuffled) < 2 or val_ratio == 0.0:
            train_samples.extend(shuffled)
            continue

        val_count = max(1, int(round(len(shuffled) * val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)
        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


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
        output_image_name = sample.image_path.name
        output_stem = sample.image_path.stem
        if sample.scene_name:
            output_image_name = f"{sample.scene_name}__{sample.image_path.name}"
            output_stem = f"{sample.scene_name}__{sample.image_path.stem}"

        shutil.copy2(sample.image_path, images_dir / output_image_name)
        output_label_path = labels_dir / f"{output_stem}.txt"
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

    if args.scene_dirs is not None:
        scene_dirs = [Path(path) for path in args.scene_dirs]
        samples, warnings = collect_scene_samples(scene_dirs, args.include_unlabeled)
    elif raw_images_dir.exists():
        if not raw_labels_dir.exists() and not args.include_unlabeled:
            raise FileNotFoundError(f"Raw labels directory does not exist: {raw_labels_dir}")
        samples, warnings = collect_samples(raw_images_dir, raw_labels_dir, args.include_unlabeled)
    else:
        scene_dirs = discover_scene_dirs(output_dir)
        if not scene_dirs:
            raise FileNotFoundError(
                f"Raw images directory does not exist and no scene folders were found: {raw_images_dir}"
            )
        samples, warnings = collect_scene_samples(scene_dirs, args.include_unlabeled)

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
