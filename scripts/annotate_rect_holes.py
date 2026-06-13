#!/usr/bin/env python3
"""Annotate hole polygons and save YOLO Seg labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_ID = 0
CROSSHAIR_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCV annotator for glass hole segmentation polygons."
    )
    parser.add_argument(
        "--images-dir",
        default="datasets/glass_rect/raw_images",
        help="Directory containing captured images.",
    )
    parser.add_argument(
        "--labels-dir",
        default="datasets/glass_rect/raw_labels",
        help="Directory used to save YOLO Seg txt labels.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Radius of displayed polygon point markers in pixels.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    images = [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def label_path_for(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def denormalize_point(x_norm: float, y_norm: float, width: int, height: int) -> tuple[int, int]:
    x = int(round(x_norm * width))
    y = int(round(y_norm * height))
    return x, y


def normalize_point(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    return x / width, y / height


def load_labels(label_path: Path, width: int, height: int) -> list[list[tuple[int, int]]]:
    objects: list[list[tuple[int, int]]] = []
    if not label_path.exists():
        return objects

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or len(parts[1:]) % 2 != 0:
            print(
                f"Skip invalid label line {line_number} in {label_path}: "
                "expected class_id followed by at least 3 x/y pairs"
            )
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            print(f"Skip invalid label line {line_number} in {label_path}: non-numeric value")
            continue
        if any(value < 0.0 or value > 1.0 for value in values[1:]):
            print(
                f"Skip invalid label line {line_number} in {label_path}: "
                "YOLO Seg polygon coordinates must be normalized to 0-1"
            )
            continue

        points: list[tuple[int, int]] = []
        for offset in range(1, len(values), 2):
            x_norm, y_norm = values[offset], values[offset + 1]
            points.append(denormalize_point(x_norm, y_norm, width, height))
        objects.append(points)
    return objects


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def save_labels(
    label_path: Path,
    objects: list[list[tuple[int, int]]],
    width: int,
    height: int,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for points in objects:
        if len(points) < 3:
            continue

        values: list[str] = [str(CLASS_ID)]
        for x, y in points:
            x_norm, y_norm = normalize_point(x, y, width, height)
            values.extend(
                [
                    f"{clamp(x_norm):.6f}",
                    f"{clamp(y_norm):.6f}",
                ]
            )
        lines.append(" ".join(values))

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_crosshair(
    canvas,
    cursor_point: tuple[int, int],
) -> None:
    height, width = canvas.shape[:2]
    x, y = cursor_point
    if not (0 <= x < width and 0 <= y < height):
        return

    cv2.line(canvas, (max(0, x - CROSSHAIR_SIZE), y), (min(width - 1, x + CROSSHAIR_SIZE), y), (0, 0, 255), 1)
    cv2.line(canvas, (x, max(0, y - CROSSHAIR_SIZE)), (x, min(height - 1, y + CROSSHAIR_SIZE)), (0, 0, 255), 1)


def draw_annotations(
    image,
    objects: list[list[tuple[int, int]]],
    current_points: list[tuple[int, int]],
    image_path: Path,
    index: int,
    total: int,
    dirty: bool,
    cursor_point: tuple[int, int] | None,
    point_radius: int,
):
    canvas = image.copy()
    colors = [(0, 255, 0), (0, 180, 255), (255, 180, 0), (255, 0, 180)]
    line_thickness = max(1, point_radius - 1)

    for object_index, points in enumerate(objects, 1):
        for point_index, point in enumerate(points):
            color = colors[point_index % len(colors)]
            cv2.circle(canvas, point, point_radius, color, -1)
            cv2.putText(
                canvas,
                str(point_index + 1),
                (point[0] + point_radius + 3, point[1] - point_radius - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                line_thickness,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [polygon], True, (0, 255, 0), line_thickness)
        center_x = int(sum(point[0] for point in points) / len(points))
        center_y = int(sum(point[1] for point in points) / len(points))
        cv2.putText(
            canvas,
            f"#{object_index}",
            (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            line_thickness,
            cv2.LINE_AA,
        )

    for point_index, point in enumerate(current_points):
        color = colors[point_index % len(colors)]
        cv2.circle(canvas, point, point_radius, color, -1)
        cv2.putText(
            canvas,
            str(point_index + 1),
            (point[0] + point_radius + 3, point[1] - point_radius - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            line_thickness,
            cv2.LINE_AA,
        )
        if point_index > 0:
            cv2.line(canvas, current_points[point_index - 1], point, (255, 255, 0), line_thickness)

    status = "modified" if dirty else "saved"
    help_lines = [
        f"{index + 1}/{total} {image_path.name} [{status}]",
        "Click polygon points | f/Enter: finish instance | s: save | u: undo | d: delete",
        "Move mouse: red crosshair | n/Right: next | p/Left: previous | q/Esc: quit",
    ]
    for line_index, text in enumerate(help_lines):
        y = 28 + line_index * 26
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if cursor_point is not None:
        draw_crosshair(canvas, cursor_point)
    return canvas


class AnnotationSession:
    def __init__(
        self,
        images: list[Path],
        labels_dir: Path,
        point_radius: int,
    ) -> None:
        self.images = images
        self.labels_dir = labels_dir
        self.point_radius = max(1, point_radius)
        self.index = 0
        self.image = None
        self.objects: list[list[tuple[int, int]]] = []
        self.current_points: list[tuple[int, int]] = []
        self.cursor_point: tuple[int, int] | None = None
        self.dirty = False
        self.window_name = "Rect Hole Annotator"

    @property
    def image_path(self) -> Path:
        return self.images[self.index]

    @property
    def label_path(self) -> Path:
        return label_path_for(self.labels_dir, self.image_path)

    def load_current(self) -> None:
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise RuntimeError(f"Failed to load image: {self.image_path}")
        height, width = self.image.shape[:2]
        self.objects = load_labels(self.label_path, width, height)
        self.current_points = []
        self.dirty = False

    def save_current(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        save_labels(self.label_path, self.objects, width, height)
        self.dirty = False
        print(f"Saved {self.label_path}")

    def show(self) -> None:
        if self.image is None:
            return
        canvas = draw_annotations(
            self.image,
            self.objects,
            self.current_points,
            self.image_path,
            self.index,
            len(self.images),
            self.dirty,
            self.cursor_point,
            self.point_radius,
        )
        cv2.imshow(self.window_name, canvas)

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        self.cursor_point = (
            max(0, min(width - 1, x)),
            max(0, min(height - 1, y)),
        )
        if event == cv2.EVENT_MOUSEMOVE:
            self.show()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.current_points.append(self.cursor_point)
        self.dirty = True
        self.show()

    def finish_current_target(self) -> None:
        if len(self.current_points) < 3:
            print("Current instance needs at least 3 points before finishing.")
            return

        self.objects.append(self.current_points.copy())
        self.current_points = []
        self.dirty = True
        self.show()

    def undo(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self.dirty = True
        elif self.objects:
            self.current_points = self.objects.pop()
            self.current_points.pop()
            self.dirty = True
        self.show()

    def delete_last_target(self) -> None:
        if self.current_points:
            self.current_points = []
            self.dirty = True
        elif self.objects:
            self.objects.pop()
            self.dirty = True
        self.show()

    def move(self, offset: int) -> None:
        if self.dirty:
            self.save_current()
        self.index = (self.index + offset) % len(self.images)
        self.load_current()
        self.show()

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.load_current()
        self.show()

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                if self.dirty:
                    self.save_current()
                break
            if key == ord("s"):
                self.save_current()
                self.show()
            elif key == ord("u"):
                self.undo()
            elif key == ord("d"):
                self.delete_last_target()
            elif key in (ord("f"), 10, 13):
                self.finish_current_target()
            elif key in (ord("n"), 83):
                self.move(1)
            elif key in (ord("p"), 81):
                self.move(-1)

        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    session = AnnotationSession(
        images,
        labels_dir,
        args.point_radius,
    )
    session.run()


if __name__ == "__main__":
    main()
