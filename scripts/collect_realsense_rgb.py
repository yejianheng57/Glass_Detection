#!/usr/bin/env python3
"""Collect RGB images from an Intel RealSense D435i camera."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pyrealsense2 as rs


IMAGE_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<index>\d{6})\.jpg$")
LOG_FIELDS = [
    "image_name",
    "timestamp",
    "camera_serial",
    "camera_mode",
    "width",
    "height",
    "fps",
    "intrinsics_json",
    "robot_pose_x",
    "robot_pose_y",
    "robot_pose_z",
    "robot_pose_rx",
    "robot_pose_ry",
    "robot_pose_rz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview RealSense RGB frames and save images with Enter."
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/glass_rect/raw_images",
        help="Directory used to save captured jpg images.",
    )
    parser.add_argument(
        "--log-file",
        default="datasets/glass_rect/capture_log.csv",
        help="CSV capture log path.",
    )
    parser.add_argument("--prefix", default="glass", help="Image file prefix.")
    parser.add_argument("--width", type=int, default=1280, help="RGB stream width.")
    parser.add_argument("--height", type=int, default=720, help="RGB stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RGB stream FPS.")
    parser.add_argument(
        "--mode",
        choices=("fixed", "robot"),
        default="fixed",
        help="Capture mode recorded in the log.",
    )
    parser.add_argument(
        "--robot-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Optional robot end-effector pose saved with every image.",
    )
    return parser.parse_args()


def next_image_index(output_dir: Path, prefix: str) -> int:
    max_index = 0
    for image_path in output_dir.glob(f"{prefix}_*.jpg"):
        match = IMAGE_PATTERN.match(image_path.name)
        if match and match.group("prefix") == prefix:
            max_index = max(max_index, int(match.group("index")))
    return max_index + 1


def ensure_log_header(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists() and log_file.stat().st_size > 0:
        return

    with log_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        writer.writeheader()


def intrinsics_to_json(intrinsics: rs.intrinsics) -> str:
    payload = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def pose_values(robot_pose: Iterable[float] | None) -> dict[str, str | float]:
    keys = [
        "robot_pose_x",
        "robot_pose_y",
        "robot_pose_z",
        "robot_pose_rx",
        "robot_pose_ry",
        "robot_pose_rz",
    ]
    if robot_pose is None:
        return {key: "" for key in keys}
    return dict(zip(keys, robot_pose, strict=True))


def append_log(
    log_file: Path,
    *,
    image_name: str,
    camera_serial: str,
    args: argparse.Namespace,
    intrinsics_json: str,
) -> None:
    row = {
        "image_name": image_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "camera_serial": camera_serial,
        "camera_mode": args.mode,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "intrinsics_json": intrinsics_json,
        **pose_values(args.robot_pose),
    }
    with log_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_file = Path(args.log_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_log_header(log_file)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    device = profile.get_device()
    camera_serial = device.get_info(rs.camera_info.serial_number)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics_json = intrinsics_to_json(color_profile.get_intrinsics())

    image_index = next_image_index(output_dir, args.prefix)
    window_name = "RealSense RGB Capture - Enter: save, q/Esc: quit"

    print(f"Saving images to: {output_dir}")
    print(f"Writing log to: {log_file}")
    print("Press Enter to save the current frame; press q or Esc to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            preview = image.copy()
            cv2.putText(
                preview,
                f"Next: {args.prefix}_{image_index:06d}.jpg",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (10, 13):
                image_name = f"{args.prefix}_{image_index:06d}.jpg"
                image_path = output_dir / image_name
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Failed to save image: {image_path}")
                append_log(
                    log_file,
                    image_name=image_name,
                    camera_serial=camera_serial,
                    args=args,
                    intrinsics_json=intrinsics_json,
                )
                print(f"Saved {image_path}")
                image_index += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
