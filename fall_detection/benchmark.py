from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .detector import Yolo26ByteTracker
from .pose import RTMPose17


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO26 + RTMPose throughput without GCN"
    )
    parser.add_argument("--source", required=True, help="Video path or camera index")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Stop after this many processed frames",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=15,
        help="Ignore the first N frames when computing averages",
    )
    return parser


def benchmark(source: str | int, config_path: str | Path, max_frames: int, warmup_frames: int) -> None:
    config = load_config(config_path)
    detector = Yolo26ByteTracker(config.detector)
    pose = RTMPose17(config.pose)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    frame_index = 0
    measured_frames = 0
    detector_seconds = 0.0
    pose_seconds = 0.0
    total_boxes = 0
    started = time.perf_counter()
    is_camera = isinstance(source, int)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp = (
                time.monotonic()
                if is_camera
                else _video_timestamp(capture, frame_index)
            )

            detector_start = time.perf_counter()
            tracked_boxes = detector(frame)
            detector_elapsed = time.perf_counter() - detector_start

            pose_start = time.perf_counter()
            pose(frame, tracked_boxes, timestamp=timestamp)
            pose_elapsed = time.perf_counter() - pose_start

            frame_index += 1
            if frame_index > warmup_frames:
                measured_frames += 1
                detector_seconds += detector_elapsed
                pose_seconds += pose_elapsed
                total_boxes += len(tracked_boxes)

            if frame_index >= max_frames:
                break
    finally:
        capture.release()

    total_seconds = time.perf_counter() - started
    detector_fps = measured_frames / max(detector_seconds, 1e-6)
    pose_fps = measured_frames / max(pose_seconds, 1e-6)
    end_to_end_fps = measured_frames / max(total_seconds, 1e-6)
    avg_people = total_boxes / max(measured_frames, 1)

    print(f"Frames measured: {measured_frames}")
    print(f"Average people per frame: {avg_people:.2f}")
    print(f"YOLO FPS: {detector_fps:.2f}")
    print(f"Pose FPS: {pose_fps:.2f}")
    print(f"YOLO->Pose wall FPS: {end_to_end_fps:.2f}")


def _video_timestamp(capture: cv2.VideoCapture, frame_index: int) -> float:
    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if np.isfinite(position_ms) and position_ms > 0:
        return position_ms / 1000.0
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 25.0
    return frame_index / fps


def _source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def main() -> None:
    args = build_parser().parse_args()
    benchmark(
        _source(args.source),
        Path(args.config),
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
    )


if __name__ == "__main__":
    main()