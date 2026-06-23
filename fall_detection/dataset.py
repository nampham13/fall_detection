from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig, load_config
from .detector import Yolo26ByteTracker
from .identity import SkeletonIdentityResolver
from .pose import RTMPose17
from .temporal import SkeletonHistory
from .types import PoseObservation


LABELS = {"normal": 0, "falling": 1, "lying": 2}


@dataclass(slots=True)
class ManifestRow:
    video: str
    label: str
    split: str
    subject_id: str
    start_seconds: float
    end_seconds: float
    target_x: float | None
    target_y: float | None


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def read_manifest(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            label = raw["label"].strip().lower()
            if label not in LABELS:
                raise ValueError(f"Unsupported label {label!r}; expected {tuple(LABELS)}")
            split = raw["split"].strip().lower()
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unsupported split {split!r}")
            rows.append(
                ManifestRow(
                    video=raw["video"],
                    label=label,
                    split=split,
                    subject_id=raw["subject_id"],
                    start_seconds=float(raw["start_seconds"]),
                    end_seconds=float(raw["end_seconds"]),
                    target_x=_optional_float(raw.get("target_x")),
                    target_y=_optional_float(raw.get("target_y")),
                )
            )
    return rows


class SkeletonDatasetBuilder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = Yolo26ByteTracker(config.detector)
        self.pose = RTMPose17(config.pose)

    def build(self, rows: list[ManifestRow], output_root: str | Path) -> None:
        output_root = Path(output_root)
        for index, row in enumerate(rows):
            features, metadata = self._extract(row)
            destination = (
                output_root
                / row.split
                / _safe_name(row.subject_id)
                / f"sample_{index:06d}.npz"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination,
                x=features,
                y=np.int64(LABELS[row.label]),
                metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            print(f"[{index + 1}/{len(rows)}] {destination}")

    def _extract(self, row: ManifestRow) -> tuple[np.ndarray, dict[str, object]]:
        if row.end_seconds - row.start_seconds < self.config.temporal.window_seconds:
            raise ValueError(
                f"{row.video}: sample duration must be >= "
                f"{self.config.temporal.window_seconds:.2f}s"
            )
        capture = cv2.VideoCapture(row.video)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {row.video}")
        capture.set(cv2.CAP_PROP_POS_MSEC, row.start_seconds * 1000.0)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        self.detector.reset()
        identity = SkeletonIdentityResolver(self.config.identity)
        history = SkeletonHistory(self.config.temporal)
        target_track_id: int | None = None
        last_timestamp = row.start_seconds

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = (
                    position_ms / 1000.0
                    if np.isfinite(position_ms) and position_ms > 0
                    else last_timestamp + 1.0 / fps
                )
                last_timestamp = timestamp
                if timestamp > row.end_seconds:
                    break

                boxes = self.detector(frame)
                poses = self.pose(frame, boxes)
                observations = identity.resolve(
                    timestamp, frame.shape[:2], poses
                )
                target = self._select_target(
                    observations, target_track_id, row.target_x, row.target_y
                )
                if target is not None:
                    target_track_id = target.track_id
                    history.add(target)
        finally:
            capture.release()

        if target_track_id is None:
            raise RuntimeError(f"{row.video}: no target person found")
        features = history.model_input(target_track_id)
        if features is None:
            raise RuntimeError(
                f"{row.video}: insufficient/fragmented pose history for sample"
            )
        return features, {
            "video": row.video,
            "label": row.label,
            "split": row.split,
            "subject_id": row.subject_id,
            "start_seconds": row.start_seconds,
            "end_seconds": row.end_seconds,
            "target_track_id": target_track_id,
        }

    @staticmethod
    def _select_target(
        observations: list[PoseObservation],
        target_track_id: int | None,
        target_x: float | None,
        target_y: float | None,
    ) -> PoseObservation | None:
        if not observations:
            return None
        if target_track_id is not None:
            for observation in observations:
                if observation.track_id == target_track_id:
                    return observation
            return None
        if target_x is not None and target_y is not None:
            height, width = observations[0].frame_size
            target = np.array([target_x * width, target_y * height], dtype=np.float32)
            return min(
                observations,
                key=lambda item: float(
                    np.linalg.norm((item.bbox[:2] + item.bbox[2:]) * 0.5 - target)
                ),
            )
        return max(
            observations,
            key=lambda item: float(
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
        )


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build ST-GCN NPZ samples from an explicitly split CSV manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    builder = SkeletonDatasetBuilder(load_config(args.config))
    builder.build(read_manifest(args.manifest), args.output)


if __name__ == "__main__":
    main()

