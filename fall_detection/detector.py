from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import DetectorConfig
from .types import TrackedBox


class Yolo26ByteTracker:
    """YOLO26 person-only detector with ByteTrack identity assignment."""

    PERSON_CLASS_ID = 0

    def __init__(self, config: DetectorConfig):
        from ultralytics import YOLO

        self.config = config
        self.device = self._resolve_device(config.device)
        self.half = bool(config.half and self.device != "cpu")
        self.model = YOLO(config.model)
        tracker_path = Path(config.tracker_config)
        self.tracker_config = str(tracker_path.resolve()) if tracker_path.exists() else config.tracker_config

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        if requested.startswith("cuda:"):
            return requested.split(":", 1)[1]
        return requested

    def __call__(self, frame: np.ndarray) -> list[TrackedBox]:
        results = self.model.track(
            source=frame,
            persist=True,
            classes=[self.PERSON_CLASS_ID],
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            tracker=self.tracker_config,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)

        return [
            TrackedBox(int(track_id), box.astype(np.float32), float(confidence))
            for box, track_id, confidence, class_id in zip(xyxy, ids, confidences, classes)
            if class_id == self.PERSON_CLASS_ID
        ]

    def reset(self) -> None:
        """Reset video-specific tracker state while retaining loaded weights."""
        self.model.predictor = None
