from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


@dataclass(slots=True)
class TrackedBox:
    detector_track_id: int
    bbox: np.ndarray
    confidence: float


@dataclass(slots=True)
class PoseObservation:
    track_id: int
    detector_track_id: int
    timestamp: float
    bbox: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray
    detector_confidence: float
    frame_size: tuple[int, int]

    @property
    def pose_quality(self) -> float:
        valid = self.scores[self.scores > 0]
        return float(np.mean(valid)) if valid.size else 0.0


class AlertState(StrEnum):
    NORMAL = "normal"
    WATCH = "watch"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    RECOVERING = "recovering"


@dataclass(slots=True)
class FallDecision:
    track_id: int
    timestamp: float
    state: AlertState
    rule_score: float
    model_probabilities: dict[str, float]
    lying_duration: float
    reason: str
    model_ready: bool

    @property
    def is_event(self) -> bool:
        return self.state in {AlertState.SUSPECTED, AlertState.CONFIRMED}

