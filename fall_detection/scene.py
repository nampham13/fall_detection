from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SceneCutConfig


@dataclass(slots=True, frozen=True)
class SceneCutResult:
    detected: bool
    pixel_difference: float
    histogram_correlation: float


class SceneCutDetector:
    """Detect hard cuts so identities and temporal evidence never cross scenes."""

    def __init__(self, config: SceneCutConfig):
        self.config = config
        self._previous_gray: np.ndarray | None = None
        self._previous_histogram: np.ndarray | None = None
        self._last_cut_timestamp = float("-inf")

    def update(self, frame: np.ndarray, timestamp: float) -> SceneCutResult:
        gray = cv2.cvtColor(
            cv2.resize(frame, tuple(self.config.input_size), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        histogram = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)

        if self._previous_gray is None or self._previous_histogram is None:
            self._previous_gray = gray
            self._previous_histogram = histogram
            return SceneCutResult(False, 0.0, 1.0)

        pixel_difference = float(
            np.mean(cv2.absdiff(gray, self._previous_gray)) / 255.0
        )
        histogram_correlation = float(
            cv2.compareHist(
                self._previous_histogram, histogram, cv2.HISTCMP_CORREL
            )
        )
        interval_ok = (
            timestamp - self._last_cut_timestamp
            >= self.config.minimum_interval_seconds
        )
        detected = bool(
            self.config.enabled
            and interval_ok
            and pixel_difference >= self.config.pixel_difference_threshold
            and histogram_correlation <= self.config.histogram_correlation_threshold
        )
        if detected:
            self._last_cut_timestamp = timestamp

        self._previous_gray = gray
        self._previous_histogram = histogram
        return SceneCutResult(detected, pixel_difference, histogram_correlation)

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_histogram = None
        self._last_cut_timestamp = float("-inf")
