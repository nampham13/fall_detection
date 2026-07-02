from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from .config import TemporalConfig
from .types import PoseObservation


COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


def pelvis_point(observation: PoseObservation) -> np.ndarray:
    indices = np.array([COCO_LEFT_HIP, COCO_RIGHT_HIP])
    valid = observation.scores[indices] >= 0.25
    if np.any(valid):
        return np.mean(observation.keypoints[indices[valid]], axis=0)
    return (observation.bbox[:2] + observation.bbox[2:]) * 0.5


class SkeletonHistory:
    def __init__(self, config: TemporalConfig):
        self.config = config
        self._history: dict[int, deque[PoseObservation]] = defaultdict(deque)

    def add(self, observation: PoseObservation) -> None:
        history = self._history[observation.track_id]
        history.append(observation)
        cutoff = observation.timestamp - self.config.history_seconds
        while history and history[0].timestamp < cutoff:
            history.popleft()

    def get(self, track_id: int) -> list[PoseObservation]:
        return list(self._history.get(track_id, ()))

    def remove_stale(self, timestamp: float, stale_seconds: float) -> list[int]:
        stale = [
            track_id
            for track_id, history in self._history.items()
            if not history or timestamp - history[-1].timestamp > stale_seconds
        ]
        for track_id in stale:
            self._history.pop(track_id, None)
        return stale

    def clear(self) -> None:
        self._history.clear()

    def get_gcn_input(self, track_id: int) -> dict[str, Any] | None:
        """Get skeleton input for HPI-GCN model."""
        sampled = self._sample_track(track_id)
        if sampled is None:
            return None

        height, width = sampled["frame_size"]
        keypoints = sampled["keypoints"][None].astype(np.float32)
        scores = np.clip(sampled["scores"], 0.0, 1.0)[None].astype(np.float32)
        return {
            "keypoint": keypoints,
            "keypoint_score": scores,
            "img_shape": (int(height), int(width)),
            "original_shape": (int(height), int(width)),
        }

    def _sample_track(self, track_id: int) -> dict[str, Any] | None:
        history = self.get(track_id)
        if len(history) < self.config.minimum_observations:
            return None

        timestamps = np.asarray([item.timestamp for item in history], dtype=np.float64)
        if timestamps[-1] - timestamps[0] < self.config.window_seconds * 0.75:
            return None

        start = timestamps[-1] - self.config.window_seconds
        selected = [item for item in history if item.timestamp >= start]
        if len(selected) < self.config.minimum_observations:
            return None

        times = np.asarray([item.timestamp for item in selected], dtype=np.float64)
        if np.max(np.diff(times)) > self.config.maximum_sample_gap_seconds:
            return None

        target_times = np.linspace(
            times[-1] - self.config.window_seconds,
            times[-1],
            self.config.sequence_length,
            dtype=np.float64,
        )
        keypoints = np.stack([item.keypoints for item in selected])
        scores = np.stack([item.scores for item in selected])

        return {
            "keypoints": _interpolate(times, keypoints, target_times),
            "scores": np.clip(_interpolate(times, scores, target_times), 0.0, 1.0),
            "frame_size": selected[-1].frame_size,
        }


def _interpolate(
    source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    flat = values.reshape(values.shape[0], -1)
    output = np.empty((len(target_times), flat.shape[1]), dtype=np.float32)
    for index in range(flat.shape[1]):
        output[:, index] = np.interp(
            target_times,
            source_times,
            flat[:, index],
            left=flat[0, index],
            right=flat[-1, index],
        )
    return output.reshape((len(target_times),) + values.shape[1:])
