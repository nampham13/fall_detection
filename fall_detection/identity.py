from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import IdentityConfig
from .types import PoseObservation, TrackedBox


@dataclass(slots=True)
class _IdentityState:
    stable_id: int
    detector_id: int
    bbox: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray
    last_seen: float


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_b = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _center_distance(first: np.ndarray, second: np.ndarray) -> float:
    center_a = (first[:2] + first[2:]) * 0.5
    center_b = (second[:2] + second[2:]) * 0.5
    diagonal = np.linalg.norm(second[2:] - second[:2])
    return float(np.linalg.norm(center_a - center_b) / max(diagonal, 1.0))


def _pose_distance(
    points_a: np.ndarray,
    scores_a: np.ndarray,
    points_b: np.ndarray,
    scores_b: np.ndarray,
    bbox: np.ndarray,
) -> float:
    visible = (scores_a >= 0.25) & (scores_b >= 0.25)
    if np.count_nonzero(visible) < 4:
        return 1.0
    scale = max(float(np.linalg.norm(bbox[2:] - bbox[:2])), 1.0)
    distances = np.linalg.norm(points_a[visible] - points_b[visible], axis=1) / scale
    return float(np.clip(np.median(distances) * 2.0, 0.0, 1.5))


class SkeletonIdentityResolver:
    """Repairs short ByteTrack ID switches using bbox motion and pose similarity."""

    def __init__(self, config: IdentityConfig):
        self.config = config
        self._detector_to_stable: dict[int, int] = {}
        self._states: dict[int, _IdentityState] = {}
        self._next_stable_id = 1

    def resolve(
        self,
        timestamp: float,
        frame_size: tuple[int, int],
        poses: list[tuple[TrackedBox, np.ndarray, np.ndarray]],
    ) -> list[PoseObservation]:
        current_detector_ids = {tracked.detector_track_id for tracked, _, _ in poses}
        assignments: dict[int, int] = {}

        for tracked, _, _ in poses:
            stable_id = self._detector_to_stable.get(tracked.detector_track_id)
            if stable_id is not None:
                assignments[tracked.detector_track_id] = stable_id

        new_items = [item for item in poses if item[0].detector_track_id not in assignments]
        occupied_stable_ids = set(assignments.values())
        candidates = [
            state
            for state in self._states.values()
            if state.stable_id not in occupied_stable_ids
            and state.detector_id not in current_detector_ids
            and timestamp - state.last_seen <= self.config.maximum_gap_seconds
        ]

        if new_items and candidates:
            costs = np.full((len(new_items), len(candidates)), 1e6, dtype=np.float32)
            for row, (tracked, keypoints, scores) in enumerate(new_items):
                for column, state in enumerate(candidates):
                    center = _center_distance(state.bbox, tracked.bbox)
                    if center > self.config.maximum_center_distance:
                        continue
                    iou_cost = 1.0 - _bbox_iou(state.bbox, tracked.bbox)
                    pose_cost = _pose_distance(
                        state.keypoints, state.scores, keypoints, scores, tracked.bbox
                    )
                    costs[row, column] = (
                        self.config.iou_weight * iou_cost
                        + self.config.pose_weight * pose_cost
                        + self.config.center_weight * center
                    )

            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] > self.config.maximum_assignment_cost:
                    continue
                detector_id = new_items[row][0].detector_track_id
                state = candidates[column]
                old_detector_id = state.detector_id
                self._detector_to_stable.pop(old_detector_id, None)
                assignments[detector_id] = state.stable_id
                self._detector_to_stable[detector_id] = state.stable_id

        observations: list[PoseObservation] = []
        for tracked, keypoints, scores in poses:
            detector_id = tracked.detector_track_id
            stable_id = assignments.get(detector_id)
            if stable_id is None:
                stable_id = self._next_stable_id
                self._next_stable_id += 1
                self._detector_to_stable[detector_id] = stable_id

            self._states[stable_id] = _IdentityState(
                stable_id=stable_id,
                detector_id=detector_id,
                bbox=tracked.bbox.copy(),
                keypoints=keypoints.copy(),
                scores=scores.copy(),
                last_seen=timestamp,
            )
            observations.append(
                PoseObservation(
                    track_id=stable_id,
                    detector_track_id=detector_id,
                    timestamp=timestamp,
                    bbox=tracked.bbox.copy(),
                    keypoints=keypoints.copy(),
                    scores=scores.copy(),
                    detector_confidence=tracked.confidence,
                    frame_size=frame_size,
                )
            )

        self.expire(timestamp)
        return observations

    def expire(self, timestamp: float) -> None:
        expired = [
            stable_id
            for stable_id, state in self._states.items()
            if timestamp - state.last_seen > self.config.maximum_gap_seconds
        ]
        for stable_id in expired:
            detector_id = self._states[stable_id].detector_id
            self._states.pop(stable_id, None)
            if self._detector_to_stable.get(detector_id) == stable_id:
                self._detector_to_stable.pop(detector_id, None)

    def reset(self) -> None:
        """Clear scene-specific associations while keeping globally unique IDs."""
        self._detector_to_stable.clear()
        self._states.clear()
