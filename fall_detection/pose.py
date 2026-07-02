from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PoseConfig
from .types import TrackedBox


RTMPOSE_S_BODY7_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
)


@dataclass(slots=True)
class _CachedPose:
    timestamp: float
    bbox: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray


class RTMPose17:
    """Top-down 17-keypoint RTMPose using YOLO boxes as person proposals."""

    def __init__(self, config: PoseConfig):
        from rtmlib import RTMPose

        self.config = config
        cache_dir = Path(config.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_dir))

        device = self._resolve_device(config.backend, config.device)
        model = RTMPOSE_S_BODY7_ONNX if config.model == "auto" else config.model
        self.model = RTMPose(
            model,
            model_input_size=tuple(config.input_size),
            to_openpose=False,
            backend=config.backend,
            device=device,
        )
        self.device = device
        self._cache: dict[int, _CachedPose] = {}

    @staticmethod
    def _resolve_device(backend: str, requested: str) -> str:
        if backend != "onnxruntime":
            return "cpu" if requested == "auto" else requested
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        if requested == "auto":
            return "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        if requested.startswith("cuda") and "CUDAExecutionProvider" not in providers:
            return "cpu"
        return requested

    def __call__(
        self,
        frame: np.ndarray,
        tracked_boxes: list[TrackedBox],
        timestamp: float | None = None,
    ) -> list[tuple[TrackedBox, np.ndarray, np.ndarray]]:
        if not tracked_boxes:
            return []

        max_age = float(self.config.cache_max_age_seconds)
        min_iou = float(self.config.cache_min_bbox_iou)
        current_time = float(timestamp) if timestamp is not None else None

        fresh_indices: list[int] = []
        fresh_boxes: list[TrackedBox] = []
        reused: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for index, tracked in enumerate(tracked_boxes):
            cached = self._cache.get(tracked.detector_track_id)
            if cached is None:
                fresh_indices.append(index)
                fresh_boxes.append(tracked)
                continue

            if current_time is not None and current_time - cached.timestamp > max_age:
                fresh_indices.append(index)
                fresh_boxes.append(tracked)
                continue

            if self._bbox_iou(cached.bbox, tracked.bbox) < min_iou:
                fresh_indices.append(index)
                fresh_boxes.append(tracked)
                continue

            reused[index] = (cached.keypoints, cached.scores)

        fresh_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if fresh_boxes:
            bboxes = [tracked.bbox.tolist() for tracked in fresh_boxes]
            keypoints, scores = self.model(frame, bboxes=bboxes)
            for index, tracked, points, confidence in zip(
                fresh_indices, fresh_boxes, keypoints, scores
            ):
                points = points.astype(np.float32)
                confidence = confidence.astype(np.float32)
                fresh_results[index] = (points, confidence)
                self._cache[tracked.detector_track_id] = _CachedPose(
                    timestamp=current_time if current_time is not None else 0.0,
                    bbox=tracked.bbox.copy(),
                    keypoints=points,
                    scores=confidence,
                )

        results: list[tuple[TrackedBox, np.ndarray, np.ndarray]] = []
        for index, tracked in enumerate(tracked_boxes):
            if index in fresh_results:
                points, confidence = fresh_results[index]
            elif index in reused:
                points, confidence = reused[index]
            else:
                continue
            results.append((tracked, points, confidence))
        return results

    @staticmethod
    def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
        area_b = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
        return intersection / max(area_a + area_b - intersection, 1e-6)

