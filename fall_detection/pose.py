from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import PoseConfig
from .types import TrackedBox


RTMPOSE_S_BODY7_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
)


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
        self, frame: np.ndarray, tracked_boxes: list[TrackedBox]
    ) -> list[tuple[TrackedBox, np.ndarray, np.ndarray]]:
        if not tracked_boxes:
            return []
        bboxes = [tracked.bbox.tolist() for tracked in tracked_boxes]
        keypoints, scores = self.model(frame, bboxes=bboxes)
        return [
            (tracked, points.astype(np.float32), confidence.astype(np.float32))
            for tracked, points, confidence in zip(tracked_boxes, keypoints, scores)
        ]

