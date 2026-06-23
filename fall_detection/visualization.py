from __future__ import annotations

import cv2
import numpy as np

from .stgcn import COCO_EDGES
from .types import AlertState, FallDecision, PoseObservation


STATE_COLORS = {
    AlertState.NORMAL: (60, 200, 60),
    AlertState.WATCH: (0, 210, 255),
    AlertState.SUSPECTED: (0, 110, 255),
    AlertState.CONFIRMED: (0, 0, 255),
    AlertState.RECOVERING: (255, 180, 0),
}


def draw_observation(
    frame: np.ndarray,
    observation: PoseObservation,
    decision: FallDecision,
    keypoint_threshold: float,
) -> None:
    color = STATE_COLORS[decision.state]
    x1, y1, x2, y2 = observation.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for source, target in COCO_EDGES:
        if (
            observation.scores[source] >= keypoint_threshold
            and observation.scores[target] >= keypoint_threshold
        ):
            start = tuple(observation.keypoints[source].astype(int))
            end = tuple(observation.keypoints[target].astype(int))
            cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
    for point, score in zip(observation.keypoints, observation.scores):
        if score >= keypoint_threshold:
            cv2.circle(frame, tuple(point.astype(int)), 3, color, -1, cv2.LINE_AA)

    model_probability = sum(
        decision.model_probabilities.get(name, 0.0) for name in ("falling", "lying")
    )
    model_text = f"p={model_probability:.2f}" if decision.model_ready else "model=NOT_READY"
    label = (
        f"ID {observation.track_id} {decision.state.value.upper()} "
        f"rule={decision.rule_score:.2f} {model_text}"
    )
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status_bar(
    frame: np.ndarray, fps: float, model_ready: bool, pose_device: str
) -> None:
    text = (
        f"Pipeline FPS: {fps:.1f} | ST-GCN: "
        f"{'READY' if model_ready else 'NOT TRAINED'} | RTMPose: {pose_device}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(
        frame,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

