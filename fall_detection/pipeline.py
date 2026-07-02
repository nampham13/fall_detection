from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import AppConfig
from .detector import Yolo26ByteTracker
from .identity import SkeletonIdentityResolver
from .pose import RTMPose17
from .rules import FallRuleEngine
from .scene import SceneCutDetector, SceneCutResult
from .gcn import GCNRuntime
from .temporal import SkeletonHistory
from .types import AlertState, FallDecision
from .visualization import draw_observation, draw_status_bar


class EventLogger:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._last_states: dict[int, AlertState] = {}
        self.scene_index = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, decision: FallDecision) -> None:
        previous = self._last_states.get(decision.track_id)
        self._last_states[decision.track_id] = decision.state
        if self.path is None or previous == decision.state:
            return
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_type": "state_transition",
            "scene_index": self.scene_index,
            "track_id": decision.track_id,
            "timestamp_seconds": round(decision.timestamp, 3),
            "state": decision.state.value,
            "rule_score": round(decision.rule_score, 4),
            "model_probabilities": decision.model_probabilities,
            "model_ready": decision.model_ready,
            "lying_duration_seconds": round(decision.lying_duration, 3),
            "reason": decision.reason,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_scene_cut(self, timestamp: float, result: SceneCutResult) -> None:
        self.scene_index += 1
        if self.path is None:
            return
        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_type": "scene_cut",
            "scene_index": self.scene_index,
            "timestamp_seconds": round(timestamp, 3),
            "pixel_difference": round(result.pixel_difference, 4),
            "histogram_correlation": round(result.histogram_correlation, 4),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class FallDetectionPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.detector = Yolo26ByteTracker(config.detector)
        self.pose = RTMPose17(config.pose)
        self.identity = SkeletonIdentityResolver(config.identity)
        self.history = SkeletonHistory(config.temporal)
        self.scene_cut = SceneCutDetector(config.scene_cut)
        self.gcn = GCNRuntime(config.gcn)
        self.rules = FallRuleEngine(
            config.rules, model_threshold=config.gcn.probability_threshold
        )
        self.events = EventLogger(config.runtime.event_log)

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> tuple[np.ndarray, list[FallDecision]]:
        scene_result = self.scene_cut.update(frame, timestamp)
        if scene_result.detected:
            self._reset_for_scene_cut(timestamp, scene_result)

        height, width = frame.shape[:2]
        tracked_boxes = self.detector(frame)
        poses = self.pose(frame, tracked_boxes, timestamp=timestamp)
        observations = self.identity.resolve(timestamp, (height, width), poses)

        output = frame.copy()
        decisions: list[FallDecision] = []
        for observation in observations:
            self.history.add(observation)
            model_sample = self.history.get_gcn_input(observation.track_id)
            probabilities = self.gcn.predict(model_sample)
            decision = self.rules.evaluate(
                self.history.get(observation.track_id),
                model_probabilities=probabilities,
                model_ready=self.gcn.ready,
            )
            decisions.append(decision)
            self.events.write(decision)
            draw_observation(
                output,
                observation,
                decision,
                self.config.runtime.draw_keypoint_threshold,
            )

        stale = self.history.remove_stale(
            timestamp, self.config.runtime.stale_track_seconds
        )
        self.rules.remove(stale)
        return output, decisions

    def _reset_for_scene_cut(
        self, timestamp: float, scene_result: SceneCutResult
    ) -> None:
        self.detector.reset()
        self.identity.reset()
        self.history.clear()
        self.rules.reset()
        self.events.write_scene_cut(timestamp, scene_result)

    def run(self, source: str | int) -> None:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 1e-3:
            source_fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = self._create_writer(source_fps, width, height)
        frame_index = 0
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
                    else self._video_timestamp(capture, frame_index, source_fps)
                )
                rendered, _ = self.process_frame(frame, timestamp)
                frame_index += 1
                pipeline_fps = frame_index / max(time.perf_counter() - started, 1e-6)
                draw_status_bar(rendered, pipeline_fps, self.gcn.ready, self.pose.device)

                if writer is not None:
                    writer.write(rendered)
                if self.config.runtime.display:
                    cv2.imshow("Fall detection", rendered)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if self.config.runtime.display:
                cv2.destroyAllWindows()

    @staticmethod
    def _video_timestamp(
        capture: cv2.VideoCapture, frame_index: int, source_fps: float
    ) -> float:
        position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if np.isfinite(position_ms) and position_ms > 0:
            return position_ms / 1000.0
        return frame_index / source_fps

    def _create_writer(
        self, fps: float, width: int, height: int
    ) -> cv2.VideoWriter | None:
        output_path = self.config.runtime.output_video
        if not output_path:
            return None
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {path}")
        return writer
