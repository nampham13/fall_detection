from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DetectorConfig:
    model: str = "yolo26s.pt"
    confidence: float = 0.25
    iou: float = 0.70
    image_size: int = 640
    device: str = "cuda:0"
    half: bool = True
    tracker_config: str = "configs/bytetrack_fall.yaml"


@dataclass(slots=True)
class PoseConfig:
    model: str = "auto"
    input_size: tuple[int, int] = (192, 256)
    backend: str = "onnxruntime"
    device: str = "auto"
    minimum_keypoint_score: float = 0.25
    cache_dir: str = "models/rtmlib"


@dataclass(slots=True)
class IdentityConfig:
    maximum_gap_seconds: float = 1.5
    maximum_center_distance: float = 1.25
    maximum_assignment_cost: float = 0.85
    iou_weight: float = 0.55
    pose_weight: float = 0.30
    center_weight: float = 0.15


@dataclass(slots=True)
class TemporalConfig:
    sequence_length: int = 48
    window_seconds: float = 3.2
    history_seconds: float = 12.0
    minimum_observations: int = 12
    maximum_sample_gap_seconds: float = 0.75


@dataclass(slots=True)
class STGCNConfig:
    checkpoint: str = "models/stgcn_fall.pt"
    input_channels: int = 7
    classes: tuple[str, ...] = ("normal", "falling", "lying")
    probability_threshold: float = 0.70
    device: str = "cuda:0"


@dataclass(slots=True)
class RuleConfig:
    lying_aspect_ratio: float = 1.05
    lying_axis_horizontalness: float = 0.62
    lying_score_threshold: float = 0.78
    minimum_lying_aspect_ratio: float = 0.88
    minimum_body_horizontalness: float = 0.68
    downward_velocity_body_lengths_per_second: float = 0.70
    abrupt_orientation_change: float = 0.40
    sudden_window_seconds: float = 0.80
    lying_confirmation_seconds: float = 1.25
    fall_to_lying_max_seconds: float = 3.0
    prolonged_lying_seconds: float = 8.0
    minimum_track_age_seconds: float = 0.75
    minimum_motion_observations: int = 4
    minimum_lying_observations: int = 5
    watch_score_threshold: float = 0.70
    alert_hold_seconds: float = 2.0
    recovery_seconds: float = 2.0
    minimum_pose_quality: float = 0.35


@dataclass(slots=True)
class SceneCutConfig:
    enabled: bool = True
    input_size: tuple[int, int] = (160, 90)
    pixel_difference_threshold: float = 0.18
    histogram_correlation_threshold: float = 0.65
    minimum_interval_seconds: float = 0.50


@dataclass(slots=True)
class RuntimeConfig:
    display: bool = False
    output_video: str | None = None
    event_log: str | None = "outputs/events.jsonl"
    draw_keypoint_threshold: float = 0.30
    stale_track_seconds: float = 3.0


@dataclass(slots=True)
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    stgcn: STGCNConfig = field(default_factory=STGCNConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    scene_cut: SceneCutConfig = field(default_factory=SceneCutConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _build(cls: type, values: dict[str, Any] | None):
    values = dict(values or {})
    for key, value in list(values.items()):
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return AppConfig(
        detector=_build(DetectorConfig, raw.get("detector")),
        pose=_build(PoseConfig, raw.get("pose")),
        identity=_build(IdentityConfig, raw.get("identity")),
        temporal=_build(TemporalConfig, raw.get("temporal")),
        stgcn=_build(STGCNConfig, raw.get("stgcn")),
        rules=_build(RuleConfig, raw.get("rules")),
        scene_cut=_build(SceneCutConfig, raw.get("scene_cut")),
        runtime=_build(RuntimeConfig, raw.get("runtime")),
    )
