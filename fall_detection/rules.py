from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RuleConfig
from .types import AlertState, FallDecision, PoseObservation


SHOULDERS = (5, 6)
HIPS = (11, 12)
BODY_JOINTS = (5, 6, 11, 12, 13, 14, 15, 16)


@dataclass(slots=True)
class _TrackRuleState:
    state: AlertState = AlertState.NORMAL
    first_seen: float | None = None
    lying_since: float | None = None
    candidate_since: float | None = None
    upright_since: float | None = None
    alert_since: float | None = None
    lying_observations: int = 0


def _mean_joint(observation: PoseObservation, indices: tuple[int, ...]) -> np.ndarray | None:
    selected = np.asarray(indices)
    valid = observation.scores[selected] >= 0.25
    if not np.any(valid):
        return None
    return np.mean(observation.keypoints[selected[valid]], axis=0)


def _axis_horizontalness(observation: PoseObservation) -> float | None:
    shoulders = _mean_joint(observation, SHOULDERS)
    hips = _mean_joint(observation, HIPS)
    if shoulders is None or hips is None:
        return None
    axis = shoulders - hips
    return float(abs(axis[0]) / max(np.linalg.norm(axis), 1e-6))


def _aspect_ratio(observation: PoseObservation) -> float:
    width = max(float(observation.bbox[2] - observation.bbox[0]), 1.0)
    height = max(float(observation.bbox[3] - observation.bbox[1]), 1.0)
    return width / height


def _body_horizontalness(observation: PoseObservation) -> float | None:
    indices = np.asarray(BODY_JOINTS)
    valid = observation.scores[indices] >= 0.25
    points = observation.keypoints[indices[valid]]
    if len(points) < 4:
        return None
    centered = points - np.mean(points, axis=0)
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    return float(abs(major_axis[0]) / max(np.linalg.norm(major_axis), 1e-6))


def _pelvis_y(observation: PoseObservation) -> float:
    hips = _mean_joint(observation, HIPS)
    if hips is not None:
        return float(hips[1])
    return float((observation.bbox[1] + observation.bbox[3]) * 0.5)


class FallRuleEngine:
    def __init__(self, config: RuleConfig, model_threshold: float = 0.7):
        self.config = config
        self.model_threshold = model_threshold
        self._states: dict[int, _TrackRuleState] = {}

    def evaluate(
        self,
        history: list[PoseObservation],
        model_probabilities: dict[str, float] | None = None,
        model_ready: bool = False,
    ) -> FallDecision:
        current = history[-1]
        track_state = self._states.setdefault(current.track_id, _TrackRuleState())
        now = current.timestamp
        if track_state.first_seen is None:
            track_state.first_seen = now
        track_age = now - track_state.first_seen
        aspect = _aspect_ratio(current)
        axis = _axis_horizontalness(current)
        body_horizontalness = _body_horizontalness(current)
        lying_score = self._lying_score(aspect, axis)
        pose_is_reliable = current.pose_quality >= self.config.minimum_pose_quality
        full_body_is_horizontal = (
            aspect >= self.config.minimum_lying_aspect_ratio
            or (
                body_horizontalness is not None
                and body_horizontalness
                >= self.config.minimum_body_horizontalness
            )
        )
        is_lying = (
            lying_score >= self.config.lying_score_threshold
            and pose_is_reliable
            and full_body_is_horizontal
        )

        recent = [
            item
            for item in history
            if now - item.timestamp <= self.config.sudden_window_seconds
        ]
        downward_velocity = self._downward_velocity(recent)
        orientation_change = self._orientation_change(recent)
        sudden_score = max(
            downward_velocity
            / max(self.config.downward_velocity_body_lengths_per_second, 1e-6),
            orientation_change / max(self.config.abrupt_orientation_change, 1e-6),
        )
        sudden_score = float(np.clip(sudden_score, 0.0, 1.5))
        sudden = (
            sudden_score >= 1.0
            and pose_is_reliable
            and track_age >= self.config.minimum_track_age_seconds
            and len(recent) >= self.config.minimum_motion_observations
        )

        if is_lying:
            track_state.upright_since = None
            if track_state.lying_since is None:
                track_state.lying_since = now
                track_state.lying_observations = 1
            else:
                track_state.lying_observations += 1
        else:
            track_state.lying_since = None
            track_state.lying_observations = 0
            if track_state.upright_since is None:
                track_state.upright_since = now

        candidate_active = (
            track_state.candidate_since is not None
            and now - track_state.candidate_since
            <= self.config.fall_to_lying_max_seconds
        )
        if sudden and not candidate_active:
            track_state.candidate_since = now
            candidate_active = True

        lying_duration = (
            now - track_state.lying_since if track_state.lying_since is not None else 0.0
        )
        stable_lying = (
            is_lying
            and lying_duration >= self.config.lying_confirmation_seconds
            and track_state.lying_observations
            >= self.config.minimum_lying_observations
        )
        prolonged = lying_duration >= self.config.prolonged_lying_seconds
        fall_then_lying = candidate_active and stable_lying
        evidence = fall_then_lying or prolonged

        model_fall_probability = 0.0
        if model_probabilities:
            model_fall_probability = model_probabilities.get("falling", 0.0) + model_probabilities.get(
                "lying", 0.0
            )
        model_confirms = (
            model_ready
            and evidence
            and model_fall_probability >= self.model_threshold
        )

        rule_score = float(
            np.clip(0.45 * min(sudden_score, 1.0) + 0.55 * lying_score, 0.0, 1.0)
        )
        reason = "no fall evidence"

        if evidence:
            if track_state.alert_since is None:
                track_state.alert_since = now
            if model_confirms or track_state.state == AlertState.CONFIRMED:
                track_state.state = AlertState.CONFIRMED
                reason = "HPI-GCN and fall-to-lying temporal evidence agree"
            else:
                track_state.state = AlertState.SUSPECTED
                reason = (
                    "prolonged lying; model unavailable or below threshold"
                    if prolonged and not fall_then_lying
                    else "abrupt descent followed by stable lying; "
                    "model unavailable or below threshold"
                )
        elif (
            track_state.state in {AlertState.SUSPECTED, AlertState.CONFIRMED}
            and track_state.alert_since is not None
            and now - track_state.alert_since < self.config.alert_hold_seconds
        ):
            reason = "holding alert to suppress frame-to-frame oscillation"
        elif track_state.state in {AlertState.SUSPECTED, AlertState.CONFIRMED}:
            if is_lying:
                track_state.state = AlertState.WATCH
                reason = "lying persists but fall evidence is no longer complete"
            else:
                track_state.state = AlertState.RECOVERING
                track_state.upright_since = track_state.upright_since or now
                reason = "person appears upright; waiting for stable recovery"
        elif track_state.state == AlertState.RECOVERING:
            recovered = (
                not is_lying
                and track_state.upright_since is not None
                and now - track_state.upright_since >= self.config.recovery_seconds
            )
            if recovered:
                track_state.state = AlertState.NORMAL
                track_state.candidate_since = None
                track_state.alert_since = None
            else:
                reason = "waiting for stable recovery"
        elif candidate_active or is_lying or sudden_score >= self.config.watch_score_threshold:
            track_state.state = AlertState.WATCH
            reason = (
                "fall candidate awaiting stable lying"
                if candidate_active
                else "weak or incomplete fall evidence"
            )
        else:
            track_state.state = AlertState.NORMAL
            if not candidate_active:
                track_state.candidate_since = None
            track_state.alert_since = None

        return FallDecision(
            track_id=current.track_id,
            timestamp=now,
            state=track_state.state,
            rule_score=rule_score,
            model_probabilities=model_probabilities,
            lying_duration=lying_duration,
            reason=reason,
            model_ready=model_ready,
        )

    def remove(self, track_ids: list[int]) -> None:
        for track_id in track_ids:
            self._states.pop(track_id, None)

    def reset(self) -> None:
        self._states.clear()

    def _lying_score(self, aspect: float, axis: float | None) -> float:
        aspect_score = np.clip(
            aspect / max(self.config.lying_aspect_ratio, 1e-6), 0.0, 1.0
        )
        if axis is None:
            return float(aspect_score)
        axis_score = np.clip(
            axis / max(self.config.lying_axis_horizontalness, 1e-6), 0.0, 1.0
        )
        return float(0.45 * aspect_score + 0.55 * axis_score)

    @staticmethod
    def _downward_velocity(history: list[PoseObservation]) -> float:
        if len(history) < 2:
            return 0.0
        first, last = history[0], history[-1]
        elapsed = max(last.timestamp - first.timestamp, 1e-3)
        body_height = max(float(first.bbox[3] - first.bbox[1]), 1.0)
        return max(0.0, (_pelvis_y(last) - _pelvis_y(first)) / body_height / elapsed)

    @staticmethod
    def _orientation_change(history: list[PoseObservation]) -> float:
        values = [
            value
            for value in (_axis_horizontalness(item) for item in history)
            if value is not None
        ]
        if len(values) < 2:
            return 0.0
        return max(0.0, values[-1] - min(values[:-1]))
