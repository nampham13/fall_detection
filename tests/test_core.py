from __future__ import annotations

import unittest

import numpy as np

from fall_detection.config import (
    IdentityConfig,
    RuleConfig,
    SceneCutConfig,
    TemporalConfig,
)
from fall_detection.cli import build_parser
from fall_detection.identity import SkeletonIdentityResolver
from fall_detection.rules import FallRuleEngine
from fall_detection.scene import SceneCutDetector
from fall_detection.temporal import SkeletonHistory
from fall_detection.types import AlertState, PoseObservation, TrackedBox


def observation(
    timestamp: float,
    track_id: int = 1,
    detector_id: int = 1,
    lying: bool = False,
    pelvis_y: float = 130.0,
) -> PoseObservation:
    points = np.zeros((17, 2), dtype=np.float32)
    if lying:
        points[:] = (100.0, pelvis_y)
        points[5] = (150.0, pelvis_y - 5)
        points[6] = (150.0, pelvis_y + 5)
        points[11] = (80.0, pelvis_y - 3)
        points[12] = (80.0, pelvis_y + 3)
        bbox = np.array([40, pelvis_y - 35, 190, pelvis_y + 35], dtype=np.float32)
    else:
        points[:] = (100.0, pelvis_y)
        points[5] = (88.0, pelvis_y - 55)
        points[6] = (112.0, pelvis_y - 55)
        points[11] = (92.0, pelvis_y)
        points[12] = (108.0, pelvis_y)
        bbox = np.array([60, pelvis_y - 95, 140, pelvis_y + 65], dtype=np.float32)
    return PoseObservation(
        track_id=track_id,
        detector_track_id=detector_id,
        timestamp=timestamp,
        bbox=bbox,
        keypoints=points,
        scores=np.ones(17, dtype=np.float32),
        detector_confidence=0.9,
        frame_size=(240, 320),
    )


class CLITests(unittest.TestCase):
    def test_show_enables_preview(self):
        args = build_parser().parse_args(["--source", "video.mp4", "--show"])
        self.assertTrue(args.show)

    def test_display_remains_a_backward_compatible_alias(self):
        args = build_parser().parse_args(["--source", "0", "--display"])
        self.assertTrue(args.show)


class IdentityTests(unittest.TestCase):
    def test_short_detector_id_switch_keeps_stable_id(self):
        resolver = SkeletonIdentityResolver(IdentityConfig())
        first = observation(0.0, detector_id=3)
        first_result = resolver.resolve(
            0.0,
            first.frame_size,
            [
                (
                    TrackedBox(3, first.bbox, 0.9),
                    first.keypoints,
                    first.scores,
                )
            ],
        )
        switched = observation(0.4, detector_id=99, pelvis_y=132)
        second_result = resolver.resolve(
            0.4,
            switched.frame_size,
            [
                (
                    TrackedBox(99, switched.bbox, 0.9),
                    switched.keypoints,
                    switched.scores,
                )
            ],
        )
        self.assertEqual(first_result[0].track_id, second_result[0].track_id)


class RuleTests(unittest.TestCase):
    def test_abrupt_motion_waits_for_stable_lying(self):
        engine = FallRuleEngine(RuleConfig(), model_threshold=0.7)
        history: list[PoseObservation] = []
        sequence = [
            observation(0.0, pelvis_y=100),
            observation(0.25, pelvis_y=108),
            observation(0.5, pelvis_y=125),
            observation(0.8, lying=True, pelvis_y=190),
        ]
        for item in sequence:
            history.append(item)
            decision = engine.evaluate(history, model_probabilities={}, model_ready=False)
        self.assertEqual(decision.state, AlertState.WATCH)

        for timestamp in (1.1, 1.5, 1.9, 2.2):
            history.append(observation(timestamp, lying=True, pelvis_y=190))
            decision = engine.evaluate(history, model_probabilities={}, model_ready=False)
        self.assertEqual(decision.state, AlertState.SUSPECTED)
        self.assertFalse(decision.model_ready)

    def test_startup_motion_cannot_immediately_alert(self):
        engine = FallRuleEngine(RuleConfig(), model_threshold=0.7)
        history = [
            observation(0.0, pelvis_y=100),
            observation(0.1, lying=True, pelvis_y=190),
        ]
        engine.evaluate(history[:1], model_probabilities={}, model_ready=False)
        decision = engine.evaluate(history, model_probabilities={}, model_ready=False)
        self.assertNotEqual(decision.state, AlertState.SUSPECTED)

    def test_bending_with_vertical_legs_is_not_lying(self):
        engine = FallRuleEngine(
            RuleConfig(
                lying_confirmation_seconds=0.5,
                minimum_lying_observations=2,
            ),
            0.7,
        )
        history: list[PoseObservation] = []
        for timestamp in (0.0, 0.3, 0.6):
            item = observation(timestamp, pelvis_y=120 + timestamp * 30)
            history.append(item)
            engine.evaluate(history, model_probabilities={}, model_ready=False)

        for timestamp in (0.9, 1.3, 1.7):
            item = observation(timestamp, pelvis_y=180)
            item.keypoints[5] = (150, 120)
            item.keypoints[6] = (150, 140)
            item.keypoints[11] = (90, 150)
            item.keypoints[12] = (90, 170)
            item.keypoints[13] = (85, 205)
            item.keypoints[14] = (105, 205)
            item.keypoints[15] = (80, 240)
            item.keypoints[16] = (110, 240)
            item.bbox = np.array([60, 100, 180, 250], dtype=np.float32)
            history.append(item)
            decision = engine.evaluate(history, model_probabilities={}, model_ready=False)
        self.assertNotEqual(decision.state, AlertState.SUSPECTED)

    def test_alert_is_held_to_prevent_oscillation(self):
        engine = FallRuleEngine(RuleConfig(alert_hold_seconds=2.0), 0.7)
        history: list[PoseObservation] = []
        for item in (
            observation(0.0, pelvis_y=100),
            observation(0.25, pelvis_y=108),
            observation(0.5, pelvis_y=125),
            observation(0.8, lying=True, pelvis_y=190),
            observation(1.2, lying=True, pelvis_y=190),
            observation(1.6, lying=True, pelvis_y=190),
            observation(2.0, lying=True, pelvis_y=190),
            observation(2.2, lying=True, pelvis_y=190),
        ):
            history.append(item)
            decision = engine.evaluate(history, model_probabilities={}, model_ready=False)
        self.assertEqual(decision.state, AlertState.SUSPECTED)

        history.append(observation(2.3, lying=False, pelvis_y=130))
        decision = engine.evaluate(history, model_probabilities={}, model_ready=False)
        self.assertEqual(decision.state, AlertState.SUSPECTED)

    def test_prolonged_lying_is_suspected(self):
        engine = FallRuleEngine(RuleConfig(prolonged_lying_seconds=8.0), 0.7)
        first = observation(0.0, lying=True, pelvis_y=180)
        engine.evaluate([first], model_probabilities={}, model_ready=False)
        second = observation(9.0, lying=True, pelvis_y=180)
        decision = engine.evaluate([first, second], model_probabilities={}, model_ready=False)
        self.assertEqual(decision.state, AlertState.SUSPECTED)
        self.assertGreaterEqual(decision.lying_duration, 8.0)


class SceneCutTests(unittest.TestCase):
    def test_hard_cut_is_detected_but_small_change_is_not(self):
        detector = SceneCutDetector(
            SceneCutConfig(
                input_size=(32, 18),
                pixel_difference_threshold=0.18,
                histogram_correlation_threshold=0.65,
            )
        )
        black = np.zeros((90, 160, 3), dtype=np.uint8)
        almost_black = np.full((90, 160, 3), 5, dtype=np.uint8)
        white = np.full((90, 160, 3), 255, dtype=np.uint8)
        self.assertFalse(detector.update(black, 0.0).detected)
        self.assertFalse(detector.update(almost_black, 0.1).detected)
        self.assertTrue(detector.update(white, 1.0).detected)


if __name__ == "__main__":
    unittest.main()
