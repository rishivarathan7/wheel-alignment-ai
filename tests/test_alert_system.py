"""
Unit tests for Module 15 — Alert and Decision Management Subsystem.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.alert_system import (
    AlertSystem,
    AlertEvent,
    MANDATORY_WARNING_TEXT,
    NON_DIAGNOSTIC_DISCLAIMER
)
from src.prediction import RealTimePredictionResult


class TestAlertSystem(unittest.TestCase):
    """Test suite for AlertSystem decision engine."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.tmp_dir.name) / "alerts.log"
        self.alert_system = AlertSystem(
            alert_log_path=self.log_file,
            cooldown_seconds=0.5,
            min_confidence=0.60,
            consecutive_threshold=3,
            smoothing_window=5,
            enable_audio=False
        )
        self.dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        self.dummy_bbox = (100, 100, 300, 300)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_alert_states_and_colors(self) -> None:
        """Verifies all 4 discrete system states have defined color mappings."""
        expected_states = ["NORMAL", "POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT", "INSUFFICIENT_DATA"]
        for state in expected_states:
            self.assertIn(state, AlertSystem.COLOR_MAP)
            color = AlertSystem.COLOR_MAP[state]
            self.assertEqual(len(color), 3)

    def test_confidence_threshold_filtering(self) -> None:
        """Verifies predictions with confidence below threshold resolve to INSUFFICIENT_DATA."""
        low_conf_pred = RealTimePredictionResult(
            wheel_id=1,
            predicted_condition="SEVERE_MISALIGNMENT",
            prediction_probability=0.45  # Below 0.60 threshold
        )

        state, conf, is_escalated, _ = self.alert_system.evaluate_prediction(low_conf_pred, track_id=1)
        self.assertEqual(state, "INSUFFICIENT_DATA")
        self.assertFalse(is_escalated)

    def test_temporal_smoothing(self) -> None:
        """Verifies single noisy frame is smoothed over history window."""
        normal_pred = RealTimePredictionResult(wheel_id=1, predicted_condition="NORMAL", prediction_probability=0.90)
        noisy_pred = RealTimePredictionResult(wheel_id=1, predicted_condition="SEVERE_MISALIGNMENT", prediction_probability=0.90)

        # 4 Normal frames followed by 1 noisy frame
        for _ in range(4):
            self.alert_system.evaluate_prediction(normal_pred, track_id=1)

        state, _, _, _ = self.alert_system.evaluate_prediction(noisy_pred, track_id=1)
        # 4 NORMAL vs 1 SEVERE_MISALIGNMENT => Majority vote resolves to NORMAL
        self.assertEqual(state, "NORMAL")

    def test_consecutive_abnormal_escalation(self) -> None:
        """Verifies escalation requires configurable consecutive abnormal observations."""
        abnormal_pred = RealTimePredictionResult(
            wheel_id=1,
            predicted_condition="SEVERE_MISALIGNMENT",
            prediction_probability=0.90
        )

        # Frame 1 & 2: Below consecutive_threshold=3 => Not escalated
        _, _, is_esc1, _ = self.alert_system.evaluate_prediction(abnormal_pred, track_id=1)
        _, _, is_esc2, _ = self.alert_system.evaluate_prediction(abnormal_pred, track_id=1)
        self.assertFalse(is_esc1)
        self.assertFalse(is_esc2)

        # Frame 3: Reaches consecutive_threshold=3 => Escalated alert triggered
        _, _, is_esc3, event = self.alert_system.evaluate_prediction(abnormal_pred, track_id=1)
        self.assertTrue(is_esc3)
        self.assertIsNotNone(event)
        self.assertEqual(event.state, "SEVERE_MISALIGNMENT")
        self.assertEqual(event.consecutive_count, 3)

    def test_alert_cooldown(self) -> None:
        """Verifies cooldown prevents duplicate alert event generation during cooldown duration."""
        abnormal_pred = RealTimePredictionResult(
            wheel_id=1,
            predicted_condition="SEVERE_MISALIGNMENT",
            prediction_probability=0.90
        )

        # Trigger 3 consecutive frames to escalate alert
        for _ in range(3):
            self.alert_system.evaluate_prediction(abnormal_pred, track_id=1)

        initial_alerts_count = len(self.alert_system.alert_history)
        self.assertEqual(initial_alerts_count, 1)

        # Process another abnormal frame immediately (within 0.5s cooldown)
        _, _, is_esc, event = self.alert_system.evaluate_prediction(abnormal_pred, track_id=1)
        self.assertTrue(is_esc)
        self.assertIsNone(event)  # Cooldown active => No duplicate event created
        self.assertEqual(len(self.alert_system.alert_history), initial_alerts_count)

    def test_warning_text_and_disclaimer(self) -> None:
        """Verifies mandatory warning message and non-diagnostic disclaimer are included."""
        abnormal_pred = RealTimePredictionResult(
            wheel_id=1,
            predicted_condition="POSSIBLE_MISALIGNMENT",
            prediction_probability=0.85
        )

        # Trigger escalation
        for _ in range(3):
            _, _, _, event = self.alert_system.evaluate_prediction(abnormal_pred, track_id=1)

        self.assertIsNotNone(event)
        self.assertEqual(event.message, MANDATORY_WARNING_TEXT)
        self.assertEqual(event.disclaimer, NON_DIAGNOSTIC_DISCLAIMER)
        self.assertIn("Possible wheel alignment issue detected", event.message)
        self.assertIn("Not a certified mechanical automotive diagnosis", event.disclaimer)

        # Check alert log file output
        self.assertTrue(self.log_file.exists())
        log_content = self.log_file.read_text(encoding="utf-8")
        self.assertIn("POSSIBLE_MISALIGNMENT", log_content)
        self.assertIn(MANDATORY_WARNING_TEXT, log_content)

    def test_audio_alert_callback(self) -> None:
        """Verifies optional audio alert callback execution."""
        callback_called = []
        self.alert_system.enable_audio = True
        self.alert_system.audio_callback = lambda: callback_called.append(True)

        abnormal_pred = RealTimePredictionResult(
            wheel_id=1,
            predicted_condition="SEVERE_MISALIGNMENT",
            prediction_probability=0.95
        )

        for _ in range(3):
            self.alert_system.evaluate_prediction(abnormal_pred, track_id=1)

        # Short pause for daemon thread execution
        import time
        time.sleep(0.1)
        self.assertTrue(len(callback_called) > 0)

    def test_visual_overlay_rendering(self) -> None:
        """Verifies draw_overlay renders bounding box and warning text banner."""
        pred = RealTimePredictionResult(
            wheel_id=1,
            predicted_condition="SEVERE_MISALIGNMENT",
            prediction_probability=0.92,
            camber_proxy=3.5,
            toe_proxy=2.1
        )

        annotated = self.alert_system.process_alert(pred, track_id=1, frame=self.dummy_frame, bbox=self.dummy_bbox)
        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.dummy_frame.shape)

    def test_stale_track_purging(self) -> None:
        """Verifies purge_inactive_tracks cleans up track states for missing track IDs."""
        pred = RealTimePredictionResult(wheel_id=1, predicted_condition="NORMAL", prediction_probability=0.90)
        self.alert_system.evaluate_prediction(pred, track_id=1)
        self.alert_system.evaluate_prediction(pred, track_id=99)

        self.assertIn(1, self.alert_system.track_states)
        self.assertIn(99, self.alert_system.track_states)

        # Purge track 99
        self.alert_system.purge_inactive_tracks(active_track_ids=[1])

        self.assertIn(1, self.alert_system.track_states)
        self.assertNotIn(99, self.alert_system.track_states)


if __name__ == "__main__":
    unittest.main()
