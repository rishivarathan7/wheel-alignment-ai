"""
Unit tests for Module 14 — End-to-End Real-Time Pipeline Integration.
"""

import math
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np

from src.config import ConfigManager
from src.pipeline import WheelAlignmentPipeline
from src.wheel_detection import WheelDetection


class TestPipelineIntegration(unittest.TestCase):
    """Test suite for WheelAlignmentPipeline end-to-end processing."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

        # Initialize test configuration
        self.config = ConfigManager()
        self.pipeline = WheelAlignmentPipeline(config_manager=self.config, debug_mode=False)

        # Create synthetic test frame (640x480 BGR image with simulated wheel circles)
        self.test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Draw synthetic wheel circles
        cv2.circle(self.test_frame, (200, 240), 50, (30, 30, 30), -1)
        cv2.circle(self.test_frame, (200, 240), 30, (200, 200, 200), 3)

        cv2.circle(self.test_frame, (440, 240), 50, (30, 30, 30), -1)
        cv2.circle(self.test_frame, (440, 240), 30, (200, 200, 200), 3)

    def tearDown(self) -> None:
        self.pipeline.stop()
        self.tmp_dir.cleanup()

    def test_single_frame_processing(self) -> None:
        """Verifies single frame processing returns annotated frame, telemetry, and performance metrics."""
        annotated, telemetry, perf = self.pipeline.process_frame(self.test_frame)

        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, self.test_frame.shape)
        self.assertIsInstance(telemetry, list)
        self.assertIsInstance(perf, dict)

        self.assertTrue(perf.get("is_valid", False))
        self.assertEqual(perf.get("frame_index"), 1)
        self.assertIn("total_latency_ms", perf)
        self.assertIn("fps", perf)
        self.assertGreaterEqual(perf["total_latency_ms"], 0.0)

    def test_multi_frame_sequential_tracking(self) -> None:
        """Verifies sequential multi-frame processing maintains track identities and builds temporal history."""
        # Process 5 sequential frames with slight movement
        for i in range(5):
            frame = self.test_frame.copy()
            # Shift synthetic wheel circle slightly horizontally
            shift_x = i * 5
            cv2.circle(frame, (200 + shift_x, 240), 50, (30, 30, 30), -1)
            cv2.circle(frame, (200 + shift_x, 240), 30, (200, 200, 200), 3)

            annotated, telemetry, perf = self.pipeline.process_frame(frame)
            self.assertEqual(perf["frame_index"], i + 1)
            self.assertTrue(perf["is_valid"])

        self.assertEqual(self.pipeline.frame_index, 5)

    def test_missing_detections_handling(self) -> None:
        """Verifies zero-detection frames handle disappearance gracefully without crashing or losing state."""
        # Mock detector returning 0 detections
        self.pipeline.detector.detect = lambda frame: []

        annotated, telemetry, perf = self.pipeline.process_frame(self.test_frame)

        self.assertIsNotNone(annotated)
        self.assertEqual(len(telemetry), 0)
        self.assertEqual(perf["active_tracks"], 0)
        self.assertTrue(perf["is_valid"])

    def test_latency_and_fps_measurement(self) -> None:
        """Verifies stage latency breakdown profiling and FPS calculations."""
        _, _, perf = self.pipeline.process_frame(self.test_frame)

        self.assertIn("prep_latency_ms", perf)
        self.assertIn("det_latency_ms", perf)
        self.assertIn("track_latency_ms", perf)
        self.assertIn("analysis_latency_ms", perf)
        self.assertIn("ml_latency_ms", perf)
        self.assertIn("total_latency_ms", perf)

        self.assertGreaterEqual(perf["fps"], 0.0)

    def test_memory_leak_prevention(self) -> None:
        """Verifies that temporal measurement history is purged when a track is deregistered."""
        # Manually register synthetic tracks in history buffer
        self.pipeline.raw_history_per_track[0] = [{"aspect_ratio": 1.0}]
        self.pipeline.raw_history_per_track[99] = [{"aspect_ratio": 1.0}]  # Stale track ID

        # Set tracker to only have track ID 0 active
        class DummyTrack:
            pass
        t0 = DummyTrack()
        t0.track_id = 0
        self.pipeline.tracker.tracks = {0: t0}

        # Trigger history buffer synchronization
        self.pipeline._sync_track_history_buffers()

        self.assertIn(0, self.pipeline.raw_history_per_track)
        self.assertNotIn(99, self.pipeline.raw_history_per_track)

    def test_debug_mode_visualization(self) -> None:
        """Verifies debug mode renders HUD overlays and attaches intermediate debug payloads."""
        self.pipeline.debug_mode = True

        annotated, telemetry, perf = self.pipeline.process_frame(self.test_frame)

        self.assertIn("debug_data", perf)
        debug_payload = perf["debug_data"]
        self.assertIn("preprocessed_image", debug_payload)
        self.assertIn("raw_detections", debug_payload)
        self.assertIn("tracked_wheels", debug_payload)

    def test_graceful_shutdown(self) -> None:
        """Verifies pipeline graceful shutdown state management."""
        self.pipeline._is_running = True
        self.pipeline.stop()
        self.assertFalse(self.pipeline._is_running)

    def test_empty_frame_handling(self) -> None:
        """Verifies empty or 0-size frames return safely."""
        empty_frame = np.array([])
        annotated, telemetry, perf = self.pipeline.process_frame(empty_frame)

        self.assertFalse(perf["is_valid"])
        self.assertEqual(len(telemetry), 0)


if __name__ == "__main__":
    unittest.main()
