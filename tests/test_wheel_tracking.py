"""
Unit Tests for Multi-Frame Wheel Tracking Subsystem.

Tests:
1. Persistent track ID assignment and state structure.
2. Temporal detection matching across consecutive video frames.
3. Temporary detection loss & stale track timeout removal.
4. Trajectory history window maintenance.
5. Image-space displacement and velocity feature calculations.
6. Trajectory visualization rendering.
"""

from pathlib import Path
import sys
import time
import unittest
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wheel_detection import WheelDetection
from src.wheel_tracking import WheelTracker, TrackedWheel, compute_iou


class TestWheelTrackingSubsystem(unittest.TestCase):
    """Unit test suite for WheelTracker and TrackedWheel components."""

    def setUp(self) -> None:
        """Sets up tracker instance for testing."""
        self.tracker = WheelTracker(
            max_disappeared=3,
            distance_threshold=60.0,
            iou_threshold=0.3,
            history_size=10
        )

    def test_track_creation_and_persistent_id(self) -> None:
        """Tests registration of new detections and persistent ID assignment."""
        det1 = WheelDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="wheel")
        det2 = WheelDetection(bbox=(400, 100, 500, 200), confidence=0.85, class_id=0, class_name="wheel")

        tracks = self.tracker.update([det1, det2])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(tracks[0].track_id, 0)
        self.assertEqual(tracks[1].track_id, 1)

    def test_temporal_matching(self) -> None:
        """Tests consistent track ID matching across consecutive frames."""
        det_f1 = WheelDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
        self.tracker.update([det_f1])

        # Slightly shifted detection in frame 2 (moving 10px right)
        det_f2 = WheelDetection(bbox=(110, 100, 210, 200), confidence=0.91, class_id=0)
        tracks_f2 = self.tracker.update([det_f2])

        self.assertEqual(len(tracks_f2), 1)
        self.assertEqual(tracks_f2[0].track_id, 0)  # ID preserved
        self.assertEqual(tracks_f2[0].centroid, (160, 150))
        self.assertEqual(tracks_f2[0].age, 2)

    def test_disappearance_and_stale_track_timeout(self) -> None:
        """Tests disappearance counter and stale track removal after max_disappeared timeout."""
        det = WheelDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
        self.tracker.update([det])  # Frame 1: active

        # Frames 2, 3, 4: zero detections
        t2 = self.tracker.update([])
        self.assertEqual(len(t2), 1)
        self.assertEqual(t2[0].disappeared_count, 1)

        t3 = self.tracker.update([])
        self.assertEqual(len(t3), 1)
        self.assertEqual(t3[0].disappeared_count, 2)

        t4 = self.tracker.update([])
        self.assertEqual(len(t4), 1)
        self.assertEqual(t4[0].disappeared_count, 3)

        # Frame 5: Exceeds max_disappeared (3) -> track removed
        t5 = self.tracker.update([])
        self.assertEqual(len(t5), 0)

    def test_history_window_limit(self) -> None:
        """Tests history window size constraint."""
        # Configured history_size=10
        det_base = WheelDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
        self.tracker.update([det_base])

        # Update 15 times
        for i in range(15):
            det = WheelDetection(bbox=(100 + i * 2, 100, 200 + i * 2, 200), confidence=0.9, class_id=0)
            tracks = self.tracker.update([det])

        self.assertEqual(len(tracks[0].history), 10)

    def test_motion_feature_calculations(self) -> None:
        """Tests image-space displacement and velocity feature calculation."""
        det1 = WheelDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
        t1 = time.time()
        self.tracker.update([det1], timestamp=t1)

        # Frame 2: Shift 30px right, 40px down after 0.1 sec (displacement magnitude = 50px)
        det2 = WheelDetection(bbox=(130, 140, 230, 240), confidence=0.9, class_id=0)
        t2 = t1 + 0.1
        tracks = self.tracker.update([det2], timestamp=t2)

        motion = tracks[0].calculate_motion_features()
        self.assertEqual(motion["dx"], 30.0)
        self.assertEqual(motion["dy"], 40.0)
        self.assertEqual(motion["displacement_magnitude"], 50.0)
        self.assertAlmostEqual(motion["velocity_magnitude"], 500.0, delta=1.0)  # 50px / 0.1s = 500px/s

    def test_trajectory_visualization(self) -> None:
        """Tests trajectory rendering overlay."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = WheelDetection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
        tracks = self.tracker.update([det])

        rendered = self.tracker.draw_trajectories(dummy_frame, tracks)
        self.assertEqual(rendered.shape, dummy_frame.shape)

    def test_compute_iou(self) -> None:
        """Tests IoU bounding box function."""
        box1 = (0, 0, 100, 100)
        box2 = (0, 0, 100, 100)
        self.assertEqual(compute_iou(box1, box2), 1.0)

        box3 = (50, 0, 150, 100)
        self.assertAlmostEqual(compute_iou(box1, box3), 1/3, places=2)


if __name__ == "__main__":
    unittest.main()
