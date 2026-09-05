"""
Unit Tests for Video Acquisition and Frame Management Subsystem.

Tests:
1. Valid video input processing (using synthetic video file).
2. Invalid video path handling.
3. Camera initialization failure handling.
4. End-of-video (EOF) handling.
5. Frame skipping & timestamp overlay verification.
"""

from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video_capture import VideoCaptureManager


class TestVideoCaptureSubsystem(unittest.TestCase):
    """Unit tests for VideoCaptureManager."""

    @classmethod
    def setUpClass(cls) -> None:
        """Creates a temporary synthetic MP4 video file for testing."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_video_path = Path(cls.temp_dir.name) / "test_sample.mp4"

        # Parameters for synthetic test video
        width, height = 320, 240
        fps = 10.0
        cls.total_frames = 15

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(cls.test_video_path), fourcc, fps, (width, height))

        for i in range(cls.total_frames):
            # Create a simple frame with changing color
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (i * 15 % 255, 120, 200)
            out.write(frame)
        out.release()

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up temporary directory."""
        cls.temp_dir.cleanup()

    def test_valid_video_input(self) -> None:
        """Tests reading frames and properties from a valid video file."""
        with VideoCaptureManager(source=self.test_video_path) as cap:
            self.assertTrue(cap.is_opened)
            props = cap.get_properties()
            self.assertEqual(props["width"], 320)
            self.assertEqual(props["height"], 240)
            self.assertEqual(props["frame_count"], self.total_frames)

            ret, frame = cap.read_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (240, 320, 3))

    def test_invalid_video_path(self) -> None:
        """Tests initialization failure handling for non-existent video files."""
        invalid_path = Path(self.temp_dir.name) / "non_existent_video_12345.mp4"
        cap = VideoCaptureManager(source=invalid_path)

        self.assertFalse(cap.is_opened)
        ret, frame = cap.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        cap.release()

    def test_camera_initialization_failure(self) -> None:
        """Tests initialization failure handling for invalid camera index."""
        invalid_camera_index = 99999
        cap = VideoCaptureManager(source=invalid_camera_index)

        self.assertFalse(cap.is_opened)
        ret, frame = cap.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        cap.release()

    def test_end_of_video_handling(self) -> None:
        """Tests reading all frames until End-Of-File (EOF) is reached."""
        with VideoCaptureManager(source=self.test_video_path) as cap:
            self.assertTrue(cap.is_opened)
            read_count = 0

            for frame in cap.stream_frames():
                self.assertIsNotNone(frame)
                read_count += 1

            self.assertEqual(read_count, self.total_frames)
            self.assertTrue(cap.is_eof)

            # Extra read after EOF should return (False, None)
            ret, frame = cap.read_frame()
            self.assertFalse(ret)
            self.assertIsNone(frame)

    def test_frame_skipping(self) -> None:
        """Tests optional frame skipping functionality."""
        # Skipping 1 frame means processing 1 out of every 2 frames
        with VideoCaptureManager(source=self.test_video_path, frame_skip=1) as cap:
            processed_count = 0
            while cap.is_opened and not cap.is_eof:
                ret, frame = cap.read_frame()
                if ret:
                    processed_count += 1

            # For 15 frames with skip=1, frames 1,3,5,7,9,11,13 are processed (7 frames)
            expected_processed = (self.total_frames) // 2
            self.assertEqual(processed_count, expected_processed)

    def test_timestamp_overlay_and_metadata(self) -> None:
        """Tests timestamp overlay drawing and metadata extraction."""
        with VideoCaptureManager(source=self.test_video_path, add_timestamp=True) as cap:
            success, frame, metadata = cap.read_frame_metadata()
            self.assertTrue(success)
            self.assertIsNotNone(frame)
            self.assertIn("timestamp", metadata)
            self.assertIn("frame_index", metadata)
            self.assertIn("fps", metadata)
            self.assertEqual(metadata["frame_index"], 1)


if __name__ == "__main__":
    unittest.main()
