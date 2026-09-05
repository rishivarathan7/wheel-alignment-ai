"""
Unit Tests for Wheel Detection Subsystem.

Tests:
1. WheelDetection dataclass structure, fields (bbox, confidence, class_id, class_name, timestamp, detection_id).
2. Separation of PretrainedYOLODetector and CustomYOLOWheelDetector.
3. Configurable confidence and IoU threshold execution.
4. Handling of empty images and zero detection frames.
5. Visualization drawing and saving debug images into results/.
6. Extracted wheel crop region validation.
"""

from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR
from src.wheel_detection import (
    WheelDetection,
    YOLOWheelDetector,
    PretrainedYOLODetector,
    CustomYOLOWheelDetector
)


class TestWheelDetectionSubsystem(unittest.TestCase):
    """Unit test suite for Wheel Detection subsystem."""

    def setUp(self) -> None:
        """Sets up test image frame and detector instance."""
        self.sample_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simulated wheel circle on synthetic frame
        cv2.circle(self.sample_img, (320, 240), 80, (200, 200, 200), -1)

        self.detector = YOLOWheelDetector(confidence_threshold=0.4, iou_threshold=0.45)

    def test_wheel_detection_structure(self) -> None:
        """Tests WheelDetection data fields and properties."""
        det = WheelDetection(
            bbox=(100, 150, 300, 350),
            confidence=0.92,
            class_id=0,
            class_name="wheel"
        )

        self.assertEqual(det.bbox, (100, 150, 300, 350))
        self.assertEqual(det.width, 200)
        self.assertEqual(det.height, 200)
        self.assertEqual(det.center, (200, 250))
        self.assertIsNotNone(det.timestamp)
        self.assertTrue(det.detection_id.startswith("det_"))

        data_dict = det.to_dict()
        self.assertEqual(data_dict["class_name"], "wheel")
        self.assertEqual(data_dict["confidence"], 0.92)

    def test_pretrained_vs_custom_detector_separation(self) -> None:
        """Tests clean separation between pretrained COCO and custom detectors."""
        pretrained = PretrainedYOLODetector(confidence_threshold=0.5)
        self.assertIsInstance(pretrained, PretrainedYOLODetector)

        # High-level wrapper should default to PretrainedYOLODetector when standard weights are passed
        self.assertIsInstance(self.detector.active_detector, PretrainedYOLODetector)

    def test_threshold_and_device_configuration(self) -> None:
        """Tests configurable thresholds and execution device."""
        det = YOLOWheelDetector(confidence_threshold=0.6, iou_threshold=0.3, device="cpu")
        self.assertEqual(det.confidence_threshold, 0.6)
        self.assertEqual(det.iou_threshold, 0.3)
        self.assertEqual(det.device, "cpu")

    def test_zero_detection_handling(self) -> None:
        """Tests graceful handling when frame is None or zero detections occur."""
        self.assertEqual(self.detector.detect(None), [])
        self.assertEqual(self.detector.detect(np.array([])), [])

    def test_detection_visualization_and_debug_saving(self) -> None:
        """Tests detection visualization drawing and debug image saving into results/."""
        detections = [
            WheelDetection(bbox=(50, 50, 200, 200), confidence=0.89, class_id=0, class_name="wheel_test")
        ]

        annotated = self.detector.draw_detections(self.sample_img, detections)
        self.assertEqual(annotated.shape, self.sample_img.shape)

        # Save debug image
        debug_file = self.detector.save_debug_image(
            self.sample_img,
            detections,
            output_dir=RESULTS_DIR,
            filename="unit_test_detection.jpg"
        )
        self.assertIsNotNone(debug_file)
        self.assertTrue(debug_file.exists())

        # Clean up test file
        if debug_file and debug_file.exists():
            debug_file.unlink()

    def test_extracted_wheel_crops(self) -> None:
        """Tests region-of-interest wheel crop extraction."""
        detections = [
            WheelDetection(bbox=(100, 100, 250, 250), confidence=0.95, class_id=0, class_name="wheel")
        ]
        crops = self.detector.extract_wheel_crops(self.sample_img, detections)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0].shape, (150, 150, 3))


if __name__ == "__main__":
    unittest.main()
