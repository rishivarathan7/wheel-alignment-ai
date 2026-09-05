"""
Unit Tests for Wheel Region Analysis Subsystem.

Tests:
1. Calculation of visual & geometric metrics (width, height, aspect ratio, center, circularity, edge density).
2. Fitted ellipse orientation angle estimation.
3. Measurement validation rules and corrupted input handling.
4. Temporal deltas (delta orientation, delta centroid, movement speed across frames).
5. CSV telemetry logging and export to results/.
6. Debug visualization rendering.
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

from src.feature_extraction import (
    WheelRegionAnalysis,
    WheelFeatureExtractor,
    WheelTelemetryExporter,
    validate_analysis_measurements
)


class TestWheelRegionAnalysisSubsystem(unittest.TestCase):
    """Unit test suite for Wheel Region Analysis component."""

    def setUp(self) -> None:
        """Sets up test synthetic crop images and feature extractor instance."""
        self.extractor = WheelFeatureExtractor()
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a synthetic 100x100 BGR wheel crop with a circle
        self.sample_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(self.sample_crop, (50, 50), 40, (255, 255, 255), -1)
        cv2.circle(self.sample_crop, (50, 50), 20, (50, 50, 50), -1)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_geometric_and_visual_metrics(self) -> None:
        """Tests calculation of bbox, aspect ratio, contour, and edge density metrics."""
        bbox = (50, 50, 150, 150)
        analysis = self.extractor.analyze_region(
            wheel_crop=self.sample_crop,
            bbox=bbox,
            track_id=1,
            frame_index=10
        )

        self.assertTrue(analysis.is_valid)
        self.assertEqual(analysis.bbox_width, 100)
        self.assertEqual(analysis.bbox_height, 100)
        self.assertEqual(analysis.aspect_ratio, 1.0)
        self.assertEqual((analysis.center_x, analysis.center_y), (100, 100))
        self.assertGreater(analysis.edge_density, 0.0)
        self.assertGreater(analysis.contour_area, 0.0)
        self.assertGreater(analysis.circularity, 0.0)

    def test_fitted_ellipse_orientation(self) -> None:
        """Tests fitted ellipse orientation angle estimation."""
        bbox = (0, 0, 100, 100)
        analysis = self.extractor.analyze_region(self.sample_crop, bbox=bbox, track_id=2)

        self.assertTrue(analysis.orientation_valid)
        self.assertGreaterEqual(analysis.fitted_ellipse_angle, 0.0)
        self.assertLessEqual(analysis.fitted_ellipse_angle, 180.0)

    def test_temporal_delta_calculations(self) -> None:
        """Tests temporal orientation changes and movement speed calculations across consecutive frames."""
        bbox1 = (100, 100, 200, 200)
        a1 = self.extractor.analyze_region(self.sample_crop, bbox=bbox1, track_id=5, frame_index=1)

        # Shift 10px right, 20px down in frame 2
        bbox2 = (110, 120, 210, 220)
        a2 = self.extractor.analyze_region(self.sample_crop, bbox=bbox2, track_id=5, frame_index=2)

        self.assertEqual(a2.delta_center_x, 10.0)
        self.assertEqual(a2.delta_center_y, 20.0)
        self.assertAlmostEqual(a2.movement_speed, np.sqrt(10**2 + 20**2), places=2)

    def test_measurement_validation(self) -> None:
        """Tests validation rules for checking measurement integrity."""
        # Valid analysis
        valid_a = WheelRegionAnalysis(bbox_width=100, bbox_height=100, aspect_ratio=1.0, edge_density=0.15)
        validated = validate_analysis_measurements(valid_a)
        self.assertTrue(validated.is_valid)

        # Invalid analysis (negative width)
        invalid_a = WheelRegionAnalysis(bbox_width=-10, bbox_height=100, aspect_ratio=-1.0, edge_density=1.5)
        validated_inv = validate_analysis_measurements(invalid_a)
        self.assertFalse(validated_inv.is_valid)

    def test_csv_telemetry_exporter(self) -> None:
        """Tests logging and exporting telemetry measurements to CSV."""
        csv_file = Path(self.temp_dir.name) / "test_telemetry.csv"
        exporter = WheelTelemetryExporter(output_path=csv_file)

        analysis1 = self.extractor.analyze_region(self.sample_crop, bbox=(50, 50, 150, 150), track_id=1, frame_index=1)
        analysis2 = self.extractor.analyze_region(self.sample_crop, bbox=(60, 60, 160, 160), track_id=1, frame_index=2)

        exporter.log_analysis(analysis1)
        exporter.log_analysis(analysis2)

        exported_path = exporter.export_to_csv()
        self.assertTrue(exported_path.exists())

        # Verify CSV contents via Pandas
        import pandas as pd
        df = pd.read_csv(exported_path)
        self.assertEqual(len(df), 2)
        self.assertIn("aspect_ratio", df.columns)
        self.assertIn("edge_density", df.columns)
        self.assertIn("fitted_ellipse_angle", df.columns)

    def test_analysis_visualization_rendering(self) -> None:
        """Tests debug visualization rendering."""
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)
        analysis = self.extractor.analyze_region(self.sample_crop, bbox=bbox, track_id=1)

        annotated = self.extractor.draw_analysis_visualizations(dummy_img, bbox, analysis)
        self.assertEqual(annotated.shape, dummy_img.shape)


if __name__ == "__main__":
    unittest.main()
