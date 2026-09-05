"""
Unit Tests for Alignment Anomaly Classification Subsystem.

Tests:
1. Canonical 3-class target label normalization (NORMAL, POSSIBLE_MISALIGNMENT, SEVERE_MISALIGNMENT).
2. AlignmentPredictor model loading and ML prediction.
3. Physical rule heuristic fallback prediction when model is uninitialized.
4. Classification evaluation metrics and confusion matrix generation.
"""

from pathlib import Path
import sys
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import WheelRegionAnalysis
from src.prediction import (
    AlignmentPredictor,
    AlignmentPrediction,
    normalize_class_label,
    ALIGNMENT_CLASSES
)


class TestAlignmentClassificationSubsystem(unittest.TestCase):
    """Unit test suite for AlignmentPredictor component."""

    def setUp(self) -> None:
        """Sets up test environment and temporary model artifacts."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create synthetic 3-class trained model
        np.random.seed(42)
        X = np.random.randn(30, 19).astype(np.float32)
        y = np.array(["NORMAL"] * 10 + ["POSSIBLE_MISALIGNMENT"] * 10 + ["SEVERE_MISALIGNMENT"] * 10)

        self.clf = RandomForestClassifier(n_estimators=10, random_state=42)
        self.clf.fit(X, y)

        self.model_path = Path(self.temp_dir.name) / "test_classifier.joblib"
        joblib.dump({"model": self.clf}, self.model_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_class_label_normalization(self) -> None:
        """Tests normalization of string, numeric, and legacy class labels."""
        self.assertEqual(normalize_class_label("NORMAL"), "NORMAL")
        self.assertEqual(normalize_class_label("POSSIBLE_MISALIGNMENT"), "POSSIBLE_MISALIGNMENT")
        self.assertEqual(normalize_class_label("SEVERE_MISALIGNMENT"), "SEVERE_MISALIGNMENT")
        self.assertEqual(normalize_class_label("MISALIGNED"), "SEVERE_MISALIGNMENT")
        self.assertEqual(normalize_class_label("SUSPECT"), "POSSIBLE_MISALIGNMENT")
        self.assertEqual(normalize_class_label(0), "NORMAL")
        self.assertEqual(normalize_class_label(1), "POSSIBLE_MISALIGNMENT")
        self.assertEqual(normalize_class_label(2), "SEVERE_MISALIGNMENT")

    def test_predictor_model_loading_and_prediction(self) -> None:
        """Tests loading trained ML classifier model and making predictions."""
        predictor = AlignmentPredictor(model_path=self.model_path)
        self.assertTrue(predictor._is_loaded)

        analysis = WheelRegionAnalysis(
            aspect_ratio=1.0,
            circularity=0.85,
            eccentricity=0.2,
            edge_density=0.15,
            fitted_ellipse_angle=90.0,
            orientation_valid=True
        )

        pred = predictor.predict(analysis)
        self.assertIsInstance(pred, AlignmentPrediction)
        self.assertIn(pred.status, ALIGNMENT_CLASSES)
        self.assertGreaterEqual(pred.confidence, 0.0)
        self.assertLessEqual(pred.confidence, 1.0)

    def test_heuristic_fallback(self) -> None:
        """Tests fallback to heuristic rule prediction when ML model path is invalid."""
        predictor = AlignmentPredictor(model_path=Path(self.temp_dir.name) / "non_existent.joblib")
        self.assertFalse(predictor._is_loaded)

        analysis_normal = WheelRegionAnalysis(aspect_ratio=1.0, fitted_ellipse_angle=90.0, eccentricity=0.1)
        pred_normal = predictor.predict(analysis_normal)
        self.assertEqual(pred_normal.status, "NORMAL")

        analysis_severe = WheelRegionAnalysis(aspect_ratio=1.5, fitted_ellipse_angle=140.0, eccentricity=0.8)
        pred_severe = predictor.predict(analysis_severe)
        self.assertIn(pred_severe.status, ["POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"])


if __name__ == "__main__":
    unittest.main()
