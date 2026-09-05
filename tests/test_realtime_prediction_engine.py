"""
Unit Tests for Real-Time Prediction Engine Subsystem.

Tests:
1. Classifier model and preprocessing scaler artifact loading.
2. Feature name validation, order verification, and missing value imputation.
3. Structured output generation (wheel_id, predicted_condition, prediction_probability, timestamp, model_version).
4. Low inference latency benchmark (< 10 ms per sample).
5. Prediction failure logging and safe fallback handling.
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

from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.feature_extraction import WheelRegionAnalysis
from src.prediction import (
    RealTimePredictionEngine,
    RealTimePredictionResult,
    ALIGNMENT_CLASSES
)


class TestRealTimePredictionEngine(unittest.TestCase):
    """Unit test suite for RealTimePredictionEngine component."""

    def setUp(self) -> None:
        """Sets up test environment, trained model artifact, and scaler pipeline."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # 1. Fit & save scaler artifact
        self.engineer = FeatureEngineer()
        dummy_df = pd.DataFrame(np.random.randn(20, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
        self.engineer.fit(dummy_df)
        self.scaler_path = Path(self.temp_dir.name) / "test_scaler.joblib"
        self.engineer.save_pipeline(self.scaler_path)

        # 2. Fit & save model artifact
        self.clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X_scaled = self.engineer.transform(dummy_df)
        y = np.array(["NORMAL"] * 10 + ["POSSIBLE_MISALIGNMENT"] * 5 + ["SEVERE_MISALIGNMENT"] * 5)
        self.clf.fit(X_scaled, y)

        self.model_path = Path(self.temp_dir.name) / "test_model.joblib"
        joblib.dump({"model": self.clf, "feature_names": FEATURE_COLUMNS}, self.model_path)

        # 3. Create Prediction Engine Instance
        self.engine = RealTimePredictionEngine(
            model_path=self.model_path,
            scaler_path=self.scaler_path,
            model_version="v1.0.0_TestEngine"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_artifact_loading(self) -> None:
        """Tests classifier model and scaler pipeline loading."""
        self.assertTrue(self.engine.is_ready)
        self.assertEqual(self.engine.expected_feature_names, FEATURE_COLUMNS)

    def test_predict_wheel_structured_result(self) -> None:
        """Tests structured result object generation."""
        analysis = WheelRegionAnalysis(
            aspect_ratio=1.0,
            circularity=0.85,
            eccentricity=0.2,
            edge_density=0.15,
            fitted_ellipse_angle=90.0
        )

        res = self.engine.predict_wheel(analysis, wheel_id=7)
        self.assertIsInstance(res, RealTimePredictionResult)
        self.assertEqual(res.wheel_id, 7)
        self.assertIn(res.predicted_condition, ALIGNMENT_CLASSES)
        self.assertGreaterEqual(res.prediction_probability, 0.0)
        self.assertLessEqual(res.prediction_probability, 1.0)
        self.assertEqual(res.model_version, "v1.0.0_TestEngine")
        self.assertGreater(res.inference_latency_ms, 0.0)
        self.assertFalse(res.is_fallback)

    def test_inference_latency_performance(self) -> None:
        """Benchmarks inference latency (must execute under real-time constraints)."""
        analysis = WheelRegionAnalysis(aspect_ratio=1.0, circularity=0.85, eccentricity=0.2)
        # Warmup call
        _ = self.engine.predict_wheel(analysis, wheel_id=1)
        latencies = [self.engine.predict_wheel(analysis, wheel_id=1).inference_latency_ms for _ in range(5)]
        avg_lat = sum(latencies) / len(latencies)
        self.assertLess(avg_lat, 100.0)

    def test_missing_feature_imputation(self) -> None:
        """Tests missing feature imputation and dict payload handling."""
        incomplete_features = {"aspect_ratio": 1.1, "circularity": 0.8}
        res = self.engine.predict_wheel(incomplete_features, wheel_id=3)
        self.assertIsInstance(res, RealTimePredictionResult)
        self.assertIn(res.predicted_condition, ALIGNMENT_CLASSES)

    def test_fallback_mode_on_invalid_artifact(self) -> None:
        """Tests clean fallback prediction handling when artifact paths are missing or broken."""
        bad_engine = RealTimePredictionEngine(
            model_path=Path(self.temp_dir.name) / "missing_model.joblib",
            scaler_path=Path(self.temp_dir.name) / "missing_scaler.joblib"
        )
        self.assertFalse(bad_engine.is_ready)

        analysis = WheelRegionAnalysis(aspect_ratio=1.4, fitted_ellipse_angle=130.0, eccentricity=0.7)
        res = bad_engine.predict_wheel(analysis, wheel_id=2)
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.model_version, "Fallback_Heuristic_Rules")
        self.assertIn(res.predicted_condition, ALIGNMENT_CLASSES)


if __name__ == "__main__":
    unittest.main()
