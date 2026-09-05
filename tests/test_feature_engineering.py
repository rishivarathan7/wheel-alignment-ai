"""
Unit Tests for Feature Engineering Subsystem.

Tests:
1. Feature extraction from raw measurement DataFrames.
2. Rolling window statistic calculation per track ID.
3. Fit and transform normalization pipeline without data leakage.
4. Saving and loading feature scaling pipeline artifact.
5. Feature dictionary export.
6. Correlation and feature importance plot generation.
"""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    FeatureEngineer,
    FEATURE_COLUMNS,
    export_feature_dictionary,
    analyze_feature_importance_and_correlation
)


class TestFeatureEngineeringSubsystem(unittest.TestCase):
    """Unit test suite for FeatureEngineer component."""

    def setUp(self) -> None:
        """Sets up synthetic raw telemetry DataFrame for feature extraction."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engineer = FeatureEngineer(window_size=3)

        records = []
        for track_id in range(2):
            for frame_idx in range(10):
                records.append({
                    "frame_index": frame_idx,
                    "track_id": track_id,
                    "is_valid": True,
                    "aspect_ratio": 1.0 + 0.1 * frame_idx,
                    "circularity": 0.8,
                    "eccentricity": 0.3,
                    "contour_area": 1000.0,
                    "contour_perimeter": 100.0,
                    "bbox_area": 1000,
                    "edge_density": 0.2,
                    "fitted_ellipse_angle": 90.0 + frame_idx,
                    "delta_orientation": 1.0,
                    "delta_center_x": 2.0,
                    "delta_center_y": 1.0,
                    "movement_speed": 2.23
                })
        self.df_raw = pd.DataFrame(records)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_transform_raw_measurements(self) -> None:
        """Tests transformation of raw measurements into engineered features dataframe."""
        df_eng = self.engineer.transform_raw_measurements(self.df_raw)
        self.assertEqual(len(df_eng), 20)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df_eng.columns)

    def test_rolling_window_statistics(self) -> None:
        """Tests rolling mean and std calculations per track ID."""
        df_eng = self.engineer.transform_raw_measurements(self.df_raw)
        # Check rolling mean aspect ratio for frame 2 (0-indexed) of track 0
        # values: 1.0, 1.1, 1.2 -> mean = 1.1
        self.assertAlmostEqual(df_eng.iloc[2]["roll_mean_aspect_ratio"], 1.1, places=4)

    def test_fit_transform_pipeline(self) -> None:
        """Tests leakage-free fit and transform scaling pipeline."""
        df_eng = self.engineer.transform_raw_measurements(self.df_raw)

        # Fit on first 10 rows (track 0)
        train_df = df_eng.iloc[:10]
        test_df = df_eng.iloc[10:]

        self.engineer.fit(train_df)
        self.assertTrue(self.engineer.is_fitted)

        scaled_test = self.engineer.transform(test_df)
        self.assertEqual(scaled_test.shape, (10, len(FEATURE_COLUMNS)))
        self.assertIsInstance(scaled_test, np.ndarray)

    def test_save_and_load_pipeline(self) -> None:
        """Tests serialization and deserialization of scaling pipeline."""
        df_eng = self.engineer.transform_raw_measurements(self.df_raw)
        self.engineer.fit(df_eng)

        scaler_path = Path(self.temp_dir.name) / "test_scaler.joblib"
        self.engineer.save_pipeline(scaler_path)
        self.assertTrue(scaler_path.exists())

        new_engineer = FeatureEngineer()
        success = new_engineer.load_pipeline(scaler_path)
        self.assertTrue(success)
        self.assertTrue(new_engineer.is_fitted)

    def test_export_feature_dictionary(self) -> None:
        """Tests feature dictionary export to JSON."""
        dict_path = Path(self.temp_dir.name) / "test_dictionary.json"
        exported = export_feature_dictionary(dict_path)
        self.assertTrue(exported.exists())

    def test_feature_importance_and_correlation(self) -> None:
        """Tests correlation matrix and feature importance plot generation."""
        df_eng = self.engineer.transform_raw_measurements(self.df_raw)
        labels = np.array([0] * 10 + [1] * 10)

        results = analyze_feature_importance_and_correlation(df_eng, labels, output_dir=self.temp_dir.name)
        self.assertTrue(Path(results["correlation_plot"]).exists())
        self.assertTrue(Path(results["importance_plot"]).exists())


if __name__ == "__main__":
    unittest.main()
