"""
Unit tests for Module 17 — Logging, Configuration, and Error Handling.
"""

import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import ConfigManager, setup_logger, get_execution_device, DEFAULT_CONFIG
from src.video_capture import VideoCaptureManager
from src.prediction import RealTimePredictionEngine, RealTimePredictionResult
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.feature_extraction import WheelRegionAnalysis, validate_analysis_measurements


class TestReliabilityAndConfig(unittest.TestCase):
    """Test suite verifying system reliability, error handling, and centralized configuration."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        import logging
        for name in logging.root.manager.loggerDict:
            lg = logging.getLogger(name)
            for h in list(lg.handlers):
                h.close()
                lg.removeHandler(h)
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass

    def test_config_manager_defaults_and_env(self) -> None:
        """Verifies ConfigManager default values, dot-notation access, and environment variable overrides."""
        cfg = ConfigManager()
        self.assertIsNotNone(cfg.get("video.width"))
        self.assertEqual(cfg.get("video.width"), 640)
        self.assertEqual(cfg.get("logging.level"), "INFO")

        # Set value via dot-notation
        cfg.set("alert.cooldown_seconds", 5.5)
        self.assertEqual(cfg.get("alert.cooldown_seconds"), 5.5)

        # Environment variable override check
        os.environ["WHEEL_AI_LOG_LEVEL"] = "DEBUG"
        cfg.load_config()
        self.assertEqual(cfg.get("logging.level"), "DEBUG")
        del os.environ["WHEEL_AI_LOG_LEVEL"]

    def test_config_save(self) -> None:
        """Verifies persisting configuration settings to YAML file."""
        save_file = self.tmp_path / "custom_settings.yaml"
        cfg = ConfigManager()
        cfg.set("pipeline.debug_mode", True)
        res_path = cfg.save(save_file)

        self.assertTrue(res_path.exists())
        self.assertIn("debug_mode", res_path.read_text(encoding="utf-8"))

    def test_setup_logger_creation(self) -> None:
        """Verifies setup_logger creates console, app file, and error log handlers."""
        app_log = self.tmp_path / "app_test.log"
        err_log = self.tmp_path / "err_test.log"

        logger_inst = setup_logger(
            name="test_reliability_logger",
            level="INFO",
            log_file=app_log,
            error_log_file=err_log
        )

        self.assertIsNotNone(logger_inst)
        self.assertTrue(len(logger_inst.handlers) >= 3)
        self.assertTrue(app_log.exists())
        self.assertTrue(err_log.exists())

    def test_missing_model_fallback(self) -> None:
        """Verifies RealTimePredictionEngine gracefully triggers fallback when artifacts are missing."""
        missing_model = self.tmp_path / "non_existent_model.joblib"
        missing_scaler = self.tmp_path / "non_existent_scaler.joblib"

        engine = RealTimePredictionEngine(
            model_path=missing_model,
            scaler_path=missing_scaler
        )

        self.assertFalse(engine.is_ready)

        # Predict should safely execute heuristic fallback rules without raising exception
        raw_input = {"aspect_ratio": 1.4, "eccentricity": 0.5}
        result = engine.predict_wheel(raw_input, wheel_id=10)

        self.assertIsInstance(result, RealTimePredictionResult)
        self.assertTrue(result.is_fallback)
        self.assertEqual(result.model_version, "Fallback_Heuristic_Rules")
        self.assertIn(result.predicted_condition, ["POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"])

    def test_invalid_video_file_handling(self) -> None:
        """Verifies VideoCaptureManager handles non-existent or corrupt video files gracefully."""
        bad_path = self.tmp_path / "non_existent_video.mp4"
        cap_mgr = VideoCaptureManager(source=bad_path)

        self.assertFalse(cap_mgr.is_opened)
        ret, frame = cap_mgr.read_frame()
        self.assertFalse(ret)
        self.assertIsNone(frame)

    def test_invalid_feature_nan_inf_imputation(self) -> None:
        """Verifies FeatureEngineer and WheelRegionAnalysis handle NaN, Inf, and missing values safely."""
        corrupted_data = pd.DataFrame([{
            "track_id": 1,
            "frame_index": 0,
            "is_valid": True,
            "aspect_ratio": np.nan,  # NaN value
            "circularity": np.inf,   # Inf value
            "eccentricity": None,    # None value
            "contour_area": 0,
            "bbox_area": 0,          # Division by zero risk
            "edge_density": "invalid_type",  # String type corruption
            "fitted_ellipse_angle": 90.0
        }])

        engineer = FeatureEngineer(window_size=5)
        df_clean = engineer.transform_raw_measurements(corrupted_data)

        self.assertFalse(df_clean.empty)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, df_clean.columns)
            val = df_clean[col].iloc[0]
            self.assertTrue(np.isfinite(val), f"Column {col} contains non-finite value: {val}")

    def test_validation_measurement_bounds(self) -> None:
        """Verifies validate_analysis_measurements flags invalid bounding box dimensions."""
        invalid_analysis = WheelRegionAnalysis(
            bbox_width=0,
            bbox_height=-10,
            aspect_ratio=-1.0
        )

        validated = validate_analysis_measurements(invalid_analysis)
        self.assertFalse(validated.is_valid)
        self.assertIn("Invalid bbox dimensions", validated.validation_notes)

    def test_execution_device_selection(self) -> None:
        """Verifies get_execution_device returns valid device string."""
        dev = get_execution_device()
        self.assertIn(dev, ["cpu", "cuda"])


if __name__ == "__main__":
    unittest.main()
