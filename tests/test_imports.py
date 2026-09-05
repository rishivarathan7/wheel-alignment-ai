"""
Unit Tests for Architecture Import Integrity and Component Initializations.
"""

import importlib.util
from pathlib import Path
import sys
import unittest

# Add project root directory to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectImports(unittest.TestCase):
    """Verifies that all core components load and import cleanly."""

    def _can_import(self, module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is not None

    def test_import_config(self) -> None:
        if not self._can_import("yaml"):
            self.skipTest("PyYAML not installed in Python environment")
        from src.config import ConfigManager, BASE_DIR
        config = ConfigManager()
        self.assertTrue(BASE_DIR.exists())
        self.assertIsNotNone(config.get("app.name"))

    def test_import_video_capture(self) -> None:
        if not self._can_import("cv2"):
            self.skipTest("opencv-python (cv2) not installed in Python environment")
        from src.video_capture import VideoCaptureManager
        cap = VideoCaptureManager(source=0)
        self.assertIsNotNone(cap)
        cap.release()

    def test_import_preprocessing(self) -> None:
        if not (self._can_import("cv2") and self._can_import("numpy")):
            self.skipTest("cv2 or numpy not installed in Python environment")
        import numpy as np
        from src.preprocessing import ImagePreprocessor
        preprocessor = ImagePreprocessor()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        processed, scale, pad = preprocessor.resize(dummy_img, (50, 50))
        self.assertEqual(processed.shape, (50, 50, 3))

    def test_import_wheel_detection(self) -> None:
        if not self._can_import("numpy"):
            self.skipTest("numpy not installed in Python environment")
        import numpy as np
        from src.wheel_detection import YOLOWheelDetector, WheelDetection
        detector = YOLOWheelDetector()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(dummy_img)
        self.assertIsInstance(detections, list)

    def test_import_wheel_tracking(self) -> None:
        if not self._can_import("numpy"):
            self.skipTest("numpy not installed in Python environment")
        from src.wheel_tracking import WheelTracker, WheelDetection
        tracker = WheelTracker()
        det = WheelDetection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0)
        tracks = tracker.update([det])
        self.assertEqual(len(tracks), 1)

    def test_import_feature_extraction(self) -> None:
        if not (self._can_import("cv2") and self._can_import("numpy")):
            self.skipTest("cv2 or numpy not installed in Python environment")
        import numpy as np
        from src.feature_extraction import WheelFeatureExtractor
        extractor = WheelFeatureExtractor()
        dummy_crop = np.zeros((50, 50, 3), dtype=np.uint8)
        features = extractor.extract_features(dummy_crop, bbox=(10, 10, 60, 60))
        self.assertIsNotNone(features)
        self.assertEqual(features.bbox_width, 50)

    def test_import_prediction(self) -> None:
        if not self._can_import("joblib") or not self._can_import("numpy"):
            self.skipTest("joblib or numpy not installed in Python environment")
        from src.prediction import AlignmentPredictor
        from src.feature_extraction import WheelRegionAnalysis
        predictor = AlignmentPredictor()
        feats = WheelRegionAnalysis(
            aspect_ratio=1.0, fitted_ellipse_angle=90.0, circularity=0.8,
            eccentricity=0.2, bbox_width=50, bbox_height=50, bbox_area=2500, contour_perimeter=200
        )
        pred = predictor.predict(feats)
        self.assertIn(pred.status, ["NORMAL", "POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"])

    def test_import_alert_system(self) -> None:
        if not (self._can_import("cv2") and self._can_import("numpy")):
            self.skipTest("cv2 or numpy not installed in Python environment")
        from src.alert_system import AlertSystem
        alert_sys = AlertSystem()
        self.assertIsNotNone(alert_sys)

    def test_import_pipeline(self) -> None:
        if not (self._can_import("cv2") and self._can_import("numpy") and self._can_import("yaml")):
            self.skipTest("Dependencies (cv2, numpy, pyyaml) not installed in Python environment")
        import numpy as np
        from src.pipeline import WheelAlignmentPipeline
        pipeline = WheelAlignmentPipeline()
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        res = pipeline.process_frame(dummy_img)
        annotated, telemetry = res[0], res[1]
        self.assertEqual(annotated.shape, dummy_img.shape)

    def test_import_gui(self) -> None:
        if not (self._can_import("PIL") and self._can_import("cv2") and self._can_import("yaml")):
            self.skipTest("GUI dependencies (Pillow, cv2, yaml) not installed in Python environment")
        import gui.dashboard
        self.assertIsNotNone(gui.dashboard.DashboardApp)


if __name__ == "__main__":
    unittest.main()
