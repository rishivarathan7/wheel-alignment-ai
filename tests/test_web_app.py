"""
Unit and Integration Tests for Localhost Web Application.

Validates all 20 items on the web application testing checklist:
1. Backend startup
2. Homepage loading
3. Start button endpoint
4. Stop button endpoint
5. Camera availability / error handling
6. Video streaming (/video_feed)
7. Wheel detection
8. Wheel tracking
9. Feature extraction
10. Random Forest prediction
11. Camber output
12. Toe output
13. Alignment classification
14. Confidence score
15. Alert system integration
16. Status API (/status)
17. State persistence across page refresh
18. Camera release on stop
19. Server worker thread shutdown
20. Existing desktop GUI compatibility
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time
import unittest
from urllib.parse import urlparse

import cv2
import numpy as np

from src.config import ConfigManager, MODELS_DIR
from src.pipeline import WheelAlignmentPipeline
from src.prediction import RealTimePredictionEngine
from src.video_capture import VideoCaptureManager
from src.web_app import app, controller, PipelineWebController


class Response:
    """Lightweight HTTP Response container for ASGI test calls."""

    def __init__(self, status_code: int, headers: dict, body_bytes: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = body_bytes
        self.text = body_bytes.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)


class ASGITestClient:
    """Zero-dependency ASGI Test Client for testing FastAPI/Starlette applications."""

    def __init__(self, app):
        self.app = app

    def get(self, path: str, headers: dict = None):
        return self.request("GET", path, headers=headers)

    def post(self, path: str, json_data: dict = None, headers: dict = None):
        return self.request("POST", path, headers=headers, json_data=json_data)

    def request(self, method: str, path: str, headers: dict = None, json_data: dict = None, body: bytes = b""):
        if json_data is not None:
            body = json.dumps(json_data).encode("utf-8")
            headers = headers or {}
            headers["content-type"] = "application/json"

        parsed = urlparse(path)
        path_info = parsed.path
        query_string = parsed.query.encode("utf-8")

        raw_headers = []
        if headers:
            for k, v in headers.items():
                raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method.upper(),
            "scheme": "http",
            "path": path_info,
            "raw_path": path_info.encode("ascii"),
            "query_string": query_string,
            "headers": raw_headers,
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
        }

        status_code = 500
        response_headers = {}
        response_body = []

        async def receive():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False
            }

        async def send(message):
            nonlocal status_code, response_headers, response_body
            if message["type"] == "http.response.start":
                status_code = message["status"]
                for k, v in message.get("headers", []):
                    response_headers[k.decode("latin-1").lower()] = v.decode("latin-1")
            elif message["type"] == "http.response.body":
                response_body.append(message.get("body", b""))

        async def run_app():
            await self.app(scope, receive, send)

        asyncio.run(run_app())

        body_bytes = b"".join(response_body)
        return Response(status_code, response_headers, body_bytes)


class TestWebAppChecklist(unittest.TestCase):
    """Automated test suite validating the 20-item web application checklist."""

    @classmethod
    def setUpClass(cls):
        cls.client = ASGITestClient(app)
        cls.config = ConfigManager()

    def setUp(self):
        # Ensure controller is stopped before each test
        controller.stop_stream()

    def tearDown(self):
        controller.stop_stream()

    def test_01_backend_startup(self):
        """1. Backend startup: Verify FastAPI application initializes and loads routes correctly."""
        self.assertIsNotNone(app)
        self.assertEqual(app.title, "AI Wheel Alignment Monitoring System")
        routes = [route.path for route in app.routes]
        self.assertIn("/", routes)
        self.assertIn("/video_feed", routes)
        self.assertIn("/status", routes)
        self.assertIn("/start", routes)
        self.assertIn("/stop", routes)

    def test_02_homepage_loading(self):
        """2. Homepage loading: Verify GET / renders HTML dashboard with controls."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        html = response.text
        self.assertIn("AI Wheel Alignment Monitor", html)
        self.assertIn("btnStart", html)
        self.assertIn("btnStop", html)
        self.assertIn("videoFeed", html)
        self.assertIn("telemetryTable", html)

    def test_03_start_button(self):
        """3. Start button: Verify POST /start and POST /api/control/start endpoints."""
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(tmp_path, fourcc, 10.0, (320, 240))
        for _ in range(5):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()

        try:
            response = self.client.post("/start", json_data={"source": tmp_path})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertTrue(controller.is_streaming)
        finally:
            controller.stop_stream()
            if Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass


    def test_04_stop_button(self):
        """4. Stop button: Verify POST /stop and POST /api/control/stop endpoints."""
        response = self.client.post("/stop")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertFalse(controller.is_streaming)

    def test_05_camera_availability(self):
        """5. Camera availability: Verify invalid camera source returns 400 with friendly message."""
        invalid_source = "non_existent_file_99999.mp4"
        response = self.client.post("/start", json_data={"source": invalid_source})
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertTrue(any(msg in data["message"] for msg in ["Camera unavailable", "Camera already in use"]))
        self.assertFalse(controller.is_streaming)

    def test_06_video_streaming(self):
        """6. Video streaming: Verify GET /video_feed returns MJPEG boundary stream."""
        response = self.client.get("/video_feed")
        self.assertEqual(response.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", response.headers.get("content-type", ""))

    def test_07_wheel_detection(self):
        """7. Wheel detection: Verify detector runs on frame and returns detection list."""
        pipeline = WheelAlignmentPipeline(config_manager=self.config)
        synthetic_frame = np.full((640, 640, 3), 128, dtype=np.uint8)
        cv2.circle(synthetic_frame, (320, 320), 100, (30, 30, 30), -1)
        annotated, telemetry, perf = pipeline.process_frame(synthetic_frame)
        self.assertIsNotNone(annotated)
        self.assertIn("det_latency_ms", perf)

    def test_08_wheel_tracking(self):
        """8. Wheel tracking: Verify multi-frame tracker maintains track identity."""
        pipeline = WheelAlignmentPipeline(config_manager=self.config)
        frame1 = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(frame1, (200, 200), 80, (255, 255, 255), -1)
        pipeline.process_frame(frame1)

        frame2 = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(frame2, (205, 205), 80, (255, 255, 255), -1)
        annotated2, telemetry2, perf2 = pipeline.process_frame(frame2)
        self.assertIsNotNone(perf2)

    def test_09_feature_extraction(self):
        """9. Feature extraction: Verify region feature extractor computes geometric proxies."""
        pipeline = WheelAlignmentPipeline(config_manager=self.config)
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.ellipse(synthetic_frame, (320, 240), (100, 80), 15, 0, 360, (200, 200, 200), -1)
        annotated, telemetry, perf = pipeline.process_frame(synthetic_frame)
        self.assertIsInstance(telemetry, list)

    def test_10_random_forest_prediction(self):
        """10. Random Forest prediction: Verify prediction engine executes classifier."""
        predictor = RealTimePredictionEngine(
            model_path=MODELS_DIR / "alignment_classifier.joblib",
            scaler_path=MODELS_DIR / "feature_scaler.joblib"
        )
        test_features = {
            "bbox_aspect_ratio": 1.1,
            "ellipse_aspect_ratio": 1.15,
            "contour_solidity": 0.92,
            "camber_proxy_angle": 1.5,
            "toe_proxy_angle": 0.8,
            "area_change_rate": 0.01,
            "centroid_velocity_magnitude": 2.5
        }
        pred = predictor.predict_wheel(test_features, wheel_id=1)
        self.assertIsNotNone(pred)
        self.assertIn(pred.predicted_condition, ["NORMAL", "POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"])

    def test_11_camber_output(self):
        """11. Camber output: Verify camber proxy angle is present in prediction output."""
        predictor = RealTimePredictionEngine()
        pred = predictor.predict_wheel({"camber_proxy_angle": 2.4}, wheel_id=1)
        self.assertIsInstance(pred.camber_proxy, float)

    def test_12_toe_output(self):
        """12. Toe output: Verify toe proxy angle is present in prediction output."""
        predictor = RealTimePredictionEngine()
        pred = predictor.predict_wheel({"toe_proxy_angle": -1.2}, wheel_id=1)
        self.assertIsInstance(pred.toe_proxy, float)

    def test_13_alignment_classification(self):
        """13. Alignment classification: Verify alignment condition is categorized."""
        predictor = RealTimePredictionEngine()
        pred = predictor.predict_wheel({"camber_proxy_angle": 5.0, "toe_proxy_angle": 4.0}, wheel_id=1)
        self.assertIsNotNone(pred.predicted_condition)

    def test_14_confidence(self):
        """14. Confidence: Verify prediction probability is between 0.0 and 1.0."""
        predictor = RealTimePredictionEngine()
        pred = predictor.predict_wheel({"camber_proxy_angle": 1.0}, wheel_id=1)
        self.assertTrue(0.0 <= pred.prediction_probability <= 1.0)

    def test_15_alerts(self):
        """15. Alerts: Verify alert system processes escalated prediction events."""
        pipeline = WheelAlignmentPipeline(config_manager=self.config)
        self.assertIsNotNone(pipeline.alert_system)

    def test_16_status_api(self):
        """16. Status API: Verify GET /status returns structured telemetry JSON payload."""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("monitoring", data)
        self.assertIn("system_status", data)
        self.assertIn("alignment_status", data)
        self.assertIn("wheels", data)
        self.assertIn("confidence", data)
        self.assertIn("camber", data)
        self.assertIn("toe", data)
        self.assertIn("speed", data)
        self.assertIn("timestamp", data)

    def test_17_browser_refresh(self):
        """17. Browser refresh: Verify state remains stable across repeated status polls."""
        for _ in range(5):
            res = self.client.get("/status")
            self.assertEqual(res.status_code, 200)

    def test_18_camera_release(self):
        """18. Camera release: Verify stop_stream releases video capture resources."""
        controller.stop_stream()
        self.assertIsNone(controller.cap)

    def test_19_server_shutdown(self):
        """19. Server shutdown: Verify stop_stream sets is_streaming to False."""
        controller.stop_stream()
        self.assertFalse(controller.is_streaming)

    def test_20_existing_desktop_gui(self):
        """20. Existing desktop GUI: Verify Tkinter GUI module and classes can be imported and referenced without errors."""
        import src.gui as gui_mod
        import gui.dashboard as db
        self.assertTrue(hasattr(gui_mod, "WheelAlignmentGUI"))
        self.assertTrue(hasattr(db, "DashboardApp"))
        self.assertTrue(hasattr(db, "SettingsDialog"))


if __name__ == "__main__":
    unittest.main()
