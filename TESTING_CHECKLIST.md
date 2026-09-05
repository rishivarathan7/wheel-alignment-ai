# Localhost Web Application Testing Checklist

Comprehensive testing checklist for the **AI-Based Real-Time Wheel Alignment Monitoring System** localhost web application (`http://localhost:8000`).

---

## Testing Results Summary

- **Total Test Cases**: 20 / 20 PASSED
- **Automated Test Suite**: `tests/test_web_app.py` (20 dedicated unit/integration tests)
- **Full Project Test Suite**: 124 / 124 PASSED (`Ran 124 tests in 51.782s OK`)
- **Existing Desktop GUI Compatibility**: Verified (`test_20_existing_desktop_gui`)

---

## 20-Point Testing Checklist

| # | Checklist Item | Test Description & Method | Status | Automated Test |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Backend startup** | Verify FastAPI app initializes with Uvicorn on `http://localhost:8000` with all routes mounted. | **PASSED** | `test_01_backend_startup` |
| **2** | **Homepage loading** | Verify `GET /` returns HTML dashboard template with video feed, controls, HUD, and telemetry table. | **PASSED** | `test_02_homepage_loading` |
| **3** | **Start button** | Verify `POST /start` and `POST /api/control/start` initialize camera acquisition worker thread. | **PASSED** | `test_03_start_button` |
| **4** | **Stop button** | Verify `POST /stop` and `POST /api/control/stop` terminate camera worker thread cleanly. | **PASSED** | `test_04_stop_button` |
| **5** | **Camera availability** | Verify unavailable or in-use camera sources return HTTP 400 with user-friendly error message (`"Camera unavailable"`). | **PASSED** | `test_05_camera_availability` |
| **6** | **Video streaming** | Verify `GET /video_feed` streams MJPEG frames (`multipart/x-mixed-replace`) with live bounding boxes. | **PASSED** | `test_06_video_streaming` |
| **7** | **Wheel detection** | Verify `YOLOWheelDetector` locates wheel ROI bounding boxes on video frames. | **PASSED** | `test_07_wheel_detection` |
| **8** | **Wheel tracking** | Verify `WheelTracker` assigns persistent track IDs across sequential frames. | **PASSED** | `test_08_wheel_tracking` |
| **9** | **Feature extraction** | Verify `WheelFeatureExtractor` extracts aspect ratio, ellipse fitting, and geometric proxies. | **PASSED** | `test_09_feature_extraction` |
| **10** | **Random Forest prediction**| Verify `RealTimePredictionEngine` runs Random Forest classifier on engineered 19-feature vectors. | **PASSED** | `test_10_random_forest_prediction` |
| **11** | **Camber output** | Verify telemetry record outputs precise camber proxy angle value (°). | **PASSED** | `test_11_camber_output` |
| **12** | **Toe output** | Verify telemetry record outputs precise toe proxy angle value (°). | **PASSED** | `test_12_toe_output` |
| **13** | **Alignment classification**| Verify alignment condition status (`NORMAL`, `POSSIBLE_MISALIGNMENT`, `SEVERE_MISALIGNMENT`). | **PASSED** | `test_13_alignment_classification` |
| **14** | **Confidence** | Verify prediction confidence probability is calculated (0.0 to 1.0 / 0% to 100%). | **PASSED** | `test_14_confidence` |
| **15** | **Alerts** | Verify `AlertSystem` generates timestamped alerts on persistent misalignment anomalies. | **PASSED** | `test_15_alerts` |
| **16** | **Status API** | Verify `GET /status` returns structured JSON telemetry payload without browser freezing. | **PASSED** | `test_16_status_api` |
| **17** | **Browser refresh** | Verify state remains synchronized across page refreshes or periodic status polling. | **PASSED** | `test_17_browser_refresh` |
| **18** | **Camera release** | Verify stopping stream releases OpenCV hardware camera resources immediately. | **PASSED** | `test_18_camera_release` |
| **19** | **Server shutdown** | Verify background worker thread terminates cleanly without resource leaks. | **PASSED** | `test_19_server_shutdown` |
| **20** | **Existing desktop GUI** | Verify desktop Tkinter GUI (`src/gui.py`) remains 100% functional and unimpacted. | **PASSED** | `test_20_existing_desktop_gui` |

---

## How to Run Automated Checklist Tests

Run the dedicated web application test suite:
```bash
python -m unittest tests/test_web_app.py
```

Run the entire project test suite (124 tests):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## Manual Step-by-Step Verification Procedure

1. **Launch Web Server**: Run `python src/main.py --web` in terminal.
2. **Open Dashboard**: Navigate to `http://localhost:8000` in Google Chrome or Microsoft Edge.
3. **Check Homepage**: Confirm title reads "AI Wheel Alignment Monitor", status is "IDLE", and camera select is visible.
4. **Test Start**: Select Camera 0 or enter sample video path, then click **Start Monitor**. Confirm stream starts.
5. **Verify AI Overlays**: Observe live video feed for bounding boxes, Wheel ID #, Camber (°), Toe (°), and status.
6. **Verify Telemetry Table**: Check that tracked wheel rows display Wheel ID, Status, Confidence, Camber, Toe, and Speed.
7. **Verify Event Log**: Confirm timestamped events appear in console log.
8. **Test Settings**: Click **Settings** icon, adjust Camber/Toe limits, click **Save Settings**, verify validation.
9. **Test Stop**: Click **Stop Monitor**. Verify video stops, status changes to "IDLE", and camera is released.
10. **Test Desktop GUI**: Run `python src/main.py --gui` to confirm desktop Tkinter window still opens without errors.
