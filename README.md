# AI-Based Real-Time Wheel Alignment Monitoring and Alert System Using Computer Vision

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Tests-124%2F124%20Passing-brightgreen.svg)](tests/)
[![FPS Throughput](https://img.shields.io/badge/Throughput-91.7%20FPS-orange.svg)](results/)

An automated computer vision screening application for detecting vehicle wheel posture anomalies in real-time. Built with **YOLOv8**, **OpenCV**, **Scikit-Learn**, **PyTorch**, **FastAPI**, **HTML5/CSS3/JavaScript**, and a dark-themed **Tkinter** desktop GUI dashboard.

Full technical documentation: [TECHNICAL_DOCUMENTATION.md](file:///c:/Users/deva%20dharshihi/OneDrive/Desktop/CAR%20PTOJ/wheel_alignment_ai/TECHNICAL_DOCUMENTATION.md)  
Testing checklist: [TESTING_CHECKLIST.md](file:///c:/Users/deva%20dharshihi/OneDrive/Desktop/CAR%20PTOJ/wheel_alignment_ai/TESTING_CHECKLIST.md)

---

> [!IMPORTANT]
> **Non-Diagnostic Disclaimer Notice**  
> This software system performs visual screening of wheel posture to detect potential geometrical anomalies. It is **not** a certified mechanical automotive diagnostic tool and does **not** provide direct physical measurement of mechanical alignment parameters (toe, camber, or caster) in degrees without calibrated multi-camera setup.  
> **Mandatory System Warning:** *"Possible wheel alignment issue detected. Please inspect the vehicle."*

---

## 🚀 Quick Start & Launch Commands (Windows)

### 1. Installation & Environment Setup
```cmd
pip install -r requirements.txt
python scripts/validate_environment.py
```

---

### 2. Launch Localhost Web Application (Primary Method)
Run Uvicorn with auto-reload from the project root:
```cmd
python -m uvicorn src.web_app:app --reload
```
Alternatively, launch via `main.py`:
```cmd
python src/main.py --web
```

Open your web browser at:
- 🌐 **Primary Web Address**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 🌐 **Alternative Web Address**: [http://localhost:8000](http://localhost:8000)

---

### 3. Launch Desktop GUI Dashboard (Tkinter)
To run the native Tkinter desktop interface:
```cmd
python src/main.py --gui
```

---

### 4. Run Automated Test Suite
To run all 124 unit and web integration tests:
```cmd
python -m unittest discover -s tests -p "test_*.py"
```

To run the web application checklist tests specifically:
```cmd
python -m unittest tests/test_web_app.py
```

---

## Key System Features

- **8-Stage Real-Time Pipeline**: Video Acquisition → Image Preprocessing → YOLOv8 Detection → Centroid/IoU Tracking → Region Feature Extraction → 19-Feature Temporal Engineering → Random Forest ML Classification → Alert Management.
- **Dual User Interfaces**:
  - **Browser-Based Localhost Web Dashboard** (FastAPI + HTML5/CSS3/Vanilla JS + MJPEG Stream + Polling/WebSocket Telemetry + Settings Modal + Event Log Console).
  - **Desktop Tkinter Dashboard** (`src/gui.py` / `gui/dashboard.py`).
- **Real-Time Visual Overlays**: Bounding boxes, Wheel ID #, Camber proxy angle (°), Toe proxy angle (°), Alignment Status, and Motion Speed (px/s).
- **Production-Quality Error Handling**:
  - Catches camera unavailable / camera already in use, missing model files, invalid frames, 0 wheels detected, and server errors without crashing.
  - Friendly browser messages (`"Camera unavailable"`, `"AI model not found"`, `"No wheel detected"`, `"Waiting for valid frame"`).
  - Complete python stack traces logged to backend logs (`logs/web_app.log`) and terminal.
- **Leak-Free Preprocessing**: `StandardScaler` pipeline fitted strictly on training splits.
- **Temporal Decision Smoothing**: Sliding window majority voting and consecutive observation thresholds prevent single-frame false alerts.

---

## Repository Directory Structure

```
wheel_alignment_ai/
├── config/
│   └── settings.yaml                      # Configuration settings
├── data/                                  # Datasets & splits
├── gui/
│   └── dashboard.py                       # Tkinter GUI monitoring application
├── logs/                                  # Application, alert, and error log files
├── models/                                # YOLO and Scikit-Learn model artifacts
├── results/                               # Benchmark CSV and Markdown reports
├── scripts/                               # CLI execution scripts
├── src/                                   # Core computer vision & ML pipeline modules
│   ├── web_app.py                         # FastAPI Web application backend server
│   ├── pipeline.py                        # End-to-end AI monitoring pipeline
│   ├── video_capture.py                   # Video acquisition manager
│   ├── wheel_detection.py                 # YOLOv8 wheel detector
│   ├── wheel_tracking.py                  # Multi-frame wheel tracker
│   ├── feature_extraction.py              # Region feature extractor
│   ├── feature_engineering.py             # 19-feature vector engineering
│   ├── prediction.py                      # Random Forest prediction engine
│   ├── alert_system.py                    # Escalated alert manager
│   ├── config.py                          # Config manager
│   ├── gui.py                             # Desktop GUI wrapper
│   └── main.py                            # Unified CLI entrypoint
├── tests/                                 # 124 unit & integration test modules
├── web/                                   # Localhost web dashboard templates & static assets
│   ├── templates/index.html               # Jinja2 HTML dashboard template
│   └── static/                            # CSS styles, JS controllers, images
├── README.md                              # Main documentation file
├── TECHNICAL_DOCUMENTATION.md             # Full 33-section technical report
├── TESTING_CHECKLIST.md                   # 20-point web application testing checklist
└── requirements.txt
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
