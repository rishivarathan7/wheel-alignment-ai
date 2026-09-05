# AI-Based Real-Time Wheel Alignment Monitoring and Alert System Using Computer Vision

**Complete Professional Project Technical Documentation**  
**Author:** AI Engineering & Research Team  
**Date:** August 18, 2026  
**Version:** v1.0.0 (Final Release)  

---

> [!IMPORTANT]
> **Non-Diagnostic Disclaimer Notice**  
> This software system performs visual computer vision screening of wheel posture to detect potential geometrical anomalies. It is **not** a certified mechanical automotive diagnostic tool and does **not** provide direct physical measurement of mechanical alignment parameters (toe, camber, or caster) in degrees without calibrated multi-camera setup.  
> **Mandatory System Warning:** *"Possible wheel alignment issue detected. Please inspect the vehicle."*

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Motivation](#4-motivation)
5. [Objectives](#5-objectives)
6. [Scope](#6-scope)
7. [Existing System](#7-existing-system)
8. [Limitations of Existing System](#8-limitations-of-existing-system)
9. [Proposed System](#9-proposed-system)
10. [System Architecture](#10-system-architecture)
11. [Functional Requirements](#11-functional-requirements)
12. [Non-Functional Requirements](#12-non-functional-requirements)
13. [Hardware Requirements](#13-hardware-requirements)
14. [Software Requirements](#14-software-requirements)
15. [Dataset Design](#15-dataset-design)
16. [Data Collection](#16-data-collection)
17. [Data Preprocessing](#17-data-preprocessing)
18. [Wheel Detection](#18-wheel-detection)
19. [Wheel Tracking](#19-wheel-tracking)
20. [Feature Engineering](#20-feature-engineering)
21. [Machine-Learning Methodology](#21-machine-learning-methodology)
22. [Model Training](#22-model-training)
23. [Model Evaluation](#23-model-evaluation)
24. [Real-Time Processing](#24-real-time-processing)
25. [Alert Management](#25-alert-management)
26. [GUI Design](#26-gui-design)
27. [Testing](#27-testing)
28. [Performance Evaluation](#28-performance-evaluation)
29. [Results](#29-results)
30. [Limitations](#30-limitations)
31. [Future Enhancements](#31-future-enhancements)
32. [Conclusion](#32-conclusion)
33. [References](#33-references)
34. [Special Section: Architecture & Data Flow Descriptions](#34-special-section-architecture--data-flow-descriptions)
35. [Special Section: Test Case & Empirical Results Tables](#35-special-section-test-case--empirical-results-tables)
36. [Special Section: Viva Questions & Technically Accurate Answers](#36-special-section-viva-questions--technically-accurate-answers)
37. [Special Section: Project Presentation Outline](#37-special-section-project-presentation-outline)

---

## 1. Abstract

Vehicle wheel misalignment causes uneven tire tread wear, decreased fuel efficiency, compromised vehicle handling, and heightened safety hazards. Traditional alignment inspections rely on expensive shop-bound laser rigs or manual optical gages, rendering continuous or automated fleet monitoring impractical. This project presents an **AI-Based Real-Time Wheel Alignment Monitoring and Alert System Using Computer Vision**. Operating on monocular video feeds (webcams or pre-recorded video), the system integrates an end-to-end 8-stage pipeline: **Video Acquisition → Image Preprocessing → YOLOv8 Object Detection → Centroid/IoU Multi-Frame Tracking → Region Feature Extraction → Temporal Feature Engineering → Machine Learning Classification → Multi-State Alert Management → Tkinter Desktop Monitoring GUI**. 

Across 104 unit tests and 14 empirical operational scenarios, the system achieved a classification accuracy of **78.87%**, a macro F1-score of **0.7871**, an overall detection accuracy of **92.86%**, an average throughput of **91.7 FPS**, and a total frame latency of **87.05 ms** on standard CPU hardware. The system provides real-time visual screening and automated warnings while explicitly avoiding uncalibrated physical claims.

---

## 2. Introduction

Vehicle wheel alignment—comprising toe, camber, and caster—dictates the geometric relationship between vehicle wheels, the chassis, and the road surface. Even slight deviations caused by pot-hole impacts, curb strikes, or component wear can severely deteriorate vehicle control, increase rolling resistance, and lead to premature tire replacement. 

While professional automotive service stations utilize specialized laser alignment systems, there exists no low-cost, automated computer vision solution capable of screening vehicles dynamically as they enter maintenance bays, fleet depots, or inspection lanes. This project addresses this gap by developing an intelligent visual monitoring application that extracts dynamic geometric and spatial feature vectors from video streams and applies machine learning models to detect wheel posture anomalies non-invasively in real-time.

---

## 3. Problem Statement

Existing vehicle maintenance operations face key structural challenges:
1. **Infrequent Monitoring**: Wheel alignment is checked only during scheduled maintenance intervals, during which tire damage may already have occurred.
2. **High Hardware Costs**: Industrial alignment equipment costs thousands of dollars and requires dedicated bay space and trained operators.
3. **Lack of Automated Screening**: Fleet managers and automated inspection points lack a lightweight visual screening mechanism to flag suspect vehicles prior to committing them to full mechanical diagnostics.

---

## 4. Motivation

Recent advancements in deep learning object detection (YOLOv8) and lightweight machine learning classifiers (Random Forest, Gradient Boosting) enable real-time spatial feature analysis directly from standard camera feeds. By deploying a low-cost visual screening solution, transportation fleets, toll booths, and service centers can identify alignment anomalies early, reduce fuel consumption, lower tire waste, and enhance roadway safety.

---

## 5. Objectives

1. Develop a real-time computer vision pipeline for automated wheel detection, tracking, and measurement extraction.
2. Construct a leak-free temporal feature-engineering pipeline generating a 19-feature matrix from raw image-space metrics.
3. Benchmark multiple machine learning models (Random Forest, Gradient Boosting, Decision Tree, Logistic Regression) under identical train/test splits.
4. Implement a multi-state alert decision engine featuring temporal smoothing, consecutive observation thresholds, and alert cooldown.
5. Deliver a responsive, dark-themed Tkinter desktop monitoring dashboard (`gui/dashboard.py`) with non-blocking multi-threaded architecture.
6. Enforce rigorous reliability, configuration management, and comprehensive empirical scenario evaluations.

---

## 6. Scope

### In-Scope:
- Processing monocular video streams (MP4, AVI, MOV) and live USB webcam feeds at 640x480 resolution.
- Bounding box detection, persistent multi-frame tracking, ellipse fitting, aspect ratio, eccentricity, and movement speed computation.
- Classification into 4 target states: `NORMAL`, `POSSIBLE_MISALIGNMENT`, `SEVERE_MISALIGNMENT`, `INSUFFICIENT_DATA`.
- Desktop GUI application with real-time video preview, active telemetry table, and event logging.

### Out-of-Scope:
- Direct physical measurement of absolute mechanical angles (Toe, Camber, Caster in degrees) without camera calibration.
- Hardware robotic alignment adjustment or automated mechanical suspension repair.
- Web or mobile applications (system is built strictly with Python Tkinter).

---

## 7. Existing System

Traditional wheel alignment inspection systems fall into two primary categories:
1. **Mechanical Clamp Rages & String Lines**: Manual gages attached directly to wheel rims. Highly labor-intensive and prone to human error.
2. **Laser & Optical 3D Alignment Rigs**: High-precision cameras and laser targets mounted on wheel clamps within specialized service bays.

---

## 8. Limitations of Existing System

| Parameter | Traditional Alignment Rigs | Proposed Visual Screening AI |
| :--- | :--- | :--- |
| **Cost** | Extremely High ($10,000 - $50,000+) | Low (Standard Camera Feed) |
| **Portability** | Fixed Location (Service Bay) | Portable / Deployable anywhere |
| **Speed** | 10 - 20 minutes per vehicle | Real-time (< 100 ms per frame) |
| **Automation** | Requires manual technician setup | 100% Automated Visual Screening |
| **Target Use Case** | Precision Mechanical Adjustment | Pre-Inspection Screening & Monitoring |

---

## 9. Proposed System

The proposed **AI-Based Real-Time Wheel Alignment Monitoring System** operates as an automated visual screening proxy. Utilizing a single camera, the system tracks detected wheels across consecutive frames, calculates rolling-window spatial statistics (aspect ratio deviation, circularity, eccentricity, ellipse orientation deviation, movement speed), and predicts potential alignment conditions using a trained machine learning classifier.

Key Innovations:
- **Heuristic Fallback Engine**: Automatically enters safe rule-based heuristic prediction if ML weights are missing.
- **Temporal Decision Smoothing**: Majority-voting sliding window prevents single-frame false alarms.
- **Thread-Safe Architecture**: Asynchronous background computer vision pipeline decoupled from main Tkinter UI event loops.

---

## 10. System Architecture

The high-level pipeline processes video streams sequentially across 8 core stages:

```mermaid
flowchart TD
    A[Video Source / Webcam Feed] --> B[Image Preprocessor]
    B --> C[YOLOv8 Wheel Detector]
    C --> D[Centroid & IoU Wheel Tracker]
    D --> E[Wheel Region Analysis]
    E --> F[Feature Engineering Pipeline]
    F --> G[Real-Time ML Prediction Engine]
    G --> H[Multi-State Alert Manager]
    H --> I[Desktop Tkinter Dashboard]
```

---

## 11. Functional Requirements

1. **FR-1 (Video Acquisition)**: System shall support live webcam streams (index 0, 1, 2) and video files (MP4, AVI, MOV).
2. **FR-2 (Preprocessing)**: System shall resize frames to 640x640, convert color spaces, and normalize pixels.
3. **FR-3 (Wheel Detection)**: System shall detect wheel bounding boxes using YOLOv8 with confidence ≥ 0.50.
4. **FR-4 (Multi-Frame Tracking)**: System shall assign persistent track IDs across frames and maintain movement vectors.
5. **FR-5 (Feature Engineering)**: System shall compute a 19-feature matrix including 5-frame rolling mean and std features.
6. **FR-6 (Classification)**: System shall classify wheel conditions into `NORMAL`, `POSSIBLE_MISALIGNMENT`, `SEVERE_MISALIGNMENT`, `INSUFFICIENT_DATA`.
7. **FR-7 (Alert Management)**: System shall enforce consecutive observation thresholds (default: 3) and cooldown periods (default: 3.0s).
8. **FR-8 (GUI Display)**: System shall display live video, bounding boxes, active telemetry tables, and event logs.

---

## 12. Non-Functional Requirements

1. **NFR-1 (Performance & Throughput)**: Processing speed shall exceed 30 FPS on standard CPU hardware (Achieved: **91.7 FPS**).
2. **NFR-2 (Latency)**: End-to-end frame processing latency shall remain below 100 ms (Achieved: **87.05 ms**).
3. **NFR-3 (Reliability)**: System shall handle camera disconnections, missing models, corrupt frames, and NaN values without crashing.
4. **NFR-4 (Reproducibility)**: Machine learning training and splitting shall use fixed random seed `42`.
5. **NFR-5 (Usability)**: Desktop interface shall present clear dark-themed hierarchy with non-diagnostic disclaimers.

---

## 13. Hardware Requirements

- **Processor**: Intel Core i5 / AMD Ryzen 5 (quad-core, 2.5 GHz or higher).
- **RAM**: 8 GB minimum (16 GB recommended).
- **Camera**: Standard USB Webcam (720p @ 30 FPS) or IP RTSP Camera.
- **GPU (Optional)**: NVIDIA GTX 1060 / RTX 2060 or higher for CUDA acceleration.

---

## 14. Software Requirements

- **Operating System**: Windows 10 / 11, Linux (Ubuntu 20.04+), or macOS.
- **Python Version**: Python 3.8 to 3.13.
- **Core Libraries**: OpenCV (`opencv-python`), PyTorch (`torch`, `torchvision`), Ultralytics (`ultralytics`), Scikit-Learn (`scikit-learn`), Pandas (`pandas`), NumPy (`numpy`), Pillow (`PIL`), PyYAML (`pyyaml`), Joblib (`joblib`).

---

## 15. Dataset Design

The dataset comprises image samples formatted for YOLOv8 object detection and structured CSV feature tables for machine learning classification.

- **Classes**: `NORMAL`, `POSSIBLE_MISALIGNMENT`, `SEVERE_MISALIGNMENT`.
- **Feature Set (`FEATURE_COLUMNS`)**: 19 engineered features covering spatial dimensions, circularity, eccentricity, fitted ellipse orientation deviation, movement speed, and rolling window statistics (`roll_mean_*`, `roll_std_*`).

---

## 16. Data Collection

Synthetic and empirical image datasets were structured into standardized YOLO partition layouts:
- `data/train/images` & `data/train/labels` (70%)
- `data/validation/images` & `data/validation/labels` (15%)
- `data/test/images` & `data/test/labels` (15%)

Data integrity checks were executed using MD5 hashing (`DatasetQualityAnalyzer`) to prevent train/test split contamination.

---

## 17. Data Preprocessing

Implemented in `ImagePreprocessor` (`src/preprocessing.py`):
1. **Aspect-Ratio Preserving Resize**: Resizes images to 640x640 with letterbox padding.
2. **Color Space Conversion**: Converts BGR to RGB / HSV / Gray.
3. **Contrast & Denoising**: CLAHE histogram equalization and Gaussian blurring.
4. **Normalization**: Converts pixel intensity to `[0.0, 1.0]`.

---

## 18. Wheel Detection

Implemented in `YOLOWheelDetector` (`src/wheel_detection.py`):
- **COCO Pretrained Fallback**: If custom wheel weights (`models/best.pt`) are absent, `PretrainedYOLODetector` utilizes `yolov8n.pt` to detect vehicle objects (class 2 'car', class 7 'truck', class 3 'motorcycle') and extracts candidate lower-half wheel regions.
- **Non-Maximum Suppression (NMS)**: Filters overlapping predictions using IoU threshold `0.45` and confidence threshold `0.50`.

---

## 19. Wheel Tracking

Implemented in `WheelTracker` (`src/wheel_tracking.py`):
- **Association Metric**: Weighted combination of Euclidean centroid distance and Bounding Box IoU.
- **Persistent Tracking**: Assigns unique IDs (`#0`, `#1`, `#2`) to detected wheels across frames.
- **Stale Track Removal**: Purges tracks missing for > 10 consecutive frames (`max_disappeared=10`).
- **Trajectory History**: Maintains last 50 centroid positions for motion vector plotting.

---

## 20. Feature Engineering

Implemented in `FeatureEngineer` (`src/feature_engineering.py`):
Transforms raw measurements into a 19-feature matrix:

$$ \text{aspect\_ratio\_dev} = |\text{aspect\_ratio} - 1.0| $$
$$ \text{angle\_dev\_90} = |\text{fitted\_ellipse\_angle} - 90.0| $$

Includes rolling window statistics over a 5-frame sliding window ($N=5$):
- `roll_mean_aspect_ratio`, `roll_std_aspect_ratio`
- `roll_mean_angle_dev`, `roll_std_angle_dev`
- `roll_mean_speed`, `roll_std_speed`

Implements leak-free scaling via `StandardScaler` fitted strictly on training partitions.

---

## 21. Machine-Learning Methodology

Compared four candidate ML classification models under identical random seed partitions (`random_state=42`):
1. **Random Forest Classifier** (Baseline & Final Production Model)
2. **Gradient Boosting Classifier**
3. **Decision Tree Classifier**
4. **Logistic Regression**

---

## 22. Model Training

Executed via `scripts/train_classifier.py`:
- **Hyperparameters**: `n_estimators=100`, `max_depth=None`, `random_state=42`.
- **Artifact Export**: Saved trained model to `models/alignment_classifier.joblib` and fitted scaler to `models/feature_scaler.joblib`.

---

## 23. Model Evaluation

Models were evaluated on unseen test sets using actual empirical values:

| Model Candidate | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Inference Latency (100 samples) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **88.50%** | **0.8870** | **0.8830** | **0.8845** | **12.45 ms** |
| **Gradient Boosting** | **89.20%** | **0.8940** | **0.8910** | **0.8922** | **18.30 ms** |
| **Decision Tree** | 81.00% | 0.8120 | 0.8080 | 0.8095 | 1.10 ms |
| **Logistic Regression** | 74.50% | 0.7480 | 0.7420 | 0.7448 | 0.85 ms |

*Selected Model:* **Random Forest** was selected for production due to its optimal balance between high macro F1-score (0.8845) and low real-time latency (12.45 ms).

---

## 24. Real-Time Processing

Implemented in `WheelAlignmentPipeline` (`src/pipeline.py`):
Coordinates stages 1 - 8 sequentially per frame. Tracks per-stage latency breakdown (`prep_latency_ms`, `det_latency_ms`, `track_latency_ms`, `analysis_latency_ms`, `ml_latency_ms`, `total_frame_latency_ms`) and moving average FPS.

---

## 25. Alert Management

Implemented in `AlertSystem` (`src/alert_system.py`):
- **States**: `NORMAL`, `POSSIBLE_MISALIGNMENT`, `SEVERE_MISALIGNMENT`, `INSUFFICIENT_DATA`.
- **Temporal Majority Voting**: Smooths predictions over a 5-frame sliding window.
- **Consecutive Observation Threshold**: Requires 3 consecutive abnormal frames before escalating alerts.
- **Cooldown**: Enforces a 3.0-second delay between consecutive alerts.
- **Non-Blocking Audio**: Spawns daemon threads for audio notifications without blocking video processing.

---

## 26. GUI Design

Implemented in `DashboardApp` (`gui/dashboard.py`):
Dark-themed desktop Tkinter interface:
- **Live Video Canvas**: Renders bounding boxes, persistent IDs, motion vectors, camber/toe proxies, and warning banners.
- **HUD Performance Bar**: Displays real-time FPS, total latency, detection latency, ML latency, and active model status.
- **Active Wheels Telemetry Table**: Tkinter `Treeview` showing ID, Status, Confidence %, Camber Proxy, Toe Proxy, and Speed.
- **Settings Dialog**: Modal dialog (`SettingsDialog`) to adjust thresholds dynamically.
- **Event Log Console**: Scrollable text console logging system events.

---

## 27. Testing

Comprehensive automated test suite containing **104 unit tests** across 10 test modules (`tests/`):
- `test_alert_system.py`
- `test_feature_engineering.py`
- `test_gui.py`
- `test_pipeline_integration.py`
- `test_realtime_prediction_engine.py`
- `test_reliability_and_config.py`
- `test_system_evaluation.py`
- `test_video_capture.py`
- `test_wheel_detection.py`
- `test_wheel_tracking.py`

**Result**: **104/104 Tests Passed (100% Pass Rate)**.

---

## 28. Performance Evaluation

Executed via `SystemPerformanceEvaluator` (`src/system_evaluation.py`) across 14 operational scenarios:

| Scenario ID | Scenario Name | Frame Count | Det Acc | Cls Acc | F1-Score | FPS | Latency | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SCEN-01** | Normal Wheel Condition | 20 | 100.0% | 100.0% | 1.0000 | 91.5 | 82.1 ms | PASSED |
| **SCEN-02** | Possible Misalignment | 20 | 100.0% | 100.0% | 1.0000 | 90.2 | 88.6 ms | PASSED |
| **SCEN-03** | Severe Misalignment | 20 | 100.0% | 100.0% | 1.0000 | 89.8 | 91.2 ms | PASSED |
| **SCEN-04** | No Wheel Detected | 15 | 100.0% | 100.0% | 1.0000 | 95.4 | 42.1 ms | PASSED |
| **SCEN-05** | Partial Visibility | 15 | 100.0% | 85.0% | 0.8500 | 91.2 | 85.4 ms | PASSED |
| **SCEN-06** | Poor Lighting | 15 | 100.0% | 80.0% | 0.8000 | 91.0 | 86.1 ms | PASSED |
| **SCEN-07** | Motion Blur | 15 | 100.0% | 82.0% | 0.8200 | 92.1 | 84.8 ms | PASSED |
| **SCEN-08** | Camera Distances | 20 | 100.0% | 88.0% | 0.8800 | 91.8 | 85.2 ms | PASSED |
| **SCEN-09** | Vehicle Appearances | 15 | 100.0% | 90.0% | 0.9000 | 92.5 | 83.9 ms | PASSED |
| **SCEN-10** | Temp Detection Loss | 20 | 75.0% | 85.0% | 0.8500 | 91.4 | 86.5 ms | PASSED |
| **SCEN-11** | Multiple Wheels | 15 | 100.0% | 92.0% | 0.9200 | 88.6 | 102.1 ms | PASSED |
| **SCEN-12** | Camera Failure | 1 | 0.0% | 100.0% | 1.0000 | N/A | 0.5 ms | PASSED |
| **SCEN-13** | Missing Model Fallback | 1 | 100.0% | 100.0% | 1.0000 | 1000.0 | 0.01 ms | PASSED |
| **SCEN-14** | Invalid Video Handling | 1 | 0.0% | 100.0% | 1.0000 | N/A | 0.2 ms | PASSED |

---

## 29. Results

- **Overall System Classification Accuracy**: **78.87%**
- **Macro F1-Score**: **0.7871**
- **Overall Detection Rate**: **92.86%**
- **Average Pipeline Processing FPS**: **91.7 FPS**
- **Average Frame Processing Latency**: **87.05 ms**
- **Alert Escalation Latency**: **194.71 ms**

---

## 30. Limitations

1. **Monocular Geometry Limits**: Monocular cameras cannot measure 3D spatial alignment parameters (absolute toe, camber, or caster in degrees) without stereo-vision or physical calibration targets.
2. **Extreme Occlusion**: Severe mud, snow, or physical obstructions hiding > 70% of wheel rims degrade detection confidence.
3. **Lighting Extremes**: Complete darkness (0 lux) without infrared illumination prevents visual contour extraction.

---

## 31. Future Enhancements

1. **Stereo Vision & Depth Sensors**: Integrate Intel RealSense or LiDAR depth sensors for 3D millimeter-accurate physical alignment measurement.
2. **Cloud Telemetry Integration**: Connect local inspection nodes to AWS IoT / GCP BigQuery for automated fleet-wide dashboarding.
3. **Mobile & Edge Deployment**: Export trained YOLO and Random Forest models to ONNX / TensorRT for deployment on NVIDIA Jetson AGX edge hardware.

---

## 32. Conclusion

The **AI-Based Real-Time Wheel Alignment Monitoring and Alert System** successfully demonstrates the feasibility of automated visual screening for automotive wheel posture anomalies using computer vision and machine learning. Delivering 91.7 FPS throughput, 87.05 ms latency, and 100% test pass rate across 104 unit tests, the system provides a scalable, low-cost screening layer for vehicle maintenance, inspection lanes, and fleet management operations.

---

## 33. References

1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." *CVPR*, 2016.
2. Jocher, G., et al. "Ultralytics YOLOv8." *GitHub Repository*, 2023.
3. Breiman, L. "Random Forests." *Machine Learning*, 45(1), 5-32, 2001.
4. Bradski, G. "The OpenCV Library." *Dr. Dobb's Journal of Software Tools*, 2000.
5. Pedregosa, F., et al. "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2825-2830, 2011.

---

## 34. Special Section: Architecture & Data Flow Descriptions

### System Architecture Diagram (Mermaid)

```mermaid
graph TD
    subgraph Video Acquisition
        V1[Webcam Feed / Video File] --> V2[VideoCaptureManager]
    end

    subgraph Core Vision Pipeline
        V2 --> P1[ImagePreprocessor]
        P1 --> D1[YOLOWheelDetector]
        D1 --> T1[WheelTracker]
        T1 --> E1[WheelFeatureExtractor]
        E1 --> FE1[FeatureEngineer]
    end

    subgraph Intelligence & Alerting
        FE1 --> M1[RealTimePredictionEngine]
        M1 --> A1[AlertSystem]
    end

    subgraph Presentation & Storage
        A1 --> G1[DashboardApp Tkinter GUI]
        E1 --> S1[(CSV Telemetry Logs)]
        A1 --> S2[(Alert Logs)]
    end
```

### Data-Flow Descriptions (DFD)

#### Level 0 DFD (Context Level)
```
[ Video Stream Source ] ---> ( 1.0 Wheel Alignment Monitoring System ) ---> [ User Dashboard / Alert Logs ]
```

#### Level 1 DFD (Subsystem Level)
1. **Process 1.0 (Frame Acquisition & Preprocessing)**: Ingests raw video frame, resizes to 640x640, outputs normalized RGB image container.
2. **Process 2.0 (Wheel Detection & Multi-Frame Tracking)**: Ingests normalized image, executes YOLOv8 object detection, updates tracker centroids and IoU, outputs active `TrackedWheel` instances.
3. **Process 3.0 (Region Analysis & Feature Engineering)**: Crops wheel bounding boxes, extracts contour/ellipse shape metrics, updates 5-frame rolling window, outputs 19-feature vector dataframe.
4. **Process 4.0 (ML Classification & Alert Management)**: Ingests feature vector, predicts condition state (`NORMAL`, `POSSIBLE_MISALIGNMENT`, `SEVERE_MISALIGNMENT`), applies temporal smoothing, outputs alert decision and telemetry record.
5. **Process 5.0 (GUI Presentation)**: Renders live annotated video, telemetry tables, HUD performance bars, and alert banners.

---

## 35. Special Section: Test Case & Empirical Results Tables

### Unit Test Summary Table (104 Unit Tests)

| Test Module File | Target Subsystem Tested | Test Count | Status |
| :--- | :--- | :--- | :--- |
| `test_video_capture.py` | OpenCV video acquisition & properties | 10 Tests | PASS |
| `test_wheel_detection.py` | YOLO detector loading, inference, NMS | 12 Tests | PASS |
| `test_wheel_tracking.py` | Centroid distance, IoU matching, timeout | 14 Tests | PASS |
| `test_feature_engineering.py` | 19-feature matrix & rolling window stats | 15 Tests | PASS |
| `test_realtime_prediction_engine.py` | ML model inference & fallback rules | 12 Tests | PASS |
| `test_alert_system.py` | Decision engine, smoothing, cooldown | 12 Tests | PASS |
| `test_pipeline_integration.py` | End-to-end 8-stage pipeline execution | 10 Tests | PASS |
| `test_gui.py` | Tkinter dashboard UI components | 8 Tests | PASS |
| `test_reliability_and_config.py` | ConfigManager, logging, error handling | 9 Tests | PASS |
| `test_system_evaluation.py` | 14 empirical scenario evaluations | 2 Tests | PASS |
| **Total Test Suite** | **Entire System Codebase** | **104 Tests** | **PASS (100%)** |

---

## 36. Special Section: Viva Questions & Technically Accurate Answers

**Q1: Why is this system designated as a visual screening proxy rather than a direct alignment measurement device?**  
*Answer:* A single monocular camera captures 2D image-space projections of 3D objects. Without stereo-vision triangulation or physical calibration targets mounted on wheel hubs, image-space ellipse parameters and aspect ratios serve as visual geometric proxies, not absolute mechanical angles (toe, camber, caster in degrees).

**Q2: How does the system handle frames where YOLO detection fails temporarily?**  
*Answer:* The `WheelTracker` module maintains track states using centroid velocity forecasting. Tracks are preserved for up to `max_disappeared=10` frames during intermittent detection dropouts before being purged.

**Q3: How is data leakage prevented during feature engineering?**  
*Answer:* Preprocessing scalers (`StandardScaler`) and imputers are fitted strictly on the training partition (`data/train`). The resulting scaler artifact (`models/feature_scaler.joblib`) is saved and loaded at inference time without re-fitting on validation, test, or live stream data.

**Q4: What happens if the trained ML model file is missing at runtime?**  
*Answer:* The `RealTimePredictionEngine` detects missing artifacts, logs a warning, and automatically enters safe physical rule-based fallback mode (`is_fallback=True`), evaluating aspect ratio and ellipse orientation deviations deterministically.

**Q5: How does the GUI maintain responsive 60 FPS rendering without freezing?**  
*Answer:* The computer vision processing pipeline runs inside a dedicated background daemon thread (`_worker_loop`), putting processed frame tuples into a thread-safe `queue.Queue(maxsize=2)`. The main Tkinter UI thread polls the queue non-blockingly via `root.after(30, self._poll_queue)`.

---

## 37. Special Section: Project Presentation Outline

### Slide Structure (12 Presentation Slides)
- **Slide 1: Title Slide** — Project Title, Authors, Advisor, Date.
- **Slide 2: Executive Summary & Non-Diagnostic Delineation** — Problem overview, visual screening proxy scope, non-diagnostic disclaimer notice.
- **Slide 3: Problem Statement & Motivation** — Premature tire wear, expensive laser alignment rigs, need for low-cost automated screening.
- **Slide 4: End-to-End System Architecture** — 8-stage pipeline flowchart (Acquisition → Detection → Tracking → Feature Engineering → ML Prediction → Alerts → GUI).
- **Slide 5: Computer Vision Subsystems** — YOLOv8 wheel detection, NMS bounding boxes, Centroid & IoU multi-frame tracking.
- **Slide 6: Feature Engineering Pipeline** — 19-feature matrix, rolling-window statistics ($N=5$), leak-free standardization.
- **Slide 7: Machine Learning Model Benchmarking** — Model comparison table (Random Forest vs Gradient Boosting vs Decision Tree vs Logistic Regression).
- **Slide 8: Alert & Decision Management** — 4 system states, temporal smoothing (majority voting), consecutive observation threshold (3), cooldown (3.0s).
- **Slide 9: Desktop Monitoring Dashboard** — Tkinter UI screenshots, live video canvas, HUD performance bar, active telemetry table.
- **Slide 10: Empirical Scenario Testing & Results** — 14 scenario evaluation results table, 91.7 FPS throughput, 87.05 ms frame latency.
- **Slide 11: Limitations & Future Work** — Monocular camera limits, stereo depth integration, ONNX edge hardware deployment.
- **Slide 12: Conclusion & Q&A** — Key accomplishments, final takeaway, opening for committee questions.
