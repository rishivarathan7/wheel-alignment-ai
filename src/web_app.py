"""
FastAPI Web Application Backend Server.

Provides browser-based localhost access to the AI Wheel Alignment Monitoring System:
- Real-time MJPEG video streaming endpoint (`/video_feed`).
- High-frequency WebSocket telemetry broadcast channel (`/ws/telemetry`).
- REST API control endpoints (`/api/control/start`, `/api/control/stop`, `/api/config`).
- Jinja2 HTML dashboard rendering (`/`).
"""

import asyncio
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Union

import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Package import path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import ConfigManager, setup_logger, LOGS_DIR, RESULTS_DIR
from src.pipeline import WheelAlignmentPipeline
from src.video_capture import VideoCaptureManager

logger = setup_logger(log_file=LOGS_DIR / "web_app.log")

# Instantiate FastAPI Application
app = FastAPI(
    title="AI Wheel Alignment Monitoring System",
    description="Browser-Based Localhost Monitoring Dashboard for Real-Time Wheel Alignment Anomaly System",
    version="1.0.0"
)

# Directories & Static Setup
WEB_DIR = BASE_DIR / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------
# Global Exception Handlers (Production Quality Error Masking)
# ---------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Handles HTTP exceptions cleanly without exposing Python stack traces."""
    logger.warning(f"HTTP {exc.status_code} Exception on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": str(exc.detail),
            "detail": str(exc.detail)
        }
    )


@app.exception_handler(Exception)
async def custom_global_exception_handler(request: Request, exc: Exception):
    """Catches all uncaught server exceptions, logs full traceback to backend, and returns clean error JSON."""
    logger.error(f"Unhandled Server Exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error. Please check backend log files.",
            "detail": "Server error"
        }
    )



class PipelineWebController:
    """
    Thread-safe controller managing background video acquisition worker,
    frame encoding for MJPEG stream, and telemetry broadcast buffers.
    Reuses existing VideoCaptureManager from src/video_capture.py for camera acquisition.
    """

    def __init__(self) -> None:
        self.config = ConfigManager()
        self.pipeline = WheelAlignmentPipeline(config_manager=self.config)

        self.video_source: Union[int, str] = self.config.get("video.source", 0)
        self.is_streaming = False
        self.worker_thread: Optional[threading.Thread] = None
        self.cap: Optional[VideoCaptureManager] = None

        # Latest frame & telemetry buffers
        self.latest_jpeg_frame: Optional[bytes] = None
        self.latest_telemetry: List[Dict[str, Any]] = []
        self.latest_perf_stats: Dict[str, Any] = {}

        # WebSocket connection manager & thread lock
        self.active_websockets: List[WebSocket] = []
        self._lock = threading.Lock()

    def start_stream(self, source: Union[int, str] = 0) -> bool:
        """
        Starts camera acquisition and AI monitoring worker thread safely.
        Prevents multiple simultaneous camera instances.
        """
        with self._lock:
            if self.is_streaming:
                self._stop_stream_unlocked()

            self.video_source = source
            self.config.set("video.source", source)

            # Re-initialize camera source via existing VideoCaptureManager module
            cap = VideoCaptureManager(source=source)
            if not cap.is_opened:
                cap.release()
                logger.error(f"Camera or video source '{source}' is unavailable or in use.")
                raise ValueError(f"Camera device or video source '{source}' is unavailable or in use by another application.")

            self.cap = cap
            self.pipeline = WheelAlignmentPipeline(config_manager=self.config)

            self.is_streaming = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info(f"Successfully started camera stream on source: {source}")
            return True

    def stop_stream(self) -> bool:
        """Stops monitoring worker thread and releases active camera instance cleanly."""
        with self._lock:
            return self._stop_stream_unlocked()

    def _stop_stream_unlocked(self) -> bool:
        """Internal unlocked stream cleanup helper."""
        self.is_streaming = False
        if self.pipeline:
            self.pipeline.stop()
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCaptureManager: {e}")
            self.cap = None

        self.latest_jpeg_frame = None
        logger.info("Stopped camera monitoring stream and released hardware resources.")
        return True

    def _worker_loop(self) -> None:
        """Background thread processing video frames into MJPEG bytes and telemetry data."""
        try:
            if not self.cap or not self.cap.is_opened:
                logger.error("Worker loop started with uninitialized camera source.")
                return

            for frame in self.cap.stream_frames():
                if not self.is_streaming:
                    break

                annotated, telemetry, perf = self.pipeline.process_frame(frame)

                # Encode frame to JPEG for MJPEG stream
                if annotated is not None and annotated.size > 0:
                    ret, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret:
                        with self._lock:
                            self.latest_jpeg_frame = buffer.tobytes()
                            self.latest_telemetry = telemetry
                            self.latest_perf_stats = perf

                time.sleep(0.02)  # ~50 FPS target cap

        except Exception as e:
            logger.error(f"Error in camera streaming worker thread: {e}", exc_info=True)
        finally:
            with self._lock:
                self.is_streaming = False
                if self.cap:
                    self.cap.release()
                    self.cap = None

    def get_latest_mjpeg_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg_frame

    def get_latest_telemetry_payload(self) -> Dict[str, Any]:
        with self._lock:
            recent_alerts = []
            if self.pipeline and hasattr(self.pipeline, "alert_system") and self.pipeline.alert_system:
                recent_alerts = self.pipeline.alert_system.get_recent_alert_events(20)
            return {
                "timestamp": datetime.now().isoformat(),
                "is_streaming": self.is_streaming,
                "video_source": self.video_source,
                "telemetry": self.latest_telemetry,
                "performance": self.latest_perf_stats,
                "alerts": recent_alerts
            }


# Global Controller Instance
controller = PipelineWebController()


# ---------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Renders main HTML web dashboard page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed")
async def video_feed():
    """Streaming HTTP MJPEG video feed endpoint."""
    async def mjpeg_generator():
        while True:
            frame_bytes = controller.get_latest_mjpeg_frame()
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            else:
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.02)  # ~50 FPS delivery limit

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint broadcasting real-time JSON telemetry to connected browsers."""
    await websocket.accept()
    controller.active_websockets.append(websocket)
    logger.info("Browser WebSocket client connected.")

    try:
        while True:
            payload = controller.get_latest_telemetry_payload()
            await websocket.send_json(payload)
            await asyncio.sleep(0.05)  # 20 Hz broadcast rate
    except WebSocketDisconnect:
        logger.info("Browser WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket communication error: {e}")
    finally:
        if websocket in controller.active_websockets:
            controller.active_websockets.remove(websocket)


class StartStreamRequest(BaseModel):
    source: Union[int, str] = 0


@app.post("/start")
@app.post("/api/control/start")
async def api_start_stream(req: Optional[StartStreamRequest] = None):
    """Starts monitoring video stream with production-quality error handling."""
    try:
        source = req.source if req else 0
        controller.start_stream(source=source)
        return {"status": "success", "message": f"Stream started on source {source}"}
    except ValueError as ve:
        err_msg = str(ve)
        logger.warning(f"Camera start failed on source '{req.source if req else 0}': {ve}")
        friendly = "Camera already in use" if "in use" in err_msg.lower() else "Camera unavailable"
        raise HTTPException(status_code=400, detail=friendly)
    except Exception as e:
        logger.error(f"Error starting stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Camera unavailable")


@app.post("/stop")
@app.post("/api/control/stop")
async def api_stop_stream():
    """Stops monitoring video stream."""
    try:
        controller.stop_stream()
        return {"status": "success", "message": "Stream stopped"}
    except Exception as e:
        logger.error(f"Error stopping stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to stop stream")


@app.get("/status")
@app.get("/api/status")
async def api_get_status():
    """Returns current system and pipeline execution status with structured AI telemetry and friendly messages."""
    try:
        payload = controller.get_latest_telemetry_payload()
        is_monitoring = bool(controller.is_streaming)
        raw_telemetry = payload.get("telemetry", []) or []

        system_status = "MONITORING" if is_monitoring else "IDLE"

        wheels: List[Dict[str, Any]] = []
        has_misalignment = False
        total_conf = 0.0
        total_camber = 0.0
        total_toe = 0.0
        total_speed = 0.0

        for rec in raw_telemetry:
            if not isinstance(rec, dict):
                continue

            wheel_id = rec.get("track_id", 0)
            status_val = str(rec.get("status", "NORMAL"))
            conf_val = float(rec.get("confidence", 0.0) or 0.0)
            camber_val = float(rec.get("camber_proxy", 0.0) or 0.0)
            toe_val = float(rec.get("toe_proxy", 0.0) or 0.0)

            motion_data = rec.get("motion", {}) or {}
            speed_val = float(motion_data.get("velocity_magnitude", 0.0) or 0.0)

            if "MISALIGNMENT" in status_val.upper() or status_val.upper() in ["SEVERE", "POSSIBLE", "MISALIGNED"]:
                has_misalignment = True

            wheels.append({
                "wheel_id": wheel_id,
                "track_id": wheel_id,
                "status": status_val,
                "confidence": round(conf_val, 4),
                "camber": round(camber_val, 2),
                "toe": round(toe_val, 2),
                "speed": round(speed_val, 2),
                "bbox": rec.get("bbox", [])
            })

            total_conf += conf_val
            total_camber += camber_val
            total_toe += toe_val
            total_speed += speed_val

        num_wheels = len(wheels)
        if num_wheels > 0:
            avg_conf = round(total_conf / num_wheels, 4)
            avg_camber = round(total_camber / num_wheels, 2)
            avg_toe = round(total_toe / num_wheels, 2)
            avg_speed = round(total_speed / num_wheels, 2)
        else:
            avg_conf = 0.0
            avg_camber = 0.0
            avg_toe = 0.0
            avg_speed = 0.0

        if not is_monitoring:
            alignment_status = "IDLE"
        elif has_misalignment:
            alignment_status = "MISALIGNED"
        else:
            alignment_status = "ALIGNED"

        timestamp_str = payload.get("timestamp") or datetime.now().isoformat()
        perf_data = payload.get("performance", {}) or {}

        # Determine AI model availability
        model_ready = False
        if controller.pipeline and hasattr(controller.pipeline, "predictor") and controller.pipeline.predictor:
            model_ready = getattr(controller.pipeline.predictor, "is_ready", False)

        model_status_msg = "Loaded" if model_ready else "AI model not found"
        perf_data["model_status"] = model_status_msg

        # Determine friendly status note
        status_note = ""
        if is_monitoring:
            if not model_ready:
                status_note = "AI model not found"
            elif perf_data.get("is_valid") is False:
                status_note = "Waiting for valid frame"
            elif num_wheels == 0:
                status_note = "No wheel detected"
            elif has_misalignment:
                status_note = "Wheel alignment issue detected"
            else:
                status_note = "System operating normally"
        else:
            status_note = "System idle"

        return {
            "status": "success",
            "monitoring": is_monitoring,
            "system_status": system_status,
            "alignment_status": alignment_status,
            "status_note": status_note,
            "model_status": model_status_msg,
            "wheels": wheels,
            "confidence": avg_conf,
            "camber": avg_camber,
            "toe": avg_toe,
            "speed": avg_speed,
            "timestamp": timestamp_str,
            # Backward compatibility fields
            "is_streaming": is_monitoring,
            "video_source": controller.video_source,
            "active_wheels": num_wheels,
            "performance": perf_data,
            "telemetry": raw_telemetry
        }

    except Exception as e:
        logger.error(f"Error retrieving status: {e}", exc_info=True)
        return {
            "status": "error",
            "monitoring": bool(getattr(controller, "is_streaming", False)),
            "system_status": "ERROR",
            "alignment_status": "ERROR",
            "wheels": [],
            "confidence": 0.0,
            "camber": 0.0,
            "toe": 0.0,
            "speed": 0.0,
            "timestamp": datetime.now().isoformat()
        }


ALLOWED_CONFIG_KEYS = {
    "video.source",
    "prediction.camber_angle_limit",
    "prediction.toe_angle_limit",
    "alert.min_confidence",
    "alert.cooldown_seconds",
    "alert.consecutive_threshold",
    "alert.enable_audio",
    "detection.confidence_threshold",
    "pipeline.debug_mode"
}


@app.get("/api/config")
async def api_get_config():
    """Returns safe pipeline configuration settings."""
    cfg = controller.config
    return {
        "video.source": cfg.get("video.source", 0),
        "prediction.camber_angle_limit": cfg.get("prediction.camber_angle_limit", 3.0),
        "prediction.toe_angle_limit": cfg.get("prediction.toe_angle_limit", 2.0),
        "detection.confidence_threshold": cfg.get("detection.confidence_threshold", 0.50),
        "alert.min_confidence": cfg.get("alert.min_confidence", 0.60),
        "alert.consecutive_threshold": cfg.get("alert.consecutive_threshold", 3),
        "alert.cooldown_seconds": cfg.get("alert.cooldown_seconds", 3.0),
        "alert.enable_audio": cfg.get("alert.enable_audio", False),
        "pipeline.debug_mode": cfg.get("pipeline.debug_mode", False)
    }


@app.post("/api/config")
async def api_update_config(data: Dict[str, Any] = Body(...)):
    """Updates pipeline configuration settings safely with strict validation."""
    try:
        cfg = controller.config
        validated_payload = {}

        for k, v in data.items():
            # Block unsafe modification of model paths or non-whitelisted keys
            if k not in ALLOWED_CONFIG_KEYS:
                logger.warning(f"Rejected unpermitted or unsafe configuration key attempt: {k}")
                continue

            # Validate numeric values
            if k == "prediction.camber_angle_limit":
                val = float(v)
                if not (0.1 <= val <= 30.0):
                    raise ValueError(f"Camber threshold must be between 0.1° and 30.0° (got {val})")
                validated_payload[k] = val

            elif k == "prediction.toe_angle_limit":
                val = float(v)
                if not (0.1 <= val <= 20.0):
                    raise ValueError(f"Toe threshold must be between 0.1° and 20.0° (got {val})")
                validated_payload[k] = val

            elif k == "alert.min_confidence":
                val = float(v)
                if not (0.05 <= val <= 1.0):
                    raise ValueError(f"Minimum confidence must be between 0.05 and 1.0 (got {val})")
                validated_payload[k] = val

            elif k == "alert.cooldown_seconds":
                val = float(v)
                if not (0.1 <= val <= 120.0):
                    raise ValueError(f"Alert cooldown must be between 0.1 and 120.0 seconds (got {val})")
                validated_payload[k] = val

            elif k == "alert.consecutive_threshold":
                val = int(v)
                if not (1 <= val <= 50):
                    raise ValueError(f"Consecutive threshold must be between 1 and 50 (got {val})")
                validated_payload[k] = val

            elif k == "detection.confidence_threshold":
                val = float(v)
                if not (0.05 <= val <= 0.95):
                    raise ValueError(f"Detection confidence must be between 0.05 and 0.95 (got {val})")
                validated_payload[k] = val

            elif k == "video.source":
                if isinstance(v, str) and v.isdigit():
                    validated_payload[k] = int(v)
                else:
                    validated_payload[k] = v

            elif k in ("alert.enable_audio", "pipeline.debug_mode"):
                validated_payload[k] = bool(v)

        # Apply validated settings to ConfigManager
        for k, v in validated_payload.items():
            cfg.set(k, v)

        cfg.save()

        # Sync updated config to active pipeline alert_system & predictor
        if controller.pipeline:
            if hasattr(controller.pipeline, "alert_system") and controller.pipeline.alert_system:
                controller.pipeline.alert_system.update_config(
                    cooldown_seconds=cfg.get("alert.cooldown_seconds"),
                    min_confidence=cfg.get("alert.min_confidence"),
                    consecutive_threshold=cfg.get("alert.consecutive_threshold"),
                    smoothing_window=cfg.get("alert.smoothing_window"),
                    enable_audio=cfg.get("alert.enable_audio")
                )
            if hasattr(controller.pipeline, "predictor") and controller.pipeline.predictor:
                controller.pipeline.predictor.camber_limit = float(cfg.get("prediction.camber_angle_limit", 3.0))
                controller.pipeline.predictor.toe_limit = float(cfg.get("prediction.toe_angle_limit", 2.0))

        logger.info(f"Updated and validated config via API: {validated_payload}")
        return {"status": "success", "message": "Configuration updated successfully", "updated": validated_payload}

    except ValueError as ve:
        logger.warning(f"Config validation failure: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error updating config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def run_web_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launches FastAPI application server via Uvicorn."""
    import uvicorn
    logger.info(f"Launching AI Wheel Alignment Web Dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_web_server()
