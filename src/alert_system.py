"""
Alert and Decision Management Subsystem.

Manages discrete alignment anomaly alert states:
1. NORMAL
2. POSSIBLE_MISALIGNMENT
3. SEVERE_MISALIGNMENT
4. INSUFFICIENT_DATA

Applies confidence thresholding, temporal prediction smoothing, consecutive abnormal frame requirements,
alert cooldown management, structured logging, non-blocking audio notifications, and visual overlays
with mandatory non-diagnostic disclaimer disclaimers.
"""

from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.prediction import RealTimePredictionResult, normalize_class_label

logger = logging.getLogger(__name__)

# Mandatory Standardized Warning and Disclaimer Statements
MANDATORY_WARNING_TEXT = "Possible wheel alignment issue detected. Please inspect the vehicle."
NON_DIAGNOSTIC_DISCLAIMER = "Visual screening proxy only. Not a certified mechanical automotive diagnosis."


@dataclass
class AlertEvent:
    """Structured record representing an escalated alert notification event."""
    timestamp: str
    track_id: int
    state: str  # "NORMAL", "POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT", "INSUFFICIENT_DATA"
    confidence: float
    consecutive_count: int
    camber_proxy: float = 0.0
    toe_proxy: float = 0.0
    message: str = MANDATORY_WARNING_TEXT
    disclaimer: str = NON_DIAGNOSTIC_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrackAlertState:
    """Internal temporal state tracker per wheel track ID."""
    track_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=5))
    consecutive_abnormal_count: int = 0
    current_state: str = "NORMAL"
    last_alert_time: float = 0.0
    is_alert_escalated: bool = False


class AlertSystem:
    """
    Alert and Decision Management Subsystem.
    Evaluates ML predictions, applies confidence filtering and temporal smoothing,
    enforces consecutive observation thresholds, manages cooldown windows, and
    renders visual/audio alert notifications.
    """

    COLOR_MAP = {
        "NORMAL": (0, 255, 0),                   # Green
        "POSSIBLE_MISALIGNMENT": (0, 255, 255),  # Yellow
        "SEVERE_MISALIGNMENT": (0, 0, 255),      # Red
        "INSUFFICIENT_DATA": (180, 180, 180),    # Gray
        "SUSPECT": (0, 255, 255),                # Legacy Yellow
        "MISALIGNED": (0, 0, 255)                # Legacy Red
    }

    def __init__(
        self,
        alert_log_path: Optional[Union[str, Path]] = None,
        cooldown_seconds: float = 3.0,
        min_confidence: float = 0.60,
        consecutive_threshold: int = 3,
        smoothing_window: int = 5,
        enable_audio: bool = False,
        audio_callback: Optional[Callable[[], None]] = None
    ) -> None:
        self.alert_log_path = Path(alert_log_path) if alert_log_path else None
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence
        self.consecutive_threshold = consecutive_threshold
        self.smoothing_window = smoothing_window
        self.enable_audio = enable_audio
        self.audio_callback = audio_callback

        # State collections
        self.track_states: Dict[int, TrackAlertState] = {}
        self.alert_history: List[AlertEvent] = []
        self.last_global_alert_time: float = 0.0

        if self.alert_log_path:
            self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_prediction(
        self,
        prediction: Any,
        track_id: int,
        timestamp: Optional[str] = None
    ) -> Tuple[str, float, bool, AlertEvent]:
        """
        Evaluates a prediction for a single wheel track identity.

        Applies:
        1. Confidence threshold check.
        2. Temporal smoothing over sliding window.
        3. Consecutive abnormal observation requirement.
        4. Cooldown management.

        Returns:
            Tuple of (effective_state, confidence, is_escalated, alert_event_or_none)
        """
        ts_now = timestamp or datetime.now().isoformat()
        now_sec = time.time()

        # Retrieve or initialize track state
        if track_id not in self.track_states:
            self.track_states[track_id] = TrackAlertState(
                track_id=track_id,
                history=deque(maxlen=self.smoothing_window)
            )
        t_state = self.track_states[track_id]

        # 1. Raw prediction extraction & confidence threshold check
        raw_status = getattr(prediction, "predicted_condition", getattr(prediction, "status", "NORMAL"))
        raw_status = normalize_class_label(raw_status)
        conf = float(getattr(prediction, "prediction_probability", getattr(prediction, "confidence", 0.90)))

        camber = float(getattr(prediction, "camber_proxy", getattr(prediction, "camber_angle", 0.0)))
        toe = float(getattr(prediction, "toe_proxy", getattr(prediction, "toe_angle", 0.0)))

        # Force state to INSUFFICIENT_DATA if confidence is below threshold
        if conf < self.min_confidence:
            effective_raw_state = "INSUFFICIENT_DATA"
        else:
            effective_raw_state = raw_status

        # 2. Temporal Smoothing over sliding window
        t_state.history.append({
            "state": effective_raw_state,
            "conf": conf,
            "camber": camber,
            "toe": toe
        })

        smoothed_state = self._compute_smoothed_state(t_state.history)

        # 3. Consecutive Abnormal Observations Requirement
        if smoothed_state in ["POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"]:
            t_state.consecutive_abnormal_count += 1
        else:
            t_state.consecutive_abnormal_count = 0

        # Escalation Decision
        is_escalated = (t_state.consecutive_abnormal_count >= self.consecutive_threshold)
        t_state.is_alert_escalated = is_escalated
        t_state.current_state = smoothed_state

        alert_event = None

        # 4. Alert Escalation and Cooldown Check
        if is_escalated:
            time_since_last = now_sec - t_state.last_alert_time
            if time_since_last >= self.cooldown_seconds:
                # Trigger Escalated Alert Event
                t_state.last_alert_time = now_sec
                self.last_global_alert_time = now_sec

                alert_event = AlertEvent(
                    timestamp=ts_now,
                    track_id=track_id,
                    state=smoothed_state,
                    confidence=round(conf, 4),
                    consecutive_count=t_state.consecutive_abnormal_count,
                    camber_proxy=camber,
                    toe_proxy=toe,
                    message=MANDATORY_WARNING_TEXT,
                    disclaimer=NON_DIAGNOSTIC_DISCLAIMER
                )
                self.alert_history.append(alert_event)

                # Log alert event to file
                self._log_alert_event(alert_event)

                # Non-blocking Audio Notification
                if self.enable_audio:
                    self._trigger_audio_alert()

        return smoothed_state, conf, is_escalated, alert_event

    def _compute_smoothed_state(self, history: deque) -> str:
        """Computes majority vote state over temporal history window."""
        if not history:
            return "INSUFFICIENT_DATA"

        states = [rec["state"] for rec in history]

        # Priority rules: SEVERE_MISALIGNMENT > POSSIBLE_MISALIGNMENT > NORMAL > INSUFFICIENT_DATA
        severe_count = states.count("SEVERE_MISALIGNMENT")
        possible_count = states.count("POSSIBLE_MISALIGNMENT")
        normal_count = states.count("NORMAL")
        insufficient_count = states.count("INSUFFICIENT_DATA")

        half = len(states) / 2.0
        if severe_count >= half:
            return "SEVERE_MISALIGNMENT"
        if (severe_count + possible_count) >= half and (severe_count + possible_count) > 0:
            return "POSSIBLE_MISALIGNMENT"
        if normal_count >= half:
            return "NORMAL"

        # Fallback to most frequent or INSUFFICIENT_DATA
        state_counts = {st: states.count(st) for st in set(states)}
        most_common = max(state_counts, key=state_counts.get)
        return most_common

    def process_alert(
        self,
        prediction: Any,
        track_id: int,
        frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Evaluates prediction, logs alerts if needed, and draws visual graphics onto frame.
        """
        state, conf, is_escalated, _ = self.evaluate_prediction(prediction, track_id)

        if frame is not None and bbox is not None:
            return self.draw_overlay(frame, bbox, track_id, prediction, effective_state=state, is_escalated=is_escalated)

        return frame

    def draw_overlay(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: Optional[int],
        prediction: Any,
        effective_state: Optional[str] = None,
        is_escalated: bool = False
    ) -> np.ndarray:
        """
        Draws bounding boxes, state badges, telemetry text, warning banners, and disclaimers.
        If a value is unavailable, displays '--' instead of inventing a value.
        """
        if frame is None or frame.size == 0:
            return frame

        output = frame.copy()
        h_frame, w_frame = output.shape[:2]

        x1, y1, x2, y2 = bbox

        # 1. Track ID string
        tid_str = str(track_id) if track_id is not None else "--"

        # 2. Alignment status string
        raw_status = effective_state or getattr(prediction, "predicted_condition", getattr(prediction, "status", None))
        if raw_status is not None and str(raw_status).strip():
            status = normalize_class_label(str(raw_status))
        else:
            status = "--"

        color = self.COLOR_MAP.get(status, (180, 180, 180))

        # 3. Detection / Prediction Confidence string
        conf_val = getattr(prediction, "prediction_probability", getattr(prediction, "confidence", None))
        if conf_val is not None and not (isinstance(conf_val, float) and np.isnan(conf_val)):
            conf_str = f"{float(conf_val) * 100:.0f}%"
        else:
            conf_str = "--"

        # 4. Camber proxy angle string
        camber_val = getattr(prediction, "camber_proxy", getattr(prediction, "camber_angle", None))
        if camber_val is not None and not (isinstance(camber_val, float) and np.isnan(camber_val)):
            camber_str = f"{float(camber_val):+0.1f}°"
        else:
            camber_str = "--"

        # 5. Toe proxy angle string
        toe_val = getattr(prediction, "toe_proxy", getattr(prediction, "toe_angle", None))
        if toe_val is not None and not (isinstance(toe_val, float) and np.isnan(toe_val)):
            toe_str = f"{float(toe_val):+0.1f}°"
        else:
            toe_str = "--"

        # Bounding box drawing (Thicker box if alert is escalated)
        box_thickness = 3 if is_escalated else 2
        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        # Header Badge Tag (Wheel ID | Alignment Status | Confidence)
        header_text = f"ID:{tid_str} | {status} ({conf_str})"
        (w, h), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(output, (x1, max(0, y1 - h - 10)), (x1 + w + 10, y1), color, -1)
        cv2.putText(output, header_text, (x1 + 5, max(h + 2, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Telemetry Subtext (Camber | Toe)
        telemetry_text = f"Camber: {camber_str} | Toe: {toe_str}"
        sub_y = min(h_frame - 5, y2 + 18)
        (tw, th), _ = cv2.getTextSize(telemetry_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(output, (x1, max(0, sub_y - th - 3)), (x1 + tw + 6, min(h_frame, sub_y + 4)), (0, 0, 0), -1)
        cv2.putText(output, telemetry_text, (x1 + 3, sub_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Mandatory Alert Banner if Escalated Anomaly
        if is_escalated or status in ["POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"]:
            output = self._render_warning_banner(output, status)

        return output

    def _render_warning_banner(self, frame: np.ndarray, status: str) -> np.ndarray:
        """
        Renders mandatory advisory warning text banner and non-diagnostic disclaimer.
        """
        h, w = frame.shape[:2]
        banner_h = 55
        banner_bg = np.zeros((banner_h, w, 3), dtype=np.uint8)

        color = self.COLOR_MAP.get(status, (0, 0, 255))
        cv2.rectangle(banner_bg, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.line(banner_bg, (0, 0), (w, 0), color, 3)

        # Warning Text Line
        warn_line = f"WARNING: {MANDATORY_WARNING_TEXT}"
        cv2.putText(banner_bg, warn_line, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        # Non-Diagnostic Disclaimer Line
        disc_line = f"NOTICE: {NON_DIAGNOSTIC_DISCLAIMER}"
        cv2.putText(banner_bg, disc_line, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

        # Blend banner at the bottom of the frame
        frame[h - banner_h:h, 0:w] = cv2.addWeighted(frame[h - banner_h:h, 0:w], 0.2, banner_bg, 0.8, 0)
        return frame

    def _log_alert_event(self, event: AlertEvent) -> None:
        """Appends structured alert log entry to file."""
        log_entry = (
            f"[{event.timestamp}] "
            f"TRACK_ID: {event.track_id} | STATE: {event.state} | "
            f"CONF: {event.confidence:.2f} | CONSECUTIVE: {event.consecutive_count} | "
            f"CAMBER: {event.camber_proxy}° | TOE: {event.toe_proxy}° | "
            f"MSG: '{event.message}' | "
            f"DISCLAIMER: '{event.disclaimer}'\n"
        )

        logger.warning(f"ALERT ESCALATED: Track #{event.track_id} -> {event.state} ({event.confidence * 100:.0f}%)")

        if self.alert_log_path:
            try:
                with open(self.alert_log_path, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except Exception as e:
                logger.error(f"Failed to write alert log to {self.alert_log_path}: {e}")

    def _trigger_audio_alert(self) -> None:
        """Triggers audio alert in a non-blocking daemon thread."""
        def play_sound() -> None:
            try:
                if self.audio_callback:
                    self.audio_callback()
                    return
                # Windows winsound fallback
                import winsound
                winsound.Beep(1200, 250)
            except Exception:
                # System bell fallback
                print("\a", end="", flush=True)

        t = threading.Thread(target=play_sound, daemon=True)
        t.start()

    def purge_inactive_tracks(self, active_track_ids: List[int]) -> None:
        """
        Purges stored temporal alert states for track IDs that are no longer active,
        preventing memory leaks.
        """
        active_set = set(active_track_ids)
        stored_ids = list(self.track_states.keys())
        for tid in stored_ids:
            if tid not in active_set:
                del self.track_states[tid]

    def clear_history(self) -> None:
        """Clears all track states and alert log history."""
        self.track_states.clear()
        self.alert_history.clear()

    def get_recent_alert_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most recent escalated alert events as a list of dictionaries."""
        return [evt.to_dict() for evt in self.alert_history[-limit:]]

    def update_config(
        self,
        cooldown_seconds: Optional[float] = None,
        min_confidence: Optional[float] = None,
        consecutive_threshold: Optional[int] = None,
        smoothing_window: Optional[int] = None,
        enable_audio: Optional[bool] = None
    ) -> None:
        """Dynamically updates alert evaluation thresholds and settings."""
        if cooldown_seconds is not None:
            self.cooldown_seconds = float(cooldown_seconds)
        if min_confidence is not None:
            self.min_confidence = float(min_confidence)
        if consecutive_threshold is not None:
            self.consecutive_threshold = int(consecutive_threshold)
        if smoothing_window is not None:
            self.smoothing_window = int(smoothing_window)
        if enable_audio is not None:
            self.enable_audio = bool(enable_audio)
        logger.info(
            f"Updated AlertSystem config: cooldown={self.cooldown_seconds}s, "
            f"min_conf={self.min_confidence}, consecutive={self.consecutive_threshold}, "
            f"smoothing={self.smoothing_window}, audio={self.enable_audio}"
        )
