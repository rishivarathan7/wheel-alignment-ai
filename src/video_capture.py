"""
Video Capture Module.

Provides robust video stream / camera acquisition, frame management, resolution/FPS configuration,
frame skipping, live FPS measurement, timestamp overlay, and failure handling.
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Generator, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoCaptureManager:
    """
    Manages video input acquisition from USB webcams or local video files (MP4, AVI, MOV).
    Supports configurable resolution, target FPS, frame skipping, live FPS calculation, and timestamping.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        target_fps: Optional[float] = None,
        frame_skip: int = 0,
        add_timestamp: bool = False
    ) -> None:
        self.source = source
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = target_fps
        self.frame_skip = max(0, frame_skip)
        self.add_timestamp = add_timestamp

        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
        self._is_eof = False

        # Telemetry & Frame Tracking
        self.total_frames_read = 0
        self.total_frames_processed = 0
        self.start_time: Optional[float] = None
        self.last_frame_time: Optional[float] = None
        self.current_fps: float = 0.0
        self.fps_alpha: float = 0.1  # Exponential moving average factor for FPS calculation

        self._initialize_source()

    def _initialize_source(self) -> None:
        """Initializes OpenCV VideoCapture object with parameters and error checks."""
        try:
            # Resolve source format
            if isinstance(self.source, Path):
                source_path = self.source.resolve()
                if not source_path.exists():
                    logger.error(f"Initialization failure: Video file does not exist at {source_path}")
                    self._is_opened = False
                    return
                source_str = str(source_path)
            elif isinstance(self.source, str):
                if self.source.isdigit():
                    source_str = int(self.source)
                else:
                    path_obj = Path(self.source)
                    # If looks like a file path, check existence
                    if any(self.source.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"]):
                        if not path_obj.exists():
                            logger.error(f"Initialization failure: Video file not found at {self.source}")
                            self._is_opened = False
                            return
                    source_str = self.source
            else:
                source_str = int(self.source)

            # Open VideoCapture
            if isinstance(source_str, int) and getattr(cv2, "CAP_DSHOW", None) is not None:
                # Use CAP_DSHOW on Windows for fast webcam probing if backend supported
                import sys
                backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(source_str, backend)
            else:
                self.cap = cv2.VideoCapture(source_str)

            if not self.cap or not self.cap.isOpened():
                logger.error(f"Initialization failure: Unable to open video source '{self.source}'")
                self._is_opened = False
                return

            self._is_opened = True
            logger.info(f"Successfully initialized video capture source: {self.source}")

            # Apply requested camera properties if supported
            if self.requested_width and self.requested_width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            if self.requested_height and self.requested_height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            if self.requested_fps and self.requested_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

            # Log active configuration properties
            actual_props = self.get_properties()
            logger.info(
                f"Capture active settings: {actual_props['width']}x{actual_props['height']} "
                f"@ {actual_props['fps']:.1f} FPS (Source: {self.source})"
            )

        except Exception as e:
            self._is_opened = False
            logger.error(f"Exception during video capture initialization for source '{self.source}': {e}")

    @property
    def is_opened(self) -> bool:
        """Returns whether video capture device/file is opened."""
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    @property
    def is_eof(self) -> bool:
        """Returns True if end of video file has been reached."""
        return self._is_eof

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads a single frame from the video source, applying optional frame skipping,
        live FPS calculation, and optional timestamp overlay.

        Returns:
            (success: bool, frame: Optional[np.ndarray])
        """
        if not self.is_opened:
            return False, None

        try:
            # Handle frame skipping
            skip_count = self.frame_skip
            while skip_count > 0:
                ret, _ = self.cap.read()
                self.total_frames_read += 1
                if not ret:
                    self._is_eof = True
                    logger.info("End of video stream reached during frame skip.")
                    return False, None
                skip_count -= 1

            # Read target frame
            ret, frame = self.cap.read()
            self.total_frames_read += 1

            if not ret or frame is None:
                self._is_eof = True
                logger.info("End of video stream reached or frame read failed.")
                return False, None

            self.total_frames_processed += 1
            now = time.time()

            # Live FPS measurement (moving average)
            if self.start_time is None:
                self.start_time = now
            if self.last_frame_time is not None:
                delta = now - self.last_frame_time
                if delta > 0:
                    instant_fps = 1.0 / delta
                    if self.current_fps == 0.0:
                        self.current_fps = instant_fps
                    else:
                        self.current_fps = (1 - self.fps_alpha) * self.current_fps + self.fps_alpha * instant_fps
            self.last_frame_time = now

            # Optional timestamp overlay
            if self.add_timestamp:
                frame = self.add_timestamp_overlay(frame)

            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame from source '{self.source}': {e}")
            return False, None

    def read_frame_metadata(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Reads frame along with metadata dict (timestamp, frame_index, fps).
        """
        success, frame = self.read_frame()
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "frame_index": self.total_frames_read,
            "processed_count": self.total_frames_processed,
            "fps": round(self.current_fps, 2)
        }
        return success, frame, metadata

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Generator to continuously yield frames until EOF or closed."""
        while self.is_opened and not self._is_eof:
            ret, frame = self.read_frame()
            if not ret or frame is None:
                break
            yield frame

    def add_timestamp_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draws current timestamp and live FPS overlay onto frame."""
        if frame is None:
            return frame
        output = frame.copy()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        fps_str = f"FPS: {self.current_fps:.1f}" if self.current_fps > 0 else "FPS: --"
        text = f"{timestamp_str} | {fps_str}"

        # Draw semi-transparent background box
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (5, 5), (w + 15, h + 15), (0, 0, 0), -1)
        cv2.putText(output, text, (10, h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return output

    def get_properties(self) -> Dict[str, Any]:
        """Returns current parameters and state properties."""
        if not self.cap or not self._is_opened:
            return {
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "frame_count": 0,
                "source": str(self.source),
                "is_opened": False,
                "is_eof": self._is_eof
            }

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "source": str(self.source),
            "is_opened": self.is_opened,
            "is_eof": self._is_eof,
            "current_fps": round(self.current_fps, 2),
            "total_read": self.total_frames_read,
            "total_processed": self.total_frames_processed
        }

    def get_current_fps(self) -> float:
        """Returns the measured live FPS."""
        return round(self.current_fps, 2)

    def release(self) -> None:
        """Releases video capture resources and resets state."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing video capture for source '{self.source}': {e}")
            self.cap = None
        self._is_opened = False
        logger.info(f"Video capture source '{self.source}' released cleanly.")

    def __enter__(self) -> "VideoCaptureManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
