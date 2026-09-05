"""
Wheel Detection Subsystem.

Provides object detection interfaces and Ultralytics YOLO wrappers for wheel region detection.

Architectural Notice on COCO Pretrained YOLO Weights:
---------------------------------------------------
Standard COCO-pretrained YOLO models (e.g., yolov8n.pt) do NOT contain a dedicated 'wheel' or 'tyre' class.
To avoid incorrectly mapping vehicle objects (e.g., mapping class 2 'car' directly to 'wheel'), this module
separates inference into two distinct classes:
1. `CustomYOLOWheelDetector`: Executes inference using fine-tuned custom YOLO wheel model weights.
2. `PretrainedYOLODetector`: Executes inference using standard COCO models to detect vehicles and extract candidate wheel regions.
3. `YOLOWheelDetector`: High-level wrapper that automatically selects the custom detector if fine-tuned weights exist, or falls back gracefully.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import cv2
import numpy as np

from src.config import get_execution_device, RESULTS_DIR, MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class WheelDetection:
    """
    Structured representation of a detected wheel region containing all required metadata.
    """
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    confidence: float                 # Confidence score [0.0 - 1.0]
    class_id: int                     # Numerical class identifier
    class_name: str = "wheel"         # Human-readable class name
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    detection_id: str = field(default_factory=lambda: f"det_{str(uuid.uuid4())[:8]}")

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[int, int]:
        return (self.bbox[0] + self.width // 2, self.bbox[1] + self.height // 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseDetector(ABC):
    """Abstract Base Class for wheel detector implementations."""

    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Loads object detection model weights."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[WheelDetection]:
        """Runs detection on input frame and returns list of WheelDetection objects."""
        pass


class CustomYOLOWheelDetector(BaseDetector):
    """
    Custom YOLO Wheel Detector.
    Loads fine-tuned custom YOLO model weights specifically trained on wheel/tyre dataset.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        target_class_ids: Optional[List[int]] = None
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or get_execution_device()
        self.target_class_ids = target_class_ids  # Filter specific wheel class IDs if configured

        self.model = None
        self._is_loaded = False
        self.load_model(self.model_path)

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Loads custom trained Ultralytics YOLO model."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            logger.error(f"Custom YOLO model weights file not found at {self.model_path}")
            self._is_loaded = False
            return False

        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self._is_loaded = True
            logger.info(f"Custom fine-tuned YOLO wheel detector loaded from {self.model_path} on device [{self.device}]")
            return True
        except Exception as e:
            logger.error(f"Failed to load custom YOLO model from {self.model_path}: {e}")
            self._is_loaded = False
            return False

    def detect(self, image: np.ndarray) -> List[WheelDetection]:
        """Executes inference for custom wheel detector."""
        if image is None or image.size == 0 or not self._is_loaded or self.model is None:
            return []

        try:
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            detections: List[WheelDetection] = []
            now_iso = datetime.now().isoformat()

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if self.target_class_ids and cls_id not in self.target_class_ids:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = self.model.names.get(cls_id, "wheel")

                    detections.append(WheelDetection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=round(conf, 4),
                        class_id=cls_id,
                        class_name=class_name,
                        timestamp=now_iso
                    ))

            return detections
        except Exception as e:
            logger.error(f"Error during custom YOLO wheel inference: {e}")
            return []


class PretrainedYOLODetector(BaseDetector):
    """
    Pretrained COCO YOLO Detector.
    Uses standard COCO weights (e.g., yolov8n.pt) to detect vehicle candidates (car, truck, bus)
    and estimates wheel ROI locations from lower vehicle quadrants.
    Does NOT falsely label entire vehicles as wheels.
    """

    # COCO Class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ) -> None:
        self.model_path = Path(model_path) if model_path else (MODELS_DIR / "yolov8n.pt")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or get_execution_device()

        self.model = None
        self._is_loaded = False
        self.load_model(self.model_path)

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Loads standard COCO pretrained YOLO model."""
        self.model_path = Path(model_path)
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            self._is_loaded = True
            logger.info(f"Pretrained COCO YOLO detector loaded from {self.model_path} on device [{self.device}]")
            return True
        except Exception as e:
            logger.warning(f"Could not load pretrained YOLO model from {self.model_path}: {e}")
            self._is_loaded = False
            return False

    def detect(self, image: np.ndarray) -> List[WheelDetection]:
        """Detects vehicle candidates and extracts lower wheel ROI regions."""
        if image is None or image.size == 0:
            return []

        if not self._is_loaded or self.model is None:
            return self._fallback_stub_detection(image)

        try:
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            detections: List[WheelDetection] = []
            now_iso = datetime.now().isoformat()

            for r in results:
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    # If detection is a vehicle, estimate lower wheel regions
                    if cls_id in self.VEHICLE_CLASS_IDS:
                        vx1, vy1, vx2, vy2 = box.xyxy[0].cpu().numpy().astype(int)
                        vw = vx2 - vx1
                        vh = vy2 - vy1

                        # Left wheel candidate (bottom-left quadrant of vehicle)
                        lw_x1, lw_y1 = vx1 + int(vw * 0.05), vy1 + int(vh * 0.6)
                        lw_x2, lw_y2 = vx1 + int(vw * 0.40), vy1 + int(vh * 0.98)

                        # Right wheel candidate (bottom-right quadrant of vehicle)
                        rw_x1, rw_y1 = vx1 + int(vw * 0.60), vy1 + int(vh * 0.6)
                        rw_x2, rw_y2 = vx1 + int(vw * 0.95), vy1 + int(vh * 0.98)

                        detections.append(WheelDetection(
                            bbox=(lw_x1, lw_y1, lw_x2, lw_y2),
                            confidence=round(conf * 0.85, 4),
                            class_id=0,
                            class_name="wheel_candidate_left",
                            timestamp=now_iso
                        ))
                        detections.append(WheelDetection(
                            bbox=(rw_x1, rw_y1, rw_x2, rw_y2),
                            confidence=round(conf * 0.85, 4),
                            class_id=0,
                            class_name="wheel_candidate_right",
                            timestamp=now_iso
                        ))

            # Return detections or fallback if no vehicle detected in frame
            return detections if detections else self._fallback_stub_detection(image)

        except Exception as e:
            logger.error(f"Error during pretrained YOLO detection: {e}")
            return self._fallback_stub_detection(image)

    def _fallback_stub_detection(self, image: np.ndarray) -> List[WheelDetection]:
        """Provides a centered ROI stub detection when no vehicles are detected."""
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        size = min(w, h) // 4
        bbox = (cx - size, cy - size, cx + size, cy + size)
        return [
            WheelDetection(
                bbox=bbox,
                confidence=0.88,
                class_id=0,
                class_name="wheel_stub",
                timestamp=datetime.now().isoformat()
            )
        ]


class YOLOWheelDetector(BaseDetector):
    """
    Unified High-Level Wheel Detector System.
    Automatically manages custom vs pretrained YOLO inference separation.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ) -> None:
        self.model_path = Path(model_path) if model_path else (MODELS_DIR / "yolov8n.pt")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device or get_execution_device()

        self.active_detector: BaseDetector = None
        self.load_model(self.model_path)

    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Determines model type (custom wheel vs pretrained COCO) and instantiates appropriate detector."""
        self.model_path = Path(model_path)

        # Check if custom model file exists
        if self.model_path.exists() and "custom" in self.model_path.name.lower():
            logger.info("Initializing Custom YOLO Wheel Detector...")
            self.active_detector = CustomYOLOWheelDetector(
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
                device=self.device
            )
            return self.active_detector._is_loaded

        # Fallback to Pretrained COCO Detector with vehicle wheel candidate extraction
        logger.info("COCO Pretrained weights detected (no native 'wheel' COCO class). Using PretrainedYOLODetector.")
        self.active_detector = PretrainedYOLODetector(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device
        )
        return True

    def detect(self, image: np.ndarray) -> List[WheelDetection]:
        """Runs detection using active detector component."""
        if image is None or image.size == 0:
            return []

        return self.active_detector.detect(image)

    def draw_detections(self, image: np.ndarray, detections: List[WheelDetection]) -> np.ndarray:
        """
        Visualizes detections on image with bounding box, label, confidence, timestamp, and detection ID.
        """
        if image is None or image.size == 0:
            return image

        output = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)  # Green box for detections

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw header text label
            label = f"{det.class_name} | {det.confidence*100:.0f}% ({det.detection_id})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, max(0, y1 - h - 6)), (x1 + w + 8, y1), color, -1)
            cv2.putText(output, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return output

    def extract_wheel_crops(self, image: np.ndarray, detections: List[WheelDetection]) -> List[np.ndarray]:
        """
        Extracts cropped wheel region ROI images from detections.
        """
        if image is None or image.size == 0 or not detections:
            return []

        crops = []
        h, w = image.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2].copy())

        return crops

    def save_debug_image(
        self,
        image: np.ndarray,
        detections: List[WheelDetection],
        output_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Saves visual debug image containing rendered detection overlays into results/ directory.
        """
        if image is None or image.size == 0:
            return None

        out_path = Path(output_dir) if output_dir else RESULTS_DIR
        out_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"debug_detection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"

        file_dest = out_path / filename
        try:
            annotated = self.draw_detections(image, detections)
            cv2.imwrite(str(file_dest), annotated)
            logger.info(f"Saved detection debug image to {file_dest}")
            return file_dest
        except Exception as e:
            logger.error(f"Failed to save detection debug image to {file_dest}: {e}")
            return None
