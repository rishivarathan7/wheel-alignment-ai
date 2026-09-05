"""
Wheel Region Analysis and Feature Extraction Subsystem.

Calculates reliable visual and geometric measurements for detected wheel regions:
- Bounding-box width, height, aspect ratio, center coordinates, and region area.
- Contour metrics: area, perimeter, circularity, and eccentricity.
- Edge density: ratio of Canny edge pixels to total crop pixels.
- Estimated rim orientation angle (via fitted ellipse) and temporal orientation changes across frames.
- Measurement validation, CSV telemetry export, and debug visualization overlays.

Design Constraint Notice:
-------------------------
Visual measurements (aspect ratio, fitted ellipse angle, eccentricity) serve as geometric proxies.
They are NOT directly labeled as physical automotive alignment angles (Toe, Camber, Caster) unless
geometrically calibrated and mapped by downstream classification models in `prediction.py`.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class WheelRegionAnalysis:
    """
    Data structure containing validated visual and temporal measurements for a single wheel region.
    """
    frame_index: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    track_id: int = -1
    detection_id: str = "det_none"

    # Bounding Box Metrics
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    bbox_width: int = 0
    bbox_height: int = 0
    aspect_ratio: float = 1.0
    center_x: int = 0
    center_y: int = 0
    bbox_area: int = 0

    # Contour & Rim Features
    contour_area: float = 0.0
    contour_perimeter: float = 0.0
    circularity: float = 0.0
    eccentricity: float = 0.0

    # Visual & Texture Density Metrics
    edge_density: float = 0.0
    mean_brightness: float = 0.0

    # Orientation Estimates (Geometric Proxies)
    fitted_ellipse_angle: float = 0.0  # Angle in degrees [0.0 - 180.0]
    orientation_valid: bool = False

    # Temporal Dynamics (Relative to previous frame)
    delta_orientation: float = 0.0
    delta_center_x: float = 0.0
    delta_center_y: float = 0.0
    movement_speed: float = 0.0

    # Quality & Validation Status
    is_valid: bool = True
    validation_notes: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        """Converts analysis dataclass to dictionary."""
        d = asdict(self)
        d["bbox"] = str(self.bbox)  # Convert tuple to string for clean CSV output
        return d


# Backward compatibility alias
AlignmentFeatures = WheelRegionAnalysis


def validate_analysis_measurements(analysis: WheelRegionAnalysis) -> WheelRegionAnalysis:
    """
    Validates visual measurements before storing.
    Verifies finite non-negative values, non-zero dimensions, and expected ranges.
    """
    notes = []
    is_valid = True

    # 1. Dimension checks
    if analysis.bbox_width <= 0 or analysis.bbox_height <= 0:
        notes.append("Invalid bbox dimensions (<= 0)")
        is_valid = False

    if analysis.aspect_ratio <= 0.0 or not math.isfinite(analysis.aspect_ratio):
        notes.append("Invalid aspect ratio")
        is_valid = False

    # 2. Density & Ratio range checks
    if not (0.0 <= analysis.edge_density <= 1.0):
        notes.append("Edge density outside [0.0, 1.0]")
        is_valid = False

    if not math.isfinite(analysis.circularity) or analysis.circularity < 0.0:
        notes.append("Invalid circularity")
        analysis.circularity = 0.0
    else:
        analysis.circularity = min(1.0, max(0.0, analysis.circularity))

    if not math.isfinite(analysis.fitted_ellipse_angle):
        notes.append("Invalid ellipse angle")
        analysis.fitted_ellipse_angle = 0.0
        analysis.orientation_valid = False

    analysis.is_valid = is_valid
    analysis.validation_notes = "; ".join(notes) if notes else "OK"
    return analysis


class WheelFeatureExtractor:
    """
    Subsystem for extracting, validating, and logging visual measurements from cropped wheel regions.
    """

    def __init__(self) -> None:
        self.prior_analyses: Dict[int, WheelRegionAnalysis] = {}

    def analyze_region(
        self,
        wheel_crop: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        track_id: int = -1,
        frame_index: int = 0,
        detection_id: str = "det_none",
        timestamp: Optional[str] = None
    ) -> WheelRegionAnalysis:
        """
        Executes comprehensive region analysis on a wheel ROI image crop.
        Calculates bounding box dimensions, aspect ratio, contour characteristics,
        edge density, fitted ellipse orientation, and temporal deltas.
        """
        now_str = timestamp or datetime.now().isoformat()
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = float(height) / float(width) if width > 0 else 1.0
        cx = x1 + width // 2
        cy = y1 + height // 2
        bbox_area = width * height

        analysis = WheelRegionAnalysis(
            frame_index=frame_index,
            timestamp=now_str,
            track_id=track_id,
            detection_id=detection_id,
            bbox=bbox,
            bbox_width=width,
            bbox_height=height,
            aspect_ratio=round(aspect_ratio, 4),
            center_x=cx,
            center_y=cy,
            bbox_area=bbox_area
        )

        # Handle empty or low-quality crop images gracefully
        if wheel_crop is None or wheel_crop.size == 0 or len(wheel_crop.shape) < 2:
            analysis.is_valid = False
            analysis.validation_notes = "Empty or missing crop image"
            return analysis

        try:
            # Grayscale conversion
            if len(wheel_crop.shape) == 3:
                gray = cv2.cvtColor(wheel_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = wheel_crop.copy()

            analysis.mean_brightness = round(float(np.mean(gray)), 2)

            # Canny edge density calculation
            edges = cv2.Canny(gray, 50, 150)
            edge_pixel_count = np.count_nonzero(edges)
            analysis.edge_density = round(float(edge_pixel_count) / float(gray.size), 4)

            # Thresholding for contour extraction
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                c_area = float(cv2.contourArea(largest_contour))
                c_perimeter = float(cv2.arcLength(largest_contour, True))

                analysis.contour_area = round(c_area, 2)
                analysis.contour_perimeter = round(c_perimeter, 2)

                # Circularity: 4 * pi * Area / Perimeter^2
                if c_perimeter > 0:
                    circ = (4.0 * math.pi * c_area) / (c_perimeter ** 2)
                    analysis.circularity = round(min(1.0, max(0.0, circ)), 4)

                # Fit ellipse for orientation and eccentricity estimation
                if len(largest_contour) >= 5:
                    (e_cx, e_cy), (ma, MA), angle = cv2.fitEllipse(largest_contour)
                    analysis.fitted_ellipse_angle = round(float(angle), 2)
                    analysis.orientation_valid = True

                    if MA > 0 and ma > 0:
                        a, b = max(ma, MA) / 2.0, min(ma, MA) / 2.0
                        ecc = math.sqrt(max(0.0, 1.0 - (b ** 2) / (a ** 2)))
                        analysis.eccentricity = round(ecc, 4)

            # Calculate temporal deltas if prior analysis exists for this track_id
            if track_id >= 0 and track_id in self.prior_analyses:
                prior = self.prior_analyses[track_id]
                analysis.delta_center_x = round(float(cx - prior.center_x), 2)
                analysis.delta_center_y = round(float(cy - prior.center_y), 2)
                analysis.movement_speed = round(
                    math.sqrt(analysis.delta_center_x ** 2 + analysis.delta_center_y ** 2), 2
                )

                if analysis.orientation_valid and prior.orientation_valid:
                    # Angular delta considering 180-degree symmetry of fitted ellipse
                    delta_angle = analysis.fitted_ellipse_angle - prior.fitted_ellipse_angle
                    if delta_angle > 90.0:
                        delta_angle -= 180.0
                    elif delta_angle < -90.0:
                        delta_angle += 180.0
                    analysis.delta_orientation = round(delta_angle, 2)

            # Store current analysis for temporal tracking
            if track_id >= 0:
                self.prior_analyses[track_id] = analysis

            # Validate measurements before returning
            return validate_analysis_measurements(analysis)

        except Exception as e:
            logger.error(f"Exception during wheel region analysis for track #{track_id}: {e}")
            analysis.is_valid = False
            analysis.validation_notes = f"Analysis exception: {e}"
            return analysis

    # Backward compatibility alias
    def extract_features(
        self,
        wheel_crop: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        track_id: int = -1
    ) -> WheelRegionAnalysis:
        box = bbox if bbox is not None else (0, 0, wheel_crop.shape[1], wheel_crop.shape[0]) if wheel_crop is not None else (0, 0, 1, 1)
        return self.analyze_region(wheel_crop, bbox=box, track_id=track_id)

    def draw_analysis_visualizations(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        analysis: WheelRegionAnalysis
    ) -> np.ndarray:
        """
        Renders visual measurement debug overlays on frame:
        - Bounding box and center marker.
        - Fitted orientation axis vector line.
        - Telemetry text panel (aspect ratio, circularity, edge density, orientation angle).
        """
        if image is None or image.size == 0:
            return image

        output = image.copy()
        x1, y1, x2, y2 = bbox
        cx, cy = analysis.center_x, analysis.center_y

        # 1. Bounding box & Center marker
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

        # 2. Render Orientation Axis Line if valid
        if analysis.orientation_valid:
            angle_rad = math.radians(analysis.fitted_ellipse_angle)
            line_len = min(analysis.bbox_width, analysis.bbox_height) // 2
            end_x = int(cx + line_len * math.cos(angle_rad))
            end_y = int(cy + line_len * math.sin(angle_rad))
            cv2.line(output, (cx, cy), (end_x, end_y), (255, 255, 0), 2)

        # 3. Telemetry Overlay Box
        telemetry = (
            f"AR: {analysis.aspect_ratio:.2f} | "
            f"EdgeDens: {analysis.edge_density:.2f} | "
            f"Angle: {analysis.fitted_ellipse_angle:.1f}°"
        )
        cv2.putText(output, telemetry, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return output


class WheelTelemetryExporter:
    """
    Exports timestamped wheel region measurements to CSV.
    """

    def __init__(self, output_path: Optional[Union[str, Path]] = None) -> None:
        self.output_path = Path(output_path) if output_path else RESULTS_DIR / "wheel_telemetry.csv"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict[str, Any]] = []

    def log_analysis(self, analysis: WheelRegionAnalysis) -> None:
        """Appends a single analysis record to buffer."""
        self.records.append(analysis.to_dict())

    def export_to_csv(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """
        Writes accumulated telemetry records to CSV file using Pandas.
        """
        dest_file = Path(filepath) if filepath else self.output_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.records:
            logger.warning("No telemetry records to export.")
            df = pd.DataFrame()
            df.to_csv(dest_file, index=False)
            return dest_file

        df = pd.DataFrame(self.records)
        df.to_csv(dest_file, index=False)
        logger.info(f"Exported {len(self.records)} wheel region measurements to CSV: {dest_file}")
        return dest_file

    def clear(self) -> None:
        """Clears accumulated records buffer."""
        self.records.clear()
