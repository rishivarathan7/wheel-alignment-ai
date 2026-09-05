"""
Multi-Frame Wheel Tracking Subsystem.

Tracks detected wheel instances across consecutive video frames to establish stable wheel identities,
maintain temporal motion trajectories, track bounding-box dimensions & confidence, handle temporary detection loss,
and compute image-space displacement and velocity features.

Architectural Design Constraint:
--------------------------------
Image-space motion trajectories and velocity vectors track pixel-space movement and vehicle dynamics across frames.
Mechanical wheel alignment (Camber and Toe deviations) is calculated separately from geometric/contour features
in `feature_extraction.py` and classified in `prediction.py`, avoiding invalid direct inference from motion alone.
"""

from dataclasses import dataclass, field, asdict
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.wheel_detection import WheelDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackedWheel:
    """
    Structure representing a persistent wheel track identity across multiple video frames.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    confidence: float
    class_name: str = "wheel"
    history_size: int = 50

    # Trajectory History & Timestamps
    history: List[Tuple[int, int]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # State Telemetry
    disappeared_count: int = 0
    age: int = 0  # Total frame count active

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(self.centroid)
        if not self.timestamps:
            self.timestamps.append(time.time())

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    def update(self, detection: WheelDetection, timestamp: Optional[float] = None) -> None:
        """Updates tracked wheel position, bbox, confidence, and trajectory history."""
        now = timestamp if timestamp is not None else time.time()
        self.bbox = detection.bbox
        self.centroid = detection.center
        self.confidence = detection.confidence
        self.class_name = detection.class_name

        self.history.append(self.centroid)
        self.timestamps.append(now)

        if len(self.history) > self.history_size:
            self.history.pop(0)
            self.timestamps.pop(0)

        self.disappeared_count = 0
        self.age += 1

    def calculate_motion_features(self) -> Dict[str, Any]:
        """
        Calculates image-space motion features: displacement (dx, dy), displacement magnitude,
        velocity vector (vx, vy), and velocity magnitude (pixels/sec or pixels/frame).
        """
        if len(self.history) < 2:
            return {
                "dx": 0.0,
                "dy": 0.0,
                "displacement_magnitude": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "velocity_magnitude": 0.0,
                "frame_count": len(self.history)
            }

        # Calculate displacement relative to previous centroid
        curr_cx, curr_cy = self.history[-1]
        prev_cx, prev_cy = self.history[-2]
        dx = float(curr_cx - prev_cx)
        dy = float(curr_cy - prev_cy)
        disp_mag = math.sqrt(dx ** 2 + dy ** 2)

        # Calculate velocity based on frame time delta
        dt = self.timestamps[-1] - self.timestamps[-2] if len(self.timestamps) >= 2 else 0.033
        if dt <= 0:
            dt = 0.033  # Default fallback ~30 FPS (33ms)

        vx = dx / dt
        vy = dy / dt
        vel_mag = disp_mag / dt

        return {
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "displacement_magnitude": round(disp_mag, 2),
            "vx": round(vx, 2),
            "vy": round(vy, 2),
            "velocity_magnitude": round(vel_mag, 2),
            "frame_count": len(self.history)
        }

    def to_dict(self) -> Dict[str, Any]:
        motion = self.calculate_motion_features()
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "width": self.width,
            "height": self.height,
            "confidence": round(self.confidence, 4),
            "class_name": self.class_name,
            "age": self.age,
            "disappeared_count": self.disappeared_count,
            "motion": motion
        }


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class WheelTracker:
    """
    Lightweight, real-time multi-wheel tracking engine.
    Matches detections across consecutive frames using combined centroid distance & IoU metrics,
    maintains persistent track identities, and manages stale track timeouts.
    """

    def __init__(
        self,
        max_disappeared: int = 10,
        distance_threshold: float = 80.0,
        iou_threshold: float = 0.3,
        history_size: int = 50
    ) -> None:
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.history_size = history_size

        self.next_track_id = 0
        self.tracks: Dict[int, TrackedWheel] = {}

    def update(
        self,
        detections: List[WheelDetection],
        timestamp: Optional[float] = None
    ) -> List[TrackedWheel]:
        """
        Updates active tracks with new frame wheel detections.
        Handles temporal matching, new track creation, and stale track cleanup.
        """
        now = timestamp if timestamp is not None else time.time()

        # If zero detections in current frame, increment disappearance for all existing tracks
        if not detections:
            deregister_ids = []
            for track_id, track in self.tracks.items():
                track.disappeared_count += 1
                if track.disappeared_count > self.max_disappeared:
                    deregister_ids.append(track_id)
            for track_id in deregister_ids:
                logger.info(f"Removing stale wheel track ID #{track_id} (timeout reached).")
                del self.tracks[track_id]
            return list(self.tracks.values())

        # If no active tracks exist, register all current detections
        if not self.tracks:
            for det in detections:
                self._register(det, timestamp=now)
            return list(self.tracks.values())

        # Build distance cost matrix between existing tracks and new detections
        track_ids = list(self.tracks.keys())
        track_centroids = np.array([self.tracks[tid].centroid for tid in track_ids])
        det_centroids = np.array([d.center for d in detections])

        # Centroid Euclidean distance matrix (num_tracks, num_dets)
        dist_matrix = np.linalg.norm(track_centroids[:, np.newaxis] - det_centroids[np.newaxis, :], axis=2)

        # Match tracks to detections greedily by minimum distance
        matched_track_indices = set()
        matched_det_indices = set()

        rows = dist_matrix.min(axis=1).argsort()
        for r in rows:
            c = dist_matrix[r].argmin()
            if r in matched_track_indices or c in matched_det_indices:
                continue

            tid = track_ids[r]
            det = detections[c]
            dist = dist_matrix[r, c]

            # Validate match via centroid distance or bounding box IoU
            iou = compute_iou(self.tracks[tid].bbox, det.bbox)
            if dist > self.distance_threshold and iou < self.iou_threshold:
                continue

            # Update matched track
            self.tracks[tid].update(det, timestamp=now)
            matched_track_indices.add(r)
            matched_det_indices.add(c)

        # Handle unmatched active tracks (increment disappearance counter)
        for r, tid in enumerate(track_ids):
            if r not in matched_track_indices:
                self.tracks[tid].disappeared_count += 1

        # Register unmatched new detections
        for c, det in enumerate(detections):
            if c not in matched_det_indices:
                self._register(det, timestamp=now)

        # Clean up stale tracks exceeding max_disappeared timeout
        deregister_ids = [
            tid for tid, track in self.tracks.items()
            if track.disappeared_count > self.max_disappeared
        ]
        for tid in deregister_ids:
            logger.info(f"Removing stale wheel track ID #{tid} (disappeared > {self.max_disappeared} frames).")
            del self.tracks[tid]

        return list(self.tracks.values())

    def _register(self, detection: WheelDetection, timestamp: Optional[float] = None) -> None:
        """Registers a new persistent wheel track."""
        now = timestamp if timestamp is not None else time.time()
        track = TrackedWheel(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            centroid=detection.center,
            confidence=detection.confidence,
            class_name=detection.class_name,
            history_size=self.history_size,
            history=[detection.center],
            timestamps=[now],
            age=1
        )
        self.tracks[self.next_track_id] = track
        logger.info(f"Registered new active wheel track ID #{self.next_track_id} at {detection.center}")
        self.next_track_id += 1

    def draw_trajectories(self, image: np.ndarray, tracks: Optional[List[TrackedWheel]] = None) -> np.ndarray:
        """
        Visualizes persistent track IDs, bounding boxes, motion vectors, and trailing centroid trajectory paths.
        """
        if image is None or image.size == 0:
            return image

        output = image.copy()
        active_tracks = tracks if tracks is not None else list(self.tracks.values())

        # Colors for distinct tracks
        colors = [
            (0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255),
            (255, 255, 0), (0, 165, 255), (147, 20, 255)
        ]

        for track in active_tracks:
            color = colors[track.track_id % len(colors)]
            x1, y1, x2, y2 = track.bbox

            # Draw current bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw trailing trajectory line
            if len(track.history) > 1:
                for i in range(1, len(track.history)):
                    pt1 = track.history[i - 1]
                    pt2 = track.history[i]
                    # Dynamic thickness fading for older points
                    thickness = int(np.sqrt(float(i) / len(track.history)) * 2.5) + 1
                    cv2.line(output, pt1, pt2, color, thickness)

            # Draw centroid point
            cv2.circle(output, track.centroid, 4, (0, 0, 255), -1)

            # Draw header tag label
            motion = track.calculate_motion_features()
            label = f"Track #{track.track_id} | Spd:{motion['velocity_magnitude']:.0f}px/s"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, max(0, y1 - h - 6)), (x1 + w + 8, y1), color, -1)
            cv2.putText(output, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return output

    def get_active_tracks(self) -> List[TrackedWheel]:
        """Returns list of currently active tracked wheels."""
        return list(self.tracks.values())

    def reset(self) -> None:
        """Resets tracking state and ID counters."""
        self.tracks.clear()
        self.next_track_id = 0
        logger.info("Wheel tracking engine state reset.")
