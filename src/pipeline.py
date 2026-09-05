"""
Wheel Alignment Monitoring Pipeline.

End-to-End real-time orchestration pipeline integrating:
Video Acquisition -> Preprocessing -> Wheel Detection -> Wheel Tracking ->
Region Analysis -> Feature Engineering -> ML Prediction -> Alert Generation.
"""

from collections import deque
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd

from src.config import ConfigManager, LOGS_DIR, RESULTS_DIR, MODELS_DIR
from src.video_capture import VideoCaptureManager
from src.wheel_detection import YOLOWheelDetector, WheelDetection
from src.wheel_tracking import WheelTracker, TrackedWheel
from src.feature_extraction import WheelFeatureExtractor, WheelRegionAnalysis, WheelTelemetryExporter
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.prediction import RealTimePredictionEngine, RealTimePredictionResult
from src.alert_system import AlertSystem
from src.preprocessing import ImagePreprocessor, PreprocessingConfig, ProcessedFrame

logger = logging.getLogger(__name__)


class WheelAlignmentPipeline:
    """
    End-to-End Real-Time Wheel Alignment Monitoring Pipeline.
    Orchestrates video stream acquisition, frame preprocessing, detection, multi-frame tracking,
    region measurement analysis, feature engineering with temporal history, ML alignment prediction,
    and alert notification generation.
    """

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        debug_mode: bool = False
    ) -> None:
        self.config = config_manager or ConfigManager()
        self.debug_mode = debug_mode or self.config.get("pipeline.debug_mode", False)

        # 1. Preprocessor Subsystem
        prep_config = PreprocessingConfig(
            target_size=(
                self.config.get("preprocessing.target_width", 640),
                self.config.get("preprocessing.target_height", 640)
            ),
            preserve_aspect_ratio=self.config.get("preprocessing.preserve_aspect_ratio", True),
            color_space=self.config.get("preprocessing.color_space", "BGR2RGB"),
            normalize=self.config.get("preprocessing.normalize", False)
        )
        self.preprocessor = ImagePreprocessor(prep_config)

        # 2. Wheel Detector Subsystem
        self.detector = YOLOWheelDetector(
            model_path=self.config.get("detection.model_path", str(MODELS_DIR / "yolov8n.pt")),
            confidence_threshold=self.config.get("detection.confidence_threshold", 0.5),
            iou_threshold=self.config.get("detection.iou_threshold", 0.45),
            device=self.config.get("detection.device", "cpu")
        )

        # 3. Multi-Frame Wheel Tracker Subsystem
        self.tracker = WheelTracker(
            max_disappeared=self.config.get("tracking.max_disappeared", 10),
            distance_threshold=self.config.get("tracking.distance_threshold", 80.0),
            iou_threshold=self.config.get("tracking.iou_threshold", 0.3),
            history_size=self.config.get("tracking.history_size", 50)
        )

        # 4. Wheel Region Feature Extractor Subsystem
        self.feature_extractor = WheelFeatureExtractor()

        # 5. Feature Engineering Subsystem & Per-Track Temporal History Buffer
        self.window_size = self.config.get("feature_engineering.window_size", 5)
        self.feature_engineer = FeatureEngineer(window_size=self.window_size)
        scaler_path = Path(self.config.get("prediction.scaler_path", str(MODELS_DIR / "feature_scaler.joblib")))
        if scaler_path.exists():
            self.feature_engineer.load_pipeline(scaler_path)

        # Raw measurement history buffer per track ID: Dict[track_id, deque of dict records]
        self.raw_history_per_track: Dict[int, deque] = {}

        # 6. ML Real-Time Prediction Engine Subsystem
        self.predictor = RealTimePredictionEngine(
            model_path=self.config.get("prediction.model_path", str(MODELS_DIR / "alignment_classifier.joblib")),
            scaler_path=scaler_path,
            model_version=self.config.get("prediction.model_version", "v1.0.0_RandomForest"),
            camber_limit=self.config.get("prediction.camber_angle_limit", 3.0),
            toe_limit=self.config.get("prediction.toe_angle_limit", 2.0)
        )

        # 7. Alert System Subsystem
        alert_log = LOGS_DIR / "alerts.log"
        self.alert_system = AlertSystem(
            alert_log_path=alert_log,
            cooldown_seconds=self.config.get("alert.cooldown_seconds", 3.0),
            min_confidence=self.config.get("alert.min_confidence", 0.60),
            consecutive_threshold=self.config.get("alert.consecutive_threshold", 3),
            smoothing_window=self.config.get("alert.smoothing_window", 5),
            enable_audio=self.config.get("alert.enable_audio", False)
        )

        # 8. Telemetry Exporter
        telemetry_csv = RESULTS_DIR / "wheel_telemetry.csv"
        self.telemetry_exporter = WheelTelemetryExporter(output_path=telemetry_csv)

        # Execution State & Performance Counters
        self._is_running = False
        self.frame_index = 0
        self.processed_frames_count = 0
        self.start_time = 0.0
        self.current_fps = 0.0
        self._frame_times: deque = deque(maxlen=30)  # Sliding window for FPS calculation

        logger.info(f"WheelAlignmentPipeline initialized (Debug Mode: {self.debug_mode}).")

    def _sync_track_history_buffers(self) -> None:
        """
        Prevents memory leaks by purging temporal feature history buffers and alert states
        for track IDs that have been deregistered by the WheelTracker.
        """
        active_ids = set(self.tracker.tracks.keys())
        buffered_ids = list(self.raw_history_per_track.keys())
        for tid in buffered_ids:
            if tid not in active_ids:
                del self.raw_history_per_track[tid]
        self.alert_system.purge_inactive_tracks(list(active_ids))

    def _update_fps(self, frame_latency_ms: float) -> float:
        """Calculates moving average processing FPS."""
        self._frame_times.append(frame_latency_ms)
        avg_latency_sec = (sum(self._frame_times) / len(self._frame_times)) / 1000.0 if self._frame_times else 0.033
        self.current_fps = round(1.0 / avg_latency_sec, 1) if avg_latency_sec > 0 else 0.0
        return self.current_fps

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[str] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Processes a single frame sequentially through all 8 pipeline stages.

        Args:
            frame: Input BGR image array.
            timestamp: ISO 8601 timestamp string or float seconds.

        Returns:
            Tuple of:
            - annotated_frame: Output frame with visual bounding boxes, trajectories, status labels, and debug overlays.
            - telemetry_records: List of structured telemetry dictionaries for each active wheel.
            - performance_stats: Latency breakdown, FPS, and debug information dictionary.
        """
        t0 = time.perf_counter()
        self.frame_index += 1
        ts_str = timestamp if isinstance(timestamp, str) else datetime.now().isoformat()

        # Handle empty/corrupted frame validation
        if frame is None or frame.size == 0:
            logger.warning(f"Frame #{self.frame_index} is empty or invalid. Skipping pipeline stage execution.")
            perf_stats = {
                "frame_index": self.frame_index,
                "timestamp": ts_str,
                "fps": self.current_fps,
                "total_latency_ms": 0.0,
                "is_valid": False,
                "active_tracks": 0
            }
            return frame, [], perf_stats

        annotated_frame = frame.copy()
        telemetry_records: List[Dict[str, Any]] = []
        debug_data: Dict[str, Any] = {}

        try:
            # Stage 1: Frame Preprocessing
            t_prep_start = time.perf_counter()
            processed_container: ProcessedFrame = self.preprocessor.preprocess(frame)
            prep_latency_ms = (time.perf_counter() - t_prep_start) * 1000.0

            # Stage 2: Wheel Detection
            t_det_start = time.perf_counter()
            detections: List[WheelDetection] = self.detector.detect(frame)
            det_latency_ms = (time.perf_counter() - t_det_start) * 1000.0

            # Stage 3: Multi-Frame Wheel Tracking
            t_track_start = time.perf_counter()
            now_sec = time.time()
            tracked_wheels: List[TrackedWheel] = self.tracker.update(detections, timestamp=now_sec)
            track_latency_ms = (time.perf_counter() - t_track_start) * 1000.0

            # Memory Leak Prevention: Sync per-track history buffers with active tracks
            self._sync_track_history_buffers()

            analysis_latency_ms = 0.0
            ml_latency_ms = 0.0

            # Handle Missing Detections gracefully
            if not tracked_wheels:
                logger.debug(f"Frame #{self.frame_index}: 0 active wheel tracks.")
            else:
                # Process active tracked wheels through Region Analysis -> Feature Engineering -> ML Prediction -> Alerting
                for wheel in tracked_wheels:
                    tid = wheel.track_id
                    try:
                        # Stage 4: Wheel Region Analysis
                        t_analysis_start = time.perf_counter()
                        wheel_crop = self.preprocessor.crop_roi(frame, wheel.bbox)
                        region_analysis: WheelRegionAnalysis = self.feature_extractor.extract_features(
                            wheel_crop=wheel_crop,
                            bbox=wheel.bbox,
                            track_id=tid
                        )
                        region_analysis.frame_index = self.frame_index
                        region_analysis.timestamp = ts_str
                        analysis_latency_ms += (time.perf_counter() - t_analysis_start) * 1000.0

                        # Log to CSV exporter
                        self.telemetry_exporter.log_analysis(region_analysis)

                        # Stage 5: Feature Engineering & Temporal History Accumulation
                        if tid not in self.raw_history_per_track:
                            self.raw_history_per_track[tid] = deque(maxlen=self.window_size)
                        self.raw_history_per_track[tid].append(region_analysis.to_dict())

                        # Engineer 19-feature vector including rolling-window statistics
                        df_raw_history = pd.DataFrame(list(self.raw_history_per_track[tid]))
                        df_engineered = self.feature_engineer.transform_raw_measurements(df_raw_history)

                        # Extract latest engineered feature record
                        latest_feature_dict = df_engineered.iloc[-1].to_dict() if not df_engineered.empty else region_analysis.to_dict()

                        # Stage 6: ML Prediction
                        t_ml_start = time.perf_counter()
                        prediction: RealTimePredictionResult = self.predictor.predict_wheel(
                            analysis=latest_feature_dict,
                            wheel_id=tid,
                            timestamp=ts_str
                        )
                        ml_latency_ms += (time.perf_counter() - t_ml_start) * 1000.0

                    except Exception as exc:
                        logger.error(f"Prediction/feature error on wheel track #{tid}: {exc}", exc_info=True)
                        prediction = RealTimePredictionResult(
                            wheel_id=tid,
                            predicted_condition="NORMAL",
                            prediction_probability=0.50,
                            camber_proxy=0.0,
                            toe_proxy=0.0,
                            is_fallback=True,
                            message="Prediction failure / missing feature data."
                        )
                        latest_feature_dict = {}

                    # Stage 7: Alert Generation & Visual Overlay
                    annotated_frame = self.alert_system.process_alert(
                        prediction=prediction,
                        track_id=tid,
                        frame=annotated_frame,
                        bbox=wheel.bbox
                    )

                    # Stage 8: Collect Telemetry Record
                    telemetry_records.append({
                        "frame_index": self.frame_index,
                        "timestamp": ts_str,
                        "track_id": tid,
                        "bbox": wheel.bbox,
                        "status": prediction.predicted_condition,
                        "confidence": prediction.prediction_probability,
                        "camber_proxy": prediction.camber_proxy,
                        "toe_proxy": prediction.toe_proxy,
                        "inference_latency_ms": prediction.inference_latency_ms,
                        "is_fallback": prediction.is_fallback,
                        "motion": wheel.calculate_motion_features(),
                        "engineered_features": latest_feature_dict
                    })


            # Draw persistent trailing centroid trajectory paths
            annotated_frame = self.tracker.draw_trajectories(annotated_frame, tracked_wheels)

            # Total Frame Latency & FPS Calculation
            total_frame_latency_ms = (time.perf_counter() - t0) * 1000.0
            current_fps = self._update_fps(total_frame_latency_ms)

            perf_stats = {
                "frame_index": self.frame_index,
                "timestamp": ts_str,
                "fps": current_fps,
                "total_latency_ms": round(total_frame_latency_ms, 2),
                "prep_latency_ms": round(prep_latency_ms, 2),
                "det_latency_ms": round(det_latency_ms, 2),
                "track_latency_ms": round(track_latency_ms, 2),
                "analysis_latency_ms": round(analysis_latency_ms, 2),
                "ml_latency_ms": round(ml_latency_ms, 2),
                "active_tracks": len(tracked_wheels),
                "model_status": "Loaded" if self.predictor.is_ready else "Fallback_Rules",
                "is_valid": True
            }

            # Render Debug Mode Visual Overlays if enabled
            if self.debug_mode:
                annotated_frame = self._render_debug_overlay(annotated_frame, perf_stats, tracked_wheels)
                debug_data = {
                    "preprocessed_image": processed_container.processed_image,
                    "raw_detections": detections,
                    "tracked_wheels": tracked_wheels,
                    "perf_stats": perf_stats
                }
                perf_stats["debug_data"] = debug_data

        except Exception as e:
            logger.error(f"Error in pipeline during frame #{self.frame_index} processing: {e}", exc_info=True)
            total_frame_latency_ms = (time.perf_counter() - t0) * 1000.0
            perf_stats = {
                "frame_index": self.frame_index,
                "timestamp": ts_str,
                "fps": self.current_fps,
                "total_latency_ms": round(total_frame_latency_ms, 2),
                "error": str(e),
                "is_valid": False
            }

        return annotated_frame, telemetry_records, perf_stats

    def _render_debug_overlay(
        self,
        frame: np.ndarray,
        perf_stats: Dict[str, Any],
        tracked_wheels: List[TrackedWheel]
    ) -> np.ndarray:
        """
        Renders intermediate debug HUD dashboard showing latency breakdown, FPS, model state,
        and active track count directly on the output frame.
        """
        if frame is None or frame.size == 0:
            return frame

        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Top-Left HUD Dashboard Box
        panel_w, panel_h = 290, 150
        margin = 10
        sub_img = output[margin:margin + panel_h, margin:margin + panel_w]
        white_rect = np.full(sub_img.shape, 30, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.3, white_rect, 0.7, 1.0)
        output[margin:margin + panel_h, margin:margin + panel_w] = res
        cv2.rectangle(output, (margin, margin), (margin + panel_w, margin + panel_h), (0, 255, 255), 1)

        # Render HUD Dashboard Text
        lines = [
            f"PIPELINE DEBUG HUD | Frame #{perf_stats.get('frame_index', 0)}",
            f"FPS: {perf_stats.get('fps', 0.0)} | Total: {perf_stats.get('total_latency_ms', 0.0):.1f}ms",
            f"Prep: {perf_stats.get('prep_latency_ms', 0.0):.1f}ms | Det: {perf_stats.get('det_latency_ms', 0.0):.1f}ms",
            f"Track: {perf_stats.get('track_latency_ms', 0.0):.1f}ms | ML: {perf_stats.get('ml_latency_ms', 0.0):.1f}ms",
            f"Active Tracks: {perf_stats.get('active_tracks', 0)} | Model: {perf_stats.get('model_status', 'N/A')}",
            f"Memory Buffers: {len(self.raw_history_per_track)} active IDs"
        ]

        y_offset = margin + 20
        for idx, text in enumerate(lines):
            color = (0, 255, 255) if idx == 0 else (255, 255, 255)
            font_scale = 0.45 if idx == 0 else 0.40
            cv2.putText(output, text, (margin + 8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            y_offset += 20

        # 2. Render ROI Crop Thumbnail in Top-Right if active tracks exist
        if tracked_wheels:
            wheel = tracked_wheels[0]
            crop = self.preprocessor.crop_roi(frame, wheel.bbox)
            if crop is not None and crop.size > 0:
                thumb_size = (100, 100)
                thumb = cv2.resize(crop, thumb_size)
                # Position in top right
                tx1, ty1 = w - thumb_size[0] - margin, margin
                tx2, ty2 = w - margin, margin + thumb_size[1]
                output[ty1:ty2, tx1:tx2] = thumb
                cv2.rectangle(output, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
                cv2.putText(output, f"ROI #{wheel.track_id}", (tx1, ty1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return output

    def run_video(
        self,
        video_source: Union[int, str] = 0,
        display: bool = False,
        save_output: bool = False,
        output_filename: str = "monitoring_output.mp4"
    ) -> None:
        """
        Runs the monitoring pipeline continuously on a video file or live camera stream.

        Args:
            video_source: Video file path or camera index integer.
            display: Whether to render live OpenCV window.
            save_output: Whether to save processed video to disk.
            output_filename: Output video filename.
        """
        self._is_running = True
        self.start_time = time.time()
        self.processed_frames_count = 0
        logger.info(f"Starting Wheel Alignment AI Pipeline on source: {video_source}")

        writer = None

        try:
            with VideoCaptureManager(source=video_source) as cap:
                props = cap.get_properties()
                logger.info(f"Capture source initialized: {props['width']}x{props['height']} @ {props['fps']} FPS")

                if save_output:
                    out_path = RESULTS_DIR / output_filename
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, props["fps"], (props["width"], props["height"]))
                    logger.info(f"Saving processed video output to: {out_path}")

                for frame in cap.stream_frames():
                    if not self._is_running:
                        logger.info("Pipeline stop flag set. Terminating execution loop.")
                        break

                    annotated_frame, telemetry, perf_stats = self.process_frame(frame)
                    self.processed_frames_count += 1

                    if writer is not None:
                        # Write frame matching video resolution
                        if annotated_frame.shape[:2] != (props["height"], props["width"]):
                            write_frame = cv2.resize(annotated_frame, (props["width"], props["height"]))
                        else:
                            write_frame = annotated_frame
                        writer.write(write_frame)

                    if display:
                        window_title = "Wheel Alignment AI Monitor [DEBUG]" if self.debug_mode else "Wheel Alignment AI Monitor"
                        cv2.imshow(window_title, annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            logger.info("User pressed 'q'. Gracefully shutting down pipeline.")
                            break
                        elif key == ord("d"):
                            self.debug_mode = not self.debug_mode
                            logger.info(f"Toggled Debug Mode: {self.debug_mode}")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught. Shutting down gracefully.")
        except Exception as e:
            logger.error(f"Unhandled exception during video pipeline run: {e}", exc_info=True)
        finally:
            self.stop()
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            # Automatically export accumulated telemetry CSV upon completion
            self.telemetry_exporter.export_to_csv()

            elapsed = time.time() - self.start_time
            avg_fps = self.processed_frames_count / elapsed if elapsed > 0 else 0.0
            logger.info(f"Pipeline stopped. Processed {self.processed_frames_count} frames in {elapsed:.2f}s (Avg: {avg_fps:.1f} FPS).")

    def stop(self) -> None:
        """Triggers graceful shutdown of the execution loop."""
        self._is_running = False
        logger.info("Stop requested for WheelAlignmentPipeline.")
