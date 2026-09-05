"""
Prediction Subsystem.

Evaluates wheel alignment anomaly status using trained Machine Learning classifiers
(Random Forest baseline) or fallback physical heuristic rules.

Target Classes:
- NORMAL
- POSSIBLE_MISALIGNMENT
- SEVERE_MISALIGNMENT
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.feature_extraction import WheelRegionAnalysis, AlignmentFeatures

logger = logging.getLogger(__name__)

# Canonical Class Label Mapping
ALIGNMENT_CLASSES = ["NORMAL", "POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT", "INSUFFICIENT_DATA"]
CLASS_LABEL_TO_ID = {label: idx for idx, label in enumerate(ALIGNMENT_CLASSES)}
CLASS_ID_TO_LABEL = {idx: label for idx, label in enumerate(ALIGNMENT_CLASSES)}

# Legacy and auxiliary mappings
LEGACY_LABEL_MAP = {
    "MISALIGNED": "SEVERE_MISALIGNMENT",
    "SUSPECT": "POSSIBLE_MISALIGNMENT",
    "UNTRUSTED": "INSUFFICIENT_DATA",
    "UNKNOWN": "INSUFFICIENT_DATA",
    "LOW_CONFIDENCE": "INSUFFICIENT_DATA",
    "0": "NORMAL",
    "1": "POSSIBLE_MISALIGNMENT",
    "2": "SEVERE_MISALIGNMENT",
    "3": "INSUFFICIENT_DATA"
}


def normalize_class_label(label: Union[str, int]) -> str:
    """Normalizes any string or numeric label into canonical target state representation."""
    lbl_str = str(label).strip().upper()
    if lbl_str in CLASS_LABEL_TO_ID:
        return lbl_str
    if lbl_str in LEGACY_LABEL_MAP:
        return LEGACY_LABEL_MAP[lbl_str]
    try:
        idx = int(label)
        if idx in CLASS_ID_TO_LABEL:
            return CLASS_ID_TO_LABEL[idx]
    except Exception:
        pass
    return "NORMAL"


@dataclass
class RealTimePredictionResult:
    """Structured result returned by the Real-Time Prediction Engine."""
    wheel_id: int
    predicted_condition: str  # "NORMAL", "POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"
    prediction_probability: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_version: str = "v1.0.0_RandomForest"
    camber_proxy: float = 0.0
    toe_proxy: float = 0.0
    inference_latency_ms: float = 0.0
    is_fallback: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def status(self) -> str:
        return self.predicted_condition

    @property
    def confidence(self) -> float:
        return self.prediction_probability

    @property
    def camber_angle(self) -> float:
        return self.camber_proxy

    @property
    def toe_angle(self) -> float:
        return self.toe_proxy


# Backward compatibility alias
AlignmentPrediction = RealTimePredictionResult


class RealTimePredictionEngine:
    """
    High-performance real-time prediction engine.
    Ingests timestamped wheel region measurements, validates feature order,
    applies the exact trained preprocessing pipeline, and outputs structured condition predictions.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        model_version: str = "v1.0.0_RandomForest",
        camber_limit: float = 3.0,
        toe_limit: float = 2.0,
        **kwargs: Any
    ) -> None:
        self.model_path = Path(model_path) if model_path else MODELS_DIR / "alignment_classifier.joblib"
        self.scaler_path = Path(scaler_path) if scaler_path else MODELS_DIR / "feature_scaler.joblib"
        self.model_version = model_version
        self.camber_limit = camber_limit
        self.toe_limit = toe_limit

        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.expected_feature_names = FEATURE_COLUMNS
        self.is_ready = False

        self.load_artifacts()

    def load_artifacts(self) -> bool:
        """
        Loads classifier model and preprocessing scaler pipeline artifacts.
        """
        success_model = False
        success_scaler = False

        # 1. Load Preprocessing Pipeline
        if self.scaler_path.exists():
            success_scaler = self.feature_engineer.load_pipeline(self.scaler_path)

        # 2. Load Model Artifact
        if self.model_path.exists():
            try:
                loaded = joblib.load(str(self.model_path))
                if isinstance(loaded, dict) and "model" in loaded:
                    self.model = loaded["model"]
                    if "feature_names" in loaded:
                        self.expected_feature_names = loaded["feature_names"]
                else:
                    self.model = loaded
                success_model = True
                logger.info(f"Loaded classifier model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load classifier model from {self.model_path}: {e}")

        self.is_ready = success_model and success_scaler
        if not self.is_ready:
            logger.warning("Prediction engine initialized in fallback mode (model or scaler artifact missing).")
        return self.is_ready

    @property
    def _is_loaded(self) -> bool:
        """Backward compatibility property for is_ready."""
        return self.is_ready

    def _extract_raw_features(
        self,
        analysis: Union[WheelRegionAnalysis, Dict[str, Any], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Extracts and aligns raw feature columns matching trained schema.
        """
        if isinstance(analysis, WheelRegionAnalysis):
            raw_dict = analysis.to_dict()
        elif isinstance(analysis, dict):
            raw_dict = analysis.copy()
        elif isinstance(analysis, pd.DataFrame):
            raw_dict = analysis.iloc[0].to_dict()
        else:
            raw_dict = {}

        # Handle missing or NaN values
        row_dict = {}
        for col in self.expected_feature_names:
            val = raw_dict.get(col, 0.0)
            try:
                f_val = float(val) if val is not None and math.isfinite(float(val)) else 0.0
            except Exception:
                f_val = 0.0
            row_dict[col] = f_val

        return pd.DataFrame([row_dict], columns=self.expected_feature_names)

    def predict_wheel(
        self,
        analysis: Union[WheelRegionAnalysis, Dict[str, Any]],
        wheel_id: int = -1,
        timestamp: Optional[str] = None
    ) -> RealTimePredictionResult:
        """
        Executes real-time inference for a single wheel region.
        """
        t0 = time.perf_counter()
        ts_now = timestamp or datetime.now().isoformat()

        # Determine wheel track ID
        tid = wheel_id
        if tid < 0 and hasattr(analysis, "track_id"):
            tid = getattr(analysis, "track_id", -1)
        elif tid < 0 and isinstance(analysis, dict):
            tid = analysis.get("track_id", -1)

        # 1. Model Prediction Path
        if self.is_ready and self.model is not None:
            try:
                df_raw = self._extract_raw_features(analysis)

                # Apply exact trained scaling pipeline
                X_scaled = self.feature_engineer.transform(df_raw)

                # Inference
                raw_pred = self.model.predict(X_scaled)[0]
                status = normalize_class_label(raw_pred)

                if hasattr(self.model, "predict_proba"):
                    probas = self.model.predict_proba(X_scaled)[0]
                    prob = float(np.max(probas))
                else:
                    prob = 0.90

                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                camber = self._estimate_camber(analysis)
                toe = self._estimate_toe(analysis)

                return RealTimePredictionResult(
                    wheel_id=tid,
                    predicted_condition=status,
                    prediction_probability=round(prob, 4),
                    timestamp=ts_now,
                    model_version=self.model_version,
                    camber_proxy=camber,
                    toe_proxy=toe,
                    inference_latency_ms=round(elapsed_ms, 3),
                    is_fallback=False,
                    details={"raw_prediction": str(raw_pred), "features_checked": len(self.expected_feature_names)}
                )

            except Exception as e:
                logger.error(f"Inference exception for wheel #{tid}: {e}. Triggering fallback prediction.")

        # 2. Safe Fallback Path
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._heuristic_fallback(analysis, wheel_id=tid, timestamp=ts_now, latency_ms=elapsed_ms)

    def predict(
        self,
        features: Union[WheelRegionAnalysis, Dict[str, Any]],
        wheel_id: int = -1,
        timestamp: Optional[str] = None
    ) -> RealTimePredictionResult:
        """Backward compatibility alias for predict_wheel."""
        return self.predict_wheel(features, wheel_id=wheel_id, timestamp=timestamp)

    def _heuristic_fallback(
        self,
        analysis: Union[WheelRegionAnalysis, Dict[str, Any]],
        wheel_id: int,
        timestamp: str,
        latency_ms: float
    ) -> RealTimePredictionResult:
        """Physical rule-based fallback evaluation."""
        camber = self._estimate_camber(analysis)
        toe = self._estimate_toe(analysis)

        is_camber_severe = abs(camber) > 4.5
        is_toe_severe = abs(toe) > 3.0
        is_camber_abnormal = abs(camber) > 3.0
        is_toe_abnormal = abs(toe) > 2.0

        if is_camber_severe or is_toe_severe:
            status = "SEVERE_MISALIGNMENT"
            prob = 0.90
        elif is_camber_abnormal or is_toe_abnormal:
            status = "POSSIBLE_MISALIGNMENT"
            prob = 0.75
        else:
            status = "NORMAL"
            prob = 0.95

        return RealTimePredictionResult(
            wheel_id=wheel_id,
            predicted_condition=status,
            prediction_probability=prob,
            timestamp=timestamp,
            model_version="Fallback_Heuristic_Rules",
            camber_proxy=camber,
            toe_proxy=toe,
            inference_latency_ms=round(latency_ms, 3),
            is_fallback=True,
            details={"fallback_reason": "Model or scaler artifact not ready"}
        )

    def _estimate_camber(self, analysis: Union[WheelRegionAnalysis, Dict[str, Any]]) -> float:
        ar = getattr(analysis, "aspect_ratio", analysis.get("aspect_ratio", 1.0) if isinstance(analysis, dict) else 1.0)
        angle = getattr(analysis, "fitted_ellipse_angle", analysis.get("fitted_ellipse_angle", 90.0) if isinstance(analysis, dict) else 90.0)
        delta_ar = (float(ar) - 1.0) * 10.0
        angle_dev = (float(angle) - 90.0) / 15.0 if angle > 0 else 0.0
        return round(float(delta_ar + angle_dev), 2)

    def _estimate_toe(self, analysis: Union[WheelRegionAnalysis, Dict[str, Any]]) -> float:
        ecc = getattr(analysis, "eccentricity", analysis.get("eccentricity", 0.0) if isinstance(analysis, dict) else 0.0)
        return round(float(ecc) * 3.5, 2)


# Backward compatibility class wrapper
AlignmentPredictor = RealTimePredictionEngine
