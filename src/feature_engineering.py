"""
Feature Engineering Subsystem for Wheel Alignment Anomaly Classification.

Engineers a curated, physical-geometric feature matrix from raw wheel region measurements (Module 08):
- Spatial & Geometric features (Aspect ratio, circularity, eccentricity, contour ratio, edge density).
- Orientation features (Fitted ellipse angle deviation from 90°, sin/cos angular components).
- Temporal & Movement dynamics (Orientation deltas, centroid shifts, movement speed).
- Rolling-window statistics per track ID (Rolling mean, std over configurable window size).
- Leakage-free normalization & imputation scaling pipeline (`fit`, `transform`, `save_pipeline`, `load_pipeline`).
- Feature dictionary documentation export and correlation/feature importance analysis.
"""

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# Curated list of informative features
FEATURE_COLUMNS = [
    "aspect_ratio",
    "aspect_ratio_dev",
    "circularity",
    "eccentricity",
    "contour_area_ratio",
    "edge_density",
    "angle_dev_90",
    "sin_angle",
    "cos_angle",
    "delta_orientation",
    "delta_center_x",
    "delta_center_y",
    "movement_speed",
    "roll_mean_aspect_ratio",
    "roll_std_aspect_ratio",
    "roll_mean_angle_dev",
    "roll_std_angle_dev",
    "roll_mean_speed",
    "roll_std_speed"
]


@dataclass
class FeatureDefinition:
    name: str
    category: str
    description: str
    data_type: str
    physical_interpretation: str


FEATURE_DICTIONARY: List[FeatureDefinition] = [
    FeatureDefinition("aspect_ratio", "Spatial", "Height / Width ratio of bounding box", "float64", "Geometric shape ratio proxy"),
    FeatureDefinition("aspect_ratio_dev", "Spatial", "Absolute deviation from 1.0 aspect ratio", "float64", "Elongation severity"),
    FeatureDefinition("circularity", "Geometric", "4 * pi * Area / Perimeter^2", "float64", "Wheel rim circular perfection"),
    FeatureDefinition("eccentricity", "Geometric", "Fitted ellipse eccentricity sqrt(1 - (b/a)^2)", "float64", "Ellipse flatness proxy"),
    FeatureDefinition("contour_area_ratio", "Geometric", "Contour area divided by bounding box area", "float64", "Fill factor of rim inside box"),
    FeatureDefinition("edge_density", "Texture", "Ratio of Canny edge pixels to total crop size", "float64", "Rim spoke/tread texture density"),
    FeatureDefinition("angle_dev_90", "Orientation", "Abs deviation of ellipse angle from 90°", "float64", "Vertical alignment slant proxy"),
    FeatureDefinition("sin_angle", "Orientation", "Sine of fitted ellipse angle in radians", "float64", "Cyclic vertical angle component"),
    FeatureDefinition("cos_angle", "Orientation", "Cosine of fitted ellipse angle in radians", "float64", "Cyclic horizontal angle component"),
    FeatureDefinition("delta_orientation", "Temporal", "Frame-to-frame change in ellipse orientation", "float64", "Angular wobble instability"),
    FeatureDefinition("delta_center_x", "Movement", "Frame-to-frame horizontal centroid displacement", "float64", "Horizontal movement vector"),
    FeatureDefinition("delta_center_y", "Movement", "Frame-to-frame vertical centroid displacement", "float64", "Vertical movement vector"),
    FeatureDefinition("movement_speed", "Movement", "Euclidean distance of centroid shift (pixels/frame)", "float64", "Image-space motion speed"),
    FeatureDefinition("roll_mean_aspect_ratio", "Rolling Stat", "Rolling mean aspect ratio over window", "float64", "Smoothed aspect ratio trend"),
    FeatureDefinition("roll_std_aspect_ratio", "Rolling Stat", "Rolling std of aspect ratio over window", "float64", "Aspect ratio temporal variance"),
    FeatureDefinition("roll_mean_angle_dev", "Rolling Stat", "Rolling mean angle deviation over window", "float64", "Smoothed orientation slant"),
    FeatureDefinition("roll_std_angle_dev", "Rolling Stat", "Rolling std of angle deviation over window", "float64", "Orientation oscillation volatility"),
    FeatureDefinition("roll_mean_speed", "Rolling Stat", "Rolling mean movement speed over window", "float64", "Smoothed tracking speed"),
    FeatureDefinition("roll_std_speed", "Rolling Stat", "Rolling std of movement speed over window", "float64", "Motion jitter volatility")
]


class FeatureEngineer:
    """
    Feature Engineering Pipeline for Wheel Alignment Anomaly Classification.
    Constructs rolling temporal features, handles missing/invalid values,
    and applies leakage-free preprocessing scaling.
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def transform_raw_measurements(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers spatial, geometric, orientation, temporal, and rolling-window features
        from raw Module 08 measurement records.
        """
        if df_raw.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        df = df_raw.copy()

        # 1. Filter out invalid rows
        if "is_valid" in df.columns:
            df = df[df["is_valid"] == True].copy()

        if df.empty:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        # Ensure track_id and frame_index exist
        if "track_id" not in df.columns:
            df["track_id"] = 0
        if "frame_index" not in df.columns:
            df["frame_index"] = np.arange(len(df))

        # Sort by track_id and frame_index
        df = df.sort_values(by=["track_id", "frame_index"]).reset_index(drop=True)

        # Helper for robust numeric conversion and NaN/Inf/String imputation
        def _safe_numeric_col(col_name: str, default_val: float = 0.0) -> pd.Series:
            if col_name in df.columns:
                s = pd.to_numeric(df[col_name], errors="coerce")
            else:
                s = pd.Series(default_val, index=df.index)
            # Replace inf with default_val and fillna
            s = s.replace([np.inf, -np.inf], np.nan)
            return s.fillna(default_val)

        # 2. Base Spatial & Geometric Features
        df["aspect_ratio"] = _safe_numeric_col("aspect_ratio", 1.0)
        df["aspect_ratio_dev"] = (df["aspect_ratio"] - 1.0).abs()
        df["circularity"] = _safe_numeric_col("circularity", 0.0)
        df["eccentricity"] = _safe_numeric_col("eccentricity", 0.0)

        bbox_area = _safe_numeric_col("bbox_area", 1.0).replace(0, 1.0)
        contour_area = _safe_numeric_col("contour_area", 0.0)
        df["contour_area_ratio"] = (contour_area / bbox_area).clip(0.0, 1.0)
        df["edge_density"] = _safe_numeric_col("edge_density", 0.0)

        # 3. Orientation Features
        angle = _safe_numeric_col("fitted_ellipse_angle", 90.0)
        df["angle_dev_90"] = (angle - 90.0).abs()
        angle_rad = np.radians(angle)
        df["sin_angle"] = np.sin(angle_rad)
        df["cos_angle"] = np.cos(angle_rad)

        # 4. Temporal & Movement Features
        df["delta_orientation"] = _safe_numeric_col("delta_orientation", 0.0)
        df["delta_center_x"] = _safe_numeric_col("delta_center_x", 0.0)
        df["delta_center_y"] = _safe_numeric_col("delta_center_y", 0.0)
        df["movement_speed"] = _safe_numeric_col("movement_speed", 0.0)

        # 5. Groupby Rolling-Window Statistics per Track ID
        grouped = df.groupby("track_id")

        df["roll_mean_aspect_ratio"] = grouped["aspect_ratio"].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).mean()
        )
        df["roll_std_aspect_ratio"] = grouped["aspect_ratio"].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).std().fillna(0.0)
        )

        df["roll_mean_angle_dev"] = grouped["angle_dev_90"].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).mean()
        )
        df["roll_std_angle_dev"] = grouped["angle_dev_90"].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).std().fillna(0.0)
        )

        df["roll_mean_speed"] = grouped["movement_speed"].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).mean()
        )
        df["roll_std_speed"] = grouped["movement_speed"].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).std().fillna(0.0)
        )

        # Select engineered feature columns
        df_engineered = df[FEATURE_COLUMNS].copy()

        # Handle missing values
        df_engineered = df_engineered.fillna(0.0)
        return df_engineered

    def fit(self, X_train: pd.DataFrame) -> "FeatureEngineer":
        """
        Fits Imputer and StandardScaler on training dataset features to prevent data leakage.
        """
        X_imp = self.imputer.fit_transform(X_train[FEATURE_COLUMNS])
        self.scaler.fit(X_imp)
        self.is_fitted = True
        logger.info("Feature scaling pipeline successfully fitted on training dataset.")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms input feature dataframe using fitted imputer and scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer pipeline must be fitted before calling transform().")
        X_sub = X[FEATURE_COLUMNS] if isinstance(X, pd.DataFrame) else X
        X_imp = self.imputer.transform(X_sub)
        return self.scaler.transform(X_imp)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fits on training data and transforms it in one step."""
        self.fit(X_train)
        return self.transform(X_train)

    def save_pipeline(self, model_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves fitted feature engineering pipeline object to file."""
        path = Path(model_path) if model_path else MODELS_DIR / "feature_scaler.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"imputer": self.imputer, "scaler": self.scaler, "feature_names": FEATURE_COLUMNS}, path)
        logger.info(f"Saved feature engineering pipeline to {path}")
        return path

    def load_pipeline(self, model_path: Optional[Union[str, Path]] = None) -> bool:
        """Loads fitted feature engineering pipeline object from file."""
        path = Path(model_path) if model_path else MODELS_DIR / "feature_scaler.joblib"
        if not path.exists():
            logger.warning(f"Feature scaler file does not exist at {path}")
            return False
        data = joblib.load(path)
        self.imputer = data["imputer"]
        self.scaler = data["scaler"]
        self.is_fitted = True
        logger.info(f"Loaded feature engineering pipeline from {path}")
        return True


def export_feature_dictionary(output_path: Optional[Union[str, Path]] = None) -> Path:
    """Exports human-readable feature dictionary documenting all engineered features."""
    dest = Path(output_path) if output_path else RESULTS_DIR / "feature_dictionary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    dict_data = [
        {
            "name": f.name,
            "category": f.category,
            "description": f.description,
            "data_type": f.data_type,
            "physical_interpretation": f.physical_interpretation
        }
        for f in FEATURE_DICTIONARY
    ]

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(dict_data, f, indent=4)

    logger.info(f"Exported Feature Dictionary to {dest}")
    return dest


def analyze_feature_importance_and_correlation(
    df_features: pd.DataFrame,
    y: np.ndarray,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Computes feature-to-feature correlation matrix and feature importance ranking via RandomForest.
    Generates and saves visual plots to results directory.
    """
    out_dir = Path(output_dir) if output_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature Correlation Matrix
    corr_matrix = df_features[FEATURE_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=90, fontsize=8)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=8)
    plt.title("Feature Correlation Matrix", pad=20)
    plt.tight_layout()

    corr_plot_path = out_dir / "feature_correlation.png"
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()

    # 2. RandomForest Feature Importance Analysis
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(df_features[FEATURE_COLUMNS], y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(FEATURE_COLUMNS)), importances[indices], color="skyblue", align="center")
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels([FEATURE_COLUMNS[i] for i in indices], rotation=90, fontsize=8)
    ax.set_ylabel("Importance Score")
    ax.set_title("RandomForest Feature Importances for Alignment Anomaly Detection")
    plt.tight_layout()

    importance_plot_path = out_dir / "feature_importance.png"
    plt.savefig(importance_plot_path, dpi=150)
    plt.close()

    logger.info(f"Saved correlation plot to {corr_plot_path} and importance plot to {importance_plot_path}")
    return {
        "correlation_plot": str(corr_plot_path),
        "importance_plot": str(importance_plot_path),
        "top_feature": FEATURE_COLUMNS[indices[0]],
        "top_importance_score": float(importances[indices[0]])
    }
