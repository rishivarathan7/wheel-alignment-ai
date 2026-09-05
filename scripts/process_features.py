"""
Script for Processing Raw Measurements into Engineered Features.

Usage:
    python scripts/process_features.py --input results/wheel_telemetry.csv --output data/processed/wheel_alignment_features.csv
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.feature_engineering import (
    FeatureEngineer,
    export_feature_dictionary,
    analyze_feature_importance_and_correlation,
    FEATURE_COLUMNS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("process_features")


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineer features from raw wheel measurement CSV.")
    parser.add_argument("--input", type=str, default=str(RESULTS_DIR / "wheel_telemetry.csv"), help="Input raw telemetry CSV file.")
    parser.add_argument("--output", type=str, default=str(PROCESSED_DATA_DIR / "wheel_alignment_features.csv"), help="Output processed features CSV file.")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size for temporal stats.")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export Feature Dictionary Documentation
    export_feature_dictionary()

    if not input_path.exists():
        logger.warning(f"Input file {input_path} does not exist. Creating synthetic sample telemetry for feature processing...")
        # Create synthetic raw telemetry data for testing pipeline CLI
        records = []
        for track_id in range(2):
            for frame_idx in range(15):
                records.append({
                    "frame_index": frame_idx,
                    "track_id": track_id,
                    "is_valid": True,
                    "aspect_ratio": 1.0 + 0.05 * np.random.randn(),
                    "circularity": 0.85 + 0.02 * np.random.randn(),
                    "eccentricity": 0.2 + 0.01 * np.random.randn(),
                    "contour_area": 2500.0,
                    "contour_perimeter": 200.0,
                    "bbox_area": 2500,
                    "edge_density": 0.15,
                    "fitted_ellipse_angle": 90.0 + 2.0 * np.random.randn(),
                    "delta_orientation": 0.5 * np.random.randn(),
                    "delta_center_x": 1.0,
                    "delta_center_y": 0.5,
                    "movement_speed": 1.12
                })
        df_raw = pd.DataFrame(records)
    else:
        df_raw = pd.read_csv(input_path)

    logger.info(f"Loaded {len(df_raw)} raw measurement records.")

    # 2. Engineer Features
    engineer = FeatureEngineer(window_size=args.window)
    df_engineered = engineer.transform_raw_measurements(df_raw)

    # Save final feature dataset
    df_engineered.to_csv(output_path, index=False)
    logger.info(f"Saved engineered features dataset ({len(df_engineered)} rows, {df_engineered.shape[1]} features) to {output_path}")

    # 3. Fit & Save Preprocessing Pipeline
    engineer.fit(df_engineered)
    scaler_path = engineer.save_pipeline(MODELS_DIR / "feature_scaler.joblib")
    logger.info(f"Saved feature scaling pipeline to {scaler_path}")

    # 4. Perform Correlation & Feature Importance Analysis
    synthetic_labels = np.random.choice([0, 1], size=len(df_engineered))
    analysis_results = analyze_feature_importance_and_correlation(df_engineered, synthetic_labels)

    print("\n" + "=" * 70)
    print(" [FEATURE ENGINEERING COMPLETED SUCCESSFULLY]")
    print(f" Output Dataset: {output_path}")
    print(f" Scaler Pipeline: {scaler_path}")
    print(f" Correlation Plot: {analysis_results['correlation_plot']}")
    print(f" Importance Plot: {analysis_results['importance_plot']}")
    print(f" Top Feature: {analysis_results['top_feature']} (Importance: {analysis_results['top_importance_score']:.4f})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
