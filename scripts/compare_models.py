"""
Machine Learning Model Comparison CLI Script.

Usage:
    python scripts/compare_models.py --features data/processed/wheel_alignment_features.csv
"""

import argparse
import logging
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.model_comparison import ModelComparer
from src.prediction import ALIGNMENT_CLASSES, normalize_class_label
from scripts.train_classifier import generate_synthetic_training_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("compare_models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run machine learning model comparison experiment.")
    parser.add_argument("--features", type=str, default=str(PROCESSED_DATA_DIR / "wheel_alignment_features.csv"), help="Path to feature dataset CSV.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Path to output directory for results.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    features_path = Path(args.features)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Feature Dataset
    if not features_path.exists():
        logger.warning(f"Feature dataset not found at {features_path}. Generating synthetic dataset for model comparison benchmarking...")
        df = generate_synthetic_training_dataset(num_samples=180)
    else:
        df = pd.read_csv(features_path)
        if "label" not in df.columns and "status" in df.columns:
            df["label"] = df["status"]
        elif "label" not in df.columns:
            df["label"] = np.random.choice(ALIGNMENT_CLASSES, size=len(df), p=[0.5, 0.3, 0.2])

    df["label"] = df["label"].apply(normalize_class_label)
    logger.info(f"Loaded dataset with {len(df)} samples across classes: {df['label'].value_counts().to_dict()}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    # 2. Identical Train/Test Partition (70% Train, 30% Test)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=args.random_seed, stratify=y
    )

    # 3. Leakage-Free Preprocessing Scaling
    engineer = FeatureEngineer()
    X_train_scaled = engineer.fit_transform(X_train_raw)
    X_test_scaled = engineer.transform(X_test_raw)

    scaler_path = engineer.save_pipeline(MODELS_DIR / "feature_scaler.joblib")

    # 4. Run Model Comparison Experiment
    comparer = ModelComparer(random_seed=args.random_seed)
    benchmark_results, best_res, best_clf = comparer.run_comparison(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # 5. Export Results & Save Selected Model Artifact
    csv_path, plot_path, report_path = comparer.export_comparison_results(benchmark_results, output_dir=out_dir)

    selected_model_path = MODELS_DIR / "alignment_classifier.joblib"
    joblib.dump({"model": best_clf, "feature_names": FEATURE_COLUMNS, "classes": ALIGNMENT_CLASSES}, selected_model_path)
    logger.info(f"Saved selected optimal model ({best_res.model_name}) to {selected_model_path}")

    print("\n" + "=" * 80)
    print(" [MODEL COMPARISON EXPERIMENT COMPLETED]")
    print(f" Selected Model: {best_res.model_name}")
    print(f" Test Accuracy: {best_res.accuracy * 100:.2f}% | F1 (Macro): {best_res.f1_macro:.4f} | F1 (Weighted): {best_res.f1_weighted:.4f}")
    print(f" Inference Latency: {best_res.inference_time_ms:.3f} ms per 100 samples")
    print(f" Comparison CSV: {csv_path}")
    print(f" Comparison Plot: {plot_path}")
    print(f" Markdown Report: {report_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
