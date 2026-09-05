"""
Training and Evaluation Pipeline for Alignment Anomaly Classifier.

Trains a Random Forest baseline classifier for 3-class wheel alignment status classification:
- NORMAL
- POSSIBLE_MISALIGNMENT
- SEVERE_MISALIGNMENT

Enforces reproducible, leakage-free data preprocessing, generates classification reports,
computes confusion matrices, and exports feature importances.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RESULTS_DIR, PROCESSED_DATA_DIR
from src.feature_engineering import FeatureEngineer, FEATURE_COLUMNS
from src.prediction import ALIGNMENT_CLASSES, normalize_class_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_classifier")


def generate_synthetic_training_dataset(num_samples: int = 150) -> pd.DataFrame:
    """Generates synthetic feature matrix with known physical distributions for dataset bootstrapping."""
    np.random.seed(42)
    records = []
    classes = ["NORMAL", "POSSIBLE_MISALIGNMENT", "SEVERE_MISALIGNMENT"]

    for i in range(num_samples):
        cls = np.random.choice(classes, p=[0.5, 0.3, 0.2])

        if cls == "NORMAL":
            ar = np.random.normal(1.0, 0.03)
            circ = np.random.normal(0.90, 0.02)
            ecc = np.random.normal(0.15, 0.02)
            angle_dev = np.random.normal(0.0, 2.0)
            speed = np.random.normal(0.5, 0.2)
        elif cls == "POSSIBLE_MISALIGNMENT":
            ar = np.random.normal(1.12, 0.04)
            circ = np.random.normal(0.82, 0.03)
            ecc = np.random.normal(0.35, 0.03)
            angle_dev = np.random.normal(6.0, 3.0)
            speed = np.random.normal(1.2, 0.3)
        else:  # SEVERE_MISALIGNMENT
            ar = np.random.normal(1.30, 0.06)
            circ = np.random.normal(0.70, 0.05)
            ecc = np.random.normal(0.55, 0.05)
            angle_dev = np.random.normal(15.0, 4.0)
            speed = np.random.normal(2.5, 0.5)

        records.append({
            "track_id": i // 10,
            "frame_index": i % 10,
            "is_valid": True,
            "aspect_ratio": ar,
            "circularity": circ,
            "eccentricity": ecc,
            "contour_area": 2500.0,
            "contour_perimeter": 200.0,
            "bbox_area": 2500,
            "edge_density": 0.18,
            "fitted_ellipse_angle": 90.0 + angle_dev,
            "delta_orientation": angle_dev * 0.1,
            "delta_center_x": speed * 0.7,
            "delta_center_y": speed * 0.3,
            "movement_speed": abs(speed),
            "label": cls
        })

    df_raw = pd.DataFrame(records)
    engineer = FeatureEngineer(window_size=5)
    df_eng = engineer.transform_raw_measurements(df_raw)
    df_eng["label"] = df_raw["label"]
    return df_eng


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random Forest classifier for wheel alignment anomaly detection.")
    parser.add_argument("--features", type=str, default=str(PROCESSED_DATA_DIR / "wheel_alignment_features.csv"), help="Path to processed features CSV.")
    parser.add_argument("--model-out", type=str, default=str(MODELS_DIR / "alignment_classifier.joblib"), help="Path to save trained classifier model.")
    parser.add_argument("--scaler-out", type=str, default=str(MODELS_DIR / "feature_scaler.joblib"), help="Path to save feature scaler.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in Random Forest.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    features_path = Path(args.features)
    model_path = Path(args.model_out)
    scaler_path = Path(args.scaler_out)
    results_dir = RESULTS_DIR

    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest & Validate Dataset
    if not features_path.exists():
        logger.warning(f"Feature dataset not found at {features_path}. Generating synthetic dataset for training pipeline execution...")
        df = generate_synthetic_training_dataset(num_samples=180)
    else:
        df = pd.read_csv(features_path)
        if "label" not in df.columns and "status" in df.columns:
            df["label"] = df["status"]
        elif "label" not in df.columns:
            logger.warning("No label column in CSV. Generating synthetic class labels for training demonstration...")
            df["label"] = np.random.choice(ALIGNMENT_CLASSES, size=len(df), p=[0.5, 0.3, 0.2])

    # Normalize class labels
    df["label"] = df["label"].apply(normalize_class_label)
    logger.info(f"Loaded feature dataset with {len(df)} rows. Class counts:\n{df['label'].value_counts().to_dict()}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    # 2. Stratified Train / Val / Test Split (70% Train, 15% Val, 15% Test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.random_seed, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=args.random_seed, stratify=y_temp
    )

    logger.info(f"Dataset split complete: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

    # 3. Leakage-Free Preprocessing Scaling Pipeline
    engineer = FeatureEngineer()
    X_train_scaled = engineer.fit_transform(X_train_raw)
    X_val_scaled = engineer.transform(X_val_raw)
    X_test_scaled = engineer.transform(X_test_raw)

    engineer.save_pipeline(scaler_path)

    # 4. Model Training (Random Forest Baseline)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_seed,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    logger.info("Random Forest baseline classifier training complete.")

    # 5. Validation Check
    y_val_pred = clf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    # 6. Evaluation on Unseen Test Dataset
    y_test_pred = clf.predict(X_test_scaled)

    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_prec_macro = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_rec_macro = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    test_f1_macro = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))

    test_prec_weighted = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_rec_weighted = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    test_f1_weighted = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))

    # 7. Save Model Artifact
    joblib.dump({"model": clf, "feature_names": FEATURE_COLUMNS, "classes": ALIGNMENT_CLASSES}, model_path)
    logger.info(f"Saved trained classifier model artifact to {model_path}")

    # 8. Reports & Confusion Matrix Generation
    report_str = classification_report(y_test, y_test_pred, target_names=None, zero_division=0)
    report_file = results_dir / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=== ALIGNMENT ANOMALY CLASSIFIER EVALUATION REPORT ===\n\n")
        f.write(report_str)

    metrics_dict = {
        "accuracy": round(test_acc, 4),
        "precision_macro": round(test_prec_macro, 4),
        "recall_macro": round(test_rec_macro, 4),
        "f1_score_macro": round(test_f1_macro, 4),
        "precision_weighted": round(test_prec_weighted, 4),
        "recall_weighted": round(test_rec_weighted, 4),
        "f1_score_weighted": round(test_f1_weighted, 4),
        "train_samples": len(X_train_scaled),
        "test_samples": len(X_test_scaled)
    }
    metrics_file = results_dir / "classification_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)

    # 9. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred, labels=ALIGNMENT_CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ALIGNMENT_CLASSES)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    ax.set_title("Wheel Alignment Anomaly Classifier - Confusion Matrix")
    plt.tight_layout()
    cm_plot_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    # 10. Feature Importance Plot
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(FEATURE_COLUMNS)), importances[indices], color="steelblue", align="center")
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels([FEATURE_COLUMNS[i] for i in indices], rotation=90, fontsize=8)
    ax.set_ylabel("Importance Score")
    ax.set_title("Random Forest Classifier Feature Importances")
    plt.tight_layout()
    fi_plot_path = results_dir / "classifier_feature_importance.png"
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()

    print("\n" + "=" * 75)
    print(" [MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY]")
    print(f" Trained Model Artifact: {model_path}")
    print(f" Preprocessing Scaler: {scaler_path}")
    print(f" Test Accuracy: {test_acc * 100:.2f}%")
    print(f" Macro Precision: {test_prec_macro:.4f} | Recall: {test_rec_macro:.4f} | F1-Score: {test_f1_macro:.4f}")
    print(f" Classification Report: {report_file}")
    print(f" Confusion Matrix Plot: {cm_plot_path}")
    print(f" Feature Importance Plot: {fi_plot_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
