"""
Dataset Quality and Validation CLI Script.

Usage:
    python scripts/analyze_dataset_quality.py --input data/processed/wheel_alignment_features.csv
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

from src.config import DATA_DIR, RESULTS_DIR, PROCESSED_DATA_DIR
from src.dataset_quality import DatasetQualityAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analyze_dataset_quality")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset quality, class balance, outliers, and contamination.")
    parser.add_argument("--input", type=str, default=str(PROCESSED_DATA_DIR / "wheel_alignment_features.csv"), help="Path to feature dataset CSV.")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR), help="Output directory for reports.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = DatasetQualityAnalyzer(dataset_dir=DATA_DIR)

    # 1. Inspect image files and split contamination
    logger.info("Scanning dataset splits for corrupted images and contamination collisions...")
    contamination_res = analyzer.check_split_contamination()
    logger.info(f"Contamination scan result: {contamination_res}")

    # 2. Ingest feature dataset
    if not input_path.exists():
        logger.warning(f"Input file {input_path} not found. Synthesizing sample feature matrix for quality analysis...")
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "aspect_ratio": np.random.normal(1.0, 0.05, n),
            "circularity": np.random.normal(0.85, 0.02, n),
            "eccentricity": np.random.normal(0.2, 0.01, n),
            "edge_density": np.random.uniform(0.1, 0.3, n),
            "label": np.random.choice(["NORMAL", "MISALIGNED"], size=n, p=[0.75, 0.25])
        })
    else:
        df = pd.read_csv(input_path)
        if "label" not in df.columns and "status" in df.columns:
            df["label"] = df["status"]
        elif "label" not in df.columns:
            # Synthetic label for validation display if unlabelled
            df["label"] = np.random.choice(["NORMAL", "MISALIGNED"], size=len(df), p=[0.7, 0.3])

    logger.info(f"Ingested {len(df)} samples for quality validation.")

    # 3. Analyze Feature Dataset Quality
    report = analyzer.analyze_feature_dataset(df, label_col="label")
    report.contamination_count = contamination_res["total_collisions"]

    # 4. Generate Visualizations
    plot_paths = analyzer.generate_visualizations(df, label_col="label", output_dir=output_dir)

    # 5. Export Quality Report & Summary CSV
    csv_path, md_path = analyzer.export_quality_report(report, df, output_dir=output_dir)

    print("\n" + "=" * 75)
    print(" [DATASET QUALITY ANALYSIS REPORT]")
    print(f" Quality Score: {report.quality_score} / 100 ({'PASSED' if report.passed else 'WARNING'})")
    print(f" Total Samples: {report.total_samples}")
    print(f" Class Distribution: {report.class_distribution}")
    print(f" Imbalance Ratio: {report.imbalance_ratio}")
    print(f" Recommendation: {report.imbalance_recommendation}")
    print(f" Missing Values: {len(report.missing_value_counts)} columns")
    print(f" Duplicate Rows: {report.duplicate_rows}")
    print(f" Outlier Samples: {report.outliers_count}")
    print(f" Split Contamination Collisions: {report.contamination_count}")
    print(f" Summary CSV: {csv_path}")
    print(f" Markdown Report: {md_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
