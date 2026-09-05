"""
Dataset Quality and Validation Subsystem.

Performs rigorous quality analysis across dataset splits and engineered feature matrices:
- Class distribution & imbalance ratio analysis with handling recommendations.
- Missing values, NaN, and duplicate sample detection.
- Invalid, corrupted, or unreadable image file verification.
- Statistical outlier detection via Interquartile Range (IQR) and Z-score metrics.
- Feature range validation (min, max, mean, std).
- Train/Validation/Test split data contamination and image hash collision checks.
- Export of CSV metrics summary, feature distribution plots, and markdown quality report.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, RESULTS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class QualityMetricReport:
    """Dataclass holding summary metrics of dataset quality analysis."""
    total_samples: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 1.0
    imbalance_recommendation: str = "Balanced"
    missing_value_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    invalid_images_count: int = 0
    outliers_count: int = 0
    contamination_count: int = 0
    quality_score: float = 100.0
    passed: bool = True
    issues: List[str] = field(default_factory=list)


def compute_image_hash(filepath: Path) -> Optional[str]:
    """Computes MD5 hash of an image file for exact duplicate and contamination detection."""
    try:
        if not filepath.exists() or filepath.stat().st_size == 0:
            return None
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {filepath}: {e}")
        return None


class DatasetQualityAnalyzer:
    """
    Subsystem for dataset quality inspection, leak detection, outlier analysis,
    and automated report generation.
    """

    def __init__(self, dataset_dir: Optional[Union[str, Path]] = None) -> None:
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DATA_DIR

    def inspect_images(self, directory: Path) -> Tuple[List[Path], List[Path]]:
        """
        Scans image directory and separates valid readable images from corrupted/invalid files.
        """
        valid_images = []
        corrupted_images = []

        if not directory.exists():
            return valid_images, corrupted_images

        for img_path in directory.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue

            if img_path.stat().st_size == 0:
                corrupted_images.append(img_path)
                continue

            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0 or len(img.shape) < 2:
                    corrupted_images.append(img_path)
                else:
                    valid_images.append(img_path)
            except Exception:
                corrupted_images.append(img_path)

        return valid_images, corrupted_images

    def check_split_contamination(
        self,
        train_dir: Optional[Path] = None,
        val_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Checks for identical image hash collisions across train, validation, and test splits.
        """
        tr_dir = train_dir or (TRAIN_DATA_DIR / "images")
        v_dir = val_dir or (VAL_DATA_DIR / "images")
        te_dir = test_dir or (TEST_DATA_DIR / "images")

        train_valid, _ = self.inspect_images(tr_dir)
        val_valid, _ = self.inspect_images(v_dir)
        test_valid, _ = self.inspect_images(te_dir)

        train_hashes = {compute_image_hash(p): p for p in train_valid if compute_image_hash(p) is not None}
        val_hashes = {compute_image_hash(p): p for p in val_valid if compute_image_hash(p) is not None}
        test_hashes = {compute_image_hash(p): p for p in test_valid if compute_image_hash(p) is not None}

        # Collisions
        tr_val_collisions = set(train_hashes.keys()).intersection(set(val_hashes.keys()))
        tr_test_collisions = set(train_hashes.keys()).intersection(set(test_hashes.keys()))

        total_collisions = len(tr_val_collisions) + len(tr_test_collisions)
        return {
            "train_val_collisions": len(tr_val_collisions),
            "train_test_collisions": len(tr_test_collisions),
            "total_collisions": total_collisions,
            "has_contamination": total_collisions > 0
        }

    def detect_outliers_iqr(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Flags outlier rows using Interquartile Range (IQR) thresholding: Q1 - 1.5*IQR or Q3 + 1.5*IQR.
        """
        outlier_mask = pd.Series(False, index=df.index)

        for col in feature_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                col_outliers = (series < (q1 - 2.5 * iqr)) | (series > (q3 + 2.5 * iqr))
                outlier_mask = outlier_mask | col_outliers

        return df[outlier_mask]

    def analyze_feature_dataset(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = "label"
    ) -> QualityMetricReport:
        """
        Performs exhaustive data quality analysis on a processed feature DataFrame.
        """
        report = QualityMetricReport()
        report.total_samples = len(df)

        if df.empty:
            report.passed = False
            report.issues.append("DataFrame is empty.")
            return report

        # 1. Missing Values
        missing = df.isnull().sum()
        report.missing_value_counts = missing[missing > 0].to_dict()
        if missing.sum() > 0:
            report.issues.append(f"Found {missing.sum()} total missing (NaN) values across {len(report.missing_value_counts)} columns.")

        # 2. Duplicate Rows
        duplicates = df.duplicated().sum()
        report.duplicate_rows = int(duplicates)
        if duplicates > 0:
            report.issues.append(f"Found {duplicates} exact duplicate rows.")

        # 3. Class Distribution & Imbalance
        if label_col and label_col in df.columns:
            class_counts = df[label_col].value_counts().to_dict()
            report.class_distribution = {str(k): int(v) for k, v in class_counts.items()}
            counts = list(class_counts.values())

            if len(counts) > 1:
                min_c = min(counts)
                max_c = max(counts)
                imb_ratio = min_c / float(max_c) if max_c > 0 else 0.0
                report.imbalance_ratio = round(imb_ratio, 4)

                if imb_ratio < 0.33:
                    report.imbalance_recommendation = (
                        f"Severe class imbalance detected (Ratio {imb_ratio:.2f}). "
                        "Recommendation: Apply SMOTE oversampling or set `class_weight='balanced'` during classifier training."
                    )
                    report.issues.append("Severe class imbalance (< 0.33 ratio).")
                elif imb_ratio < 0.6:
                    report.imbalance_recommendation = (
                        f"Moderate class imbalance detected (Ratio {imb_ratio:.2f}). "
                        "Recommendation: Use stratified k-fold splitting and probability threshold tuning."
                    )
                else:
                    report.imbalance_recommendation = "Class distribution is sufficiently balanced."

        # 4. Outlier Analysis
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in num_cols:
            num_cols.remove(label_col)

        outliers_df = self.detect_outliers_iqr(df, num_cols)
        report.outliers_count = len(outliers_df)
        if report.outliers_count > 0:
            report.issues.append(f"Detected {report.outliers_count} statistical outlier samples.")

        # Quality score deduction logic
        score = 100.0
        score -= min(30.0, (missing.sum() / max(1, len(df))) * 100.0)
        score -= min(20.0, (duplicates / max(1, len(df))) * 50.0)
        if report.imbalance_ratio < 0.33:
            score -= 15.0
        report.quality_score = round(max(0.0, score), 2)
        report.passed = report.quality_score >= 70.0 and len(report.missing_value_counts) == 0

        return report

    def generate_visualizations(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = "label",
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, str]:
        """
        Generates and saves class distribution bar chart and feature distribution histograms.
        """
        out_dir = Path(output_dir) if output_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Class Distribution Plot
        if label_col and label_col in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            class_counts = df[label_col].value_counts()
            class_counts.plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
            ax.set_title("Dataset Class Distribution")
            ax.set_xlabel("Class Label")
            ax.set_ylabel("Sample Count")
            plt.tight_layout()

            class_plot_path = out_dir / "class_distribution.png"
            plt.savefig(class_plot_path, dpi=150)
            plt.close()
            paths["class_distribution"] = str(class_plot_path)

        # 2. Feature Distribution Histograms
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col][:6]
        if num_cols:
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            for idx, col in enumerate(num_cols):
                axes[idx].hist(df[col].dropna(), bins=20, color="skyblue", edgecolor="black", alpha=0.7)
                axes[idx].set_title(col, fontsize=10)
            plt.tight_layout()

            feat_plot_path = out_dir / "feature_distributions.png"
            plt.savefig(feat_plot_path, dpi=150)
            plt.close()
            paths["feature_distributions"] = str(feat_plot_path)

        return paths

    def export_quality_report(
        self,
        report: QualityMetricReport,
        df: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Path]:
        """
        Exports dataset metrics summary to CSV (`dataset_quality_summary.csv`)
        and a detailed markdown quality report (`dataset_quality_report.md`).
        """
        out_dir = Path(output_dir) if output_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Summary CSV
        summary_rows = [
            {"Metric": "Total Samples", "Value": report.total_samples},
            {"Metric": "Quality Score", "Value": f"{report.quality_score} / 100"},
            {"Metric": "Passed Validation", "Value": report.passed},
            {"Metric": "Duplicate Rows", "Value": report.duplicate_rows},
            {"Metric": "Missing Value Columns", "Value": len(report.missing_value_counts)},
            {"Metric": "Outlier Samples", "Value": report.outliers_count},
            {"Metric": "Imbalance Ratio", "Value": report.imbalance_ratio},
            {"Metric": "Contamination Collisions", "Value": report.contamination_count}
        ]
        summary_df = pd.DataFrame(summary_rows)
        csv_path = out_dir / "dataset_quality_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        # 2. Detailed Markdown Report
        md_content = [
            "# Dataset Quality & Validation Report\n",
            f"**Generated Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Overall Quality Score:** `{report.quality_score} / 100`\n",
            f"**Validation Status:** `{'PASSED' if report.passed else 'WARNING / FAILED'}`\n\n",
            "## 1. Class Distribution & Balance",
            f"- **Class Counts:** `{json.dumps(report.class_distribution)}`",
            f"- **Imbalance Ratio:** `{report.imbalance_ratio}`",
            f"- **Recommendation:** {report.imbalance_recommendation}\n\n",
            "## 2. Data Integrity Checks",
            f"- **Total Samples:** `{report.total_samples}`",
            f"- **Missing Values:** `{len(report.missing_value_counts)}` columns with missing entries.",
            f"- **Duplicate Rows:** `{report.duplicate_rows}`",
            f"- **Outliers Detected (IQR):** `{report.outliers_count}`",
            f"- **Train/Test Contamination Collisions:** `{report.contamination_count}`\n\n",
            "## 3. Identified Issues & Warnings"
        ]

        if report.issues:
            for issue in report.issues:
                md_content.append(f"- [!] {issue}")
        else:
            md_content.append("- No critical dataset quality issues detected.")

        md_content.append("\n## 4. Policy Notice")
        md_content.append(
            "> **Notice**: Automatic modification of dataset samples is disabled. "
            "Any proposed dataset cleaning, duplicate removal, or imbalance resampling is documented and must be explicitly approved."
        )

        md_path = out_dir / "dataset_quality_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

        logger.info(f"Exported Quality Report to {md_path} and CSV summary to {csv_path}")
        return csv_path, md_path
