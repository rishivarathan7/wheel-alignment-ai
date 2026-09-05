"""
Unit Tests for Dataset Quality and Validation Subsystem.

Tests:
1. Class distribution and imbalance ratio calculations.
2. Missing value and duplicate row detection.
3. IQR statistical outlier detection.
4. Split contamination and MD5 image hash collision checking.
5. Quality report generation (CSV summary & Markdown report).
6. Class and feature distribution plot generation.
"""

from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np
import pandas as pd

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_quality import (
    DatasetQualityAnalyzer,
    QualityMetricReport,
    compute_image_hash
)


class TestDatasetQualitySubsystem(unittest.TestCase):
    """Unit test suite for DatasetQualityAnalyzer component."""

    def setUp(self) -> None:
        """Sets up test synthetic datasets and temp directories."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.analyzer = DatasetQualityAnalyzer(dataset_dir=self.temp_dir.name)

        # Synthetic feature dataframe
        np.random.seed(42)
        self.df_clean = pd.DataFrame({
            "aspect_ratio": np.random.normal(1.0, 0.05, 50),
            "circularity": np.random.normal(0.85, 0.02, 50),
            "eccentricity": np.random.normal(0.2, 0.01, 50),
            "label": ["NORMAL"] * 25 + ["MISALIGNED"] * 25
        })

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_class_distribution_and_imbalance(self) -> None:
        """Tests class distribution counting and imbalance ratio calculation."""
        # Balanced dataset (25 vs 25 -> ratio 1.0)
        rep_balanced = self.analyzer.analyze_feature_dataset(self.df_clean, label_col="label")
        self.assertEqual(rep_balanced.class_distribution, {"NORMAL": 25, "MISALIGNED": 25})
        self.assertEqual(rep_balanced.imbalance_ratio, 1.0)

        # Imbalanced dataset (90 vs 10 -> ratio 0.11)
        df_imbalanced = pd.DataFrame({
            "aspect_ratio": np.ones(100),
            "label": ["NORMAL"] * 90 + ["MISALIGNED"] * 10
        })
        rep_imb = self.analyzer.analyze_feature_dataset(df_imbalanced, label_col="label")
        self.assertAlmostEqual(rep_imb.imbalance_ratio, 0.1111, places=3)
        self.assertIn("Severe class imbalance", rep_imb.imbalance_recommendation)

    def test_missing_values_and_duplicates(self) -> None:
        """Tests missing value and duplicate row detection."""
        df_dirty = self.df_clean.copy()
        # Add NaNs
        df_dirty.loc[0, "aspect_ratio"] = np.nan
        df_dirty.loc[1, "circularity"] = np.nan
        # Add duplicate row
        df_dirty = pd.concat([df_dirty, df_dirty.iloc[[2]]], ignore_index=True)

        report = self.analyzer.analyze_feature_dataset(df_dirty, label_col="label")
        self.assertGreater(len(report.missing_value_counts), 0)
        self.assertGreater(report.duplicate_rows, 0)
        self.assertFalse(report.passed)

    def test_outlier_detection_iqr(self) -> None:
        """Tests statistical outlier detection using IQR."""
        df_outliers = self.df_clean.copy()
        # Inject extreme outlier
        df_outliers.loc[0, "aspect_ratio"] = 100.0

        outliers = self.analyzer.detect_outliers_iqr(df_outliers, ["aspect_ratio"])
        self.assertGreaterEqual(len(outliers), 1)

    def test_split_contamination(self) -> None:
        """Tests image hash collision detection across train/val/test split directories."""
        tr_dir = Path(self.temp_dir.name) / "train" / "images"
        val_dir = Path(self.temp_dir.name) / "val" / "images"
        tr_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Create identical image in train and val
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img_path1 = tr_dir / "img1.png"
        img_path2 = val_dir / "img2.png"
        cv2.imwrite(str(img_path1), img)
        cv2.imwrite(str(img_path2), img)

        res = self.analyzer.check_split_contamination(train_dir=tr_dir, val_dir=val_dir, test_dir=val_dir)
        self.assertTrue(res["has_contamination"])
        self.assertEqual(res["train_val_collisions"], 1)

    def test_quality_report_export(self) -> None:
        """Tests CSV summary and Markdown report generation."""
        report = self.analyzer.analyze_feature_dataset(self.df_clean, label_col="label")
        csv_path, md_path = self.analyzer.export_quality_report(report, self.df_clean, output_dir=self.temp_dir.name)

        self.assertTrue(csv_path.exists())
        self.assertTrue(md_path.exists())

        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("Dataset Quality & Validation Report", content)
            self.assertIn("Overall Quality Score", content)

    def test_quality_visualizations(self) -> None:
        """Tests generation of class distribution and feature distribution plots."""
        paths = self.analyzer.generate_visualizations(self.df_clean, label_col="label", output_dir=self.temp_dir.name)
        self.assertIn("class_distribution", paths)
        self.assertTrue(Path(paths["class_distribution"]).exists())
        self.assertIn("feature_distributions", paths)
        self.assertTrue(Path(paths["feature_distributions"]).exists())


if __name__ == "__main__":
    unittest.main()
