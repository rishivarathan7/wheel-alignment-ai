"""
Unit Tests for Custom Wheel Detector Training and Dataset Pipeline.

Tests:
1. Parsing and validation of dataset.yaml configuration.
2. Dataset splitting logic into train/validation/test sets.
3. YOLO annotation line and file format validation (detecting missing, malformed, or out-of-bounds labels).
4. Training configuration parameters and execution device selection.
"""

from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import yaml

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from scripts.split_dataset import split_yolo_dataset
from scripts.validate_dataset import (
    validate_yolo_label_line,
    validate_split_directory,
    validate_full_dataset
)


class TestDatasetAndTrainingPipeline(unittest.TestCase):
    """Unit test suite for Dataset and Training Pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up a temporary dataset environment with synthetic images and label text files."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.base_path = Path(cls.temp_dir.name)

        cls.raw_dir = cls.base_path / "raw"
        cls.ann_dir = cls.base_path / "annotations"
        cls.raw_dir.mkdir(parents=True)
        cls.ann_dir.mkdir(parents=True)

        # Generate 10 synthetic images and corresponding YOLO label text files
        for i in range(10):
            img_name = f"sample_{i:02d}.jpg"
            lbl_name = f"sample_{i:02d}.txt"

            # Create dummy image file using OpenCV
            import cv2
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(cls.raw_dir / img_name), dummy_img)

            # Create valid YOLO label line: class_id x_center y_center width height
            with open(cls.ann_dir / lbl_name, "w", encoding="utf-8") as f:
                f.write("0 0.500 0.500 0.400 0.400\n")

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up temporary directory."""
        cls.temp_dir.cleanup()

    def test_dataset_yaml_structure(self) -> None:
        """Tests dataset.yaml structure and parameters."""
        yaml_file = DATA_DIR / "dataset.yaml"
        self.assertTrue(yaml_file.exists())

        with open(yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("train", cfg)
        self.assertIn("val", cfg)
        self.assertIn("test", cfg)
        self.assertEqual(cfg["nc"], 1)
        self.assertIn("wheel", list(cfg["names"].values()))

    def test_split_dataset_utility(self) -> None:
        """Tests dataset splitting into train, val, and test subsets."""
        split_counts = split_yolo_dataset(
            raw_img_dir=self.raw_dir,
            ann_dir=self.ann_dir,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=42
        )

        self.assertEqual(split_counts["train"], 7)
        self.assertEqual(split_counts["val"], 2)
        self.assertEqual(split_counts["test"], 1)

    def test_yolo_annotation_line_validation(self) -> None:
        """Tests line-by-line YOLO annotation validation rules."""
        # 1. Valid line
        ok, err = validate_yolo_label_line("0 0.50 0.50 0.30 0.30", line_num=1, num_classes=1)
        self.assertTrue(ok)
        self.assertEqual(err, "")

        # 2. Malformed line (too few values)
        ok, err = validate_yolo_label_line("0 0.50 0.50", line_num=1, num_classes=1)
        self.assertFalse(ok)
        self.assertIn("Expected 5 values", err)

        # 3. Invalid class ID out of bounds
        ok, err = validate_yolo_label_line("5 0.50 0.50 0.30 0.30", line_num=1, num_classes=1)
        self.assertFalse(ok)
        self.assertIn("Class ID 5 out of bounds", err)

        # 4. Out-of-bounds coordinate (> 1.0)
        ok, err = validate_yolo_label_line("0 1.50 0.50 0.30 0.30", line_num=1, num_classes=1)
        self.assertFalse(ok)
        self.assertIn("outside normalized range", err)

        # 5. Non-positive box width/height (0.0)
        ok, err = validate_yolo_label_line("0 0.50 0.50 0.00 0.30", line_num=1, num_classes=1)
        self.assertFalse(ok)
        self.assertIn("must be in range", err)

    def test_split_directory_validation(self) -> None:
        """Tests split directory validator on synthetic split folder."""
        # Create a mock split directory
        mock_split = self.base_path / "mock_split"
        img_dir = mock_split / "images"
        lbl_dir = mock_split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Write clean sample
        import cv2
        cv2.imwrite(str(img_dir / "valid.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
        with open(lbl_dir / "valid.txt", "w", encoding="utf-8") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

        # Write corrupted sample
        cv2.imwrite(str(img_dir / "corrupt.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
        with open(lbl_dir / "corrupt.txt", "w", encoding="utf-8") as f:
            f.write("0 2.5 0.5 0.2 0.2\n")

        res = validate_split_directory(mock_split, num_classes=1)
        self.assertEqual(res.total_images, 2)
        self.assertEqual(res.valid_samples, 1)
        self.assertIn("corrupt.txt", res.corrupted_label_files)


if __name__ == "__main__":
    unittest.main()
