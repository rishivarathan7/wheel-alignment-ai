"""
Dataset Splitting Utility Script.

Partitions raw image files and YOLO annotation label files into train, validation,
and test sets according to configurable ratios.
"""

import argparse
import logging
from pathlib import Path
import random
import shutil
import sys
from typing import Dict, List, Tuple

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RAW_DATA_DIR, ANNOTATIONS_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SplitDataset")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def split_yolo_dataset(
    raw_img_dir: Path = RAW_DATA_DIR,
    ann_dir: Path = ANNOTATIONS_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, int]:
    """
    Partitions raw images and label files into train, val, and test dataset splits.

    Returns:
        Dict mapping split names ('train', 'val', 'test') to count of assigned samples.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0. Got: {train_ratio} + {val_ratio} + {test_ratio}")

    if not raw_img_dir.exists():
        logger.error(f"Raw image directory not found: {raw_img_dir}")
        return {"train": 0, "val": 0, "test": 0}

    # Discover valid image files
    image_files = [f for f in raw_img_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    if not image_files:
        logger.warning(f"No image files found in {raw_img_dir}")
        return {"train": 0, "val": 0, "test": 0}

    # Shuffle deterministically
    random.seed(seed)
    image_files.sort()
    random.shuffle(image_files)

    total = len(image_files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    splits = {
        "train": (train_files, TRAIN_DATA_DIR),
        "val": (val_files, VAL_DATA_DIR),
        "test": (test_files, TEST_DATA_DIR)
    }

    counts = {"train": 0, "val": 0, "test": 0}

    for split_name, (files, dest_dir) in splits.items():
        img_dest = dest_dir / "images"
        lbl_dest = dest_dir / "labels"
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # Copy image
            shutil.copy2(img_path, img_dest / img_path.name)
            counts[split_name] += 1

            # Copy corresponding label file if present
            label_name = f"{img_path.stem}.txt"
            label_path = ann_dir / label_name
            if label_path.exists():
                shutil.copy2(label_path, lbl_dest / label_name)
            else:
                # Create empty label file for negative samples if missing
                (lbl_dest / label_name).touch()

    logger.info(
        f"Dataset split complete: Total={total} | "
        f"Train={counts['train']} | Val={counts['val']} | Test={counts['test']}"
    )

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition dataset into train/val/test splits for YOLO training.")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DATA_DIR), help="Directory containing raw images")
    parser.add_argument("--ann-dir", type=str, default=str(ANNOTATIONS_DIR), help="Directory containing label text files")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_yolo_dataset(
        raw_img_dir=Path(args.raw_dir),
        ann_dir=Path(args.ann_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
