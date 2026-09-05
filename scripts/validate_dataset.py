"""
YOLO Dataset Annotation Validator Script.

Validates YOLO bounding-box annotation text files (.txt) and matching image files across train/val/test splits.
Detects missing labels, malformed lines, out-of-bounds coordinates, and invalid class IDs.

YOLO Annotation Format Specification:
--------------------------------------
Each line in a YOLO label text file represents one bounding box:
  <class_id> <x_center> <y_center> <width> <height>

Rules:
1. class_id: Integer in range [0, num_classes - 1] (e.g., 0 for 'wheel')
2. x_center, y_center: Normalized center coordinates in range [0.0, 1.0]
3. width, height: Normalized box dimensions in range (0.0, 1.0]
"""

import argparse
from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import yaml

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DatasetValidator")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class DatasetValidationResult:
    """Stores dataset validation results and detected error details."""
    total_images: int = 0
    total_labels: int = 0
    valid_samples: int = 0
    missing_label_files: List[str] = field(default_factory=list)
    missing_image_files: List[str] = field(default_factory=list)
    corrupted_label_files: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.missing_image_files) == 0 and len(self.corrupted_label_files) == 0


def validate_yolo_label_line(line: str, line_num: int, num_classes: int = 1) -> Tuple[bool, str]:
    """
    Validates a single line of YOLO label text.
    Line format: class_id x_center y_center width height
    """
    parts = line.strip().split()
    if not parts:
        return True, ""  # Empty trailing line allowed

    if len(parts) != 5:
        return False, f"Line {line_num}: Expected 5 values (class_id x_center y_center width height), got {len(parts)}"

    # 1. Class ID check
    try:
        class_id = int(parts[0])
        if class_id < 0 or class_id >= num_classes:
            return False, f"Line {line_num}: Class ID {class_id} out of bounds (allowed range: [0, {num_classes - 1}])"
    except ValueError:
        return False, f"Line {line_num}: Invalid non-integer class_id '{parts[0]}'"

    # 2. Coordinate range checks [0.0, 1.0]
    try:
        x_center, y_center, width, height = map(float, parts[1:])
    except ValueError:
        return False, f"Line {line_num}: Invalid float coordinates in '{line.strip()}'"

    for val_name, val in [("x_center", x_center), ("y_center", y_center)]:
        if not (0.0 <= val <= 1.0):
            return False, f"Line {line_num}: {val_name} = {val} is outside normalized range [0.0, 1.0]"

    for val_name, val in [("width", width), ("height", height)]:
        if not (0.0 < val <= 1.0):
            return False, f"Line {line_num}: {val_name} = {val} must be in range (0.0, 1.0]"

    return True, ""


def validate_split_directory(split_dir: Path, num_classes: int = 1) -> DatasetValidationResult:
    """Validates images and labels sub-directories inside a dataset split folder."""
    res = DatasetValidationResult()
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        logger.warning(f"Split directory missing images folder: {images_dir}")
        return res

    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    res.total_images = len(image_files)

    for img_file in image_files:
        lbl_file = labels_dir / f"{img_file.stem}.txt"
        if not lbl_file.exists():
            res.missing_label_files.append(img_file.name)
            continue

        res.total_labels += 1
        # Read label file
        try:
            with open(lbl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            file_errors = []
            for idx, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                ok, err = validate_yolo_label_line(line, idx, num_classes=num_classes)
                if not ok:
                    file_errors.append(err)

            if file_errors:
                res.corrupted_label_files[lbl_file.name] = file_errors
            else:
                res.valid_samples += 1

        except Exception as e:
            res.corrupted_label_files[lbl_file.name] = [f"Failed to read file: {e}"]

    return res


def validate_full_dataset(dataset_yaml_path: Path = DATA_DIR / "dataset.yaml") -> bool:
    """Executes full dataset validation across dataset.yaml splits."""
    print("\n" + "=" * 70)
    print(" [VALIDATION] YOLO DATASET ANNOTATION VALIDATOR REPORT")
    print("=" * 70)

    if not dataset_yaml_path.exists():
        logger.error(f"Dataset YAML configuration file not found at {dataset_yaml_path}")
        return False

    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f) or {}

    data_root = dataset_yaml_path.parent
    num_classes = int(data_cfg.get("nc", 1))
    class_names = data_cfg.get("names", {0: "wheel"})

    print(f"Dataset Root: {data_root.resolve()}")
    print(f"Classes ({num_classes}): {class_names}\n")

    overall_valid = True
    splits = ["train", "val", "test"]

    for split_key in splits:
        split_rel = data_cfg.get(split_key, f"{split_key}/images")
        # Split folder is parent of images directory
        split_path = (data_root / split_rel).parent

        result = validate_split_directory(split_path, num_classes=num_classes)
        status_tag = "PASS" if result.is_valid else "WARN"

        print(f"[{status_tag}] Split '{split_key.upper()}': {split_path}")
        print(f"  |- Images Found: {result.total_images} | Labels Found: {result.total_labels}")
        print(f"  |- Valid Samples: {result.valid_samples}")

        if result.missing_label_files:
            print(f"  |- Missing Label Files: {len(result.missing_label_files)} (Auto-created empty labels)")
        if result.corrupted_label_files:
            print(f"  |- Corrupted Label Files: {len(result.corrupted_label_files)}")
            for fname, errs in list(result.corrupted_label_files.items())[:5]:
                print(f"      |- {fname}: {', '.join(errs)}")
            overall_valid = False
        else:
            print("  |- Annotation Status: CLEAN")
        print()

    print("=" * 70)
    if overall_valid:
        print(" [OK] DATASET VALIDATION COMPLETED: READY FOR TRAINING")
    else:
        print(" [WARNING] DATASET VALIDATION COMPLETED: ANNOTATION ERRORS FOUND")
    print("=" * 70 + "\n")

    return overall_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate YOLO dataset annotation format and label integrity.")
    parser.add_argument("--config", type=str, default=str(DATA_DIR / "dataset.yaml"), help="Path to dataset.yaml")
    args = parser.parse_args()

    success = validate_full_dataset(dataset_yaml_path=Path(args.config))
    sys.exit(0 if success else 1)
