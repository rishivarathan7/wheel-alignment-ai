"""
Custom YOLO Wheel Detector Training Script.

Prepares and executes training for custom YOLO wheel/tyre detector on custom annotated dataset.
Saves checkpoints, best performing model, training metrics, and loss plots. Works on CPU or CUDA GPU.
"""

import argparse
import logging
from pathlib import Path
import shutil
import sys
from typing import Dict, Any, Optional

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager, get_execution_device, MODELS_DIR, RESULTS_DIR, DATA_DIR
from scripts.validate_dataset import validate_full_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainYOLO")


def train_custom_yolo_wheel_detector(
    base_model: Optional[str] = None,
    dataset_yaml: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    imgsz: Optional[int] = None,
    lr0: Optional[float] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    best_model_dest: Optional[str] = None
) -> Optional[Path]:
    """
    Executes training of custom YOLO wheel detector.

    Returns:
        Path to best trained weights (.pt) if successful, None otherwise.
    """
    config = ConfigManager()

    # Resolve settings from config or function parameters
    model_weights = base_model or config.get("training.base_model", "models/yolov8n.pt")
    data_yaml = Path(dataset_yaml or config.get("training.dataset_config", "data/dataset.yaml")).resolve()
    num_epochs = epochs or config.get("training.epochs", 50)
    batch = batch_size or config.get("training.batch_size", 16)
    image_size = imgsz or config.get("training.imgsz", 640)
    learning_rate = lr0 or config.get("training.learning_rate", 0.01)
    target_device = device or get_execution_device()
    save_dir = Path(output_dir or config.get("training.output_dir", "results/train")).resolve()
    final_best_dest = Path(best_model_dest or config.get("training.best_model_dest", "models/wheel_yolo_custom.pt")).resolve()

    logger.info("Starting Custom YOLO Wheel Detector Training Pipeline...")
    logger.info(f"Target Execution Device: [{target_device.upper()}]")
    logger.info(f"Dataset Configuration: {data_yaml}")

    # 1. Validate dataset annotations before starting training
    logger.info("Running dataset annotation validation check...")
    if not data_yaml.exists():
        logger.error(f"Training aborted: dataset.yaml missing at {data_yaml}")
        return None

    dataset_valid = validate_full_dataset(data_yaml)
    if not dataset_valid:
        logger.warning("Dataset validation flagged warnings. Proceeding with training on available clean samples...")

    # 2. Initialize YOLO Model
    try:
        from ultralytics import YOLO
        model = YOLO(model_weights)
        logger.info(f"Base model loaded from {model_weights}")
    except ImportError:
        logger.error("Training aborted: ultralytics package not installed. Run: pip install -r requirements.txt")
        return None
    except Exception as e:
        logger.error(f"Error loading base YOLO model {model_weights}: {e}")
        return None

    # 3. Execute Training
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=num_epochs,
            batch=batch,
            imgsz=image_size,
            lr0=learning_rate,
            device=target_device,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
            save=True,
            save_period=config.get("training.save_period", 5),
            patience=config.get("training.patience", 10),
            verbose=True
        )

        logger.info("YOLO training loop completed successfully.")

        # 4. Save and copy best performing model weights
        run_save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else save_dir
        best_pt = run_save_dir / "weights" / "best.pt"

        if best_pt.exists():
            final_best_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_pt, final_best_dest)
            logger.info(f"Best model weights saved to {best_pt}")
            logger.info(f"Copied best model weights to target deployment path: {final_best_dest}")
            return final_best_dest
        else:
            logger.warning(f"best.pt not found at expected location {best_pt}")
            return None

    except Exception as e:
        logger.error(f"Error occurred during YOLO training execution: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO wheel/tyre detection model.")
    parser.add_argument("--base-model", type=str, default=None, help="Base model weights (e.g. models/yolov8n.pt)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (e.g., 640)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate lr0")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    args = parser.parse_args()

    trained_weights = train_custom_yolo_wheel_detector(
        base_model=args.base_model,
        dataset_yaml=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=args.device
    )

    if trained_weights:
        print(f"\n[OK] Training completed successfully. Best model saved to: {trained_weights}\n")
    else:
        print("\n[FAIL] Training did not complete successfully. Check logs for details.\n")
