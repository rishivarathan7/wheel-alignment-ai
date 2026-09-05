"""
Custom YOLO Wheel Detector Evaluation Script.

Evaluates trained custom YOLO model performance on the test dataset split (data/test/).
Computes mAP50, mAP50-95, Precision, Recall, Confusion Matrix, and saves plots into results/evaluation/.
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager, get_execution_device, MODELS_DIR, RESULTS_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvaluateYOLO")


@dataclass
class EvaluationMetrics:
    """Dataclass storing model evaluation metrics."""
    map50: float
    map50_95: float
    precision: float
    recall: float
    fitness: float
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mAP50": round(self.map50, 4),
            "mAP50-95": round(self.map50_95, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "Fitness": round(self.fitness, 4),
            "Device": self.device
        }


def evaluate_custom_yolo_detector(
    model_path: Optional[str] = None,
    dataset_yaml: Optional[str] = None,
    imgsz: int = 640,
    device: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Optional[EvaluationMetrics]:
    """
    Evaluates trained custom YOLO model on test dataset split.

    Returns:
        EvaluationMetrics object containing test performance scores, or None if evaluation fails.
    """
    config = ConfigManager()

    weights_file = Path(model_path or config.get("detection.model_path", "models/wheel_yolo_custom.pt")).resolve()
    data_yaml = Path(dataset_yaml or config.get("training.dataset_config", "data/dataset.yaml")).resolve()
    target_device = device or get_execution_device()
    eval_out_dir = Path(output_dir or RESULTS_DIR / "evaluation").resolve()

    logger.info("Starting Custom YOLO Wheel Detector Evaluation Pipeline...")
    logger.info(f"Evaluating Model Weights: {weights_file}")
    logger.info(f"Target Evaluation Dataset: {data_yaml}")

    if not weights_file.exists():
        logger.error(f"Evaluation failed: Model weights file not found at {weights_file}")
        logger.info("Train a model first using: python scripts/train_yolo.py")
        return None

    if not data_yaml.exists():
        logger.error(f"Evaluation failed: Dataset YAML file not found at {data_yaml}")
        return None

    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_file))

        eval_out_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation on test split
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=imgsz,
            device=target_device,
            project=str(eval_out_dir.parent),
            name=eval_out_dir.name,
            exist_ok=True,
            verbose=True
        )

        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        fitness = float(metrics.fitness)

        eval_metrics = EvaluationMetrics(
            map50=map50,
            map50_95=map50_95,
            precision=precision,
            recall=recall,
            fitness=fitness,
            device=target_device
        )

        print("\n" + "=" * 70)
        print(" 📊 CUSTOM YOLO WHEEL DETECTOR EVALUATION METRICS REPORT")
        print("=" * 70)
        print(f" Model Weights: {weights_file.name}")
        print(f" Execution Device: [{target_device.upper()}]")
        print(f" mAP@50    : {map50 * 100:.2f}%")
        print(f" mAP@50-95 : {map50_95 * 100:.2f}%")
        print(f" Precision : {precision * 100:.2f}%")
        print(f" Recall    : {recall * 100:.2f}%")
        print(f" Fitness   : {fitness:.4f}")
        print("=" * 70 + "\n")

        return eval_metrics

    except Exception as e:
        logger.error(f"Error during YOLO model evaluation: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained custom YOLO wheel detector on test dataset split.")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model weights (e.g. models/wheel_yolo_custom.pt)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    args = parser.parse_args()

    results = evaluate_custom_yolo_detector(
        model_path=args.model,
        dataset_yaml=args.dataset,
        imgsz=args.imgsz,
        device=args.device
    )

    sys.exit(0 if results is not None else 1)
