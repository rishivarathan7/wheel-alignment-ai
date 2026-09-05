"""
Centralized Configuration and Logging Management Module.

Provides singleton configuration management (`ConfigManager`), environment variable overrides,
default settings resolution, YAML persistence, directory structure enforcement, and structured
application logging setup (`setup_logger`).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
try:
    import yaml
except ImportError:
    yaml = None

# Base directory relative to this file: root of wheel_alignment_ai package
BASE_DIR = Path(__file__).resolve().parent.parent

# Core Directory Structure
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "validation"
TEST_DATA_DIR = DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
VIDEOS_DIR = BASE_DIR / "videos"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

DEFAULT_CONFIG_PATH = CONFIG_DIR / "settings.yaml"

# Default System Configuration Dictionary
DEFAULT_CONFIG: Dict[str, Any] = {
    "video": {
        "source": 0,
        "width": 640,
        "height": 480,
        "fps": 30.0,
        "frame_skip": 1
    },
    "preprocessing": {
        "target_width": 640,
        "target_height": 640,
        "preserve_aspect_ratio": True,
        "color_space": "BGR2RGB",
        "normalize": False
    },
    "detection": {
        "model_path": str(MODELS_DIR / "yolov8n.pt"),
        "confidence_threshold": 0.50,
        "iou_threshold": 0.45,
        "device": "cpu"
    },
    "tracking": {
        "max_disappeared": 10,
        "distance_threshold": 80.0,
        "iou_threshold": 0.30,
        "history_size": 50
    },
    "feature_engineering": {
        "window_size": 5
    },
    "prediction": {
        "model_path": str(MODELS_DIR / "alignment_classifier.joblib"),
        "scaler_path": str(MODELS_DIR / "feature_scaler.joblib"),
        "model_version": "v1.0.0_RandomForest",
        "camber_angle_limit": 3.0,
        "toe_angle_limit": 2.0
    },
    "alert": {
        "cooldown_seconds": 3.0,
        "min_confidence": 0.60,
        "consecutive_threshold": 3,
        "smoothing_window": 5,
        "enable_audio": False
    },
    "pipeline": {
        "debug_mode": False
    },
    "logging": {
        "level": "INFO",
        "log_file": "logs/app.log",
        "error_log_file": "logs/errors.log"
    }
}

logger = logging.getLogger(__name__)


def setup_logger(
    name: str = "wheel_alignment_ai",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    error_log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configures structured, multi-handler Python application logging.

    Args:
        name: Logger module name identifier.
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file: Optional path to main application log file.
        error_log_file: Optional path to dedicated error log file.

    Returns:
        Configured Logger instance.
    """
    logger_inst = logging.getLogger(name)
    target_level = getattr(logging, level.upper(), logging.INFO)
    logger_inst.setLevel(target_level)

    # Standardized Log Formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if not logger_inst.handlers:
        # 1. Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(target_level)
        console_handler.setFormatter(formatter)
        logger_inst.addHandler(console_handler)

        # 2. Main App File Handler
        main_log = log_file or LOGS_DIR / "app.log"
        main_log.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(main_log, encoding="utf-8")
        file_handler.setLevel(target_level)
        file_handler.setFormatter(formatter)
        logger_inst.addHandler(file_handler)

        # 3. Dedicated Error File Handler
        err_log = error_log_file or LOGS_DIR / "errors.log"
        err_log.parent.mkdir(parents=True, exist_ok=True)
        err_handler = logging.FileHandler(err_log, encoding="utf-8")
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(formatter)
        logger_inst.addHandler(err_handler)

    return logger_inst


def get_execution_device() -> str:
    """
    Determines hardware execution device for PyTorch and deep learning inference.
    Automatically detects CUDA GPU availability and falls back to CPU if unavailable.
    """
    env_device = os.getenv("WHEEL_AI_DEVICE", "").strip().lower()
    if env_device in ["cpu", "cuda"]:
        return env_device

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU acceleration available: {device_name} (cuda:0)")
            return "cuda"
    except Exception as e:
        logger.debug(f"PyTorch CUDA check skipped: {e}")

    logger.info("GPU acceleration unavailable or disabled. Selected execution device: CPU.")
    return "cpu"


class ConfigManager:
    """
    Singleton Configuration Manager class for managing application settings,
    YAML persistence, environment overrides, and directory structure.
    """

    _instance: Optional["ConfigManager"] = None

    def __new__(cls, config_path: Optional[Path] = None) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[Path] = None) -> None:
        if self._initialized:
            return

        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.settings: Dict[str, Any] = {}
        self.ensure_directories()
        self.load_config()
        self._initialized = True

    def ensure_directories(self) -> None:
        """Ensures all required project directories exist."""
        directories = [
            DATA_DIR, RAW_DATA_DIR, ANNOTATIONS_DIR, PROCESSED_DATA_DIR,
            TRAIN_DATA_DIR / "images", TRAIN_DATA_DIR / "labels",
            VAL_DATA_DIR / "images", VAL_DATA_DIR / "labels",
            TEST_DATA_DIR / "images", TEST_DATA_DIR / "labels",
            MODELS_DIR, VIDEOS_DIR, IMAGES_DIR, RESULTS_DIR, LOGS_DIR, CONFIG_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """
        Loads configuration parameters from YAML file with fallback to DEFAULT_CONFIG
        and environment variable overrides.
        """
        # Start with deep copy of DEFAULT_CONFIG
        import copy
        self.settings = copy.deepcopy(DEFAULT_CONFIG)

        if yaml is not None and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_settings = yaml.safe_load(f) or {}
                self._update_dict_recursive(self.settings, file_settings)
                logger.info(f"Loaded YAML configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration file {self.config_path}: {e}. Falling back to default settings.")

        # Environment variable overrides
        env_log_level = os.getenv("WHEEL_AI_LOG_LEVEL")
        if env_log_level:
            self.settings["logging"]["level"] = env_log_level.upper()

        return self.settings

    def _update_dict_recursive(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        for k, v in update.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._update_dict_recursive(base[k], v)
            else:
                base[k] = v

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a setting value using dot-notation pathing.
        Example: config.get('logging.level', 'INFO')
        """
        keys = key_path.split(".")
        val = self.settings
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key_path: str, value: Any) -> None:
        """
        Sets a setting value using dot-notation pathing.
        Example: config.set('alert.cooldown_seconds', 5.0)
        """
        keys = key_path.split(".")
        d = self.settings
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def save(self, config_path: Optional[Union[str, Path]] = None) -> Path:
        """Persists settings dictionary to YAML file."""
        target_path = Path(config_path) if config_path else self.config_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if yaml is None:
            logger.warning("PyYAML not installed. Cannot save YAML settings file.")
            return target_path

        try:
            with open(target_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.settings, f, default_flow_style=False)
            logger.info(f"Saved configuration to {target_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {target_path}: {e}")

        return target_path


# Module-level default instance
config = ConfigManager()
