"""
Image Preprocessing Pipeline Module.

Provides configurable, reproducible image transformations for training and inference.
Includes frame validation, aspect-ratio preserving resizing, ROI cropping, color space conversion,
edge-preserving noise reduction, CLAHE contrast enhancement, and normalization.

Documentation of Preprocessing Operations:
------------------------------------------
1. Frame Validation: Ensures input arrays are non-empty, uncorrupted, and valid 2D/3D matrices.
2. Aspect-Ratio Preserving Resize (Letterboxing): Prevents geometric distortion of wheel rim ellipses,
   which is critical for accurate camber and toe angle estimation.
3. Color-Space Conversion: Converts raw BGR OpenCV frames to BGR2RGB for deep learning models (YOLO/PyTorch)
   or BGR2GRAY for contour/ellipse fitting algorithms.
4. Bilateral/Gaussian Noise Reduction: Removes sensor noise in dark wheel wells while preserving sharp rim edges.
5. CLAHE Contrast Enhancement: Adapts local contrast in dark wheel arches without over-exposing highlights.
6. Normalization: Scales pixel values to [0, 1] or standardizes them for model tensor inputs.
7. ROI Extraction: Isolates specific wheel bounding box regions with optional context margins.
8. Original Frame Preservation: Retains pristine input frame for visual debugging and UI overlays.
"""

from dataclasses import dataclass, field, asdict
import logging
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """
    Configuration parameters for image preprocessing pipeline.
    Reusable across training data pipelines and runtime inference to prevent preprocessing mismatch.
    """
    target_size: Optional[Tuple[int, int]] = (640, 640)
    preserve_aspect_ratio: bool = True
    color_space: str = "BGR2RGB"  # "BGR2RGB", "BGR2GRAY", "BGR2HSV", "NONE"
    noise_reduction: Optional[str] = None  # None, "gaussian", "bilateral"
    kernel_size: int = 3
    contrast_enhancement: Optional[str] = None  # None, "clahe"
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    normalize: bool = False
    normalization_mode: str = "scale"  # "scale" ([0,1]), "z_score"
    crop_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


@dataclass
class ProcessedFrame:
    """Container holding processed output along with pristine original frame for debugging."""
    processed_image: np.ndarray
    original_image: np.ndarray
    scale_factor: float = 1.0
    padding: Tuple[int, int] = (0, 0)  # (pad_w, pad_h)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImagePreprocessor:
    """
    Production image preprocessor executing standardized computer vision transformations.
    Usable in both dataset creation / model training and real-time video inference.
    """

    def __init__(self, config: Optional[Union[PreprocessingConfig, Dict[str, Any]]] = None) -> None:
        if isinstance(config, dict):
            self.config = PreprocessingConfig.from_dict(config)
        elif isinstance(config, PreprocessingConfig):
            self.config = config
        else:
            self.config = PreprocessingConfig()

    @staticmethod
    def validate_frame(image: Any) -> bool:
        """
        Validates if an image array is non-empty, uncorrupted, and valid.
        Returns True if valid, False if empty/corrupted.
        """
        if image is None:
            logger.warning("Frame validation failed: Image object is None.")
            return False

        if not isinstance(image, np.ndarray):
            logger.warning(f"Frame validation failed: Expected numpy.ndarray, got {type(image)}.")
            return False

        if image.size == 0 or len(image.shape) < 2:
            logger.warning(f"Frame validation failed: Invalid shape {image.shape} or empty size.")
            return False

        if not np.isfinite(image).all():
            logger.warning("Frame validation failed: Image contains NaN or Inf values.")
            return False

        return True

    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        preserve_aspect_ratio: Optional[bool] = None,
        pad_color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes frame with optional aspect-ratio preservation (letterboxing).

        Rationale: Preserving wheel aspect ratio prevents false geometric distortion
        of the wheel rim, ensuring accurate camber and toe angle calculations.

        Returns:
            (resized_image, scale_factor, (pad_width, pad_height))
        """
        if not self.validate_frame(image):
            return image, 1.0, (0, 0)

        size = target_size or self.config.target_size
        keep_aspect = (
            preserve_aspect_ratio
            if preserve_aspect_ratio is not None
            else self.config.preserve_aspect_ratio
        )

        if size is None:
            return image, 1.0, (0, 0)

        target_w, target_h = size
        h, w = image.shape[:2]

        if (w, h) == (target_w, target_h):
            return image, 1.0, (0, 0)

        if not keep_aspect:
            # Direct resize (may stretch aspect ratio)
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            scale = min(target_w / w, target_h / h)
            return resized, scale, (0, 0)

        # Aspect-ratio preserving letterbox resize
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Add colored border
        if len(image.shape) == 3:
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        else:
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color[0])

        return padded, scale, (left, top)

    def convert_color_space(self, image: np.ndarray, color_space: Optional[str] = None) -> np.ndarray:
        """
        Converts image color space.

        Rationale: Models (YOLO/PyTorch) expect RGB inputs, OpenCV reads BGR by default,
        and traditional contour algorithms require Grayscale.
        """
        if not self.validate_frame(image):
            return image

        cs = color_space or self.config.color_space
        if cs is None or cs.upper() in ["NONE", "BGR"]:
            return image

        cs_upper = cs.upper()
        try:
            if cs_upper == "BGR2RGB":
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif cs_upper == "BGR2GRAY":
                if len(image.shape) == 2:
                    return image
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif cs_upper == "BGR2HSV":
                return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif cs_upper == "BGR2LAB":
                return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            elif cs_upper == "RGB2BGR":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                logger.warning(f"Unsupported color space '{cs}'. Returning original frame.")
                return image
        except Exception as e:
            logger.error(f"Color space conversion error: {e}")
            return image

    def reduce_noise(self, image: np.ndarray, method: Optional[str] = None, kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Applies edge-preserving noise reduction.

        Rationale: Bilateral filtering removes sensor noise in shadowed wheel wells
        without smudging sharp tire/rim boundary edges necessary for alignment geometry.
        """
        if not self.validate_frame(image):
            return image

        m = method or self.config.noise_reduction
        k = kernel_size or self.config.kernel_size
        if m is None:
            return image

        try:
            if m.lower() == "gaussian":
                k_val = max(1, k if k % 2 == 1 else k + 1)
                return cv2.GaussianBlur(image, (k_val, k_val), 0)
            elif m.lower() == "bilateral":
                # Bilateral filter preserves sharp edges
                return cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
            elif m.lower() == "median":
                k_val = max(1, k if k % 2 == 1 else k + 1)
                return cv2.medianBlur(image, k_val)
            else:
                return image
        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return image

    def enhance_contrast(
        self,
        image: np.ndarray,
        method: Optional[str] = None,
        clip_limit: Optional[float] = None,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Rationale: Enhances local contrast under vehicle wheel arches where shadow variance
        hides rim spoke features, avoiding over-saturation of surrounding bright areas.
        """
        if not self.validate_frame(image):
            return image

        m = method or self.config.contrast_enhancement
        if m is None:
            return image

        c_limit = clip_limit or self.config.clahe_clip_limit
        g_size = grid_size or self.config.clahe_grid_size

        try:
            if len(image.shape) == 3:
                # Apply CLAHE on L-channel in LAB space to avoid changing colors
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=c_limit, tileGridSize=g_size)
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=c_limit, tileGridSize=g_size)
                return clahe.apply(image)
        except Exception as e:
            logger.error(f"Contrast enhancement error: {e}")
            return image

    def normalize_image(
        self,
        image: np.ndarray,
        mode: Optional[str] = None,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
    ) -> np.ndarray:
        """
        Normalizes pixel range for downstream model compatibility.

        Rationale: Deep learning neural networks require inputs scaled to [0.0, 1.0] or z-score standardized.
        """
        if not self.validate_frame(image):
            return image

        m = mode or self.config.normalization_mode
        try:
            img_float = image.astype(np.float32)

            if m == "scale":
                return img_float / 255.0

            elif m == "z_score":
                m_val = np.array(mean or [0.485, 0.456, 0.406], dtype=np.float32)
                s_val = np.array(std or [0.229, 0.224, 0.225], dtype=np.float32)
                scaled = img_float / 255.0
                return (scaled - m_val) / s_val

            elif m == "min_max":
                min_val, max_val = img_float.min(), img_float.max()
                if max_val > min_val:
                    return (img_float - min_val) / (max_val - min_val)
                return img_float

            return img_float
        except Exception as e:
            logger.error(f"Normalization error: {e}")
            return image

    def crop_roi(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin_percent: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        Crops Region of Interest (ROI) with optional percentage margin expansion.

        Rationale: Isolates specific wheel bounding boxes for localized feature extraction,
        with optional margin to ensure rim edges are not clipped.
        """
        if not self.validate_frame(image) or len(bbox) != 4:
            return None

        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)

            if margin_percent > 0:
                bw, bh = x2 - x1, y2 - y1
                mw, mh = int(bw * margin_percent), int(bh * margin_percent)
                x1, y1 = x1 - mw, y1 - mh
                x2, y2 = x2 + mw, y2 + mh

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid ROI crop bounds: {(x1, y1, x2, y2)}")
                return None

            return image[y1:y2, x1:x2].copy()
        except Exception as e:
            logger.error(f"ROI crop error for bbox {bbox}: {e}")
            return None

    def preprocess(self, image: np.ndarray) -> ProcessedFrame:
        """
        Executes complete configured preprocessing pipeline, returning ProcessedFrame container
        with both pristine original frame and transformed image.
        """
        if not self.validate_frame(image):
            empty_arr = np.array([], dtype=np.uint8)
            return ProcessedFrame(processed_image=empty_arr, original_image=empty_arr)

        original = image.copy()
        working = image.copy()

        # 1. Optional Cropping
        if self.config.crop_box:
            cropped = self.crop_roi(working, self.config.crop_box)
            if cropped is not None:
                working = cropped

        # 2. Contrast Enhancement
        if self.config.contrast_enhancement:
            working = self.enhance_contrast(working)

        # 3. Noise Reduction
        if self.config.noise_reduction:
            working = self.reduce_noise(working)

        # 4. Color-Space Conversion
        if self.config.color_space:
            working = self.convert_color_space(working)

        # 5. Aspect-Ratio Preserving Resize
        scale_factor = 1.0
        padding = (0, 0)
        if self.config.target_size:
            working, scale_factor, padding = self.resize(working)

        # 6. Normalization
        if self.config.normalize:
            working = self.normalize_image(working)

        return ProcessedFrame(
            processed_image=working,
            original_image=original,
            scale_factor=scale_factor,
            padding=padding,
            metadata={"config": self.config.to_dict()}
        )
