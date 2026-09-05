"""
Unit Tests for Image Preprocessing Subsystem.

Tests:
1. Empty and corrupted frame validation.
2. Frame resizing with aspect-ratio preservation (letterboxing).
3. Optional cropping and ROI extraction with margin.
4. Color space conversions (BGR2RGB, BGR2GRAY, BGR2HSV).
5. Noise reduction (Gaussian, Bilateral filter).
6. Contrast enhancement (CLAHE).
7. Image normalization (Scale [0, 1], Z-score).
8. Original frame preservation in ProcessedFrame container.
9. PreprocessingConfig serialization consistency.
"""

from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import ImagePreprocessor, PreprocessingConfig, ProcessedFrame


class TestImagePreprocessingPipeline(unittest.TestCase):
    """Unit test suite for ImagePreprocessor component."""

    def setUp(self) -> None:
        """Sets up test synthetic image frames."""
        self.preprocessor = ImagePreprocessor()
        # Create synthetic 400x300 BGR color frame
        self.sample_bgr = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(self.sample_bgr, (50, 50), (250, 200), (0, 255, 0), -1)
        cv2.circle(self.sample_bgr, (200, 150), 40, (255, 0, 0), -1)

    def test_frame_validation(self) -> None:
        """Tests validation of empty, corrupted, and valid frames."""
        self.assertFalse(ImagePreprocessor.validate_frame(None))
        self.assertFalse(ImagePreprocessor.validate_frame(np.array([])))
        self.assertFalse(ImagePreprocessor.validate_frame("invalid_string_input"))

        # Corrupted frame with NaN
        nan_frame = np.zeros((100, 100, 3), dtype=np.float32)
        nan_frame[10, 10, 0] = np.nan
        self.assertFalse(ImagePreprocessor.validate_frame(nan_frame))

        # Valid frame
        self.assertTrue(ImagePreprocessor.validate_frame(self.sample_bgr))

    def test_aspect_ratio_preserving_resize(self) -> None:
        """Tests letterbox resize with aspect-ratio preservation."""
        config = PreprocessingConfig(target_size=(640, 640), preserve_aspect_ratio=True)
        prep = ImagePreprocessor(config)

        # Input: 400x300 (aspect ratio 4:3)
        resized, scale, (pad_w, pad_h) = prep.resize(self.sample_bgr)

        self.assertEqual(resized.shape, (640, 640, 3))
        self.assertGreater(pad_h, 0)  # Vertical padding added due to 4:3 input into 1:1 box
        self.assertGreater(scale, 0.0)

    def test_direct_resize(self) -> None:
        """Tests standard resize without aspect-ratio letterboxing."""
        config = PreprocessingConfig(target_size=(640, 640), preserve_aspect_ratio=False)
        prep = ImagePreprocessor(config)

        resized, scale, (pad_w, pad_h) = prep.resize(self.sample_bgr)
        self.assertEqual(resized.shape, (640, 640, 3))
        self.assertEqual((pad_w, pad_h), (0, 0))

    def test_cropping(self) -> None:
        """Tests region cropping using bounding box."""
        crop_box = (50, 50, 250, 200)
        cropped = self.preprocessor.crop_roi(self.sample_bgr, bbox=crop_box)

        self.assertIsNotNone(cropped)
        self.assertEqual(cropped.shape, (150, 200, 3))  # Height: 200-50=150, Width: 250-50=200

    def test_roi_extraction_with_margin(self) -> None:
        """Tests ROI crop with percentage margin expansion."""
        bbox = (100, 100, 200, 200)
        cropped_normal = self.preprocessor.crop_roi(self.sample_bgr, bbox=bbox, margin_percent=0.0)
        cropped_margin = self.preprocessor.crop_roi(self.sample_bgr, bbox=bbox, margin_percent=0.2)

        self.assertIsNotNone(cropped_normal)
        self.assertIsNotNone(cropped_margin)
        self.assertGreater(cropped_margin.shape[0], cropped_normal.shape[0])
        self.assertGreater(cropped_margin.shape[1], cropped_normal.shape[1])

    def test_color_space_conversion(self) -> None:
        """Tests color space conversions."""
        rgb = self.preprocessor.convert_color_space(self.sample_bgr, "BGR2RGB")
        self.assertEqual(rgb.shape, self.sample_bgr.shape)

        gray = self.preprocessor.convert_color_space(self.sample_bgr, "BGR2GRAY")
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (300, 400))

        hsv = self.preprocessor.convert_color_space(self.sample_bgr, "BGR2HSV")
        self.assertEqual(hsv.shape, self.sample_bgr.shape)

    def test_noise_reduction(self) -> None:
        """Tests Gaussian and Bilateral noise reduction."""
        gaussian = self.preprocessor.reduce_noise(self.sample_bgr, method="gaussian", kernel_size=3)
        self.assertEqual(gaussian.shape, self.sample_bgr.shape)

        bilateral = self.preprocessor.reduce_noise(self.sample_bgr, method="bilateral")
        self.assertEqual(bilateral.shape, self.sample_bgr.shape)

    def test_contrast_enhancement(self) -> None:
        """Tests CLAHE contrast enhancement."""
        clahe_out = self.preprocessor.enhance_contrast(self.sample_bgr, method="clahe")
        self.assertEqual(clahe_out.shape, self.sample_bgr.shape)

        gray = cv2.cvtColor(self.sample_bgr, cv2.COLOR_BGR2GRAY)
        clahe_gray = self.preprocessor.enhance_contrast(gray, method="clahe")
        self.assertEqual(clahe_gray.shape, gray.shape)

    def test_normalization(self) -> None:
        """Tests scaling and z-score normalization."""
        # Scale mode [0.0, 1.0]
        scaled = self.preprocessor.normalize_image(self.sample_bgr, mode="scale")
        self.assertEqual(scaled.dtype, np.float32)
        self.assertLessEqual(scaled.max(), 1.0)
        self.assertGreaterEqual(scaled.min(), 0.0)

        # Z-score mode
        z_scored = self.preprocessor.normalize_image(self.sample_bgr, mode="z_score")
        self.assertEqual(z_scored.dtype, np.float32)

    def test_original_frame_preservation(self) -> None:
        """Tests that ProcessedFrame retains pristine original image for debugging."""
        config = PreprocessingConfig(
            target_size=(640, 640),
            color_space="BGR2GRAY",
            normalize=True
        )
        prep = ImagePreprocessor(config)
        result: ProcessedFrame = prep.preprocess(self.sample_bgr)

        self.assertIsInstance(result, ProcessedFrame)

        # Processed image is grayscale & normalized
        self.assertEqual(len(result.processed_image.shape), 2)

        # Original image remains BGR 3D uint8 array
        self.assertEqual(result.original_image.shape, (300, 400, 3))
        self.assertEqual(result.original_image.dtype, np.uint8)
        np.testing.assert_array_equal(result.original_image, self.sample_bgr)

    def test_config_serialization(self) -> None:
        """Tests PreprocessingConfig serialization consistency between training and inference."""
        original_config = PreprocessingConfig(
            target_size=(512, 512),
            preserve_aspect_ratio=True,
            color_space="BGR2RGB",
            contrast_enhancement="clahe",
            normalize=True
        )
        dict_repr = original_config.to_dict()
        reconstructed = PreprocessingConfig.from_dict(dict_repr)

        self.assertEqual(original_config.target_size, reconstructed.target_size)
        self.assertEqual(original_config.color_space, reconstructed.color_space)
        self.assertEqual(original_config.contrast_enhancement, reconstructed.contrast_enhancement)
        self.assertEqual(original_config.normalize, reconstructed.normalize)


if __name__ == "__main__":
    unittest.main()
