"""Image preprocessing for robust feature matching.

Handles varying illumination, contrast, and exposure conditions
between drone camera frames and satellite reference imagery.
"""

from __future__ import annotations

import cv2
import numpy as np


class FramePreprocessor:
    """Preprocesses drone camera frames for better feature matching.

    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    and optional denoising to handle:
    - Shadow/sun differences between drone and satellite imagery
    - Low-contrast scenes (overcast, dawn/dusk)
    - Camera exposure variations
    """

    def __init__(
        self,
        clahe_clip: float = 3.0,
        clahe_grid: int = 8,
        denoise: bool = False,
        denoise_strength: float = 3.0,
        target_size: tuple[int, int] | None = None,
    ):
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )
        self._denoise = denoise
        self._denoise_strength = denoise_strength
        self._target_size = target_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a BGR camera frame.

        Returns a grayscale, contrast-enhanced image ready for feature extraction.
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # CLAHE for local contrast enhancement
        gray = self._clahe.apply(gray)

        # Optional denoising (expensive — only for low-light)
        if self._denoise:
            gray = cv2.fastNlMeansDenoising(
                gray, None,
                h=self._denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        # Resize if needed
        if self._target_size is not None:
            gray = cv2.resize(gray, self._target_size)

        return gray

    def process_pair(
        self, drone_frame: np.ndarray, tile_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess both drone and tile images for matching.

        Normalizes contrast between the two images for better feature matching.
        """
        drone_gray = self.process(drone_frame)
        tile_gray = self.process(tile_image)
        return drone_gray, tile_gray


def estimate_blur(image: np.ndarray) -> float:
    """Estimate image blur using Laplacian variance.

    Returns a sharpness score — lower values indicate more blur.
    Useful for rejecting frames that are too blurry for matching.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def estimate_exposure(image: np.ndarray) -> float:
    """Estimate image exposure level (0.0=black, 1.0=white).

    Returns mean brightness normalized to [0, 1].
    Values below 0.2 or above 0.8 suggest poor exposure.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(image.mean()) / 255.0
