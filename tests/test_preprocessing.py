"""Tests for image preprocessing module."""

import numpy as np
import pytest

from onboard.preprocessing import FramePreprocessor, estimate_blur, estimate_exposure


class TestFramePreprocessor:
    def test_returns_grayscale(self):
        proc = FramePreprocessor()
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = proc.process(bgr)
        assert len(result.shape) == 2
        assert result.shape == (100, 100)

    def test_accepts_grayscale_input(self):
        proc = FramePreprocessor()
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = proc.process(gray)
        assert result.shape == (100, 100)

    def test_clahe_improves_contrast(self):
        proc = FramePreprocessor(clahe_clip=3.0)
        # Low-contrast image (values 100-150)
        low_contrast = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        result = proc.process(low_contrast)
        # CLAHE should increase dynamic range
        assert result.std() > low_contrast.std()

    def test_resize(self):
        proc = FramePreprocessor(target_size=(64, 64))
        img = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = proc.process(img)
        assert result.shape == (64, 64)

    def test_process_pair(self):
        proc = FramePreprocessor()
        drone = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tile = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        d, t = proc.process_pair(drone, tile)
        assert len(d.shape) == 2
        assert len(t.shape) == 2

    def test_denoise_mode(self):
        proc = FramePreprocessor(denoise=True, denoise_strength=5.0)
        noisy = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = proc.process(noisy)
        assert result.shape == noisy.shape


class TestBlurEstimation:
    def test_sharp_image_higher_score(self):
        # Sharp = high-frequency content
        sharp = np.zeros((100, 100), dtype=np.uint8)
        sharp[::2, :] = 255  # alternating lines
        blur_score = estimate_blur(sharp)

        # Blurred version
        blurred = np.full((100, 100), 128, dtype=np.uint8)
        blur_score_blurred = estimate_blur(blurred)

        assert blur_score > blur_score_blurred

    def test_accepts_bgr(self):
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = estimate_blur(bgr)
        assert score >= 0


class TestExposureEstimation:
    def test_black_image(self):
        black = np.zeros((100, 100), dtype=np.uint8)
        assert estimate_exposure(black) == pytest.approx(0.0, abs=0.01)

    def test_white_image(self):
        white = np.full((100, 100), 255, dtype=np.uint8)
        assert estimate_exposure(white) == pytest.approx(1.0, abs=0.01)

    def test_mid_gray(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        exp = estimate_exposure(gray)
        assert 0.45 < exp < 0.55

    def test_accepts_bgr(self):
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        exp = estimate_exposure(bgr)
        assert 0.0 <= exp <= 1.0
