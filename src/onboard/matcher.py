"""Feature matching via SuperPoint + LightGlue (ONNX) or ORB fallback.

Extracts keypoints and descriptors from drone and tile images,
then matches them to produce point correspondences for homography.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MatchResult:
    """Matched keypoint pairs between drone and tile images."""
    drone_pts: np.ndarray   # (N, 2) keypoints in drone image
    tile_pts: np.ndarray    # (N, 2) keypoints in tile image
    scores: np.ndarray      # (N,) match confidence scores
    num_matches: int


class OnnxMatcher:
    """SuperPoint + LightGlue via ONNX Runtime."""

    def __init__(self, superpoint_path: Path, lightglue_path: Path):
        self._sp_path = superpoint_path
        self._lg_path = lightglue_path
        self._sp_session = None
        self._lg_session = None

    def load(self) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._sp_session = ort.InferenceSession(str(self._sp_path), opts)
        self._lg_session = ort.InferenceSession(str(self._lg_path), opts)
        logger.info("ONNX models loaded: SuperPoint + LightGlue")

    def _extract_features(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract keypoints and descriptors from a grayscale image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Normalize to [0, 1] float32, add batch + channel dims
        inp = gray.astype(np.float32) / 255.0
        inp = inp[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

        outputs = self._sp_session.run(None, {"image": inp})
        keypoints = outputs[0][0]     # (N, 2)
        descriptors = outputs[1][0]   # (N, D)
        return keypoints, descriptors

    def match(self, drone_image: np.ndarray, tile_image: np.ndarray) -> MatchResult:
        """Match features between drone and tile images."""
        kp0, desc0 = self._extract_features(drone_image)
        kp1, desc1 = self._extract_features(tile_image)

        # LightGlue matching
        outputs = self._lg_session.run(None, {
            "kpts0": kp0[np.newaxis].astype(np.float32),
            "kpts1": kp1[np.newaxis].astype(np.float32),
            "desc0": desc0[np.newaxis].astype(np.float32),
            "desc1": desc1[np.newaxis].astype(np.float32),
        })

        matches = outputs[0][0]   # (M, 2) index pairs
        scores = outputs[1][0]    # (M,) confidence

        drone_pts = kp0[matches[:, 0]]
        tile_pts = kp1[matches[:, 1]]

        return MatchResult(
            drone_pts=drone_pts,
            tile_pts=tile_pts,
            scores=scores,
            num_matches=len(matches),
        )

    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Extract a global descriptor by average-pooling SuperPoint descriptors."""
        _, descriptors = self._extract_features(image)
        if len(descriptors) == 0:
            return np.zeros(256, dtype=np.float32)
        return descriptors.mean(axis=0).astype(np.float32)


class OrbMatcher:
    """Fallback matcher using OpenCV ORB (no neural network needed)."""

    def __init__(self, max_features: int = 1000):
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def load(self) -> None:
        pass  # ORB is ready immediately

    def match(self, drone_image: np.ndarray, tile_image: np.ndarray) -> MatchResult:
        """Match ORB features between drone and tile images."""
        gray0 = cv2.cvtColor(drone_image, cv2.COLOR_BGR2GRAY) if len(drone_image.shape) == 3 else drone_image
        gray1 = cv2.cvtColor(tile_image, cv2.COLOR_BGR2GRAY) if len(tile_image.shape) == 3 else tile_image

        kp0, desc0 = self._orb.detectAndCompute(gray0, None)
        kp1, desc1 = self._orb.detectAndCompute(gray1, None)

        if desc0 is None or desc1 is None or len(kp0) < 4 or len(kp1) < 4:
            return MatchResult(
                drone_pts=np.empty((0, 2)),
                tile_pts=np.empty((0, 2)),
                scores=np.empty(0),
                num_matches=0,
            )

        # Lowe's ratio test
        matches_knn = self._bf.knnMatch(desc0, desc1, k=2)
        good = []
        for pair in matches_knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if not good:
            return MatchResult(
                drone_pts=np.empty((0, 2)),
                tile_pts=np.empty((0, 2)),
                scores=np.empty(0),
                num_matches=0,
            )

        drone_pts = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float32)
        tile_pts = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32)
        scores = np.array([1.0 - m.distance / 256.0 for m in good], dtype=np.float32)

        return MatchResult(
            drone_pts=drone_pts,
            tile_pts=tile_pts,
            scores=scores,
            num_matches=len(good),
        )

    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Rough global descriptor from ORB — mean of binary descriptors cast to float."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, desc = self._orb.detectAndCompute(gray, None)
        if desc is None or len(desc) == 0:
            return np.zeros(32, dtype=np.float32)
        return desc.astype(np.float32).mean(axis=0)
