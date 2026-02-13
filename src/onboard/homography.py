"""Homography estimation and GPS extraction from matched features.

Given point correspondences between a drone frame and a satellite tile,
computes the geometric transform and extracts the drone's GPS position.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from shared.tile_math import GeoPoint, TileCoord, tile_pixel_to_gps


@dataclass(slots=True)
class HomographyResult:
    """Result of homography estimation."""
    H: np.ndarray               # 3x3 homography matrix
    inlier_mask: np.ndarray     # boolean mask of inlier matches
    inlier_ratio: float         # fraction of matches that are inliers
    position: GeoPoint          # estimated GPS position
    confidence: float           # overall confidence [0, 1]


def estimate_homography(
    drone_pts: np.ndarray,
    tile_pts: np.ndarray,
    ransac_threshold: float = 5.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Compute homography from drone keypoints to tile keypoints.

    Args:
        drone_pts: (N, 2) keypoint coordinates in drone image
        tile_pts: (N, 2) corresponding coordinates in satellite tile
        ransac_threshold: RANSAC reprojection error threshold in pixels
        confidence: RANSAC confidence level

    Returns:
        (H, inlier_mask, inlier_ratio) or None if estimation fails
    """
    if len(drone_pts) < 4:
        return None

    H, mask = cv2.findHomography(
        drone_pts.reshape(-1, 1, 2).astype(np.float32),
        tile_pts.reshape(-1, 1, 2).astype(np.float32),
        cv2.RANSAC,
        ransac_threshold,
        confidence=confidence,
    )

    if H is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)
    inlier_ratio = float(np.sum(mask)) / len(mask)
    return H, mask, inlier_ratio


def extract_gps(
    H: np.ndarray,
    drone_image_size: tuple[int, int],
    tile: TileCoord,
) -> GeoPoint:
    """Extract GPS position by transforming drone image center through homography.

    Args:
        H: 3x3 homography mapping drone pixels → tile pixels
        drone_image_size: (width, height) of drone image
        tile: tile coordinate of the matched satellite tile
    """
    w, h = drone_image_size
    center = np.array([[[w / 2.0, h / 2.0]]], dtype=np.float32)
    sat_point = cv2.perspectiveTransform(center, H)
    px, py = sat_point[0, 0]
    return tile_pixel_to_gps(tile, float(px), float(py))


def match_and_localize(
    drone_pts: np.ndarray,
    tile_pts: np.ndarray,
    drone_image_size: tuple[int, int],
    tile: TileCoord,
    min_inlier_ratio: float = 0.3,
) -> HomographyResult | None:
    """Full pipeline: estimate homography and extract GPS.

    Returns HomographyResult if successful, None if match quality too low.
    """
    result = estimate_homography(drone_pts, tile_pts)
    if result is None:
        return None

    H, mask, inlier_ratio = result
    if inlier_ratio < min_inlier_ratio:
        return None

    position = extract_gps(H, drone_image_size, tile)

    return HomographyResult(
        H=H,
        inlier_mask=mask,
        inlier_ratio=inlier_ratio,
        position=position,
        confidence=inlier_ratio,
    )
