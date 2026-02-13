"""Altitude estimation from homography scale factor.

When the drone camera FOV is known, the scale difference between
the drone frame and the satellite tile reveals the drone's altitude.

Ground sampling distance (GSD) = altitude * sensor_width / (focal_length * image_width)
Tile GSD = meters_per_pixel at the tile's zoom level

Scale = drone_GSD / tile_GSD = (H determinant)^0.5 approximately

Therefore: altitude = scale * tile_GSD * focal_length * image_width / sensor_width
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shared.tile_math import TileCoord


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Camera lens/sensor parameters for altitude estimation.

    These can be calibrated or looked up from the camera spec sheet.
    """
    focal_length_mm: float = 4.74   # typical RPi Camera Module 3
    sensor_width_mm: float = 6.287  # RPi Camera Module 3 sensor
    image_width_px: int = 640       # capture resolution width

    @property
    def fov_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return 2 * math.degrees(math.atan(self.sensor_width_mm / (2 * self.focal_length_mm)))

    @property
    def gsd_per_meter(self) -> float:
        """Ground sampling distance in pixels per meter at 1m altitude."""
        return self.focal_length_mm * self.image_width_px / self.sensor_width_mm / 1000.0


def estimate_altitude_from_homography(
    H: np.ndarray,
    tile: TileCoord,
    camera: CameraIntrinsics,
    latitude: float = 0.0,
) -> float | None:
    """Estimate drone altitude from the homography matrix.

    The homography H maps drone pixels → tile pixels.
    The scale factor relates the two image resolutions:
      scale = sqrt(|det(H)|)

    Args:
        H: 3x3 homography matrix (drone → tile)
        tile: matched satellite tile coordinate
        camera: camera intrinsics
        latitude: drone latitude (for GSD latitude correction)

    Returns:
        Estimated altitude in meters, or None if estimation fails
    """
    det = np.linalg.det(H)
    if det <= 0:
        return None

    # Scale factor: how many tile pixels per drone pixel
    scale = math.sqrt(abs(det))

    # Tile ground sampling distance at this latitude
    # meters_per_pixel at equator, corrected for latitude
    mpp_equator = tile.meters_per_pixel
    lat_factor = math.cos(math.radians(latitude)) if latitude != 0 else 1.0
    tile_gsd = mpp_equator * lat_factor  # meters per tile pixel

    # Drone GSD = scale * tile_gsd (meters per drone pixel)
    drone_gsd = scale * tile_gsd

    # GSD = alt * sensor_width / (focal_length * image_width)
    # So: alt = GSD * focal_length * image_width / sensor_width
    # focal_length and sensor_width are both in mm, so they cancel units
    altitude = drone_gsd * camera.focal_length_mm * camera.image_width_px / camera.sensor_width_mm

    # Sanity check
    if altitude < 1.0 or altitude > 10000.0:
        return None

    return altitude


def estimate_altitude_from_scale(
    matched_distance_px: float,
    true_distance_m: float,
    camera: CameraIntrinsics,
) -> float:
    """Estimate altitude from a known ground distance.

    If we know the real-world distance between two matched points,
    and their pixel distance in the drone image, we can compute altitude.

    Args:
        matched_distance_px: distance between points in drone image (pixels)
        true_distance_m: real-world distance between those points (meters)
        camera: camera intrinsics

    Returns:
        Estimated altitude in meters
    """
    # GSD = true_distance / pixel_distance (meters per pixel)
    gsd = true_distance_m / matched_distance_px

    # altitude = GSD * focal_length * image_width / sensor_width
    # focal_length and sensor_width both in mm, units cancel
    altitude = gsd * camera.focal_length_mm * camera.image_width_px / camera.sensor_width_mm

    return altitude
