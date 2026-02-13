"""Tests for altitude estimation module."""

import math

import numpy as np
import pytest

from onboard.altitude import (
    CameraIntrinsics,
    estimate_altitude_from_homography,
    estimate_altitude_from_scale,
)
from shared.tile_math import TileCoord


class TestCameraIntrinsics:
    def test_default_values(self):
        cam = CameraIntrinsics()
        assert cam.focal_length_mm > 0
        assert cam.sensor_width_mm > 0
        assert cam.image_width_px > 0

    def test_fov(self):
        cam = CameraIntrinsics()
        fov = cam.fov_deg
        assert 50 < fov < 80  # typical for RPi camera

    def test_gsd_per_meter(self):
        cam = CameraIntrinsics()
        gsd = cam.gsd_per_meter
        assert gsd > 0


class TestAltitudeFromHomography:
    def test_identity_homography(self):
        """Identity H means drone pixel = tile pixel → altitude depends on tile zoom."""
        cam = CameraIntrinsics(focal_length_mm=4.74, sensor_width_mm=6.287, image_width_px=640)
        tile = TileCoord(z=17, x=70000, y=43000)
        H = np.eye(3, dtype=np.float64)

        alt = estimate_altitude_from_homography(H, tile, cam, latitude=52.0)
        assert alt is not None
        # At z17, mpp ≈ 1.19m at equator, ~0.73m at 52°N
        # Identity means drone_gsd = tile_gsd
        # alt = gsd * f * W / s / 1000
        # This should give a reasonable altitude
        assert 0 < alt < 5000

    def test_scaled_homography(self):
        """Scaling H by 2x should roughly double the altitude."""
        cam = CameraIntrinsics()
        tile = TileCoord(z=17, x=70000, y=43000)

        H1 = np.eye(3, dtype=np.float64)
        H2 = np.diag([2.0, 2.0, 1.0])

        alt1 = estimate_altitude_from_homography(H1, tile, cam, latitude=52.0)
        alt2 = estimate_altitude_from_homography(H2, tile, cam, latitude=52.0)

        assert alt1 is not None and alt2 is not None
        ratio = alt2 / alt1
        assert 1.8 < ratio < 2.2, f"Scale 2x should ~double altitude, got ratio {ratio:.2f}"

    def test_negative_det_returns_none(self):
        """Negative determinant (reflection) should fail."""
        cam = CameraIntrinsics()
        tile = TileCoord(z=17, x=70000, y=43000)
        H = np.diag([-1.0, 1.0, 1.0])  # reflection
        assert estimate_altitude_from_homography(H, tile, cam) is None

    def test_zoom19_lower_altitude(self):
        """Higher zoom = finer resolution → same H should give lower altitude."""
        cam = CameraIntrinsics()
        H = np.eye(3)

        alt17 = estimate_altitude_from_homography(
            H, TileCoord(z=17, x=70000, y=43000), cam, latitude=52.0,
        )
        alt19 = estimate_altitude_from_homography(
            H, TileCoord(z=19, x=280000, y=172000), cam, latitude=52.0,
        )

        assert alt17 is not None and alt19 is not None
        assert alt17 > alt19, "z17 should give higher altitude than z19"
        ratio = alt17 / alt19
        assert 3.5 < ratio < 4.5  # 2^2 = 4x resolution difference


class TestAltitudeFromScale:
    def test_known_distance(self):
        """If 10m on ground spans 100px, altitude should be calculable."""
        cam = CameraIntrinsics(focal_length_mm=4.74, sensor_width_mm=6.287, image_width_px=640)
        alt = estimate_altitude_from_scale(
            matched_distance_px=100.0,
            true_distance_m=10.0,
            camera=cam,
        )
        assert alt > 0
        # GSD = 10/100 = 0.1 m/px
        # alt = 0.1 * 4.74 * 640 / 6.287 / 1000 ≈ 48.2m
        assert 40 < alt < 60

    def test_higher_altitude_bigger_gsd(self):
        """At higher altitude, same ground distance spans fewer pixels."""
        cam = CameraIntrinsics()
        alt_low = estimate_altitude_from_scale(100.0, 10.0, cam)
        alt_high = estimate_altitude_from_scale(50.0, 10.0, cam)  # same distance, fewer pixels
        assert alt_high > alt_low
