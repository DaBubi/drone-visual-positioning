"""End-to-end pipeline test: EKF + matching + geofence + telemetry.

Tests the full positioning pipeline with synthetic data to verify
all components integrate correctly.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from onboard.altitude import CameraIntrinsics, estimate_altitude_from_homography
from onboard.ekf import EKFConfig, PositionEKF
from onboard.geofence import CircleGeofence, GeofenceChecker
from onboard.health import HealthMonitor
from onboard.homography import estimate_homography, extract_gps, match_and_localize
from onboard.matcher import OrbMatcher
from onboard.msp import MSPGPSData, encode_set_raw_gps, msp_checksum
from onboard.nmea import PositionFix, format_gga, format_rmc, nmea_checksum
from onboard.preprocessing import FramePreprocessor, estimate_blur
from onboard.telemetry import FrameRecord, TelemetryLogger
from onboard.tile_cache import TileCache
from programmer.simulate import generate_circle, generate_line
from shared.tile_math import GeoPoint, TileCoord, haversine_km, gps_to_tile


class TestFullPipeline:
    """Test the complete VPS pipeline with synthetic data."""

    def test_ekf_with_noisy_circle_trajectory(self):
        """Simulate a circular flight with noisy position fixes."""
        center = GeoPoint(52.5200, 13.4050)
        ekf = PositionEKF(EKFConfig(gate_threshold=25.0))
        fence = CircleGeofence(center=center, radius_km=0.5)
        checker = GeofenceChecker(fence)
        health = HealthMonitor(window_size=50, min_fix_rate=0.5)

        trajectory = generate_circle(center, radius_m=200, speed_mps=10, duration=30, dt=0.3)
        rng = np.random.RandomState(42)

        errors = []
        for i, pt in enumerate(trajectory):
            t = pt.t
            true_pos = GeoPoint(pt.lat, pt.lon)

            # Simulate noisy position fix (80% fix rate)
            if rng.random() > 0.2:
                noise_lat = rng.normal(0, 1e-5)  # ~1m noise
                noise_lon = rng.normal(0, 1e-5)
                measured = GeoPoint(pt.lat + noise_lat, pt.lon + noise_lon)
                hdop = 1.0 + rng.random()

                accepted = ekf.update(measured, hdop, t)
                geofence_ok = checker.check(measured)
                health.record_frame(fix=True, latency_ms=80.0,
                                    ekf_accepted=accepted, geofence_ok=geofence_ok)
            else:
                health.record_frame(fix=False, latency_ms=80.0)

            if ekf.state.initialized:
                err = haversine_km(true_pos, ekf.position) * 1000
                errors.append(err)

        # EKF should track the circle with good accuracy
        late_errors = errors[len(errors) // 2:]
        mean_err = np.mean(late_errors)
        assert mean_err < 10.0, f"Mean tracking error {mean_err:.1f}m"

        # Health should be good
        status = health.status
        assert status.healthy
        assert status.fix_rate > 0.5

        # Geofence should have no violations (circle is within 0.5km)
        assert checker.consecutive_violations == 0

    def test_nmea_and_msp_output_consistency(self):
        """Both NMEA and MSP should encode the same position."""
        pos = GeoPoint(52.5200, 13.4050)

        # NMEA
        fix = PositionFix(lat=pos.lat, lon=pos.lon, hdop=1.2, speed_knots=5.0)
        gga = format_gga(fix)
        rmc = format_rmc(fix)
        assert gga.startswith("$GPGGA")
        assert rmc.startswith("$GPRMC")

        # MSP
        msp_data = MSPGPSData.from_position(
            lat=pos.lat, lon=pos.lon, speed_mps=5.0 / 1.94384, hdop=1.2,
        )
        frame = encode_set_raw_gps(msp_data)
        assert frame[:3] == b"$M<"

        # Both should encode roughly the same position
        assert msp_data.lat == 525200000  # 52.52 * 1e7
        assert msp_data.lon == 134050000

    def test_preprocessing_with_synthetic_image(self):
        """Preprocessing should handle various image conditions."""
        proc = FramePreprocessor(clahe_clip=3.0)

        # Create a synthetic low-contrast image
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        # Add some texture
        for i in range(0, 256, 16):
            img[i:i+8, :] = 140

        processed = proc.process(img)
        assert processed.shape == (256, 256)
        # CLAHE should increase contrast
        assert processed.std() > img[:, :, 0].std()

        blur = estimate_blur(processed)
        assert blur > 0

    def test_tile_cache_with_real_files(self, tmp_path):
        """Cache should serve tile images efficiently."""
        cache = TileCache(max_tiles=5)

        # Create test tiles
        for i in range(3):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"tile_{i}.png"), img)

        # First load (all misses)
        for i in range(3):
            img = cache.get(tmp_path / f"tile_{i}.png")
            assert img is not None

        # Second load (all hits)
        for i in range(3):
            img = cache.get(tmp_path / f"tile_{i}.png")
            assert img is not None

        assert cache.hit_rate == pytest.approx(0.5)  # 3 hits / 6 total

    def test_telemetry_records_full_pipeline(self, tmp_path):
        """Telemetry should record all frame data."""
        telemetry = TelemetryLogger(tmp_path, prefix="pipeline")
        path = telemetry.start()

        for i in range(10):
            rec = FrameRecord(
                timestamp=float(i) * 0.3,
                frame_num=i,
                fix=i % 3 != 0,
                lat=52.52 + i * 0.0001 if i % 3 != 0 else 0.0,
                lon=13.405 if i % 3 != 0 else 0.0,
                hdop=1.5,
                inlier_ratio=0.8,
                num_matches=42,
                tile_z=17,
                tile_x=70406,
                tile_y=42987,
                retrieval_ms=5.0,
                match_ms=75.0,
                total_ms=85.0,
                ekf_lat=52.52 + i * 0.0001,
                ekf_lon=13.405,
                ekf_vlat=0.0001,
                ekf_vlon=0.0,
                ekf_speed_mps=11.0,
                ekf_gate=1.5,
                ekf_accepted=True,
            )
            telemetry.log(rec)

        telemetry.stop()

        # Verify file contents
        import csv
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 11  # header + 10 records

    def test_altitude_from_known_setup(self):
        """Altitude estimation with known parameters."""
        cam = CameraIntrinsics(
            focal_length_mm=4.74,
            sensor_width_mm=6.287,
            image_width_px=640,
        )

        # A 2x zoom homography at z17 should give ~2x the altitude
        H1 = np.eye(3)
        H2 = np.diag([2.0, 2.0, 1.0])
        tile = TileCoord(z=17, x=70000, y=43000)

        alt1 = estimate_altitude_from_homography(H1, tile, cam, latitude=52.0)
        alt2 = estimate_altitude_from_homography(H2, tile, cam, latitude=52.0)

        assert alt1 is not None and alt2 is not None
        assert alt1 > 100  # should be ~355m for z17 at 52°N
        assert alt2 > alt1  # 2x scale = higher altitude


class TestHomographyPipeline:
    """Test homography estimation with synthetic point correspondences."""

    def test_identity_homography(self):
        """Matching points should give identity homography."""
        pts = np.array([[50, 50], [200, 50], [200, 200], [50, 200],
                        [128, 128], [100, 200]], dtype=np.float32)
        result = estimate_homography(pts, pts)
        assert result is not None
        H, mask, ratio = result
        assert ratio > 0.9
        # H should be close to identity
        assert np.allclose(H, np.eye(3), atol=0.1)

    def test_translated_homography(self):
        """Translated points should give translation homography."""
        drone_pts = np.array([[50, 50], [200, 50], [200, 200], [50, 200],
                             [128, 128], [100, 200]], dtype=np.float32)
        # Shift by (10, 20)
        tile_pts = drone_pts + np.array([10, 20], dtype=np.float32)

        result = estimate_homography(drone_pts, tile_pts)
        assert result is not None
        H, mask, ratio = result
        assert ratio > 0.9

        # Transform drone center through H
        center = np.array([[[128.0, 128.0]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(center, H)
        expected = np.array([138.0, 148.0])
        assert np.allclose(transformed[0, 0], expected, atol=1.0)

    def test_extract_gps_from_tile(self):
        """GPS extraction from tile coordinates."""
        tile = TileCoord(z=17, x=70406, y=42987)
        H = np.eye(3, dtype=np.float64)
        pos = extract_gps(H, (256, 256), tile)
        # Should be somewhere in Berlin
        assert 52.0 < pos.lat < 53.0
        assert 13.0 < pos.lon < 14.0
