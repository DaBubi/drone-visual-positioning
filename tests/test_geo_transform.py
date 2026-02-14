"""Tests for coordinate transformations."""

import math

import numpy as np
import pytest

from onboard.geo_transform import (
    PixelCoord, tile_pixel_to_gps, gps_to_tile_pixel,
    homography_to_gps, meters_per_pixel, pixel_distance_to_meters,
    TILE_SIZE,
)
from shared.tile_math import GeoPoint, TileCoord, haversine_km


class TestTilePixelToGps:
    def test_tile_origin(self):
        # Top-left corner of tile (0,0) at zoom 0
        gps = tile_pixel_to_gps(TileCoord(0, 0, 0), PixelCoord(0, 0))
        assert gps.lat == pytest.approx(85.05, abs=0.1)  # max Mercator lat
        assert gps.lon == pytest.approx(-180.0)

    def test_tile_center(self):
        # Center of tile (0,0) at zoom 0 should be roughly 0,0 is not right
        # at zoom 1: tile (1,1) center should be near 0,0
        gps = tile_pixel_to_gps(TileCoord(1, 1, 1), PixelCoord(0, 0))
        assert gps.lat == pytest.approx(0.0, abs=1.0)
        assert gps.lon == pytest.approx(0.0, abs=1.0)

    def test_known_location(self):
        # Berlin: approximately tile (17, 70406, 42987)
        gps = tile_pixel_to_gps(TileCoord(17, 70406, 42987), PixelCoord(128, 128))
        assert 52.0 < gps.lat < 53.0
        assert 13.0 < gps.lon < 14.0


class TestGpsToTilePixel:
    def test_roundtrip(self):
        original = GeoPoint(lat=52.52, lon=13.405)
        tile, pixel = gps_to_tile_pixel(original, zoom=17)
        recovered = tile_pixel_to_gps(tile, pixel)
        # Should be very close
        dist = haversine_km(original, recovered) * 1000
        assert dist < 1.0, f"Roundtrip error: {dist:.2f}m"

    def test_roundtrip_southern_hemisphere(self):
        original = GeoPoint(lat=-33.87, lon=151.21)
        tile, pixel = gps_to_tile_pixel(original, zoom=17)
        recovered = tile_pixel_to_gps(tile, pixel)
        dist = haversine_km(original, recovered) * 1000
        assert dist < 1.0

    def test_pixel_in_range(self):
        _, pixel = gps_to_tile_pixel(GeoPoint(52.52, 13.405), zoom=17)
        assert 0 <= pixel.x < TILE_SIZE
        assert 0 <= pixel.y < TILE_SIZE

    def test_zoom_19(self):
        tile, pixel = gps_to_tile_pixel(GeoPoint(52.52, 13.405), zoom=19)
        assert tile.z == 19
        recovered = tile_pixel_to_gps(tile, pixel)
        dist = haversine_km(GeoPoint(52.52, 13.405), recovered) * 1000
        assert dist < 0.5


class TestHomographyToGps:
    def test_identity_maps_center(self):
        H = np.eye(3)
        tile = TileCoord(17, 70406, 42987)
        gps = homography_to_gps(H, tile, image_center=(128, 128))
        # Identity H maps center to tile center pixel
        tile_center = tile_pixel_to_gps(tile, PixelCoord(128, 128))
        dist = haversine_km(gps, tile_center) * 1000
        assert dist < 0.1

    def test_translation(self):
        # Shift 10 pixels right
        H = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=float)
        tile = TileCoord(17, 70406, 42987)
        gps1 = homography_to_gps(H, tile, image_center=(128, 128))
        gps2 = homography_to_gps(np.eye(3), tile, image_center=(128, 128))
        # Shifted GPS should be east of original
        assert gps1.lon > gps2.lon

    def test_degenerate_returns_zero(self):
        H = np.zeros((3, 3))
        tile = TileCoord(17, 70406, 42987)
        gps = homography_to_gps(H, tile)
        assert gps.lat == 0.0
        assert gps.lon == 0.0


class TestMetersPerPixel:
    def test_equator_z17(self):
        mpp = meters_per_pixel(0.0, 17)
        assert 1.0 < mpp < 1.5  # ~1.19m at equator z17

    def test_higher_lat_smaller_mpp(self):
        mpp_eq = meters_per_pixel(0.0, 17)
        mpp_52 = meters_per_pixel(52.0, 17)
        assert mpp_52 < mpp_eq  # higher latitude = less ground per pixel

    def test_higher_zoom_smaller_mpp(self):
        mpp17 = meters_per_pixel(52.0, 17)
        mpp19 = meters_per_pixel(52.0, 19)
        assert mpp19 < mpp17
        assert mpp19 == pytest.approx(mpp17 / 4, rel=0.01)

    def test_z19_submeter(self):
        mpp = meters_per_pixel(52.0, 19)
        assert mpp < 0.5


class TestPixelDistanceToMeters:
    def test_zero_distance(self):
        d = pixel_distance_to_meters(0, 0, 52.0, 17)
        assert d == 0.0

    def test_10_pixels(self):
        d = pixel_distance_to_meters(10, 0, 52.0, 17)
        mpp = meters_per_pixel(52.0, 17)
        assert d == pytest.approx(10 * mpp, rel=0.01)

    def test_diagonal(self):
        d = pixel_distance_to_meters(3, 4, 52.0, 17)
        mpp = meters_per_pixel(52.0, 17)
        assert d == pytest.approx(5 * mpp, rel=0.01)
