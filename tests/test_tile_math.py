"""Tests for GPS ↔ tile ↔ pixel coordinate conversions."""

import math

import pytest

from shared.tile_math import (
    GeoPoint,
    TileCoord,
    TilePixel,
    gps_to_tile,
    gps_to_tile_pixel,
    haversine_km,
    tile_center_gps,
    tile_pixel_to_gps,
    tiles_in_radius,
)


class TestGpsToTile:
    def test_known_location_seattle(self):
        """Seattle Space Needle at zoom 17."""
        pt = GeoPoint(lat=47.6205, lon=-122.3493)
        tile = gps_to_tile(pt, 17)
        assert tile.z == 17
        # Known tile coords for this location at z17
        assert 20900 <= tile.x <= 21100
        assert 45700 <= tile.y <= 45800

    def test_origin(self):
        """Null Island (0, 0) should be in the center tile."""
        pt = GeoPoint(lat=0.0, lon=0.0)
        tile = gps_to_tile(pt, 1)
        assert tile == TileCoord(z=1, x=1, y=1)

    def test_zoom_0(self):
        """Zoom 0 has only one tile."""
        pt = GeoPoint(lat=45.0, lon=90.0)
        tile = gps_to_tile(pt, 0)
        assert tile == TileCoord(z=0, x=0, y=0)


class TestRoundTrip:
    """GPS → tile + pixel → GPS should be identity (within precision)."""

    @pytest.mark.parametrize("lat,lon,zoom", [
        (47.6205, -122.3493, 17),
        (0.0, 0.0, 15),
        (51.5074, -0.1278, 19),    # London
        (-33.8688, 151.2093, 17),  # Sydney
        (35.6762, 139.6503, 17),   # Tokyo
    ])
    def test_round_trip(self, lat, lon, zoom):
        pt = GeoPoint(lat=lat, lon=lon)
        tp = gps_to_tile_pixel(pt, zoom)
        recovered = tile_pixel_to_gps(tp.tile, tp.px, tp.py)

        # At zoom 17, 1 pixel ≈ 1.19m, so tolerance of ~0.0001° ≈ 11m
        assert abs(recovered.lat - pt.lat) < 0.0001, f"lat: {recovered.lat} vs {pt.lat}"
        assert abs(recovered.lon - pt.lon) < 0.0001, f"lon: {recovered.lon} vs {pt.lon}"


class TestTileCenterGps:
    def test_returns_geopoint(self):
        tile = TileCoord(z=17, x=20850, y=45850)
        center = tile_center_gps(tile)
        assert isinstance(center, GeoPoint)
        assert -90 <= center.lat <= 90
        assert -180 <= center.lon <= 180


class TestHaversine:
    def test_same_point(self):
        pt = GeoPoint(lat=47.6, lon=-122.3)
        assert haversine_km(pt, pt) == pytest.approx(0.0, abs=1e-10)

    def test_known_distance(self):
        # Seattle to Portland ≈ 233 km
        seattle = GeoPoint(lat=47.6062, lon=-122.3321)
        portland = GeoPoint(lat=45.5155, lon=-122.6789)
        dist = haversine_km(seattle, portland)
        assert 230 < dist < 240

    def test_antipodal(self):
        # Half earth circumference ≈ 20015 km
        a = GeoPoint(lat=0.0, lon=0.0)
        b = GeoPoint(lat=0.0, lon=180.0)
        dist = haversine_km(a, b)
        assert 20000 < dist < 20100


class TestTilesInRadius:
    def test_small_area(self):
        center = GeoPoint(lat=47.6062, lon=-122.3321)
        tiles = tiles_in_radius(center, 1.0, 17)
        # 1km radius at zoom 17 should give a manageable number of tiles
        assert 1 < len(tiles) < 150
        # All tiles should be zoom 17
        assert all(t.z == 17 for t in tiles)

    def test_larger_area(self):
        center = GeoPoint(lat=47.6062, lon=-122.3321)
        tiles_1km = tiles_in_radius(center, 1.0, 17)
        tiles_5km = tiles_in_radius(center, 5.0, 17)
        assert len(tiles_5km) > len(tiles_1km)


class TestMetersPerPixel:
    def test_zoom_17(self):
        tile = TileCoord(z=17, x=0, y=0)
        # At equator, zoom 17 ≈ 1.19 m/pixel
        assert 1.0 < tile.meters_per_pixel < 1.3

    def test_zoom_19(self):
        tile = TileCoord(z=19, x=0, y=0)
        assert 0.2 < tile.meters_per_pixel < 0.4
