"""Tests for geofence safety module."""

import pytest

from onboard.geofence import CircleGeofence, RectGeofence, GeofenceChecker
from shared.tile_math import GeoPoint


class TestCircleGeofence:
    def test_center_is_inside(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        assert fence.contains(GeoPoint(52.52, 13.405))

    def test_nearby_point_inside(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        # ~500m north
        assert fence.contains(GeoPoint(52.5245, 13.405))

    def test_far_point_outside(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        # ~5km north
        assert not fence.contains(GeoPoint(52.565, 13.405))

    def test_boundary_with_margin(self):
        fence = CircleGeofence(
            center=GeoPoint(52.52, 13.405),
            radius_km=1.0,
            margin_km=0.5,
        )
        # Point ~1.2km away should be inside (within margin)
        assert fence.contains(GeoPoint(52.531, 13.405))

    def test_distance_to_boundary_inside(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        dist = fence.distance_to_boundary(GeoPoint(52.52, 13.405))
        assert dist > 0  # inside

    def test_distance_to_boundary_outside(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        dist = fence.distance_to_boundary(GeoPoint(52.565, 13.405))
        assert dist < 0  # outside


class TestRectGeofence:
    def test_center_inside(self):
        fence = RectGeofence(
            nw=GeoPoint(52.53, 13.39),
            se=GeoPoint(52.51, 13.42),
        )
        assert fence.contains(GeoPoint(52.52, 13.405))

    def test_outside_north(self):
        fence = RectGeofence(
            nw=GeoPoint(52.53, 13.39),
            se=GeoPoint(52.51, 13.42),
        )
        assert not fence.contains(GeoPoint(52.54, 13.405))

    def test_outside_east(self):
        fence = RectGeofence(
            nw=GeoPoint(52.53, 13.39),
            se=GeoPoint(52.51, 13.42),
        )
        assert not fence.contains(GeoPoint(52.52, 13.43))

    def test_corner_inside(self):
        fence = RectGeofence(
            nw=GeoPoint(52.53, 13.39),
            se=GeoPoint(52.51, 13.42),
        )
        assert fence.contains(GeoPoint(52.53, 13.39))  # NW corner
        assert fence.contains(GeoPoint(52.51, 13.42))  # SE corner


class TestGeofenceChecker:
    def test_valid_position(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        checker = GeofenceChecker(fence)
        assert checker.check(GeoPoint(52.52, 13.405))
        assert checker.consecutive_violations == 0

    def test_invalid_position(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        checker = GeofenceChecker(fence)
        assert not checker.check(GeoPoint(53.0, 13.405))  # way outside
        assert checker.consecutive_violations == 1

    def test_breach_after_max_violations(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=0.1)
        checker = GeofenceChecker(fence, max_violations=3)

        outside = GeoPoint(53.0, 13.405)
        checker.check(outside)
        checker.check(outside)
        assert not checker.is_breached
        checker.check(outside)
        assert checker.is_breached

    def test_reset_after_valid(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        checker = GeofenceChecker(fence, max_violations=5)

        # Some violations
        checker.check(GeoPoint(53.0, 13.405))
        checker.check(GeoPoint(53.0, 13.405))
        assert checker.consecutive_violations == 2

        # Valid fix resets counter
        checker.check(GeoPoint(52.52, 13.405))
        assert checker.consecutive_violations == 0
        assert not checker.is_breached

    def test_violation_rate(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        checker = GeofenceChecker(fence)

        checker.check(GeoPoint(52.52, 13.405))  # inside
        checker.check(GeoPoint(53.0, 13.405))   # outside
        checker.check(GeoPoint(52.52, 13.405))  # inside
        checker.check(GeoPoint(53.0, 13.405))   # outside

        assert checker.violation_rate == pytest.approx(0.5)

    def test_reset(self):
        fence = CircleGeofence(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        checker = GeofenceChecker(fence)
        checker.check(GeoPoint(53.0, 13.405))
        checker.reset()
        assert checker.consecutive_violations == 0
        assert checker.violation_rate == 0.0
