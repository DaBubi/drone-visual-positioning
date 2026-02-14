"""Tests for dead reckoning fallback module."""

import pytest

from onboard.dead_reckoning import DeadReckoning
from shared.tile_math import GeoPoint, haversine_km


class TestDeadReckoning:
    def test_no_reference_returns_none(self):
        dr = DeadReckoning()
        assert dr.extrapolate(t=0.0) is None
        assert not dr.has_reference

    def test_update_reference(self):
        dr = DeadReckoning()
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=10.0, ve_mps=0.0, hdop=1.0, t=0.0,
        )
        assert dr.has_reference

    def test_stationary_extrapolation(self):
        dr = DeadReckoning()
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=0.0, ve_mps=0.0, hdop=1.0, t=0.0,
        )
        result = dr.extrapolate(t=1.0)
        assert result is not None
        pos, hdop = result
        assert abs(pos.lat - 52.52) < 1e-6
        assert abs(pos.lon - 13.405) < 1e-6

    def test_northward_extrapolation(self):
        dr = DeadReckoning()
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=10.0, ve_mps=0.0, hdop=1.0, t=0.0,
        )
        result = dr.extrapolate(t=5.0)
        assert result is not None
        pos, hdop = result
        # 10 m/s * 5s = 50m north
        dist = haversine_km(GeoPoint(52.52, 13.405), pos) * 1000
        assert 45 < dist < 55, f"Expected ~50m, got {dist:.1f}m"
        assert pos.lat > 52.52  # moved north

    def test_eastward_extrapolation(self):
        dr = DeadReckoning()
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=0.0, ve_mps=15.0, hdop=1.0, t=0.0,
        )
        result = dr.extrapolate(t=2.0)
        assert result is not None
        pos, _ = result
        assert pos.lon > 13.405  # moved east

    def test_hdop_increases_with_time(self):
        dr = DeadReckoning(hdop_growth_rate=2.0)
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=10.0, ve_mps=0.0, hdop=1.0, t=0.0,
        )
        _, hdop1 = dr.extrapolate(t=1.0)
        _, hdop5 = dr.extrapolate(t=5.0)
        assert hdop5 > hdop1
        assert hdop1 == pytest.approx(3.0)  # 1.0 + 2.0 * 1s
        assert hdop5 == pytest.approx(11.0)  # 1.0 + 2.0 * 5s

    def test_exceeds_max_extrapolation(self):
        dr = DeadReckoning(max_extrapolation_s=5.0)
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=10.0, ve_mps=0.0, hdop=1.0, t=0.0,
        )
        assert dr.extrapolate(t=4.0) is not None
        assert dr.extrapolate(t=6.0) is None  # too far

    def test_negative_time_returns_none(self):
        dr = DeadReckoning()
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=10.0, ve_mps=0.0, hdop=1.0, t=5.0,
        )
        assert dr.extrapolate(t=3.0) is None  # before reference

    def test_reference_update_resets(self):
        dr = DeadReckoning()
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=10.0, ve_mps=0.0, hdop=1.0, t=0.0,
        )
        # Update to new position
        dr.update_reference(
            GeoPoint(52.53, 13.405), vn_mps=5.0, ve_mps=0.0, hdop=0.8, t=10.0,
        )
        result = dr.extrapolate(t=12.0)
        assert result is not None
        pos, hdop = result
        # Should extrapolate from new reference
        assert pos.lat > 52.53

    def test_diagonal_movement(self):
        dr = DeadReckoning()
        dr.update_reference(
            GeoPoint(52.52, 13.405), vn_mps=10.0, ve_mps=10.0, hdop=1.0, t=0.0,
        )
        result = dr.extrapolate(t=3.0)
        assert result is not None
        pos, _ = result
        assert pos.lat > 52.52  # moved north
        assert pos.lon > 13.405  # moved east
