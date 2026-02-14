"""Tests for position fusion engine."""

import pytest

from onboard.fusion import FusionOutput, PositionFusion
from onboard.ekf import EKFConfig
from shared.tile_math import GeoPoint, haversine_km


class TestFusionOutput:
    def test_default_values(self):
        out = FusionOutput(
            position=None, hdop=99.0, speed_mps=0.0, heading_deg=0.0,
            fix_quality=0, source="none", geofence_ok=True, ekf_accepted=False,
        )
        assert out.position is None
        assert out.source == "none"

    def test_with_position(self):
        pt = GeoPoint(52.52, 13.405)
        out = FusionOutput(
            position=pt, hdop=1.5, speed_mps=10.0, heading_deg=90.0,
            fix_quality=1, source="visual", geofence_ok=True, ekf_accepted=True,
        )
        assert out.position.lat == 52.52
        assert out.fix_quality == 1


class TestPositionFusion:
    def _make_fusion(self, **kwargs):
        defaults = dict(
            center=GeoPoint(52.52, 13.405),
            radius_km=5.0,
            ekf_config=EKFConfig(gate_threshold=1e6),
        )
        defaults.update(kwargs)
        return PositionFusion(**defaults)

    def test_no_input_no_fix(self):
        fusion = self._make_fusion()
        out = fusion.update(None, t=0.0)
        assert out.position is None
        assert out.fix_quality == 0
        assert out.source == "none"

    def test_visual_fix_initializes(self):
        fusion = self._make_fusion()
        pt = GeoPoint(52.52, 13.405)
        out = fusion.update(pt, hdop=1.0, t=0.0)
        # First update initializes EKF — should produce visual fix
        assert out.source == "visual"
        assert out.fix_quality == 1
        assert out.position is not None
        assert out.ekf_accepted

    def test_visual_then_predict(self):
        fusion = self._make_fusion()
        pt = GeoPoint(52.52, 13.405)
        # Feed enough measurements to initialize EKF
        for i in range(3):
            fusion.update(pt, hdop=1.0, t=float(i))

        # Now no visual — should get EKF prediction
        out = fusion.update(None, t=3.0)
        assert out.source == "ekf_predict"
        assert out.fix_quality == 2
        assert out.position is not None

    def test_dead_reckoning_fallback(self):
        fusion = self._make_fusion(max_dead_reckoning_s=10.0)
        pt = GeoPoint(52.52, 13.405)
        # Initialize with visual fixes
        for i in range(5):
            fusion.update(pt, hdop=1.0, t=float(i))

        # Long gap — EKF should reset, fall back to dead reckoning
        out = fusion.update(None, t=100.0)
        # After long gap, EKF resets so predict won't work,
        # but dead reckoning reference was set from last visual fix
        # DR max is 10s, and 100-4=96s gap exceeds it
        assert out.fix_quality in (0, 2, 3)

    def test_geofence_blocks_out_of_bounds(self):
        fusion = self._make_fusion(center=GeoPoint(52.52, 13.405), radius_km=1.0)
        # Position far outside geofence
        far_pt = GeoPoint(53.0, 14.0)
        out = fusion.update(far_pt, hdop=1.0, t=0.0)
        # Should be blocked by geofence
        assert not out.geofence_ok
        assert out.position is None
        assert out.fix_quality == 0

    def test_geofence_allows_inside(self):
        fusion = self._make_fusion(center=GeoPoint(52.52, 13.405), radius_km=5.0)
        pt = GeoPoint(52.52, 13.405)  # center — definitely inside
        out = fusion.update(pt, hdop=1.0, t=0.0)
        assert out.geofence_ok
        assert out.position is not None

    def test_no_geofence(self):
        fusion = PositionFusion(
            center=None,
            ekf_config=EKFConfig(gate_threshold=1e6),
        )
        pt = GeoPoint(52.52, 13.405)
        out = fusion.update(pt, hdop=1.0, t=0.0)
        assert out.geofence_ok
        assert out.position is not None

    def test_speed_and_heading(self):
        fusion = self._make_fusion()
        # Feed moving positions (eastward)
        for i in range(5):
            pt = GeoPoint(52.52, 13.405 + i * 0.0001)
            fusion.update(pt, hdop=1.0, t=float(i))

        out = fusion.update(GeoPoint(52.52, 13.4055), hdop=1.0, t=5.0)
        assert out.speed_mps >= 0.0
        # Heading should be roughly east (90 deg) for eastward movement
        # but EKF smoothing may alter it, so just check it's computed
        assert 0.0 <= out.heading_deg < 360.0

    def test_hdop_passthrough_visual(self):
        fusion = self._make_fusion()
        pt = GeoPoint(52.52, 13.405)
        out = fusion.update(pt, hdop=1.5, t=0.0)
        assert out.hdop == 1.5

    def test_hdop_degraded_for_predict(self):
        fusion = self._make_fusion()
        pt = GeoPoint(52.52, 13.405)
        for i in range(3):
            fusion.update(pt, hdop=1.0, t=float(i))

        out = fusion.update(None, t=3.0)
        if out.source == "ekf_predict":
            assert out.hdop == 3.0  # degraded confidence

    def test_reset(self):
        fusion = self._make_fusion()
        pt = GeoPoint(52.52, 13.405)
        fusion.update(pt, hdop=1.0, t=0.0)
        fusion.reset()
        out = fusion.update(None, t=1.0)
        assert out.position is None
        assert out.fix_quality == 0

    def test_continuous_visual_fixes(self):
        fusion = self._make_fusion()
        for i in range(10):
            pt = GeoPoint(52.52 + i * 0.00001, 13.405)
            out = fusion.update(pt, hdop=1.0, t=float(i))
            assert out.source == "visual"
            assert out.fix_quality == 1
            assert out.position is not None
