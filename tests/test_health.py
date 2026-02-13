"""Tests for health monitoring module."""

import pytest

from onboard.health import HealthMonitor


class TestHealthMonitor:
    def test_initial_healthy(self):
        mon = HealthMonitor()
        assert mon.status.healthy
        assert mon.status.frames_total == 0

    def test_records_fix(self):
        mon = HealthMonitor()
        mon.record_frame(fix=True, latency_ms=50.0)
        assert mon.status.fixes_total == 1
        assert mon.status.frames_total == 1

    def test_records_miss(self):
        mon = HealthMonitor()
        mon.record_frame(fix=False, latency_ms=50.0)
        assert mon.status.misses_total == 1

    def test_fix_rate_calculation(self):
        mon = HealthMonitor(window_size=10)
        for _ in range(7):
            mon.record_frame(fix=True, latency_ms=50.0)
        for _ in range(3):
            mon.record_frame(fix=False, latency_ms=50.0)
        assert mon.status.fix_rate == pytest.approx(0.7)

    def test_low_fix_rate_warning(self):
        mon = HealthMonitor(window_size=20, min_fix_rate=0.5)
        for _ in range(20):
            mon.record_frame(fix=False, latency_ms=50.0)
        s = mon.status
        assert not s.healthy
        assert any("fix rate" in w.lower() for w in s.warnings)

    def test_high_latency_warning(self):
        mon = HealthMonitor(max_latency_ms=100.0)
        for _ in range(20):
            mon.record_frame(fix=True, latency_ms=200.0)
        s = mon.status
        assert not s.healthy
        assert any("latency" in w.lower() for w in s.warnings)

    def test_consecutive_misses_warning(self):
        mon = HealthMonitor(max_consecutive_misses=5)
        for _ in range(5):
            mon.record_frame(fix=False, latency_ms=50.0)
        s = mon.status
        assert not s.healthy
        assert any("lost fix" in w.lower() for w in s.warnings)

    def test_consecutive_misses_reset_on_fix(self):
        mon = HealthMonitor(max_consecutive_misses=5)
        for _ in range(4):
            mon.record_frame(fix=False, latency_ms=50.0)
        mon.record_frame(fix=True, latency_ms=50.0)
        s = mon.status
        assert s.healthy

    def test_geofence_violation_tracked(self):
        mon = HealthMonitor()
        mon.record_frame(fix=True, latency_ms=50.0, geofence_ok=False)
        assert mon.status.geofence_violations == 1

    def test_outlier_rejection_tracked(self):
        mon = HealthMonitor()
        mon.record_frame(fix=True, latency_ms=50.0, ekf_accepted=False)
        assert mon.status.outliers_rejected == 1

    def test_latency_stats(self):
        mon = HealthMonitor()
        mon.record_frame(fix=True, latency_ms=50.0)
        mon.record_frame(fix=True, latency_ms=100.0)
        mon.record_frame(fix=True, latency_ms=150.0)
        s = mon.status
        assert s.avg_latency_ms == pytest.approx(100.0)
        assert s.max_latency_ms == 150.0

    def test_uptime(self):
        mon = HealthMonitor()
        s = mon.status
        assert s.uptime_s >= 0
