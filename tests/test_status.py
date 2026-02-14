"""Tests for status dashboard."""

import time

import pytest

from onboard.status import StatusDashboard, SubsystemStatus, SystemSnapshot


class TestSubsystemStatus:
    def test_defaults(self):
        s = SubsystemStatus(name="test")
        assert s.ok
        assert s.message == ""

    def test_age_without_update(self):
        s = SubsystemStatus(name="test")
        assert s.age_s == -1.0
        assert not s.stale

    def test_stale_detection(self):
        s = SubsystemStatus(name="test", last_update_t=time.monotonic() - 10.0)
        assert s.stale

    def test_fresh(self):
        s = SubsystemStatus(name="test", last_update_t=time.monotonic())
        assert not s.stale


class TestSystemSnapshot:
    def test_all_ok_empty(self):
        snap = SystemSnapshot()
        assert snap.all_ok

    def test_all_ok_with_healthy(self):
        snap = SystemSnapshot(
            subsystems={"cam": SubsystemStatus(name="cam", ok=True)},
        )
        assert snap.all_ok

    def test_not_ok_with_failure(self):
        snap = SystemSnapshot(
            subsystems={"cam": SubsystemStatus(name="cam", ok=False, message="offline")},
        )
        assert not snap.all_ok

    def test_warnings_from_failures(self):
        snap = SystemSnapshot(
            subsystems={"uart": SubsystemStatus(name="uart", ok=False, message="disconnected")},
        )
        w = snap.warnings
        assert len(w) == 1
        assert "uart" in w[0]

    def test_summary_contains_status(self):
        snap = SystemSnapshot(
            uptime_s=120.0,
            position_source="visual",
            fix_rate=0.85,
            fps=3.0,
        )
        s = snap.summary()
        assert "OK" in s
        assert "visual" in s


class TestStatusDashboard:
    def test_initial_snapshot(self):
        dash = StatusDashboard()
        snap = dash.snapshot()
        assert snap.all_ok
        assert snap.fps == 0.0

    def test_update_subsystem(self):
        dash = StatusDashboard()
        dash.update("camera", ok=True, message="30fps")
        snap = dash.snapshot()
        assert "camera" in snap.subsystems
        assert snap.subsystems["camera"].ok

    def test_update_existing_subsystem(self):
        dash = StatusDashboard()
        dash.update("uart", ok=True)
        dash.update("uart", ok=False, message="disconnected")
        snap = dash.snapshot()
        assert not snap.subsystems["uart"].ok

    def test_fps_calculation(self):
        dash = StatusDashboard()
        t = time.monotonic()
        # Simulate 10 frames at 5 FPS
        for i in range(10):
            dash.record_frame(t + i * 0.2)
        snap = dash.snapshot()
        assert 4.5 < snap.fps < 5.5

    def test_position_info(self):
        dash = StatusDashboard()
        dash.set_position_info("visual", 0.9)
        snap = dash.snapshot()
        assert snap.position_source == "visual"
        assert snap.fix_rate == 0.9

    def test_uptime(self):
        dash = StatusDashboard()
        snap = dash.snapshot()
        assert snap.uptime_s >= 0.0

    def test_multiple_subsystems(self):
        dash = StatusDashboard()
        dash.update("camera", ok=True)
        dash.update("matcher", ok=True)
        dash.update("uart", ok=False, message="error")
        snap = dash.snapshot()
        assert len(snap.subsystems) == 3
        assert not snap.all_ok
