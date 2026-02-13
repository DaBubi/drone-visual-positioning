"""Tests for the flight path simulator."""

import json
import math

import numpy as np
import pytest

from programmer.simulate import (
    generate_circle,
    generate_line,
    generate_lawnmower,
    TrajectoryPoint,
    SimulationResult,
    save_results,
)
from shared.tile_math import GeoPoint, haversine_km


class TestCircleTrajectory:
    def test_returns_points(self):
        pts = generate_circle(GeoPoint(52.52, 13.405), duration=10, dt=0.5)
        assert len(pts) > 0
        assert all(isinstance(p, TrajectoryPoint) for p in pts)

    def test_points_near_center(self):
        center = GeoPoint(52.52, 13.405)
        pts = generate_circle(center, radius_m=200, duration=30)
        for p in pts:
            dist = haversine_km(center, GeoPoint(p.lat, p.lon)) * 1000
            assert 150 < dist < 250, f"Point at {dist:.0f}m from center, expected ~200m"

    def test_full_circle(self):
        """After one revolution, should return near the start."""
        center = GeoPoint(52.52, 13.405)
        radius = 100.0
        speed = 10.0
        circumference = 2 * math.pi * radius
        period = circumference / speed

        pts = generate_circle(center, radius_m=radius, speed_mps=speed,
                              duration=period, dt=0.1)
        start = GeoPoint(pts[0].lat, pts[0].lon)
        end = GeoPoint(pts[-1].lat, pts[-1].lon)
        dist = haversine_km(start, end) * 1000
        assert dist < 5.0, f"Circle didn't close: {dist:.1f}m gap"

    def test_heading_tangent(self):
        pts = generate_circle(GeoPoint(52.52, 13.405), duration=5)
        for p in pts:
            assert 0 <= p.heading < 360


class TestLineTrajectory:
    def test_returns_points(self):
        pts = generate_line(GeoPoint(52.52, 13.405), duration=10)
        assert len(pts) > 0

    def test_northward(self):
        pts = generate_line(GeoPoint(52.52, 13.405), heading_deg=0, speed_mps=15, duration=10)
        # Should move north
        assert pts[-1].lat > pts[0].lat
        assert abs(pts[-1].lon - pts[0].lon) < 1e-6  # no east/west drift

    def test_eastward(self):
        pts = generate_line(GeoPoint(52.52, 13.405), heading_deg=90, speed_mps=15, duration=10)
        assert pts[-1].lon > pts[0].lon

    def test_distance_matches_speed(self):
        speed = 20.0
        duration = 10.0
        pts = generate_line(GeoPoint(52.52, 13.405), speed_mps=speed, duration=duration)
        start = GeoPoint(pts[0].lat, pts[0].lon)
        end = GeoPoint(pts[-1].lat, pts[-1].lon)
        dist = haversine_km(start, end) * 1000
        expected = speed * duration
        assert abs(dist - expected) < 5.0, f"Distance {dist:.0f}m vs expected {expected:.0f}m"


class TestLawnmowerTrajectory:
    def test_returns_points(self):
        pts = generate_lawnmower(GeoPoint(52.52, 13.405))
        assert len(pts) > 10

    def test_stays_in_area(self):
        center = GeoPoint(52.52, 13.405)
        pts = generate_lawnmower(center, width_m=400, height_m=400)
        for p in pts:
            dist = haversine_km(center, GeoPoint(p.lat, p.lon)) * 1000
            assert dist < 500, f"Point {dist:.0f}m from center, exceeds area"


class TestSaveResults:
    def test_saves_json(self, tmp_path):
        pts = [TrajectoryPoint(t=0, lat=52.52, lon=13.405, heading=0, altitude=50)]
        result = SimulationResult(
            trajectory=pts,
            matched=[True],
            errors_m=[0.5],
            mean_error_m=0.5,
            max_error_m=0.5,
            fix_rate=1.0,
            total_time_s=1.0,
        )
        out = tmp_path / "results.json"
        save_results(result, out)

        data = json.loads(out.read_text())
        assert data["num_frames"] == 1
        assert data["fix_rate"] == 1.0
        assert len(data["points"]) == 1
        assert data["points"][0]["true_lat"] == 52.52
