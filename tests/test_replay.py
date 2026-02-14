"""Tests for flight replay and analysis."""

import json

import pytest

from onboard.flight_recorder import (
    FlightRecord, FlightRecorder,
    SOURCE_VISUAL, SOURCE_EKF_PREDICT, SOURCE_DEAD_RECKONING, SOURCE_NONE,
    FLAG_GEOFENCE_OK, FLAG_EKF_ACCEPTED,
)
from programmer.replay import analyze_flight, flight_to_geojson, save_analysis


def _make_records(n=50, source=SOURCE_VISUAL):
    """Create sample flight records."""
    records = []
    for i in range(n):
        records.append(FlightRecord(
            timestamp=float(i),
            lat=52.52 + i * 0.0001,
            lon=13.405 + i * 0.00005,
            vn_mps=5.0, ve_mps=2.5,
            hdop=1.5, speed_mps=5.59,
            heading_deg=27.0,
            fix_quality=1 if source == SOURCE_VISUAL else 2,
            source=source,
            match_count=40, inlier_ratio=0.6,
            latency_ms=150,
            flags=FLAG_GEOFENCE_OK | FLAG_EKF_ACCEPTED,
        ))
    return records


class TestAnalyzeFlight:
    def test_empty_records(self):
        stats = analyze_flight([])
        assert stats.total_frames == 0
        assert stats.fix_rate == 0.0

    def test_all_visual_fixes(self):
        records = _make_records(20, source=SOURCE_VISUAL)
        stats = analyze_flight(records)
        assert stats.visual_fixes == 20
        assert stats.fix_rate == 1.0
        assert stats.no_fix == 0

    def test_mixed_sources(self):
        records = []
        records.extend(_make_records(10, SOURCE_VISUAL))
        for r in _make_records(5, SOURCE_EKF_PREDICT):
            r.source = SOURCE_EKF_PREDICT
            records.append(r)
        for r in _make_records(3, SOURCE_DEAD_RECKONING):
            r.source = SOURCE_DEAD_RECKONING
            records.append(r)
        for r in _make_records(2, SOURCE_NONE):
            r.source = SOURCE_NONE
            r.lat = 0
            r.lon = 0
            records.append(r)

        stats = analyze_flight(records)
        assert stats.visual_fixes == 10
        assert stats.ekf_predictions == 5
        assert stats.dead_reckoning == 3
        assert stats.no_fix == 2
        assert stats.total_frames == 20

    def test_duration(self):
        records = _make_records(100)
        stats = analyze_flight(records)
        assert stats.duration_s == pytest.approx(99.0)

    def test_hdop_average(self):
        records = _make_records(10)
        stats = analyze_flight(records)
        assert stats.mean_hdop == pytest.approx(1.5)

    def test_latency_stats(self):
        records = _make_records(10)
        stats = analyze_flight(records)
        assert stats.mean_latency_ms == 150.0
        assert stats.max_latency_ms == 150

    def test_geofence_violations(self):
        records = _make_records(10)
        records[3].flags = 0  # no geofence OK flag
        records[7].flags = 0
        stats = analyze_flight(records)
        assert stats.geofence_violations == 2

    def test_summary_string(self):
        records = _make_records(10)
        stats = analyze_flight(records)
        s = stats.summary()
        assert "Flight Duration" in s
        assert "Fix Rate" in s


class TestFlightToGeoJSON:
    def test_produces_features(self):
        records = _make_records(5)
        geo = flight_to_geojson(records)
        assert geo["type"] == "FeatureCollection"
        assert len(geo["features"]) > 0

    def test_trajectory_line(self):
        records = _make_records(5)
        geo = flight_to_geojson(records)
        lines = [f for f in geo["features"] if f["geometry"]["type"] == "LineString"]
        assert len(lines) == 1

    def test_point_features(self):
        records = _make_records(5)
        geo = flight_to_geojson(records)
        points = [f for f in geo["features"] if f["geometry"]["type"] == "Point"]
        assert len(points) == 5

    def test_point_properties(self):
        records = _make_records(3)
        geo = flight_to_geojson(records)
        points = [f for f in geo["features"] if f["geometry"]["type"] == "Point"]
        props = points[0]["properties"]
        assert "source" in props
        assert "hdop" in props
        assert "speed_mps" in props

    def test_skips_zero_positions(self):
        records = _make_records(5)
        records[2].lat = 0
        records[2].lon = 0
        geo = flight_to_geojson(records)
        points = [f for f in geo["features"] if f["geometry"]["type"] == "Point"]
        assert len(points) == 4


class TestSaveAnalysis:
    def test_creates_files(self, tmp_path):
        records = _make_records(10)
        outputs = save_analysis(records, tmp_path / "analysis")
        assert "stats" in outputs
        assert "geojson" in outputs
        assert "stats_json" in outputs
        assert outputs["stats"].exists()
        assert outputs["geojson"].exists()

    def test_geojson_valid(self, tmp_path):
        records = _make_records(10)
        outputs = save_analysis(records, tmp_path / "analysis")
        with open(outputs["geojson"]) as f:
            geo = json.load(f)
        assert geo["type"] == "FeatureCollection"

    def test_stats_json_valid(self, tmp_path):
        records = _make_records(10)
        outputs = save_analysis(records, tmp_path / "analysis")
        with open(outputs["stats_json"]) as f:
            data = json.load(f)
        assert "fix_rate" in data
        assert "duration_s" in data
