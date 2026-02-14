"""Tests for visualization tools."""

import csv
import json
from pathlib import Path

import pytest

from programmer.visualize import (
    telemetry_to_geojson,
    simulation_to_geojson,
    generate_stats_html,
)


@pytest.fixture
def telemetry_csv(tmp_path):
    """Create a sample telemetry CSV."""
    path = tmp_path / "telemetry.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "frame_num", "fix", "lat", "lon", "hdop",
            "inlier_ratio", "num_matches", "tile_z", "tile_x", "tile_y",
            "retrieval_ms", "match_ms", "total_ms",
            "ekf_lat", "ekf_lon", "ekf_vlat", "ekf_vlon",
            "ekf_speed_mps", "ekf_gate", "ekf_accepted",
        ])
        for i in range(20):
            is_fix = i % 3 != 0
            writer.writerow([
                f"{i * 0.3:.3f}", i,
                1 if is_fix else 0,
                f"{52.52 + i * 0.0001:.8f}" if is_fix else "",
                f"{13.405:.8f}" if is_fix else "",
                "1.5" if is_fix else "",
                "0.8" if is_fix else "",
                42 if is_fix else 0,
                17 if is_fix else "",
                70406 if is_fix else "",
                42987 if is_fix else "",
                "5.0", "75.0", "85.0",
                f"{52.52 + i * 0.0001:.8f}",
                "13.40500000",
                "0.0000100000",
                "0.0000000000",
                "11.13",
                "1.5",
                1 if is_fix else 0,
            ])
    return path


@pytest.fixture
def simulation_json(tmp_path):
    """Create a sample simulation results JSON."""
    data = {
        "num_frames": 10,
        "fix_rate": 0.8,
        "mean_error_m": 2.5,
        "max_error_m": 5.0,
        "total_time_s": 3.0,
        "points": [
            {
                "t": i * 0.3,
                "true_lat": 52.52 + i * 0.0001,
                "true_lon": 13.405,
                "heading": 45.0,
                "matched": i % 2 == 0,
                "error_m": 1.0 + i * 0.5,
            }
            for i in range(10)
        ],
    }
    path = tmp_path / "sim_results.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestTelemetryToGeoJSON:
    def test_produces_feature_collection(self, telemetry_csv):
        geojson = telemetry_to_geojson(telemetry_csv)
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) > 0

    def test_has_trajectory_line(self, telemetry_csv):
        geojson = telemetry_to_geojson(telemetry_csv)
        lines = [f for f in geojson["features"]
                 if f["geometry"]["type"] == "LineString"]
        assert len(lines) >= 1

    def test_saves_to_file(self, telemetry_csv, tmp_path):
        out = tmp_path / "output.geojson"
        telemetry_to_geojson(telemetry_csv, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["type"] == "FeatureCollection"


class TestSimulationToGeoJSON:
    def test_produces_features(self, simulation_json):
        geojson = simulation_to_geojson(simulation_json)
        assert len(geojson["features"]) == 11  # 10 points + 1 line

    def test_saves_to_file(self, simulation_json, tmp_path):
        out = tmp_path / "sim.geojson"
        simulation_to_geojson(simulation_json, out)
        assert out.exists()


class TestStatsHTML:
    def test_generates_html(self, telemetry_csv, tmp_path):
        out = tmp_path / "stats.html"
        generate_stats_html(telemetry_csv, out)
        assert out.exists()
        content = out.read_text()
        assert "VPS Flight Statistics" in content
        assert "Fix Rate" in content
        assert "Total Frames" in content
