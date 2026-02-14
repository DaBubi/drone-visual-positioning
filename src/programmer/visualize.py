"""Visualization tools for flight telemetry and map packs.

Generates static HTML/SVG visualizations of:
- Flight trajectories over satellite tiles
- Position error heatmaps
- EKF state evolution
- Fix rate over time

No matplotlib dependency — outputs standalone HTML files.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from shared.tile_math import GeoPoint, gps_to_tile_pixel, tile_pixel_to_gps, TILE_SIZE


def telemetry_to_geojson(csv_path: Path, output_path: Path | None = None) -> dict:
    """Convert a telemetry CSV to GeoJSON for visualization.

    Creates a GeoJSON FeatureCollection with:
    - A LineString for the EKF-smoothed trajectory
    - Points for each frame with fix/miss status and error data
    """
    features = []
    coords_ekf = []
    coords_raw = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_fix = row["fix"] == "1"
            ekf_lat = float(row["ekf_lat"]) if row["ekf_lat"] else 0
            ekf_lon = float(row["ekf_lon"]) if row["ekf_lon"] else 0

            if ekf_lat != 0 and ekf_lon != 0:
                coords_ekf.append([ekf_lon, ekf_lat])

            if is_fix:
                lat = float(row["lat"])
                lon = float(row["lon"])
                coords_raw.append([lon, lat])

                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "frame": int(row["frame_num"]),
                        "fix": True,
                        "hdop": float(row["hdop"]) if row["hdop"] else 0,
                        "inlier_ratio": float(row["inlier_ratio"]) if row["inlier_ratio"] else 0,
                        "total_ms": float(row["total_ms"]),
                        "ekf_speed": float(row["ekf_speed_mps"]) if row["ekf_speed_mps"] else 0,
                        "ekf_accepted": row["ekf_accepted"] == "1",
                    },
                })

    # EKF trajectory line
    if coords_ekf:
        features.insert(0, {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords_ekf},
            "properties": {"type": "ekf_trajectory", "stroke": "#00ff00", "stroke-width": 2},
        })

    # Raw fix trajectory line
    if coords_raw:
        features.insert(0, {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords_raw},
            "properties": {"type": "raw_trajectory", "stroke": "#ff6600", "stroke-width": 1},
        })

    geojson = {"type": "FeatureCollection", "features": features}

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

    return geojson


def simulation_to_geojson(sim_json_path: Path, output_path: Path | None = None) -> dict:
    """Convert simulation results JSON to GeoJSON."""
    with open(sim_json_path) as f:
        data = json.load(f)

    features = []
    coords = []

    for pt in data["points"]:
        lon, lat = pt["true_lon"], pt["true_lat"]
        coords.append([lon, lat])
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "t": pt["t"],
                "heading": pt["heading"],
                "matched": pt["matched"],
                "error_m": pt["error_m"],
            },
        })

    if coords:
        features.insert(0, {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"type": "trajectory", "stroke": "#3388ff", "stroke-width": 2},
        })

    geojson = {"type": "FeatureCollection", "features": features}

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

    return geojson


def generate_stats_html(csv_path: Path, output_path: Path) -> None:
    """Generate a standalone HTML statistics page from telemetry CSV.

    Shows key metrics in a clean dashboard without any dependencies.
    """
    frames = 0
    fixes = 0
    latencies = []
    speeds = []
    hdops = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames += 1
            if row["fix"] == "1":
                fixes += 1
                if row["hdop"]:
                    hdops.append(float(row["hdop"]))
            latencies.append(float(row["total_ms"]))
            if row["ekf_speed_mps"]:
                speeds.append(float(row["ekf_speed_mps"]))

    fix_rate = fixes / max(1, frames) * 100
    avg_lat = sum(latencies) / max(1, len(latencies))
    max_lat = max(latencies) if latencies else 0
    avg_speed = sum(speeds) / max(1, len(speeds))
    avg_hdop = sum(hdops) / max(1, len(hdops))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VPS Flight Stats</title>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
.card {{ background: #16213e; padding: 20px; border-radius: 8px; text-align: center; }}
.value {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
.label {{ color: #888; margin-top: 8px; }}
h1 {{ color: #00ff88; }}
</style></head><body>
<h1>VPS Flight Statistics</h1>
<p>Source: {csv_path.name}</p>
<div class="grid">
<div class="card"><div class="value">{frames}</div><div class="label">Total Frames</div></div>
<div class="card"><div class="value">{fixes}</div><div class="label">Fixes</div></div>
<div class="card"><div class="value">{fix_rate:.1f}%</div><div class="label">Fix Rate</div></div>
<div class="card"><div class="value">{avg_lat:.0f}ms</div><div class="label">Avg Latency</div></div>
<div class="card"><div class="value">{max_lat:.0f}ms</div><div class="label">Max Latency</div></div>
<div class="card"><div class="value">{avg_speed:.1f}m/s</div><div class="label">Avg Speed</div></div>
<div class="card"><div class="value">{avg_hdop:.1f}</div><div class="label">Avg HDOP</div></div>
<div class="card"><div class="value">{1000/max(1,avg_lat):.1f}Hz</div><div class="label">Effective Rate</div></div>
</div>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
