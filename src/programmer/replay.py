"""Flight replay tool for post-flight analysis.

Reads binary flight recorder files and produces analysis reports,
GeoJSON trajectories, and statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from onboard.flight_recorder import (
    FlightRecorder, FlightRecord,
    SOURCE_VISUAL, SOURCE_EKF_PREDICT, SOURCE_DEAD_RECKONING,
    FLAG_GEOFENCE_OK, FLAG_EKF_ACCEPTED,
)


@dataclass(slots=True)
class FlightStats:
    """Statistics from a flight recording."""
    duration_s: float = 0.0
    total_frames: int = 0
    visual_fixes: int = 0
    ekf_predictions: int = 0
    dead_reckoning: int = 0
    no_fix: int = 0
    fix_rate: float = 0.0
    mean_hdop: float = 0.0
    mean_latency_ms: float = 0.0
    max_latency_ms: int = 0
    mean_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    geofence_violations: int = 0
    ekf_rejections: int = 0
    mean_inlier_ratio: float = 0.0

    def summary(self) -> str:
        return "\n".join([
            f"Flight Duration: {self.duration_s:.1f}s",
            f"Total Frames: {self.total_frames}",
            f"Fix Rate: {self.fix_rate:.1%}",
            f"  Visual: {self.visual_fixes} | EKF: {self.ekf_predictions} | DR: {self.dead_reckoning} | None: {self.no_fix}",
            f"HDOP: {self.mean_hdop:.1f} avg",
            f"Latency: {self.mean_latency_ms:.0f}ms avg, {self.max_latency_ms}ms max",
            f"Speed: {self.mean_speed_mps:.1f} m/s avg, {self.max_speed_mps:.1f} m/s max",
            f"Inlier Ratio: {self.mean_inlier_ratio:.2f} avg",
            f"Geofence Violations: {self.geofence_violations}",
            f"EKF Rejections: {self.ekf_rejections}",
        ])


def analyze_flight(records: list[FlightRecord]) -> FlightStats:
    """Compute statistics from flight records."""
    if not records:
        return FlightStats()

    stats = FlightStats()
    stats.total_frames = len(records)
    stats.duration_s = records[-1].timestamp - records[0].timestamp

    hdops = []
    latencies = []
    speeds = []
    inlier_ratios = []

    for r in records:
        if r.source == SOURCE_VISUAL:
            stats.visual_fixes += 1
        elif r.source == SOURCE_EKF_PREDICT:
            stats.ekf_predictions += 1
        elif r.source == SOURCE_DEAD_RECKONING:
            stats.dead_reckoning += 1
        else:
            stats.no_fix += 1

        if r.hdop < 99:
            hdops.append(r.hdop)
        latencies.append(r.latency_ms)
        speeds.append(r.speed_mps)

        if r.source == SOURCE_VISUAL:
            inlier_ratios.append(r.inlier_ratio)

        if not (r.flags & FLAG_GEOFENCE_OK):
            stats.geofence_violations += 1
        if r.source == SOURCE_VISUAL and not (r.flags & FLAG_EKF_ACCEPTED):
            stats.ekf_rejections += 1

    fixes = stats.visual_fixes + stats.ekf_predictions + stats.dead_reckoning
    stats.fix_rate = fixes / stats.total_frames if stats.total_frames > 0 else 0.0
    stats.mean_hdop = sum(hdops) / len(hdops) if hdops else 99.0
    stats.mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    stats.max_latency_ms = max(latencies) if latencies else 0
    stats.mean_speed_mps = sum(speeds) / len(speeds) if speeds else 0.0
    stats.max_speed_mps = max(speeds) if speeds else 0.0
    stats.mean_inlier_ratio = sum(inlier_ratios) / len(inlier_ratios) if inlier_ratios else 0.0

    return stats


def flight_to_geojson(records: list[FlightRecord]) -> dict:
    """Convert flight records to GeoJSON FeatureCollection."""
    features = []

    # Trajectory line
    coords = []
    for r in records:
        if r.lat != 0 or r.lon != 0:
            coords.append([r.lon, r.lat])

    if coords:
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"type": "trajectory"},
        })

    # Individual fix points with metadata
    for r in records:
        if r.lat == 0 and r.lon == 0:
            continue
        source_names = {
            SOURCE_VISUAL: "visual",
            SOURCE_EKF_PREDICT: "ekf_predict",
            SOURCE_DEAD_RECKONING: "dead_reckoning",
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
            "properties": {
                "timestamp": r.timestamp,
                "source": source_names.get(r.source, "none"),
                "hdop": r.hdop,
                "speed_mps": r.speed_mps,
                "heading_deg": r.heading_deg,
                "match_count": r.match_count,
                "inlier_ratio": r.inlier_ratio,
                "latency_ms": r.latency_ms,
            },
        })

    return {"type": "FeatureCollection", "features": features}


def save_analysis(records: list[FlightRecord], output_dir: Path) -> dict[str, Path]:
    """Run full analysis and save results.

    Returns dict of output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Stats
    stats = analyze_flight(records)
    stats_path = output_dir / "flight_stats.txt"
    stats_path.write_text(stats.summary())
    outputs["stats"] = stats_path

    # GeoJSON
    geojson = flight_to_geojson(records)
    geojson_path = output_dir / "flight_track.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    outputs["geojson"] = geojson_path

    # Stats JSON
    stats_json_path = output_dir / "flight_stats.json"
    with open(stats_json_path, "w") as f:
        json.dump({
            "duration_s": stats.duration_s,
            "total_frames": stats.total_frames,
            "fix_rate": stats.fix_rate,
            "visual_fixes": stats.visual_fixes,
            "mean_hdop": stats.mean_hdop,
            "mean_latency_ms": stats.mean_latency_ms,
            "max_speed_mps": stats.max_speed_mps,
        }, f, indent=2)
    outputs["stats_json"] = stats_json_path

    return outputs
