"""Flight path simulator for offline VPS testing.

Generates synthetic drone trajectories over a map pack, simulates
camera frames by cropping satellite tiles, and runs the matching
pipeline to evaluate accuracy without real hardware.

Usage:
    vps-program simulate --pack-dir ./map_pack --trajectory circle \
        --center 52.5163,13.3777 --duration 60 --speed 10
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shared.tile_math import (
    GeoPoint, TileCoord, gps_to_tile, gps_to_tile_pixel,
    tile_pixel_to_gps, haversine_km, TILE_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrajectoryPoint:
    """A single point in a simulated flight path."""
    t: float        # time in seconds
    lat: float
    lon: float
    heading: float  # degrees, 0=north, clockwise
    altitude: float  # meters AGL


def generate_circle(
    center: GeoPoint,
    radius_m: float = 200.0,
    speed_mps: float = 10.0,
    altitude: float = 50.0,
    duration: float = 60.0,
    dt: float = 0.333,
) -> list[TrajectoryPoint]:
    """Generate a circular flight path.

    Args:
        center: center of the circle
        radius_m: radius in meters
        speed_mps: ground speed in m/s
        altitude: altitude AGL in meters
        duration: total time in seconds
        dt: time step between points
    """
    circumference = 2 * math.pi * radius_m
    period = circumference / speed_mps
    omega = 2 * math.pi / period  # rad/s

    # Convert radius to degrees
    radius_lat = radius_m / 111320.0
    radius_lon = radius_m / (111320.0 * math.cos(math.radians(center.lat)))

    points = []
    t = 0.0
    while t <= duration:
        angle = omega * t
        lat = center.lat + radius_lat * math.cos(angle)
        lon = center.lon + radius_lon * math.sin(angle)
        heading = math.degrees(angle + math.pi / 2) % 360  # tangent to circle
        points.append(TrajectoryPoint(t=t, lat=lat, lon=lon, heading=heading, altitude=altitude))
        t += dt

    return points


def generate_line(
    start: GeoPoint,
    heading_deg: float = 0.0,
    speed_mps: float = 15.0,
    altitude: float = 50.0,
    duration: float = 60.0,
    dt: float = 0.333,
) -> list[TrajectoryPoint]:
    """Generate a straight-line flight path.

    Args:
        start: starting position
        heading_deg: heading in degrees (0=north, 90=east)
        speed_mps: ground speed in m/s
        altitude: altitude AGL
        duration: total time in seconds
        dt: time step
    """
    heading_rad = math.radians(heading_deg)
    vlat = speed_mps * math.cos(heading_rad) / 111320.0  # deg/s north
    vlon = speed_mps * math.sin(heading_rad) / (111320.0 * math.cos(math.radians(start.lat)))

    points = []
    t = 0.0
    while t <= duration:
        lat = start.lat + vlat * t
        lon = start.lon + vlon * t
        points.append(TrajectoryPoint(t=t, lat=lat, lon=lon, heading=heading_deg, altitude=altitude))
        t += dt

    return points


def generate_lawnmower(
    center: GeoPoint,
    width_m: float = 400.0,
    height_m: float = 400.0,
    speed_mps: float = 10.0,
    altitude: float = 50.0,
    spacing_m: float = 50.0,
    dt: float = 0.333,
) -> list[TrajectoryPoint]:
    """Generate a lawnmower/zigzag survey pattern.

    Args:
        center: center of the survey area
        width_m: area width in meters (east-west)
        height_m: area height in meters (north-south)
        speed_mps: ground speed
        altitude: altitude AGL
        spacing_m: distance between parallel lines
        dt: time step
    """
    half_w = width_m / 2
    half_h = height_m / 2

    # Convert to degrees
    dlat = half_h / 111320.0
    dlon = half_w / (111320.0 * math.cos(math.radians(center.lat)))

    # Generate waypoints
    waypoints = []
    num_lines = int(width_m / spacing_m) + 1
    for i in range(num_lines):
        x_frac = i / max(1, num_lines - 1)  # 0 to 1
        lon = center.lon - dlon + 2 * dlon * x_frac

        if i % 2 == 0:
            # North to south
            waypoints.append((center.lat + dlat, lon))
            waypoints.append((center.lat - dlat, lon))
        else:
            # South to north
            waypoints.append((center.lat - dlat, lon))
            waypoints.append((center.lat + dlat, lon))

    # Interpolate along waypoints at speed
    points = []
    t = 0.0
    for i in range(len(waypoints) - 1):
        lat0, lon0 = waypoints[i]
        lat1, lon1 = waypoints[i + 1]
        seg_dist = haversine_km(GeoPoint(lat0, lon0), GeoPoint(lat1, lon1)) * 1000
        seg_time = seg_dist / speed_mps

        if seg_dist < 1e-3:
            continue

        heading = math.degrees(math.atan2(lon1 - lon0, lat1 - lat0)) % 360
        steps = max(1, int(seg_time / dt))

        for s in range(steps):
            frac = s / steps
            lat = lat0 + (lat1 - lat0) * frac
            lon = lon0 + (lon1 - lon0) * frac
            points.append(TrajectoryPoint(t=t, lat=lat, lon=lon, heading=heading, altitude=altitude))
            t += dt

    return points


def crop_synthetic_frame(
    position: GeoPoint,
    heading: float,
    map_pack_dir: Path,
    frame_size: int = 256,
    zoom: int = 19,
) -> np.ndarray | None:
    """Crop a synthetic drone camera frame from satellite tiles.

    Simulates a nadir-looking camera by extracting a region from
    the satellite tile at the drone's position.

    Args:
        position: drone GPS position
        heading: drone heading in degrees (for rotation)
        map_pack_dir: path to map pack with tiles/
        frame_size: output frame size (square)
        zoom: tile zoom level to crop from

    Returns:
        BGR image or None if tile not available
    """
    tp = gps_to_tile_pixel(position, zoom)
    tile_path = map_pack_dir / "tiles" / str(tp.tile.z) / str(tp.tile.x) / f"{tp.tile.y}.png"

    if not tile_path.exists():
        return None

    tile_img = cv2.imread(str(tile_path))
    if tile_img is None:
        return None

    # Extract region centered on drone position
    cx, cy = int(tp.px), int(tp.py)
    half = frame_size // 2

    # Handle edge cases — pad if needed
    th, tw = tile_img.shape[:2]
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    if y0 < 0 or x0 < 0 or y1 > th or x1 > tw:
        # Crop what we can and pad
        crop_y0 = max(0, y0)
        crop_y1 = min(th, y1)
        crop_x0 = max(0, x0)
        crop_x1 = min(tw, x1)
        crop = tile_img[crop_y0:crop_y1, crop_x0:crop_x1]

        frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        paste_y = max(0, -y0)
        paste_x = max(0, -x0)
        frame[paste_y:paste_y + crop.shape[0], paste_x:paste_x + crop.shape[1]] = crop
    else:
        frame = tile_img[y0:y1, x0:x1]

    # Rotate by heading (simulates drone yaw)
    if abs(heading) > 1.0:
        M = cv2.getRotationMatrix2D((half, half), -heading, 1.0)
        frame = cv2.warpAffine(frame, M, (frame_size, frame_size))

    return frame


@dataclass(slots=True)
class SimulationResult:
    """Results from a trajectory simulation."""
    trajectory: list[TrajectoryPoint]
    matched: list[bool]        # was each frame matched?
    errors_m: list[float]      # position error per frame (meters)
    mean_error_m: float
    max_error_m: float
    fix_rate: float            # fraction of frames with a fix
    total_time_s: float


def save_results(results: SimulationResult, output_path: Path) -> None:
    """Save simulation results to JSON."""
    data = {
        "num_frames": len(results.trajectory),
        "fix_rate": results.fix_rate,
        "mean_error_m": results.mean_error_m,
        "max_error_m": results.max_error_m,
        "total_time_s": results.total_time_s,
        "points": [
            {
                "t": p.t,
                "true_lat": p.lat,
                "true_lon": p.lon,
                "heading": p.heading,
                "matched": m,
                "error_m": e,
            }
            for p, m, e in zip(results.trajectory, results.matched, results.errors_m)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)
