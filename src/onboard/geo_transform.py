"""Coordinate transformations between pixel, tile, and GPS coordinates.

Provides the math to convert a matched pixel position within a satellite
tile into a GPS coordinate, accounting for tile coordinate system and
Web Mercator projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.tile_math import GeoPoint, TileCoord


# Web Mercator constants
TILE_SIZE = 256
EARTH_CIRCUMFERENCE_M = 40_075_016.686


@dataclass(slots=True)
class PixelCoord:
    """Pixel position within a tile image."""
    x: float  # 0..TILE_SIZE
    y: float  # 0..TILE_SIZE


def tile_pixel_to_gps(tile: TileCoord, pixel: PixelCoord) -> GeoPoint:
    """Convert a pixel position within a tile to GPS coordinates.

    Args:
        tile: tile coordinate (z, x, y)
        pixel: pixel position within the tile (0..256)

    Returns:
        GPS coordinate
    """
    n = 2 ** tile.z

    # Global pixel coordinate
    global_x = (tile.x + pixel.x / TILE_SIZE)
    global_y = (tile.y + pixel.y / TILE_SIZE)

    # Convert to longitude (linear)
    lon = global_x / n * 360.0 - 180.0

    # Convert to latitude (inverse Mercator)
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * global_y / n)))
    lat = math.degrees(lat_rad)

    return GeoPoint(lat=lat, lon=lon)


def gps_to_tile_pixel(point: GeoPoint, zoom: int) -> tuple[TileCoord, PixelCoord]:
    """Convert GPS to tile coordinate + pixel within that tile.

    Args:
        point: GPS coordinate
        zoom: zoom level

    Returns:
        (tile_coord, pixel_within_tile)
    """
    n = 2 ** zoom

    # Longitude to x
    x_global = (point.lon + 180.0) / 360.0 * n
    tile_x = int(x_global)
    pixel_x = (x_global - tile_x) * TILE_SIZE

    # Latitude to y (Mercator)
    lat_rad = math.radians(point.lat)
    y_global = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    tile_y = int(y_global)
    pixel_y = (y_global - tile_y) * TILE_SIZE

    return TileCoord(z=zoom, x=tile_x, y=tile_y), PixelCoord(x=pixel_x, y=pixel_y)


def homography_to_gps(
    H,  # 3x3 numpy array
    tile: TileCoord,
    image_center: tuple[float, float] = (128.0, 128.0),
) -> GeoPoint:
    """Extract GPS position from a homography mapping drone→tile.

    The homography maps drone image pixels to tile pixels.
    We project the center of the drone image through H to get the
    corresponding tile pixel, then convert to GPS.

    Args:
        H: 3x3 homography matrix (drone→tile)
        tile: the matched tile coordinate
        image_center: center of the drone image in pixels

    Returns:
        Estimated GPS position
    """
    import numpy as np

    cx, cy = image_center
    src = np.array([cx, cy, 1.0])
    dst = H @ src
    if abs(dst[2]) < 1e-10:
        return GeoPoint(lat=0.0, lon=0.0)

    px = dst[0] / dst[2]
    py = dst[1] / dst[2]

    return tile_pixel_to_gps(tile, PixelCoord(x=px, y=py))


def meters_per_pixel(lat: float, zoom: int) -> float:
    """Ground resolution in meters per pixel at given latitude and zoom."""
    return (EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat))) / (TILE_SIZE * 2 ** zoom)


def pixel_distance_to_meters(
    dx: float, dy: float, lat: float, zoom: int,
) -> float:
    """Convert pixel displacement to meters."""
    mpp = meters_per_pixel(lat, zoom)
    return math.sqrt(dx * dx + dy * dy) * mpp
