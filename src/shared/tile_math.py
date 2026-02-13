"""GPS ↔ Web Mercator tile ↔ pixel coordinate conversions.

Web Mercator (EPSG:3857) tile system:
  - Zoom z → 2^z × 2^z tiles, each 256×256 pixels
  - Tile (0,0) is top-left (NW corner)
  - x increases eastward, y increases southward
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TILE_SIZE = 256


@dataclass(frozen=True, slots=True)
class GeoPoint:
    """WGS84 coordinate."""
    lat: float  # degrees, [-90, 90]
    lon: float  # degrees, [-180, 180]


@dataclass(frozen=True, slots=True)
class TileCoord:
    """Tile index at a given zoom level."""
    z: int
    x: int
    y: int

    @property
    def meters_per_pixel(self) -> float:
        """Ground resolution at equator for this zoom level."""
        return 156543.03392 / (2 ** self.z)


@dataclass(frozen=True, slots=True)
class TilePixel:
    """Pixel position within a specific tile."""
    tile: TileCoord
    px: float  # pixel x within tile [0, TILE_SIZE)
    py: float  # pixel y within tile [0, TILE_SIZE)


def gps_to_tile(point: GeoPoint, zoom: int) -> TileCoord:
    """Convert GPS coordinate to tile index at given zoom."""
    n = 2.0 ** zoom
    x = int((point.lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(point.lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    # Clamp to valid range
    x = max(0, min(int(n) - 1, x))
    y = max(0, min(int(n) - 1, y))
    return TileCoord(z=zoom, x=x, y=y)


def gps_to_tile_pixel(point: GeoPoint, zoom: int) -> TilePixel:
    """Convert GPS to exact pixel position within its tile."""
    n = 2.0 ** zoom
    x_float = (point.lon + 180.0) / 360.0 * n
    lat_rad = math.radians(point.lat)
    y_float = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

    tile_x = int(x_float)
    tile_y = int(y_float)
    # Clamp
    tile_x = max(0, min(int(n) - 1, tile_x))
    tile_y = max(0, min(int(n) - 1, tile_y))

    px = (x_float - tile_x) * TILE_SIZE
    py = (y_float - tile_y) * TILE_SIZE

    return TilePixel(
        tile=TileCoord(z=zoom, x=tile_x, y=tile_y),
        px=px,
        py=py,
    )


def tile_pixel_to_gps(tile: TileCoord, px: float, py: float) -> GeoPoint:
    """Convert tile + pixel offset to GPS coordinate."""
    n = 2.0 ** tile.z
    lon = (tile.x + px / TILE_SIZE) / n * 360.0 - 180.0
    lat = math.degrees(
        math.atan(math.sinh(math.pi * (1 - 2 * (tile.y + py / TILE_SIZE) / n)))
    )
    return GeoPoint(lat=lat, lon=lon)


def tile_center_gps(tile: TileCoord) -> GeoPoint:
    """GPS coordinate of the center of a tile."""
    return tile_pixel_to_gps(tile, TILE_SIZE / 2, TILE_SIZE / 2)


def tiles_in_radius(center: GeoPoint, radius_km: float, zoom: int) -> list[TileCoord]:
    """Return all tiles within radius_km of center at given zoom.

    Uses a bounding box approximation then filters by actual distance.
    """
    # Approximate bounding box
    # 1 degree latitude ≈ 111.32 km
    dlat = radius_km / 111.32
    # 1 degree longitude varies with latitude
    dlon = radius_km / (111.32 * math.cos(math.radians(center.lat)))

    nw = GeoPoint(lat=center.lat + dlat, lon=center.lon - dlon)
    se = GeoPoint(lat=center.lat - dlat, lon=center.lon + dlon)

    tile_nw = gps_to_tile(nw, zoom)
    tile_se = gps_to_tile(se, zoom)

    tiles = []
    for x in range(tile_nw.x, tile_se.x + 1):
        for y in range(tile_nw.y, tile_se.y + 1):
            tile = TileCoord(z=zoom, x=x, y=y)
            # Check actual distance from center to tile center
            tc = tile_center_gps(tile)
            dist = haversine_km(center, tc)
            if dist <= radius_km * 1.2:  # 20% margin for tile overlap
                tiles.append(tile)
    return tiles


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance between two GPS points in kilometers."""
    R = 6371.0  # Earth radius km
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(h))
