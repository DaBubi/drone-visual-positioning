"""Onboard map pack loader.

Loads a map pack (tiles + index + metadata) from disk for the onboard
matching pipeline. Provides tile lookup by coordinate and image loading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from shared.tile_math import GeoPoint, TileCoord, tile_center_gps

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TileEntry:
    """A single tile in the map pack."""
    coord: TileCoord
    path: str
    center: GeoPoint


@dataclass(slots=True)
class MapPackInfo:
    """Metadata about a loaded map pack."""
    center: GeoPoint | None = None
    radius_km: float = 0.0
    zoom_levels: list[int] = field(default_factory=list)
    tile_count: int = 0
    has_index: bool = False
    pack_dir: Path | None = None


class MapLoader:
    """Loads and provides access to a map pack on the drone.

    Usage:
        loader = MapLoader(Path("/opt/vps/maps/current"))
        loader.load()
        tile = loader.get_tile(TileCoord(17, 70405, 43000))
        img = loader.load_image(tile)
    """

    def __init__(self, pack_dir: Path):
        self._pack_dir = pack_dir
        self._tiles: dict[tuple[int, int, int], TileEntry] = {}
        self._tiles_by_zoom: dict[int, list[TileEntry]] = {}
        self._info = MapPackInfo(pack_dir=pack_dir)
        self._loaded = False

    @property
    def info(self) -> MapPackInfo:
        return self._info

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def tile_count(self) -> int:
        return len(self._tiles)

    def load(self) -> bool:
        """Load map pack metadata and tile list. Returns True on success."""
        if not self._pack_dir.is_dir():
            logger.error("Map pack directory not found: %s", self._pack_dir)
            return False

        # Load metadata
        meta_path = self._pack_dir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if "center_lat" in meta and "center_lon" in meta:
                    self._info.center = GeoPoint(
                        lat=meta["center_lat"], lon=meta["center_lon"],
                    )
                self._info.radius_km = meta.get("radius_km", 0.0)
                self._info.zoom_levels = meta.get("zoom_levels", [])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse metadata: %s", e)

        # Load tile list
        tile_list_path = self._pack_dir / "index" / "tile_list.json"
        if not tile_list_path.exists():
            logger.error("tile_list.json not found in %s", self._pack_dir)
            return False

        try:
            with open(tile_list_path) as f:
                entries = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid tile_list.json: %s", e)
            return False

        for entry in entries:
            z = entry.get("z", 0)
            x = entry.get("x", 0)
            y = entry.get("y", 0)
            coord = TileCoord(z=z, x=x, y=y)
            center = tile_center_gps(coord)
            te = TileEntry(coord=coord, path=entry.get("path", ""), center=center)

            self._tiles[(z, x, y)] = te
            if z not in self._tiles_by_zoom:
                self._tiles_by_zoom[z] = []
            self._tiles_by_zoom[z].append(te)

        self._info.tile_count = len(self._tiles)
        self._info.has_index = (self._pack_dir / "index" / "faiss.index").exists()

        if not self._info.zoom_levels and self._tiles_by_zoom:
            self._info.zoom_levels = sorted(self._tiles_by_zoom.keys())

        self._loaded = True
        logger.info(
            "Map pack loaded: %d tiles, zoom=%s, index=%s",
            len(self._tiles), self._info.zoom_levels, self._info.has_index,
        )
        return True

    def get_tile(self, coord: TileCoord) -> TileEntry | None:
        """Get a tile entry by coordinate."""
        return self._tiles.get((coord.z, coord.x, coord.y))

    def get_tiles_at_zoom(self, zoom: int) -> list[TileEntry]:
        """Get all tiles at a zoom level."""
        return self._tiles_by_zoom.get(zoom, [])

    def load_image(self, tile: TileEntry) -> np.ndarray | None:
        """Load a tile image from disk."""
        path = self._pack_dir / tile.path
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        return img

    def get_tile_by_index(self, index: int, zoom: int | None = None) -> TileEntry | None:
        """Get a tile by its index in the tile list (for FAISS results)."""
        if zoom is not None:
            tiles = self.get_tiles_at_zoom(zoom)
            if 0 <= index < len(tiles):
                return tiles[index]
            return None

        # Without zoom, use insertion order
        all_tiles = list(self._tiles.values())
        if 0 <= index < len(all_tiles):
            return all_tiles[index]
        return None

    def nearest_tiles(self, point: GeoPoint, zoom: int, k: int = 5) -> list[TileEntry]:
        """Find the k nearest tiles to a GPS point at given zoom level.

        Simple brute-force search — for FAISS-less fallback.
        """
        from shared.tile_math import haversine_km

        tiles = self.get_tiles_at_zoom(zoom)
        if not tiles:
            return []

        dists = [(haversine_km(point, t.center), t) for t in tiles]
        dists.sort(key=lambda x: x[0])
        return [t for _, t in dists[:k]]
