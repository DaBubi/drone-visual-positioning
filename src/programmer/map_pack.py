"""Map pack format definition.

A map pack is a directory containing satellite tiles, a FAISS index,
and metadata — everything the onboard module needs to localize.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from shared.tile_math import GeoPoint, TileCoord, tile_center_gps


@dataclass
class MapPackMetadata:
    """Metadata for a map pack."""
    center_lat: float
    center_lon: float
    radius_km: float
    zoom_levels: list[int]
    tile_count: int
    created_at: str = ""
    version: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class TileListEntry:
    """An entry in the tile list JSON."""
    z: int
    x: int
    y: int
    path: str       # relative path within map pack
    lat: float      # center latitude
    lon: float      # center longitude


def tile_image_path(tile: TileCoord) -> str:
    """Relative path for a tile image within the map pack."""
    return f"tiles/{tile.z}/{tile.x}/{tile.y}.png"


def save_metadata(pack_dir: Path, metadata: MapPackMetadata) -> None:
    """Save map pack metadata."""
    meta_path = pack_dir / "metadata.json"
    meta_path.write_text(json.dumps(asdict(metadata), indent=2))


def load_metadata(pack_dir: Path) -> MapPackMetadata:
    """Load map pack metadata."""
    meta_path = pack_dir / "metadata.json"
    data = json.loads(meta_path.read_text())
    return MapPackMetadata(**data)


def save_tile_list(pack_dir: Path, entries: list[TileListEntry]) -> None:
    """Save the tile list for the FAISS index."""
    index_dir = pack_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    tile_list_path = index_dir / "tile_list.json"
    tile_list_path.write_text(json.dumps([asdict(e) for e in entries], indent=2))


def load_tile_list(pack_dir: Path) -> list[TileListEntry]:
    """Load the tile list."""
    tile_list_path = pack_dir / "index" / "tile_list.json"
    data = json.loads(tile_list_path.read_text())
    return [TileListEntry(**e) for e in data]


def make_tile_entry(tile: TileCoord) -> TileListEntry:
    """Create a tile list entry with computed center GPS."""
    center = tile_center_gps(tile)
    return TileListEntry(
        z=tile.z, x=tile.x, y=tile.y,
        path=tile_image_path(tile),
        lat=center.lat, lon=center.lon,
    )
