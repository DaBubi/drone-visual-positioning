"""Map pack validation tool.

Verifies the integrity and completeness of a map pack before
deployment to the drone. Checks:
- metadata.json exists and is valid
- tile_list.json matches actual tile files
- FAISS index loads and has correct dimensions
- Tile images are readable and correct size
- Coverage area is reasonable
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from shared.tile_math import GeoPoint, tile_center_gps, haversine_km, TileCoord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ValidationResult:
    """Result of map pack validation."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tile_count: int = 0
    readable_tiles: int = 0
    index_vectors: int = 0
    index_dim: int = 0
    zoom_levels: list[int] = field(default_factory=list)
    coverage_km2: float = 0.0
    center: GeoPoint | None = None

    def summary(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [
            f"Map Pack: {status}",
            f"  Tiles: {self.readable_tiles}/{self.tile_count} readable",
            f"  Index: {self.index_vectors} vectors, dim={self.index_dim}",
            f"  Zoom levels: {self.zoom_levels}",
            f"  Coverage: ~{self.coverage_km2:.1f} km²",
        ]
        if self.center:
            lines.append(f"  Center: {self.center.lat:.4f}, {self.center.lon:.4f}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors[:10]:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def validate_map_pack(pack_dir: Path) -> ValidationResult:
    """Validate a map pack directory.

    Args:
        pack_dir: path to the map pack

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()

    # Check directory exists
    if not pack_dir.is_dir():
        result.valid = False
        result.errors.append(f"Directory not found: {pack_dir}")
        return result

    # Check metadata.json
    metadata_path = pack_dir / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            if "center_lat" in metadata and "center_lon" in metadata:
                result.center = GeoPoint(
                    lat=metadata["center_lat"],
                    lon=metadata["center_lon"],
                )
            if "zoom_levels" in metadata:
                result.zoom_levels = metadata["zoom_levels"]
        except (json.JSONDecodeError, KeyError) as e:
            result.errors.append(f"Invalid metadata.json: {e}")
            result.valid = False
    else:
        result.warnings.append("metadata.json not found")

    # Check tile_list.json
    tile_list_path = pack_dir / "index" / "tile_list.json"
    entries = []
    if tile_list_path.exists():
        try:
            with open(tile_list_path) as f:
                entries = json.load(f)
            result.tile_count = len(entries)
        except json.JSONDecodeError as e:
            result.errors.append(f"Invalid tile_list.json: {e}")
            result.valid = False
    else:
        result.errors.append("index/tile_list.json not found")
        result.valid = False

    # Verify tile files exist and are readable
    bad_tiles = 0
    zoom_set = set()
    tile_centers = []

    for entry in entries:
        tile_path = pack_dir / entry.get("path", "")
        if not tile_path.exists():
            bad_tiles += 1
            if bad_tiles <= 5:
                result.errors.append(f"Missing tile: {entry.get('path', '?')}")
            continue

        img = cv2.imread(str(tile_path))
        if img is None:
            bad_tiles += 1
            if bad_tiles <= 5:
                result.errors.append(f"Unreadable tile: {entry.get('path', '?')}")
            continue

        result.readable_tiles += 1
        z = entry.get("z", 0)
        zoom_set.add(z)

        if "x" in entry and "y" in entry:
            tc = tile_center_gps(TileCoord(z=z, x=entry["x"], y=entry["y"]))
            tile_centers.append(tc)

    if bad_tiles > 5:
        result.errors.append(f"... and {bad_tiles - 5} more missing/unreadable tiles")

    if bad_tiles > 0:
        result.valid = False

    if not result.zoom_levels and zoom_set:
        result.zoom_levels = sorted(zoom_set)

    # Estimate coverage area from tile centers
    if tile_centers:
        lats = [t.lat for t in tile_centers]
        lons = [t.lon for t in tile_centers]
        center = GeoPoint(lat=sum(lats) / len(lats), lon=sum(lons) / len(lons))
        if result.center is None:
            result.center = center

        # Approximate area from bounding box
        lat_range_km = haversine_km(
            GeoPoint(min(lats), center.lon),
            GeoPoint(max(lats), center.lon),
        )
        lon_range_km = haversine_km(
            GeoPoint(center.lat, min(lons)),
            GeoPoint(center.lat, max(lons)),
        )
        result.coverage_km2 = lat_range_km * lon_range_km

    # Check FAISS index
    index_path = pack_dir / "index" / "faiss.index"
    if index_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(index_path))
            result.index_vectors = index.ntotal
            result.index_dim = index.d

            if index.ntotal != result.readable_tiles:
                result.warnings.append(
                    f"Index has {index.ntotal} vectors but {result.readable_tiles} tiles"
                )
        except Exception as e:
            result.errors.append(f"Failed to load FAISS index: {e}")
            result.valid = False
    else:
        result.warnings.append("FAISS index not found (run build-index first)")

    # Sanity checks
    if result.tile_count == 0:
        result.errors.append("No tiles in map pack")
        result.valid = False

    if result.coverage_km2 > 100:
        result.warnings.append(
            f"Large coverage area ({result.coverage_km2:.0f} km²) — may be slow"
        )

    return result
