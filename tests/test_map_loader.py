"""Tests for map pack loader."""

import json

import cv2
import numpy as np
import pytest

from onboard.map_loader import MapLoader, TileEntry, MapPackInfo
from shared.tile_math import GeoPoint, TileCoord


def _create_pack(tmp_path, num_tiles=5, zoom=17, create_index=False):
    """Create a minimal map pack for testing."""
    pack = tmp_path / "map_pack"
    pack.mkdir()
    tiles_dir = pack / "tiles" / str(zoom)
    tiles_dir.mkdir(parents=True)
    index_dir = pack / "index"
    index_dir.mkdir()

    entries = []
    for i in range(num_tiles):
        tile_x = 70400 + i
        tile_dir = tiles_dir / str(tile_x)
        tile_dir.mkdir()
        tile_path = tile_dir / "43000.png"
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(tile_path), img)
        entries.append({
            "z": zoom, "x": tile_x, "y": 43000,
            "path": f"tiles/{zoom}/{tile_x}/43000.png",
        })

    with open(index_dir / "tile_list.json", "w") as f:
        json.dump(entries, f)

    metadata = {
        "center_lat": 52.52,
        "center_lon": 13.405,
        "radius_km": 1.0,
        "zoom_levels": [zoom],
        "tile_count": num_tiles,
    }
    with open(pack / "metadata.json", "w") as f:
        json.dump(metadata, f)

    if create_index:
        (index_dir / "faiss.index").touch()

    return pack


class TestMapLoader:
    def test_load_success(self, tmp_path):
        pack = _create_pack(tmp_path)
        loader = MapLoader(pack)
        assert loader.load()
        assert loader.is_loaded
        assert loader.tile_count == 5

    def test_load_nonexistent(self, tmp_path):
        loader = MapLoader(tmp_path / "nonexistent")
        assert not loader.load()

    def test_load_metadata(self, tmp_path):
        pack = _create_pack(tmp_path)
        loader = MapLoader(pack)
        loader.load()
        assert loader.info.center is not None
        assert loader.info.center.lat == pytest.approx(52.52)
        assert loader.info.zoom_levels == [17]

    def test_get_tile(self, tmp_path):
        pack = _create_pack(tmp_path)
        loader = MapLoader(pack)
        loader.load()
        tile = loader.get_tile(TileCoord(17, 70400, 43000))
        assert tile is not None
        assert tile.coord.x == 70400

    def test_get_missing_tile(self, tmp_path):
        pack = _create_pack(tmp_path)
        loader = MapLoader(pack)
        loader.load()
        assert loader.get_tile(TileCoord(17, 99999, 99999)) is None

    def test_get_tiles_at_zoom(self, tmp_path):
        pack = _create_pack(tmp_path, num_tiles=3)
        loader = MapLoader(pack)
        loader.load()
        tiles = loader.get_tiles_at_zoom(17)
        assert len(tiles) == 3
        assert loader.get_tiles_at_zoom(19) == []

    def test_load_image(self, tmp_path):
        pack = _create_pack(tmp_path)
        loader = MapLoader(pack)
        loader.load()
        tile = loader.get_tile(TileCoord(17, 70400, 43000))
        img = loader.load_image(tile)
        assert img is not None
        assert img.shape == (256, 256, 3)

    def test_get_tile_by_index(self, tmp_path):
        pack = _create_pack(tmp_path, num_tiles=5)
        loader = MapLoader(pack)
        loader.load()
        tile = loader.get_tile_by_index(0)
        assert tile is not None
        assert loader.get_tile_by_index(100) is None

    def test_has_index(self, tmp_path):
        pack = _create_pack(tmp_path, create_index=True)
        loader = MapLoader(pack)
        loader.load()
        assert loader.info.has_index

    def test_no_index(self, tmp_path):
        pack = _create_pack(tmp_path, create_index=False)
        loader = MapLoader(pack)
        loader.load()
        assert not loader.info.has_index

    def test_nearest_tiles(self, tmp_path):
        pack = _create_pack(tmp_path, num_tiles=5)
        loader = MapLoader(pack)
        loader.load()
        center = GeoPoint(52.52, 13.405)
        nearest = loader.nearest_tiles(center, zoom=17, k=3)
        assert len(nearest) == 3

    def test_tile_entry_has_center(self, tmp_path):
        pack = _create_pack(tmp_path)
        loader = MapLoader(pack)
        loader.load()
        tile = loader.get_tile(TileCoord(17, 70400, 43000))
        assert tile.center is not None
        assert isinstance(tile.center, GeoPoint)
