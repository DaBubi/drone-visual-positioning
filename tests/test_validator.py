"""Tests for map pack validator."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from programmer.validator import validate_map_pack


def _create_map_pack(tmp_path, num_tiles=5, create_index=False):
    """Helper to create a minimal valid map pack."""
    pack = tmp_path / "map_pack"
    pack.mkdir()
    tiles_dir = pack / "tiles" / "17"
    tiles_dir.mkdir(parents=True)
    index_dir = pack / "index"
    index_dir.mkdir()

    entries = []
    for i in range(num_tiles):
        tile_dir = tiles_dir / str(70400 + i)
        tile_dir.mkdir()
        tile_path = tile_dir / "43000.png"
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(tile_path), img)
        entries.append({
            "z": 17, "x": 70400 + i, "y": 43000,
            "path": f"tiles/17/{70400 + i}/43000.png",
        })

    with open(index_dir / "tile_list.json", "w") as f:
        json.dump(entries, f)

    metadata = {
        "center_lat": 52.52,
        "center_lon": 13.405,
        "radius_km": 1.0,
        "zoom_levels": [17],
        "tile_count": num_tiles,
    }
    with open(pack / "metadata.json", "w") as f:
        json.dump(metadata, f)

    if create_index:
        import faiss
        dim = 32
        index = faiss.IndexFlatL2(dim)
        vecs = np.random.randn(num_tiles, dim).astype(np.float32)
        index.add(vecs)
        faiss.write_index(index, str(index_dir / "faiss.index"))

    return pack


class TestValidator:
    def test_valid_pack(self, tmp_path):
        pack = _create_map_pack(tmp_path, num_tiles=5, create_index=True)
        result = validate_map_pack(pack)
        assert result.valid
        assert result.tile_count == 5
        assert result.readable_tiles == 5
        assert result.index_vectors == 5
        assert len(result.errors) == 0

    def test_missing_directory(self, tmp_path):
        result = validate_map_pack(tmp_path / "nonexistent")
        assert not result.valid
        assert any("not found" in e.lower() for e in result.errors)

    def test_missing_tile_list(self, tmp_path):
        pack = tmp_path / "map_pack"
        pack.mkdir()
        (pack / "metadata.json").write_text("{}")
        result = validate_map_pack(pack)
        assert not result.valid

    def test_missing_tiles(self, tmp_path):
        pack = tmp_path / "map_pack"
        pack.mkdir()
        index_dir = pack / "index"
        index_dir.mkdir()
        # Create tile_list with entries pointing to non-existent files
        entries = [{"z": 17, "x": 99999, "y": 99999, "path": "tiles/17/99999/99999.png"}]
        with open(index_dir / "tile_list.json", "w") as f:
            json.dump(entries, f)
        result = validate_map_pack(pack)
        assert not result.valid
        assert result.readable_tiles == 0

    def test_metadata_parsed(self, tmp_path):
        pack = _create_map_pack(tmp_path)
        result = validate_map_pack(pack)
        assert result.center is not None
        assert abs(result.center.lat - 52.52) < 0.01
        assert result.zoom_levels == [17]

    def test_no_index_warning(self, tmp_path):
        pack = _create_map_pack(tmp_path, create_index=False)
        result = validate_map_pack(pack)
        assert result.valid  # still valid, just no index yet
        assert any("faiss" in w.lower() for w in result.warnings)

    def test_summary_string(self, tmp_path):
        pack = _create_map_pack(tmp_path, create_index=True)
        result = validate_map_pack(pack)
        s = result.summary()
        assert "VALID" in s
        assert "Tiles:" in s
