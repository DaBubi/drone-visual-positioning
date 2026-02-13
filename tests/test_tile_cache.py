"""Tests for tile image cache."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from onboard.tile_cache import TileCache


@pytest.fixture
def tile_dir(tmp_path):
    """Create temporary tile images."""
    for i in range(5):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        path = tmp_path / f"tile_{i}.png"
        cv2.imwrite(str(path), img)
    return tmp_path


class TestTileCache:
    def test_loads_tile(self, tile_dir):
        cache = TileCache(max_tiles=10)
        img = cache.get(tile_dir / "tile_0.png")
        assert img is not None
        assert img.shape == (256, 256, 3)

    def test_cache_hit(self, tile_dir):
        cache = TileCache(max_tiles=10)
        path = tile_dir / "tile_0.png"
        img1 = cache.get(path)
        img2 = cache.get(path)
        assert img1 is img2  # same object (cached)
        assert cache._hits == 1
        assert cache._misses == 1  # first load was a miss

    def test_missing_file_returns_none(self, tmp_path):
        cache = TileCache()
        assert cache.get(tmp_path / "nonexistent.png") is None

    def test_eviction(self, tile_dir):
        cache = TileCache(max_tiles=3)
        for i in range(5):
            cache.get(tile_dir / f"tile_{i}.png")
        assert cache.size == 3  # oldest 2 evicted

    def test_lru_order(self, tile_dir):
        cache = TileCache(max_tiles=3)
        # Load tiles 0, 1, 2
        cache.get(tile_dir / "tile_0.png")
        cache.get(tile_dir / "tile_1.png")
        cache.get(tile_dir / "tile_2.png")
        # Access tile_0 again (moves to end)
        cache.get(tile_dir / "tile_0.png")
        # Load tile_3 — should evict tile_1 (oldest)
        cache.get(tile_dir / "tile_3.png")
        assert str(tile_dir / "tile_1.png") not in cache._cache
        assert str(tile_dir / "tile_0.png") in cache._cache

    def test_hit_rate(self, tile_dir):
        cache = TileCache()
        path = tile_dir / "tile_0.png"
        cache.get(path)  # miss
        cache.get(path)  # hit
        cache.get(path)  # hit
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_memory_estimate(self, tile_dir):
        cache = TileCache()
        cache.get(tile_dir / "tile_0.png")
        assert cache.memory_mb > 0

    def test_clear(self, tile_dir):
        cache = TileCache()
        cache.get(tile_dir / "tile_0.png")
        cache.clear()
        assert cache.size == 0
        assert cache.hit_rate == 0.0

    def test_stats(self, tile_dir):
        cache = TileCache(max_tiles=50)
        cache.get(tile_dir / "tile_0.png")
        s = cache.stats()
        assert s["size"] == 1
        assert s["max"] == 50
        assert s["misses"] == 1
