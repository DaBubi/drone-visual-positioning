"""LRU tile image cache for onboard matching.

Keeps recently-accessed satellite tile images in memory to avoid
repeated disk reads during the matching loop. On RPi with 1-4GB RAM,
caching 50-100 tiles saves significant I/O latency.

Each tile at 256x256 BGR is ~192KB, so 100 tiles ≈ 19MB.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TileCache:
    """LRU cache for satellite tile images.

    Thread-safe for the single-threaded VPS loop (no locking needed).
    """

    def __init__(self, max_tiles: int = 100):
        self._max = max_tiles
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, path: Path) -> np.ndarray | None:
        """Get a tile image from cache, loading from disk if needed.

        Returns BGR numpy array or None if file doesn't exist.
        """
        key = str(path)

        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self._misses += 1
        img = cv2.imread(key)
        if img is None:
            return None

        self._cache[key] = img
        self._cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

        return img

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def memory_mb(self) -> float:
        """Approximate memory usage in MB."""
        total_bytes = sum(img.nbytes for img in self._cache.values())
        return total_bytes / (1024 * 1024)

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        return {
            "size": self.size,
            "max": self._max,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "memory_mb": f"{self.memory_mb:.1f}",
        }
