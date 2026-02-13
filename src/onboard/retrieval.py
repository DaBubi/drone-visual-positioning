"""Coarse localization via FAISS vector similarity search.

Loads a pre-built FAISS index of satellite tile descriptors (built by the
laptop programmer) and searches for the closest tiles to a drone frame descriptor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.tile_math import TileCoord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TileEntry:
    """A satellite tile with its metadata."""
    tile: TileCoord
    path: Path          # path to tile image file
    descriptor: np.ndarray | None = None  # global descriptor (loaded separately)


@dataclass(slots=True)
class RetrievalResult:
    """Result of a coarse retrieval query."""
    entries: list[TileEntry]
    distances: np.ndarray   # L2 distances to query


class TileIndex:
    """FAISS-based tile retrieval index."""

    def __init__(self, map_pack_dir: Path):
        self._map_pack = map_pack_dir
        self._index = None
        self._entries: list[TileEntry] = []

    def load(self) -> None:
        """Load FAISS index and tile metadata from map pack."""
        import faiss

        index_path = self._map_pack / "index" / "faiss.index"
        tile_list_path = self._map_pack / "index" / "tile_list.json"

        self._index = faiss.read_index(str(index_path))

        with open(tile_list_path) as f:
            tile_list = json.load(f)

        self._entries = []
        for entry in tile_list:
            self._entries.append(TileEntry(
                tile=TileCoord(z=entry["z"], x=entry["x"], y=entry["y"]),
                path=self._map_pack / entry["path"],
            ))

        logger.info("Loaded tile index: %d tiles, dim=%d",
                     len(self._entries), self._index.d)

    def search(self, descriptor: np.ndarray, k: int = 5) -> RetrievalResult:
        """Find the k nearest tiles to the query descriptor.

        Args:
            descriptor: (D,) global image descriptor
            k: number of candidates to return
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() first.")

        k = min(k, len(self._entries))
        query = descriptor.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query, k)

        entries = [self._entries[i] for i in indices[0] if i >= 0]
        return RetrievalResult(entries=entries, distances=distances[0])

    @property
    def num_tiles(self) -> int:
        return len(self._entries)
