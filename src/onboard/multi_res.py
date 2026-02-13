"""Multi-resolution tile matching strategy.

Uses zoom 17 for coarse area identification, then refines
with zoom 19 tiles around the coarse match for sub-meter precision.

Zoom 17: ~1.19 m/px at equator, tiles cover ~305m each
Zoom 19: ~0.30 m/px at equator, tiles cover ~76m each

Strategy:
1. Match drone frame against z17 index → coarse GPS (~5m accuracy)
2. Load z19 tiles near the coarse match
3. Match against z19 tiles → refined GPS (~0.3m accuracy)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from onboard.homography import match_and_localize, HomographyResult
from onboard.matcher import MatchResult
from onboard.retrieval import TileIndex, RetrievalResult
from shared.tile_math import GeoPoint, TileCoord, gps_to_tile, tile_pixel_to_gps

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MultiResResult:
    """Result from multi-resolution matching."""
    position: GeoPoint
    confidence: float
    coarse_tile: TileCoord     # z17 tile that matched
    fine_tile: TileCoord | None  # z19 tile if refinement succeeded
    coarse_inlier_ratio: float
    fine_inlier_ratio: float
    refined: bool              # True if z19 refinement was used


def refine_with_zoom19(
    frame: np.ndarray,
    coarse_position: GeoPoint,
    matcher,
    tile_index_z19: TileIndex,
    min_matches: int = 15,
    min_inlier_ratio: float = 0.3,
) -> tuple[GeoPoint, float, TileCoord] | None:
    """Attempt z19 refinement around a coarse z17 position.

    Searches for the z19 tile containing the coarse position and its
    8 neighbors, then performs fine matching.

    Args:
        frame: drone camera frame
        coarse_position: GPS from z17 coarse match
        matcher: feature matcher (OrbMatcher or OnnxMatcher)
        tile_index_z19: FAISS index for zoom 19 tiles
        min_matches: minimum feature matches
        min_inlier_ratio: minimum inlier ratio to accept

    Returns:
        (refined_position, inlier_ratio, tile) or None
    """
    # Find the z19 tile at the coarse position
    center_tile = gps_to_tile(coarse_position, 19)

    # Search 3x3 neighborhood
    neighbor_tiles = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            neighbor_tiles.append(TileCoord(z=19, x=center_tile.x + dx, y=center_tile.y + dy))

    # Try matching against each z19 tile from the index
    descriptor = matcher.extract_global_descriptor(frame)
    candidates = tile_index_z19.search(descriptor, k=min(20, tile_index_z19.num_tiles))

    # Prioritize tiles in the neighborhood of the coarse match
    neighbor_set = {(t.x, t.y) for t in neighbor_tiles}
    nearby = [e for e in candidates.entries if (e.tile.x, e.tile.y) in neighbor_set]
    others = [e for e in candidates.entries if (e.tile.x, e.tile.y) not in neighbor_set]
    ordered = nearby + others[:3]  # neighbors first, then a few more

    h, w = frame.shape[:2]
    for entry in ordered:
        tile_img = cv2.imread(str(entry.path))
        if tile_img is None:
            continue

        match_result = matcher.match(frame, tile_img)
        if match_result.num_matches < min_matches:
            continue

        result = match_and_localize(
            match_result.drone_pts,
            match_result.tile_pts,
            (w, h),
            entry.tile,
            min_inlier_ratio=min_inlier_ratio,
        )

        if result is not None:
            return result.position, result.inlier_ratio, entry.tile

    return None


def match_multi_resolution(
    frame: np.ndarray,
    matcher,
    tile_index_z17: TileIndex,
    tile_index_z19: TileIndex | None,
    max_candidates: int = 5,
    min_matches: int = 15,
    min_inlier_ratio: float = 0.3,
) -> MultiResResult | None:
    """Full multi-resolution matching pipeline.

    1. Coarse match at z17
    2. If z19 index available, refine position

    Args:
        frame: drone camera frame (BGR)
        matcher: feature matcher
        tile_index_z17: FAISS index for zoom 17
        tile_index_z19: FAISS index for zoom 19 (optional)
        max_candidates: top-k for coarse retrieval
        min_matches: minimum feature matches
        min_inlier_ratio: minimum inlier ratio

    Returns:
        MultiResResult or None if no match found
    """
    # Step 1: Coarse z17 match
    descriptor = matcher.extract_global_descriptor(frame)
    candidates = tile_index_z17.search(descriptor, k=max_candidates)

    h, w = frame.shape[:2]
    coarse_position = None
    coarse_tile = None
    coarse_inlier = 0.0

    for entry in candidates.entries:
        tile_img = cv2.imread(str(entry.path))
        if tile_img is None:
            continue

        match_result = matcher.match(frame, tile_img)
        if match_result.num_matches < min_matches:
            continue

        result = match_and_localize(
            match_result.drone_pts,
            match_result.tile_pts,
            (w, h),
            entry.tile,
            min_inlier_ratio=min_inlier_ratio,
        )

        if result is not None:
            coarse_position = result.position
            coarse_tile = entry.tile
            coarse_inlier = result.inlier_ratio
            break

    if coarse_position is None or coarse_tile is None:
        return None

    # Step 2: Refine with z19 if available
    if tile_index_z19 is not None and tile_index_z19.num_tiles > 0:
        refined = refine_with_zoom19(
            frame, coarse_position, matcher, tile_index_z19,
            min_matches=min_matches,
            min_inlier_ratio=min_inlier_ratio,
        )

        if refined is not None:
            fine_pos, fine_inlier, fine_tile = refined
            return MultiResResult(
                position=fine_pos,
                confidence=fine_inlier,
                coarse_tile=coarse_tile,
                fine_tile=fine_tile,
                coarse_inlier_ratio=coarse_inlier,
                fine_inlier_ratio=fine_inlier,
                refined=True,
            )

    # Fall back to coarse position
    return MultiResResult(
        position=coarse_position,
        confidence=coarse_inlier,
        coarse_tile=coarse_tile,
        fine_tile=None,
        coarse_inlier_ratio=coarse_inlier,
        fine_inlier_ratio=0.0,
        refined=False,
    )
