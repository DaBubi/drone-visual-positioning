"""Async satellite tile downloader.

Downloads Web Mercator tiles from Mapbox (or other providers)
for a circular area defined by center + radius.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiohttp

from shared.tile_math import GeoPoint, TileCoord, tiles_in_radius
from programmer.map_pack import tile_image_path

logger = logging.getLogger(__name__)

MAPBOX_URL = "https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token={token}"
OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


async def download_tile(
    session: aiohttp.ClientSession,
    tile: TileCoord,
    output_dir: Path,
    api_key: str,
    provider: str = "mapbox",
) -> Path | None:
    """Download a single satellite tile.

    Providers:
    - mapbox: High-res satellite (requires API key)
    - osm: OpenStreetMap (no key needed, map tiles not satellite)
    - esri: ESRI World Imagery (no key needed, satellite)

    Returns the saved file path, or None on failure.
    """
    rel_path = tile_image_path(tile)
    out_path = output_dir / rel_path
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if provider == "mapbox":
        url = MAPBOX_URL.format(z=tile.z, x=tile.x, y=tile.y, token=api_key)
    elif provider == "osm":
        url = OSM_URL.format(z=tile.z, x=tile.x, y=tile.y)
    elif provider == "esri":
        url = ESRI_URL.format(z=tile.z, x=tile.x, y=tile.y)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                out_path.write_bytes(data)
                return out_path
            else:
                logger.warning("Failed to download tile %s: HTTP %d", tile, resp.status)
                return None
    except Exception as e:
        logger.error("Error downloading tile %s: %s", tile, e)
        return None


async def download_area(
    center: GeoPoint,
    radius_km: float,
    zoom_levels: list[int],
    output_dir: Path,
    api_key: str,
    concurrency: int = 10,
    provider: str = "mapbox",
) -> list[tuple[TileCoord, Path]]:
    """Download all tiles in a circular area at specified zoom levels.

    Args:
        center: center GPS coordinate
        radius_km: radius in kilometers
        zoom_levels: list of zoom levels to download (e.g., [17, 19])
        output_dir: base directory for the map pack
        api_key: tile provider API key
        concurrency: max concurrent downloads
        provider: tile provider name

    Returns:
        List of (tile, path) pairs for successfully downloaded tiles
    """
    # Collect all tiles across zoom levels
    all_tiles: list[TileCoord] = []
    for z in zoom_levels:
        tiles = tiles_in_radius(center, radius_km, z)
        all_tiles.extend(tiles)
        logger.info("Zoom %d: %d tiles", z, len(tiles))

    logger.info("Total tiles to download: %d", len(all_tiles))

    results: list[tuple[TileCoord, Path]] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _download(tile: TileCoord):
        async with semaphore:
            path = await download_tile(
                session, tile, output_dir, api_key, provider
            )
            if path is not None:
                results.append((tile, path))

    headers = {"User-Agent": "DroneVPS/0.1 (visual positioning research)"}
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        tasks = [_download(t) for t in all_tiles]
        await asyncio.gather(*tasks)

    logger.info("Downloaded %d/%d tiles", len(results), len(all_tiles))
    return results
