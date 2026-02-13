"""CLI for the laptop map programmer.

Usage:
    vps-program download --center LAT,LON --radius-km 5 --zoom 17,19 --api-key KEY
    vps-program build-index ./map_pack/
    vps-program package ./map_pack/ --output map_pack.tar.gz
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from shared.tile_math import GeoPoint
from programmer.downloader import download_area
from programmer.indexer import build_index
from programmer.map_pack import (
    MapPackMetadata,
    make_tile_entry,
    save_metadata,
    save_tile_list,
)
from programmer.packager import package


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Drone VPS Map Programmer — prepare satellite imagery for onboard matching."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


@cli.command()
@click.option("--center", required=True, help="Center coordinate as LAT,LON")
@click.option("--radius-km", required=True, type=float, help="Area radius in km")
@click.option("--zoom", default="17,19", help="Comma-separated zoom levels")
@click.option("--api-key", required=True, envvar="MAPBOX_API_KEY", help="Mapbox API key")
@click.option("--output", "-o", default="./map_pack", help="Output directory")
@click.option("--concurrency", default=10, type=int, help="Max concurrent downloads")
def download(center: str, radius_km: float, zoom: str, api_key: str, output: str, concurrency: int):
    """Download satellite tiles for an area."""
    lat, lon = map(float, center.split(","))
    center_pt = GeoPoint(lat=lat, lon=lon)
    zoom_levels = [int(z.strip()) for z in zoom.split(",")]
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Downloading tiles: center=({lat:.4f}, {lon:.4f}), "
               f"radius={radius_km}km, zoom={zoom_levels}")

    results = asyncio.run(download_area(
        center=center_pt,
        radius_km=radius_km,
        zoom_levels=zoom_levels,
        output_dir=output_dir,
        api_key=api_key,
        concurrency=concurrency,
    ))

    # Save metadata and tile list
    metadata = MapPackMetadata(
        center_lat=lat,
        center_lon=lon,
        radius_km=radius_km,
        zoom_levels=zoom_levels,
        tile_count=len(results),
    )
    save_metadata(output_dir, metadata)

    entries = [make_tile_entry(tile) for tile, _ in results]
    save_tile_list(output_dir, entries)

    click.echo(f"Downloaded {len(results)} tiles to {output_dir}")


@cli.command("build-index")
@click.argument("pack_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--onnx/--orb", default=False, help="Use SuperPoint ONNX (default: ORB)")
@click.option("--model", type=click.Path(path_type=Path), help="SuperPoint ONNX model path")
def build_index_cmd(pack_dir: Path, onnx: bool, model: Path | None):
    """Build FAISS retrieval index from downloaded tiles."""
    click.echo(f"Building index for {pack_dir}")
    build_index(pack_dir, use_onnx=onnx, superpoint_path=model)
    click.echo("Index built successfully")


@cli.command()
@click.argument("pack_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output archive path")
@click.option("--include-models", is_flag=True, help="Include ONNX models in archive")
@click.option("--models-dir", type=click.Path(path_type=Path), help="Models directory")
def package_cmd(pack_dir: Path, output: Path | None, include_models: bool, models_dir: Path | None):
    """Package map pack into tar.gz for transfer to drone."""
    result = package(pack_dir, output, include_models, models_dir)
    click.echo(f"Package created: {result}")


if __name__ == "__main__":
    cli()
