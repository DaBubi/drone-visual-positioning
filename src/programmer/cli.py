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
@click.option("--api-key", envvar="MAPBOX_API_KEY", default="", help="Mapbox API key (required for mapbox provider)")
@click.option("--provider", type=click.Choice(["mapbox", "osm", "esri"]),
              default="esri", help="Tile provider (esri=satellite, no key needed)")
@click.option("--output", "-o", default="./map_pack", help="Output directory")
@click.option("--concurrency", default=10, type=int, help="Max concurrent downloads")
def download(center: str, radius_km: float, zoom: str, api_key: str, provider: str, output: str, concurrency: int):
    """Download satellite tiles for an area."""
    lat, lon = map(float, center.split(","))
    center_pt = GeoPoint(lat=lat, lon=lon)
    zoom_levels = [int(z.strip()) for z in zoom.split(",")]
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Downloading tiles: center=({lat:.4f}, {lon:.4f}), "
               f"radius={radius_km}km, zoom={zoom_levels}")

    if provider == "mapbox" and not api_key:
        raise click.UsageError("Mapbox provider requires --api-key or MAPBOX_API_KEY env var")

    results = asyncio.run(download_area(
        center=center_pt,
        radius_km=radius_km,
        zoom_levels=zoom_levels,
        output_dir=output_dir,
        api_key=api_key,
        concurrency=concurrency,
        provider=provider,
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


@cli.command()
@click.argument("pack_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--trajectory", type=click.Choice(["circle", "line", "lawnmower"]),
              default="circle", help="Flight path type")
@click.option("--center", required=True, help="Center as LAT,LON")
@click.option("--duration", type=float, default=60.0, help="Flight duration in seconds")
@click.option("--speed", type=float, default=10.0, help="Speed in m/s")
@click.option("--output", "-o", type=click.Path(path_type=Path),
              default=Path("simulation_results.json"))
def simulate(pack_dir: Path, trajectory: str, center: str, duration: float,
             speed: float, output: Path):
    """Simulate a flight path over a map pack and evaluate matching accuracy."""
    from programmer.simulate import (
        generate_circle, generate_line, generate_lawnmower,
        SimulationResult, save_results,
    )

    lat, lon = map(float, center.split(","))
    center_pt = GeoPoint(lat=lat, lon=lon)

    if trajectory == "circle":
        points = generate_circle(center_pt, speed_mps=speed, duration=duration)
    elif trajectory == "line":
        points = generate_line(center_pt, speed_mps=speed, duration=duration)
    else:
        points = generate_lawnmower(center_pt, speed_mps=speed, duration=duration)

    click.echo(f"Generated {len(points)} trajectory points ({trajectory}, {duration}s)")
    click.echo(f"Map pack: {pack_dir}")

    # For now, save trajectory — full simulation requires matcher setup
    results = SimulationResult(
        trajectory=points,
        matched=[False] * len(points),
        errors_m=[0.0] * len(points),
        mean_error_m=0.0,
        max_error_m=0.0,
        fix_rate=0.0,
        total_time_s=duration,
    )
    save_results(results, output)
    click.echo(f"Trajectory saved to {output}")


@cli.command("export-models")
@click.option("--output-dir", "-o", type=click.Path(path_type=Path),
              default=Path("./models"), help="Output directory for ONNX files")
@click.option("--image-size", type=int, default=640, help="Expected input image size")
@click.option("--superpoint-only", is_flag=True, help="Only export SuperPoint")
def export_models(output_dir: Path, image_size: int, superpoint_only: bool):
    """Export SuperPoint + LightGlue models to ONNX for RPi deployment."""
    from programmer.export_onnx import export_lightglue, export_superpoint

    click.echo(f"Exporting models to {output_dir}")
    export_superpoint(output_dir / "superpoint.onnx", image_size)
    click.echo("SuperPoint exported.")

    if not superpoint_only:
        try:
            export_lightglue(output_dir / "lightglue.onnx")
            click.echo("LightGlue exported.")
        except Exception as e:
            click.echo(f"LightGlue export failed: {e}")
            click.echo("Use --use-orb-fallback on RPi instead.")


if __name__ == "__main__":
    cli()
