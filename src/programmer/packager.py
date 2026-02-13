"""Package a map pack into a transferable archive.

Bundles tiles, FAISS index, metadata, and optionally ONNX models
into a tar.gz for SCP/USB transfer to the drone's RPi.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path

from programmer.map_pack import load_metadata

logger = logging.getLogger(__name__)


def package(
    pack_dir: Path,
    output_path: Path | None = None,
    include_models: bool = False,
    models_dir: Path | None = None,
) -> Path:
    """Create a tar.gz archive of the map pack.

    Args:
        pack_dir: map pack directory
        output_path: output archive path (default: pack_dir.tar.gz)
        include_models: include ONNX model files in archive
        models_dir: directory containing ONNX models

    Returns:
        Path to the created archive
    """
    if output_path is None:
        output_path = pack_dir.with_suffix(".tar.gz")

    metadata = load_metadata(pack_dir)
    logger.info(
        "Packaging map pack: center=(%.4f, %.4f), radius=%.1fkm, %d tiles",
        metadata.center_lat, metadata.center_lon,
        metadata.radius_km, metadata.tile_count,
    )

    with tarfile.open(output_path, "w:gz") as tar:
        # Add all map pack contents
        for item in pack_dir.rglob("*"):
            if item.is_file():
                arcname = f"map_pack/{item.relative_to(pack_dir)}"
                tar.add(str(item), arcname=arcname)

        # Optionally include ONNX models
        if include_models and models_dir is not None:
            for model_file in models_dir.glob("*.onnx"):
                arcname = f"map_pack/models/{model_file.name}"
                tar.add(str(model_file), arcname=arcname)
                logger.info("Included model: %s", model_file.name)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Package created: %s (%.1f MB)", output_path, size_mb)
    return output_path
