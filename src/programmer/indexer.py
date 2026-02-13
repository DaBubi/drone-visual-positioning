"""Feature extraction and FAISS index builder.

Processes downloaded satellite tiles:
1. Extracts a global descriptor per tile (for coarse retrieval)
2. Builds a FAISS index from all descriptors
3. Optionally pre-extracts keypoints per tile (for faster onboard matching)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from shared.tile_math import TileCoord
from programmer.map_pack import TileListEntry, load_tile_list, save_tile_list

logger = logging.getLogger(__name__)


def extract_orb_global_descriptor(image: np.ndarray, orb: cv2.ORB) -> np.ndarray:
    """Extract a global descriptor by averaging ORB feature descriptors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, desc = orb.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        return np.zeros(32, dtype=np.float32)
    return desc.astype(np.float32).mean(axis=0)


def extract_superpoint_global_descriptor(
    image: np.ndarray, sp_session
) -> np.ndarray:
    """Extract global descriptor via SuperPoint (ONNX) average pooling."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    inp = gray.astype(np.float32) / 255.0
    inp = inp[np.newaxis, np.newaxis, :, :]
    outputs = sp_session.run(None, {"image": inp})
    descriptors = outputs[1][0]  # (N, D)
    if len(descriptors) == 0:
        return np.zeros(256, dtype=np.float32)
    return descriptors.mean(axis=0).astype(np.float32)


def build_index(
    pack_dir: Path,
    use_onnx: bool = False,
    superpoint_path: Path | None = None,
) -> None:
    """Build FAISS index from tile images in a map pack.

    Args:
        pack_dir: map pack directory containing tiles/ and index/tile_list.json
        use_onnx: use SuperPoint ONNX for descriptors (else ORB fallback)
        superpoint_path: path to SuperPoint ONNX model (required if use_onnx)
    """
    import faiss

    entries = load_tile_list(pack_dir)
    logger.info("Building index for %d tiles", len(entries))

    # Initialize extractor
    sp_session = None
    orb = None
    if use_onnx and superpoint_path is not None:
        import onnxruntime as ort
        sp_session = ort.InferenceSession(str(superpoint_path))
        desc_dim = 256
    else:
        orb = cv2.ORB_create(nfeatures=1000)
        desc_dim = 32

    descriptors = []
    valid_entries = []

    for i, entry in enumerate(entries):
        img_path = pack_dir / entry.path
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read tile: %s", img_path)
            continue

        if sp_session is not None:
            desc = extract_superpoint_global_descriptor(img, sp_session)
        else:
            desc = extract_orb_global_descriptor(img, orb)

        descriptors.append(desc)
        valid_entries.append(entry)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d tiles", i + 1, len(entries))

    if not descriptors:
        raise RuntimeError("No valid tile descriptors extracted")

    # Build FAISS index
    desc_matrix = np.stack(descriptors).astype(np.float32)
    logger.info("Descriptor matrix: %s", desc_matrix.shape)

    # Use flat L2 index for small datasets, IVF for larger
    if len(descriptors) < 10000:
        index = faiss.IndexFlatL2(desc_dim)
    else:
        nlist = min(256, len(descriptors) // 10)
        quantizer = faiss.IndexFlatL2(desc_dim)
        index = faiss.IndexIVFFlat(quantizer, desc_dim, nlist)
        index.train(desc_matrix)

    index.add(desc_matrix)

    # Save
    index_dir = pack_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    np.save(str(index_dir / "descriptors.npy"), desc_matrix)

    # Update tile list with only valid entries
    save_tile_list(pack_dir, valid_entries)

    logger.info("Index built: %d vectors, dim=%d", index.ntotal, desc_dim)
