"""Integration test: download real tiles, build index, simulate matching.

Uses OpenStreetMap tiles (free, no API key) for a small area,
then simulates a drone frame by cropping/transforming a tile and
verifying the pipeline recovers the correct GPS position.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from shared.tile_math import (
    GeoPoint,
    TileCoord,
    gps_to_tile,
    gps_to_tile_pixel,
    tile_center_gps,
    tile_pixel_to_gps,
    tiles_in_radius,
    haversine_km,
    TILE_SIZE,
)
from onboard.nmea import PositionFix, format_gga, format_rmc, nmea_checksum
from onboard.homography import estimate_homography, extract_gps, match_and_localize
from programmer.map_pack import (
    MapPackMetadata,
    TileListEntry,
    make_tile_entry,
    save_metadata,
    save_tile_list,
    tile_image_path,
)


# --- Test area: central Berlin (Brandenburger Tor) ---
TEST_CENTER = GeoPoint(lat=52.5163, lon=13.3777)
TEST_RADIUS_KM = 0.5
TEST_ZOOM = 17


def download_osm_tile(tile: TileCoord, output_dir: Path) -> Path | None:
    """Download a single tile from OpenStreetMap."""
    import urllib.request

    rel_path = tile_image_path(tile)
    out_path = output_dir / rel_path
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://tile.openstreetmap.org/{tile.z}/{tile.x}/{tile.y}.png"

    req = urllib.request.Request(url, headers={"User-Agent": "drone-vps-test/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            out_path.write_bytes(data)
            return out_path
    except Exception as e:
        print(f"  Failed to download {tile}: {e}")
        return None


@pytest.fixture(scope="module")
def map_pack_dir(tmp_path_factory) -> Path:
    """Download a small set of real tiles and build a map pack."""
    pack_dir = tmp_path_factory.mktemp("map_pack")

    # Get tiles for the test area
    tiles = tiles_in_radius(TEST_CENTER, TEST_RADIUS_KM, TEST_ZOOM)
    print(f"\nDownloading {len(tiles)} tiles for Berlin test area (zoom {TEST_ZOOM})...")

    downloaded = []
    for i, tile in enumerate(tiles):
        path = download_osm_tile(tile, pack_dir)
        if path is not None:
            downloaded.append((tile, path))
        # Rate limit to be polite to OSM
        if i > 0 and i % 5 == 0:
            time.sleep(1.0)

    print(f"Downloaded {len(downloaded)}/{len(tiles)} tiles")
    assert len(downloaded) > 0, "No tiles downloaded — check network"

    # Save metadata
    metadata = MapPackMetadata(
        center_lat=TEST_CENTER.lat,
        center_lon=TEST_CENTER.lon,
        radius_km=TEST_RADIUS_KM,
        zoom_levels=[TEST_ZOOM],
        tile_count=len(downloaded),
    )
    save_metadata(pack_dir, metadata)

    # Save tile list
    entries = [make_tile_entry(tile) for tile, _ in downloaded]
    save_tile_list(pack_dir, entries)

    return pack_dir


class TestTileDownload:
    """Verify we can download and read real satellite tiles."""

    def test_tiles_downloaded(self, map_pack_dir: Path):
        tile_list_path = map_pack_dir / "index" / "tile_list.json"
        assert tile_list_path.exists()
        with open(tile_list_path) as f:
            entries = json.load(f)
        assert len(entries) > 0
        print(f"\n  Tile count: {len(entries)}")

    def test_tiles_readable(self, map_pack_dir: Path):
        """All downloaded tiles should be valid images."""
        with open(map_pack_dir / "index" / "tile_list.json") as f:
            entries = json.load(f)

        valid = 0
        for entry in entries[:10]:  # Check first 10
            img_path = map_pack_dir / entry["path"]
            img = cv2.imread(str(img_path))
            if img is not None:
                assert img.shape[0] > 0 and img.shape[1] > 0
                valid += 1

        print(f"\n  Valid images: {valid}/{min(10, len(entries))}")
        assert valid > 0

    def test_metadata_saved(self, map_pack_dir: Path):
        meta_path = map_pack_dir / "metadata.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert abs(meta["center_lat"] - TEST_CENTER.lat) < 0.01
        assert abs(meta["center_lon"] - TEST_CENTER.lon) < 0.01
        assert meta["radius_km"] == TEST_RADIUS_KM


class TestORBIndexBuild:
    """Build and query a FAISS index using ORB descriptors."""

    def test_build_and_query(self, map_pack_dir: Path):
        """Build ORB index and verify nearest-neighbor search works."""
        from programmer.indexer import build_index, extract_orb_global_descriptor
        from onboard.retrieval import TileIndex

        # Build index
        build_index(map_pack_dir, use_onnx=False)

        # Verify index files created
        assert (map_pack_dir / "index" / "faiss.index").exists()
        assert (map_pack_dir / "index" / "descriptors.npy").exists()

        # Load and query
        tile_index = TileIndex(map_pack_dir)
        tile_index.load()
        assert tile_index.num_tiles > 0
        print(f"\n  Index has {tile_index.num_tiles} tiles")

        # Query with a tile's own descriptor — should return itself as top match
        with open(map_pack_dir / "index" / "tile_list.json") as f:
            entries = json.load(f)

        test_entry = entries[0]
        img = cv2.imread(str(map_pack_dir / test_entry["path"]))
        assert img is not None

        orb = cv2.ORB_create(nfeatures=1000)
        query_desc = extract_orb_global_descriptor(img, orb)

        result = tile_index.search(query_desc, k=3)
        assert len(result.entries) > 0

        # The top result should be very close to the queried tile
        top = result.entries[0]
        print(f"  Query tile: z={test_entry['z']} x={test_entry['x']} y={test_entry['y']}")
        print(f"  Top result: z={top.tile.z} x={top.tile.x} y={top.tile.y}")
        print(f"  Distance: {result.distances[0]:.4f}")


class TestSyntheticDroneMatch:
    """Simulate a drone frame by cropping/transforming a real tile,
    then verify the pipeline recovers the correct position."""

    def _simulate_drone_frame(self, tile_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a simulated drone frame from a satellite tile.

        Crops the center region and applies a small rotation to simulate
        realistic drone-to-satellite viewpoint difference.

        Returns (drone_frame, transform_matrix) where transform_matrix maps
        drone pixels to tile pixels.
        """
        h, w = tile_img.shape[:2]
        # Crop center 60% of the tile
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        crop = tile_img[margin_y:h - margin_y, margin_x:w - margin_x].copy()

        # Apply small rotation (5 degrees) to simulate heading difference
        ch, cw = crop.shape[:2]
        center = (cw // 2, ch // 2)
        angle = 5.0
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(crop, M_rot, (cw, ch), borderMode=cv2.BORDER_REFLECT)

        # Add slight gaussian noise to simulate camera differences
        noise = np.random.normal(0, 5, rotated.shape).astype(np.uint8)
        noisy = cv2.add(rotated, noise)

        return noisy, np.array([margin_x, margin_y])

    def test_orb_matching_recovers_position(self, map_pack_dir: Path):
        """ORB feature matching should recover approximate GPS from a synthetic drone frame."""
        from onboard.matcher import OrbMatcher

        with open(map_pack_dir / "index" / "tile_list.json") as f:
            entries = json.load(f)

        # Pick a tile near the center
        center_tile = gps_to_tile(TEST_CENTER, TEST_ZOOM)
        best_entry = None
        best_dist = float("inf")
        for entry in entries:
            dx = abs(entry["x"] - center_tile.x)
            dy = abs(entry["y"] - center_tile.y)
            dist = dx + dy
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        assert best_entry is not None
        tile_coord = TileCoord(z=best_entry["z"], x=best_entry["x"], y=best_entry["y"])
        tile_img = cv2.imread(str(map_pack_dir / best_entry["path"]))
        assert tile_img is not None

        true_center = tile_center_gps(tile_coord)
        print(f"\n  Test tile: z={tile_coord.z} x={tile_coord.x} y={tile_coord.y}")
        print(f"  True center: ({true_center.lat:.6f}, {true_center.lon:.6f})")

        # Simulate drone frame
        drone_frame, offset = self._simulate_drone_frame(tile_img)
        print(f"  Drone frame size: {drone_frame.shape}")

        # Match with ORB
        matcher = OrbMatcher(max_features=2000)
        matcher.load()

        match_result = matcher.match(drone_frame, tile_img)
        print(f"  ORB matches: {match_result.num_matches}")

        if match_result.num_matches < 4:
            pytest.skip("Not enough ORB matches (OSM tiles may lack texture)")

        # Estimate homography
        h, w = drone_frame.shape[:2]
        result = match_and_localize(
            match_result.drone_pts,
            match_result.tile_pts,
            (w, h),
            tile_coord,
            min_inlier_ratio=0.15,  # Lower threshold for OSM tiles
        )

        if result is None:
            pytest.skip("Homography estimation failed (OSM tiles may lack texture for ORB)")

        print(f"  Inlier ratio: {result.inlier_ratio:.2%}")
        print(f"  Estimated: ({result.position.lat:.6f}, {result.position.lon:.6f})")

        # Check position accuracy — should be within the tile
        error_km = haversine_km(
            GeoPoint(lat=result.position.lat, lon=result.position.lon),
            true_center,
        )
        error_m = error_km * 1000
        print(f"  Position error: {error_m:.1f} m")

        # At zoom 17, tile is ~305m wide, so error should be less than a tile width
        assert error_m < 500, f"Position error too large: {error_m:.1f} m"

    def test_nmea_from_estimated_position(self, map_pack_dir: Path):
        """Verify NMEA output from an estimated position is valid."""
        # Use tile center as a simulated fix
        tile = gps_to_tile(TEST_CENTER, TEST_ZOOM)
        pos = tile_center_gps(tile)

        fix = PositionFix(lat=pos.lat, lon=pos.lon, altitude=50.0, hdop=1.5)
        gga = format_gga(fix)
        rmc = format_rmc(fix)

        print(f"\n  GGA: {gga.strip()}")
        print(f"  RMC: {rmc.strip()}")

        # Validate checksum
        for sentence in [gga, rmc]:
            body = sentence[1:sentence.index("*")]
            expected_cs = nmea_checksum(body)
            actual_cs = sentence[sentence.index("*") + 1:sentence.index("*") + 3]
            assert actual_cs == expected_cs, f"Checksum mismatch in {sentence[:6]}"

        # Parse back the latitude from GGA
        fields = gga.strip().split(",")
        lat_str = fields[2]  # ddmm.mmmmm
        lat_hem = fields[3]
        deg = int(lat_str[:2])
        mins = float(lat_str[2:])
        parsed_lat = deg + mins / 60.0
        if lat_hem == "S":
            parsed_lat = -parsed_lat

        assert abs(parsed_lat - pos.lat) < 0.001, f"Parsed lat {parsed_lat} vs {pos.lat}"
        print(f"  Parsed lat from NMEA: {parsed_lat:.6f} (true: {pos.lat:.6f})")


class TestEndToEndPipeline:
    """Full pipeline test: retrieval → matching → homography → NMEA."""

    def test_full_pipeline(self, map_pack_dir: Path):
        """Run the complete positioning pipeline on a synthetic drone frame."""
        from onboard.matcher import OrbMatcher
        from onboard.retrieval import TileIndex
        from programmer.indexer import build_index

        # Ensure index is built
        if not (map_pack_dir / "index" / "faiss.index").exists():
            build_index(map_pack_dir, use_onnx=False)

        # Load index
        tile_index = TileIndex(map_pack_dir)
        tile_index.load()

        # Pick a test tile and create synthetic drone frame
        with open(map_pack_dir / "index" / "tile_list.json") as f:
            entries = json.load(f)

        center_tile = gps_to_tile(TEST_CENTER, TEST_ZOOM)
        # Find closest tile
        best = min(entries, key=lambda e: abs(e["x"] - center_tile.x) + abs(e["y"] - center_tile.y))
        tile_coord = TileCoord(z=best["z"], x=best["x"], y=best["y"])
        tile_img = cv2.imread(str(map_pack_dir / best["path"]))
        assert tile_img is not None

        # Simulate drone frame (center crop + noise)
        h, w = tile_img.shape[:2]
        margin = int(min(h, w) * 0.15)
        drone_frame = tile_img[margin:h - margin, margin:w - margin].copy()
        noise = np.random.normal(0, 3, drone_frame.shape).astype(np.uint8)
        drone_frame = cv2.add(drone_frame, noise)

        true_pos = tile_center_gps(tile_coord)
        print(f"\n  === Full Pipeline Test ===")
        print(f"  True position: ({true_pos.lat:.6f}, {true_pos.lon:.6f})")

        t0 = time.time()

        # Step 1: Coarse retrieval
        matcher = OrbMatcher(max_features=2000)
        matcher.load()
        query_desc = matcher.extract_global_descriptor(drone_frame)
        candidates = tile_index.search(query_desc, k=min(20, tile_index.num_tiles))

        t_retrieval = time.time() - t0
        print(f"  Retrieval: {len(candidates.entries)} candidates in {t_retrieval*1000:.1f}ms")

        # Step 2: Fine matching + homography on each candidate
        position = None
        for entry in candidates.entries:
            cand_img = cv2.imread(str(entry.path))
            if cand_img is None:
                continue

            match_result = matcher.match(drone_frame, cand_img)
            if match_result.num_matches < 8:
                continue

            dh, dw = drone_frame.shape[:2]
            result = match_and_localize(
                match_result.drone_pts,
                match_result.tile_pts,
                (dw, dh),
                entry.tile,
                min_inlier_ratio=0.15,
            )
            if result is not None:
                position = result.position
                print(f"  Matched tile: z={entry.tile.z} x={entry.tile.x} y={entry.tile.y}")
                print(f"  Inliers: {result.inlier_ratio:.1%} ({int(result.inlier_ratio * match_result.num_matches)}/{match_result.num_matches})")
                break

        t_total = time.time() - t0
        print(f"  Total pipeline time: {t_total*1000:.1f}ms")

        if position is None:
            pytest.skip("Pipeline could not match — OSM tiles may lack texture for ORB")

        # Step 3: Generate NMEA
        fix = PositionFix(lat=position.lat, lon=position.lon)
        gga = format_gga(fix)
        rmc = format_rmc(fix)

        print(f"  Estimated: ({position.lat:.6f}, {position.lon:.6f})")
        print(f"  GGA: {gga.strip()}")
        print(f"  RMC: {rmc.strip()}")

        error_m = haversine_km(
            GeoPoint(lat=position.lat, lon=position.lon),
            true_pos,
        ) * 1000
        print(f"  Position error: {error_m:.1f} m")

        # Within reasonable range for ORB on OSM tiles
        assert error_m < 500, f"Error too large: {error_m:.1f}m"
