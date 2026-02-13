"""Main positioning loop for the onboard VPS module.

Captures frames, matches against satellite tiles, and outputs
NMEA GPS sentences to the flight controller via UART.
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from onboard.camera import create_camera
from onboard.config import VPSConfig
from onboard.homography import match_and_localize
from onboard.matcher import OnnxMatcher, OrbMatcher
from onboard.nmea import PositionFix, UartSender, format_gga, format_rmc
from onboard.retrieval import TileIndex

logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


def main() -> None:
    """Entry point for the onboard VPS service."""
    config = VPSConfig()

    # Load config from file if present
    config_path = Path("/opt/vps/config.json")
    if config_path.exists():
        config = VPSConfig.model_validate_json(config_path.read_text())

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Initialize components
    logger.info("Loading tile index from %s", config.map_pack)
    tile_index = TileIndex(config.map_pack)
    tile_index.load()
    logger.info("Loaded %d tiles", tile_index.num_tiles)

    # Choose matcher backend
    if config.matcher.use_orb_fallback:
        logger.info("Using ORB fallback matcher")
        matcher = OrbMatcher()
    else:
        logger.info("Loading ONNX models")
        matcher = OnnxMatcher(
            config.matcher.superpoint_onnx,
            config.matcher.lightglue_onnx,
        )
    matcher.load()

    camera = create_camera(config.camera)
    camera.open()

    uart: UartSender | None = None
    if config.uart.enabled:
        uart = UartSender(config.uart.port, config.uart.baudrate)
        uart.open()
        logger.info("UART open on %s @ %d baud", config.uart.port, config.uart.baudrate)

    period = 1.0 / config.target_hz
    fixes = 0
    misses = 0

    logger.info("VPS running at %.1f Hz target", config.target_hz)

    try:
        while _running:
            t0 = time.monotonic()

            frame = camera.grab()
            if frame is None:
                logger.warning("Frame capture failed")
                time.sleep(0.1)
                continue

            # Coarse retrieval
            descriptor = matcher.extract_global_descriptor(frame)
            candidates = tile_index.search(descriptor, k=config.matcher.max_candidates)

            # Fine matching + homography
            position = None
            for entry in candidates.entries:
                tile_img = cv2.imread(str(entry.path))
                if tile_img is None:
                    continue

                match_result = matcher.match(frame, tile_img)
                if match_result.num_matches < config.matcher.min_matches:
                    continue

                h, w = frame.shape[:2]
                result = match_and_localize(
                    match_result.drone_pts,
                    match_result.tile_pts,
                    (w, h),
                    entry.tile,
                    min_inlier_ratio=config.matcher.confidence_threshold,
                )

                if result is not None:
                    position = result.position
                    hdop = max(0.5, 5.0 * (1.0 - result.confidence))
                    break

            if position is not None:
                fix = PositionFix(
                    lat=position.lat,
                    lon=position.lon,
                    hdop=hdop,
                )
                if uart is not None:
                    uart.send_fix(fix)
                fixes += 1
                logger.debug("Fix: %.6f, %.6f (HDOP=%.1f)", fix.lat, fix.lon, fix.hdop)
            else:
                # Send no-fix sentence
                if uart is not None:
                    no_fix = PositionFix(lat=0, lon=0, fix_quality=0)
                    uart.send_fix(no_fix)
                misses += 1

            # Rate limiting
            elapsed = time.monotonic() - t0
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                logger.debug("Frame took %.0fms (target %.0fms)",
                             elapsed * 1000, period * 1000)

            if (fixes + misses) % 100 == 0:
                total = fixes + misses
                logger.info("Stats: %d/%d fixes (%.0f%%), %.1f Hz avg",
                            fixes, total, 100 * fixes / max(1, total),
                            1.0 / max(0.001, elapsed))

    finally:
        camera.close()
        if uart is not None:
            uart.close()
        logger.info("VPS shutdown. Total fixes: %d, misses: %d", fixes, misses)


if __name__ == "__main__":
    main()
