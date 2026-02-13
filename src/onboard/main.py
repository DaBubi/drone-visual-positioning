"""Main positioning loop for the onboard VPS module.

Captures frames, matches against satellite tiles, and outputs
NMEA GPS sentences to the flight controller via UART.

Features:
- Coarse retrieval via FAISS → fine matching via ORB/SuperPoint
- Multi-resolution: z17 for coarse area, z19 for precise position
- Extended Kalman Filter for position smoothing + velocity estimation
- Telemetry logging for post-flight analysis
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
from onboard.ekf import EKFConfig, PositionEKF
from onboard.homography import match_and_localize
from onboard.matcher import OnnxMatcher, OrbMatcher
from onboard.nmea import PositionFix, UartSender, format_gga, format_rmc
from onboard.retrieval import TileIndex
from onboard.telemetry import FrameRecord, TelemetryLogger
from shared.tile_math import GeoPoint

logger = logging.getLogger(__name__)

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received")
    _running = False


def _try_match_frame(
    frame: np.ndarray,
    matcher,
    tile_index: TileIndex,
    config: VPSConfig,
) -> tuple[GeoPoint | None, float, float, int, int, int, int, float, float]:
    """Attempt to match a drone frame against the tile index.

    Returns:
        (position, hdop, inlier_ratio, num_matches, tile_z, tile_x, tile_y,
         retrieval_ms, match_ms)
    """
    t_ret = time.monotonic()
    descriptor = matcher.extract_global_descriptor(frame)
    candidates = tile_index.search(descriptor, k=config.matcher.max_candidates)
    retrieval_ms = (time.monotonic() - t_ret) * 1000

    t_match = time.monotonic()
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
            hdop = max(0.5, 5.0 * (1.0 - result.confidence))
            match_ms = (time.monotonic() - t_match) * 1000
            return (
                result.position, hdop, result.inlier_ratio,
                match_result.num_matches,
                entry.tile.z, entry.tile.x, entry.tile.y,
                retrieval_ms, match_ms,
            )

    match_ms = (time.monotonic() - t_match) * 1000
    return None, 0.0, 0.0, 0, 0, 0, 0, retrieval_ms, match_ms


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

    # EKF for position smoothing
    ekf = PositionEKF(EKFConfig(
        measurement_noise=config.ekf_measurement_noise,
        gate_threshold=config.ekf_gate_threshold,
    ))

    # Telemetry
    telemetry: TelemetryLogger | None = None
    if config.telemetry_dir is not None:
        telemetry = TelemetryLogger(config.telemetry_dir)
        telemetry.start()

    period = 1.0 / config.target_hz
    fixes = 0
    misses = 0
    frame_num = 0

    logger.info("VPS running at %.1f Hz target (EKF=%s, telemetry=%s)",
                config.target_hz, True, telemetry is not None)

    try:
        while _running:
            t0 = time.monotonic()
            frame_num += 1

            frame = camera.grab()
            if frame is None:
                logger.warning("Frame capture failed")
                time.sleep(0.1)
                continue

            # Match frame against satellite tiles
            (position, hdop, inlier_ratio, num_matches,
             tile_z, tile_x, tile_y,
             retrieval_ms, match_ms) = _try_match_frame(
                frame, matcher, tile_index, config,
            )

            # EKF update
            ekf_accepted = False
            if position is not None:
                ekf_accepted = ekf.update(position, hdop, t0)
                fixes += 1
            else:
                misses += 1

            # Use EKF-smoothed position for NMEA output
            ekf_state = ekf.state
            if ekf_state.initialized:
                output_pos = ekf.position
                fix = PositionFix(
                    lat=output_pos.lat,
                    lon=output_pos.lon,
                    hdop=hdop if position else 2.0,
                    speed_knots=ekf.speed_mps * 1.94384,  # m/s to knots
                )
                if uart is not None:
                    uart.send_fix(fix)
                logger.debug(
                    "Fix #%d: %.6f, %.6f (HDOP=%.1f, v=%.1fm/s, gate=%.1f%s)",
                    frame_num, output_pos.lat, output_pos.lon, fix.hdop,
                    ekf.speed_mps, ekf_state.innovation_gate,
                    "" if ekf_accepted else " REJECTED",
                )
            else:
                # No fix yet — send invalid
                if uart is not None:
                    uart.send_fix(PositionFix(lat=0, lon=0, fix_quality=0))

            # Telemetry
            elapsed_ms = (time.monotonic() - t0) * 1000
            if telemetry is not None:
                rec = FrameRecord(
                    timestamp=t0,
                    frame_num=frame_num,
                    fix=position is not None,
                    lat=position.lat if position else 0.0,
                    lon=position.lon if position else 0.0,
                    hdop=hdop,
                    inlier_ratio=inlier_ratio,
                    num_matches=num_matches,
                    tile_z=tile_z,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    retrieval_ms=retrieval_ms,
                    match_ms=match_ms,
                    total_ms=elapsed_ms,
                    ekf_lat=ekf_state.lat,
                    ekf_lon=ekf_state.lon,
                    ekf_vlat=ekf_state.vlat,
                    ekf_vlon=ekf_state.vlon,
                    ekf_speed_mps=ekf.speed_mps if ekf_state.initialized else 0.0,
                    ekf_gate=ekf_state.innovation_gate,
                    ekf_accepted=ekf_accepted,
                )
                telemetry.log(rec)

            # Rate limiting
            elapsed = time.monotonic() - t0
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                logger.debug("Frame %d took %.0fms (target %.0fms)",
                             frame_num, elapsed * 1000, period * 1000)

            if frame_num % 100 == 0:
                total = fixes + misses
                logger.info("Stats: %d/%d fixes (%.0f%%), %.1f Hz, v=%.1f m/s",
                            fixes, total, 100 * fixes / max(1, total),
                            1.0 / max(0.001, elapsed),
                            ekf.speed_mps if ekf_state.initialized else 0.0)

    finally:
        camera.close()
        if uart is not None:
            uart.close()
        if telemetry is not None:
            telemetry.stop()
        logger.info("VPS shutdown. Total fixes: %d, misses: %d", fixes, misses)


if __name__ == "__main__":
    main()
