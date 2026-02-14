"""Position fusion: combines visual matching, EKF, dead reckoning, and geofence.

This is the top-level positioning engine that orchestrates all
subsystems into a single update() call. The main loop uses this
instead of calling each component directly.

Priority:
1. Visual match → EKF update → output EKF position
2. No visual match → EKF predict (if recent) → output prediction
3. EKF stale → dead reckoning → output extrapolated position
4. Dead reckoning expired → no-fix NMEA sentence
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from onboard.dead_reckoning import DeadReckoning
from onboard.ekf import EKFConfig, PositionEKF
from onboard.geofence import CircleGeofence, GeofenceChecker
from shared.tile_math import GeoPoint

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FusionOutput:
    """Result of a single fusion update."""
    position: GeoPoint | None  # fused GPS position
    hdop: float                # horizontal dilution of precision
    speed_mps: float           # estimated ground speed
    heading_deg: float         # estimated heading (0=N, clockwise)
    fix_quality: int           # 0=no fix, 1=visual, 2=EKF predict, 3=dead reckoning
    source: str                # "visual", "ekf_predict", "dead_reckoning", "none"
    geofence_ok: bool          # True if position is within geofence
    ekf_accepted: bool         # True if EKF accepted the measurement


class PositionFusion:
    """Fuses visual positioning with EKF and dead reckoning.

    Usage:
        fusion = PositionFusion(center=GeoPoint(52.52, 13.405), radius_km=1.0)

        # On each frame:
        output = fusion.update(visual_position, hdop, t)
        if output.position:
            send_nmea(output)
    """

    def __init__(
        self,
        center: GeoPoint | None = None,
        radius_km: float = 5.0,
        ekf_config: EKFConfig | None = None,
        max_dead_reckoning_s: float = 10.0,
        geofence_margin_km: float = 0.2,
    ):
        self._ekf = PositionEKF(ekf_config or EKFConfig())
        self._dr = DeadReckoning(max_extrapolation_s=max_dead_reckoning_s)

        self._fence: GeofenceChecker | None = None
        if center is not None:
            fence = CircleGeofence(
                center=center,
                radius_km=radius_km,
                margin_km=geofence_margin_km,
            )
            self._fence = GeofenceChecker(fence)

    def update(
        self,
        visual_position: GeoPoint | None,
        hdop: float = 2.0,
        t: float | None = None,
    ) -> FusionOutput:
        """Process one frame's positioning result.

        Args:
            visual_position: GPS from visual matching, or None if no match
            hdop: horizontal DOP from visual matching
            t: timestamp (monotonic seconds)

        Returns:
            FusionOutput with fused position and metadata
        """
        if t is None:
            t = time.monotonic()

        ekf_accepted = False
        output_pos: GeoPoint | None = None
        output_hdop = 99.0
        source = "none"
        fix_quality = 0

        if visual_position is not None:
            # Case 1: Visual fix available
            ekf_accepted = self._ekf.update(visual_position, hdop, t)
            if self._ekf.state.initialized:
                output_pos = self._ekf.position
                output_hdop = hdop
                source = "visual"
                fix_quality = 1

                # Update dead reckoning reference
                vn, ve = self._ekf.velocity_mps
                self._dr.update_reference(
                    self._ekf.position, vn, ve, hdop, t,
                )
        elif self._ekf.state.initialized:
            # Case 2: No visual fix, but EKF has state — use prediction
            predicted = self._ekf.predict(t)
            if predicted.lat != 0 or predicted.lon != 0:
                output_pos = predicted
                output_hdop = 3.0  # degraded confidence
                source = "ekf_predict"
                fix_quality = 2

        if output_pos is None:
            # Case 3: Try dead reckoning
            dr_result = self._dr.extrapolate(t)
            if dr_result is not None:
                output_pos, output_hdop = dr_result
                source = "dead_reckoning"
                fix_quality = 3

        # Geofence check
        geofence_ok = True
        if output_pos is not None and self._fence is not None:
            geofence_ok = self._fence.check(output_pos)
            if not geofence_ok:
                logger.warning(
                    "Geofence violation: %.6f, %.6f (source=%s)",
                    output_pos.lat, output_pos.lon, source,
                )
                # Don't send out-of-bounds positions
                output_pos = None
                fix_quality = 0
                source = "none"

        # Speed and heading
        speed_mps = 0.0
        heading_deg = 0.0
        if self._ekf.state.initialized:
            speed_mps = self._ekf.speed_mps
            vn, ve = self._ekf.velocity_mps
            if speed_mps > 0.5:
                import math
                heading_deg = math.degrees(math.atan2(ve, vn)) % 360

        return FusionOutput(
            position=output_pos,
            hdop=output_hdop,
            speed_mps=speed_mps,
            heading_deg=heading_deg,
            fix_quality=fix_quality,
            source=source,
            geofence_ok=geofence_ok,
            ekf_accepted=ekf_accepted,
        )

    def reset(self) -> None:
        """Reset all state."""
        self._ekf.reset()
        self._dr = DeadReckoning(max_extrapolation_s=self._dr._max_extrap)
        if self._fence:
            self._fence.reset()
