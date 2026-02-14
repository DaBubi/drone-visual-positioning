"""Dead reckoning fallback for when visual matching fails.

When the camera loses lock (e.g., passing over water, shadows,
clouds), the system extrapolates position using the last known
velocity from the EKF. This maintains NMEA output continuity
so the flight controller doesn't lose GPS fix entirely.

Dead reckoning accuracy degrades over time (~1-5m/s drift),
so the confidence (HDOP) increases with each extrapolated fix.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from shared.tile_math import GeoPoint


@dataclass(slots=True)
class DeadReckoningState:
    """State for dead reckoning extrapolation."""
    last_fix: GeoPoint
    last_fix_time: float         # monotonic timestamp
    velocity_north_mps: float    # m/s northward
    velocity_east_mps: float     # m/s eastward
    max_extrapolation_s: float   # max time to extrapolate before giving up
    base_hdop: float             # HDOP at last fix


class DeadReckoning:
    """Extrapolates position when visual fixes are unavailable.

    Uses constant-velocity model from the EKF's last known state.
    HDOP increases linearly with time since last fix to signal
    decreasing confidence.
    """

    def __init__(self, max_extrapolation_s: float = 10.0, hdop_growth_rate: float = 1.0):
        """
        Args:
            max_extrapolation_s: max seconds to extrapolate before returning None
            hdop_growth_rate: HDOP increase per second of extrapolation
        """
        self._max_extrap = max_extrapolation_s
        self._hdop_rate = hdop_growth_rate
        self._state: DeadReckoningState | None = None

    def update_reference(
        self,
        position: GeoPoint,
        vn_mps: float,
        ve_mps: float,
        hdop: float,
        t: float | None = None,
    ) -> None:
        """Update the reference state from a new visual fix.

        Call this every time the EKF produces a valid fix.
        """
        self._state = DeadReckoningState(
            last_fix=position,
            last_fix_time=t if t is not None else time.monotonic(),
            velocity_north_mps=vn_mps,
            velocity_east_mps=ve_mps,
            max_extrapolation_s=self._max_extrap,
            base_hdop=hdop,
        )

    def extrapolate(self, t: float | None = None) -> tuple[GeoPoint, float] | None:
        """Extrapolate position from last fix using dead reckoning.

        Args:
            t: current time (monotonic). Uses time.monotonic() if None.

        Returns:
            (position, hdop) or None if no reference or too much time elapsed
        """
        if self._state is None:
            return None

        if t is None:
            t = time.monotonic()

        dt = t - self._state.last_fix_time
        if dt < 0 or dt > self._state.max_extrapolation_s:
            return None

        # Convert velocity to degrees/second
        lat = self._state.last_fix.lat
        dlat = self._state.velocity_north_mps * dt / 111320.0
        dlon = self._state.velocity_east_mps * dt / (111320.0 * math.cos(math.radians(lat)))

        new_lat = lat + dlat
        new_lon = self._state.last_fix.lon + dlon

        # HDOP increases with time
        hdop = self._state.base_hdop + self._hdop_rate * dt

        return GeoPoint(lat=new_lat, lon=new_lon), hdop

    @property
    def has_reference(self) -> bool:
        return self._state is not None

    @property
    def time_since_fix(self) -> float:
        """Seconds since last visual fix, or inf if no reference."""
        if self._state is None:
            return float("inf")
        return time.monotonic() - self._state.last_fix_time
