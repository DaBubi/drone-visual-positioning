"""Geofence safety boundary for VPS positioning.

Defines a circular or polygon boundary within which the VPS
system is allowed to report fixes. Positions outside the geofence
are rejected — this prevents the system from reporting erroneous
positions that could cause the drone to fly to a wrong location.

The geofence should match the area covered by the loaded map pack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.tile_math import GeoPoint, haversine_km


@dataclass(frozen=True, slots=True)
class CircleGeofence:
    """Circular geofence defined by center + radius.

    Matches the map pack coverage area.
    """
    center: GeoPoint
    radius_km: float
    margin_km: float = 0.1  # extra margin beyond radius

    def contains(self, point: GeoPoint) -> bool:
        """Check if a point is inside the geofence."""
        dist = haversine_km(self.center, point)
        return dist <= (self.radius_km + self.margin_km)

    def distance_to_boundary(self, point: GeoPoint) -> float:
        """Distance from point to geofence boundary in km.

        Positive = inside, negative = outside.
        """
        dist = haversine_km(self.center, point)
        return (self.radius_km + self.margin_km) - dist


@dataclass(frozen=True, slots=True)
class RectGeofence:
    """Rectangular geofence defined by NW and SE corners."""
    nw: GeoPoint  # north-west corner (max lat, min lon)
    se: GeoPoint  # south-east corner (min lat, max lon)

    def contains(self, point: GeoPoint) -> bool:
        return (
            self.se.lat <= point.lat <= self.nw.lat
            and self.nw.lon <= point.lon <= self.se.lon
        )


class GeofenceChecker:
    """Validates positions against a geofence boundary.

    Tracks consecutive out-of-bounds fixes and can trigger
    a safety response after N violations.
    """

    def __init__(
        self,
        fence: CircleGeofence | RectGeofence,
        max_violations: int = 5,
    ):
        self._fence = fence
        self._max_violations = max_violations
        self._consecutive_violations = 0
        self._total_checks = 0
        self._total_violations = 0

    def check(self, point: GeoPoint) -> bool:
        """Check if position is within geofence.

        Returns True if position is valid (inside fence).
        """
        self._total_checks += 1
        inside = self._fence.contains(point)

        if inside:
            self._consecutive_violations = 0
        else:
            self._consecutive_violations += 1
            self._total_violations += 1

        return inside

    @property
    def is_breached(self) -> bool:
        """True if too many consecutive out-of-bounds fixes."""
        return self._consecutive_violations >= self._max_violations

    @property
    def consecutive_violations(self) -> int:
        return self._consecutive_violations

    @property
    def violation_rate(self) -> float:
        if self._total_checks == 0:
            return 0.0
        return self._total_violations / self._total_checks

    def reset(self) -> None:
        self._consecutive_violations = 0
        self._total_checks = 0
        self._total_violations = 0
