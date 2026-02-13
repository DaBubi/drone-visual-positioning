"""System health monitoring for the onboard VPS.

Tracks key performance metrics and raises alerts when the system
is degraded. Designed for headless operation on the RPi.

Monitors:
- Fix rate (fraction of frames with successful matches)
- Frame processing latency
- EKF innovation gate statistics
- Cache hit rate
- Memory usage
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HealthStatus:
    """Snapshot of system health."""
    fix_rate: float = 0.0         # rolling fix rate [0, 1]
    avg_latency_ms: float = 0.0  # average frame processing time
    max_latency_ms: float = 0.0  # worst-case latency in window
    frames_total: int = 0
    fixes_total: int = 0
    misses_total: int = 0
    outliers_rejected: int = 0   # EKF-rejected measurements
    geofence_violations: int = 0
    uptime_s: float = 0.0
    healthy: bool = True
    warnings: list[str] = field(default_factory=list)


class HealthMonitor:
    """Monitors VPS system health with rolling windows.

    Triggers warnings when:
    - Fix rate drops below threshold
    - Latency exceeds target
    - Too many consecutive misses
    - Geofence breach detected
    """

    def __init__(
        self,
        window_size: int = 100,
        min_fix_rate: float = 0.3,
        max_latency_ms: float = 500.0,
        max_consecutive_misses: int = 30,
    ):
        self._window = window_size
        self._min_fix_rate = min_fix_rate
        self._max_latency_ms = max_latency_ms
        self._max_consecutive_misses = max_consecutive_misses

        self._fixes: deque[bool] = deque(maxlen=window_size)
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._consecutive_misses = 0
        self._total_frames = 0
        self._total_fixes = 0
        self._total_misses = 0
        self._outliers_rejected = 0
        self._geofence_violations = 0
        self._start_time = time.monotonic()

    def record_frame(
        self,
        fix: bool,
        latency_ms: float,
        ekf_accepted: bool = True,
        geofence_ok: bool = True,
    ) -> None:
        """Record the result of processing one frame."""
        self._total_frames += 1
        self._fixes.append(fix)
        self._latencies.append(latency_ms)

        if fix:
            self._total_fixes += 1
            self._consecutive_misses = 0
        else:
            self._total_misses += 1
            self._consecutive_misses += 1

        if not ekf_accepted:
            self._outliers_rejected += 1

        if not geofence_ok:
            self._geofence_violations += 1

    @property
    def status(self) -> HealthStatus:
        """Get current health status."""
        warnings = []
        healthy = True

        # Fix rate
        if self._fixes:
            fix_rate = sum(self._fixes) / len(self._fixes)
        else:
            fix_rate = 0.0

        if self._total_frames > 10 and fix_rate < self._min_fix_rate:
            warnings.append(f"Low fix rate: {fix_rate:.0%} (min {self._min_fix_rate:.0%})")
            healthy = False

        # Latency
        if self._latencies:
            avg_lat = sum(self._latencies) / len(self._latencies)
            max_lat = max(self._latencies)
        else:
            avg_lat = 0.0
            max_lat = 0.0

        if avg_lat > self._max_latency_ms:
            warnings.append(f"High latency: {avg_lat:.0f}ms avg (max {self._max_latency_ms:.0f}ms)")
            healthy = False

        # Consecutive misses
        if self._consecutive_misses >= self._max_consecutive_misses:
            warnings.append(f"Lost fix: {self._consecutive_misses} consecutive misses")
            healthy = False

        # Geofence
        if self._geofence_violations > 0:
            warnings.append(f"Geofence violations: {self._geofence_violations}")

        return HealthStatus(
            fix_rate=fix_rate,
            avg_latency_ms=avg_lat,
            max_latency_ms=max_lat,
            frames_total=self._total_frames,
            fixes_total=self._total_fixes,
            misses_total=self._total_misses,
            outliers_rejected=self._outliers_rejected,
            geofence_violations=self._geofence_violations,
            uptime_s=time.monotonic() - self._start_time,
            healthy=healthy,
            warnings=warnings,
        )

    def log_status(self) -> None:
        """Log current health status."""
        s = self.status
        level = logging.INFO if s.healthy else logging.WARNING
        logger.log(
            level,
            "Health: fix=%.0f%% lat=%.0fms frames=%d fixes=%d misses=%d outliers=%d%s",
            s.fix_rate * 100,
            s.avg_latency_ms,
            s.frames_total,
            s.fixes_total,
            s.misses_total,
            s.outliers_rejected,
            f" WARNINGS: {'; '.join(s.warnings)}" if s.warnings else "",
        )
