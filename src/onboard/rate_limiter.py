"""Output rate limiter for position updates.

Controls the frequency of NMEA/MSP output to the flight controller.
Prevents flooding the UART while ensuring timely position updates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class RateLimiterStats:
    """Rate limiter statistics."""
    total_requests: int = 0
    accepted: int = 0
    throttled: int = 0
    actual_hz: float = 0.0


class RateLimiter:
    """Token-bucket rate limiter for position output.

    Usage:
        limiter = RateLimiter(max_hz=5.0)
        if limiter.allow(t):
            send_position(...)
    """

    def __init__(self, max_hz: float = 5.0, burst: int = 2):
        self._min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self._max_hz = max_hz
        self._burst = burst
        self._tokens = float(burst)
        self._last_t = 0.0
        self._last_accept_t = 0.0
        self._stats = RateLimiterStats()
        self._accept_times: list[float] = []

    @property
    def stats(self) -> RateLimiterStats:
        return self._stats

    @property
    def max_hz(self) -> float:
        return self._max_hz

    def allow(self, t: float | None = None) -> bool:
        """Check if an output is allowed at time t.

        Returns True if the output should be sent.
        """
        if t is None:
            t = time.monotonic()

        self._stats.total_requests += 1

        # Replenish tokens
        if self._last_t > 0:
            elapsed = t - self._last_t
            self._tokens = min(
                float(self._burst),
                self._tokens + elapsed * self._max_hz,
            )
        self._last_t = t

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            self._stats.accepted += 1
            self._last_accept_t = t

            # Track actual rate
            self._accept_times.append(t)
            if len(self._accept_times) > 20:
                self._accept_times = self._accept_times[-10:]
            if len(self._accept_times) >= 2:
                dt = self._accept_times[-1] - self._accept_times[0]
                if dt > 0:
                    self._stats.actual_hz = (len(self._accept_times) - 1) / dt

            return True

        self._stats.throttled += 1
        return False

    def time_until_next(self, t: float | None = None) -> float:
        """Seconds until the next output would be allowed."""
        if self._tokens >= 1.0:
            return 0.0
        if t is None:
            t = time.monotonic()
        needed = 1.0 - self._tokens
        return needed / self._max_hz if self._max_hz > 0 else 0.0

    def reset(self) -> None:
        """Reset state."""
        self._tokens = float(self._burst)
        self._last_t = 0.0
        self._stats = RateLimiterStats()
        self._accept_times.clear()
