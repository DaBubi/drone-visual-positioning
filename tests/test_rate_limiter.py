"""Tests for rate limiter."""

import pytest

from onboard.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_first_request_allowed(self):
        rl = RateLimiter(max_hz=5.0)
        assert rl.allow(t=0.0)

    def test_burst_allowed(self):
        rl = RateLimiter(max_hz=5.0, burst=3)
        assert rl.allow(t=0.0)
        assert rl.allow(t=0.0)
        assert rl.allow(t=0.0)
        assert not rl.allow(t=0.0)  # burst exhausted

    def test_rate_limiting(self):
        rl = RateLimiter(max_hz=5.0, burst=1)
        assert rl.allow(t=0.0)
        assert not rl.allow(t=0.1)  # too soon (need 0.2s)
        assert rl.allow(t=0.5)  # enough time passed

    def test_steady_rate(self):
        rl = RateLimiter(max_hz=5.0, burst=2)
        accepted = 0
        for i in range(50):
            t = i * 0.5  # 2 Hz input, well under 5 Hz limit
            if rl.allow(t):
                accepted += 1
        # All should be accepted since input < limit
        assert accepted == 50

    def test_high_rate_throttled(self):
        rl = RateLimiter(max_hz=2.0, burst=1)
        accepted = 0
        for i in range(100):
            t = i * 0.1  # 10 Hz input
            if rl.allow(t):
                accepted += 1
        # At 2 Hz limit with 10 Hz input over 10s, expect ~20 accepted
        assert 18 <= accepted <= 22

    def test_stats(self):
        rl = RateLimiter(max_hz=5.0, burst=1)
        rl.allow(t=0.0)
        rl.allow(t=0.05)
        rl.allow(t=0.1)
        assert rl.stats.total_requests == 3
        assert rl.stats.accepted >= 1
        assert rl.stats.throttled >= 0
        assert rl.stats.accepted + rl.stats.throttled == 3

    def test_time_until_next(self):
        rl = RateLimiter(max_hz=5.0, burst=1)
        assert rl.time_until_next() == 0.0  # has tokens
        rl.allow(t=0.0)
        wait = rl.time_until_next(t=0.0)
        assert wait > 0.0

    def test_reset(self):
        rl = RateLimiter(max_hz=5.0, burst=1)
        rl.allow(t=0.0)
        rl.allow(t=0.0)
        rl.reset()
        assert rl.allow(t=0.0)
        assert rl.stats.total_requests == 1

    def test_max_hz_property(self):
        rl = RateLimiter(max_hz=3.0)
        assert rl.max_hz == 3.0

    def test_actual_hz_tracking(self):
        rl = RateLimiter(max_hz=10.0, burst=1)
        for i in range(20):
            rl.allow(t=i * 0.1)
        assert rl.stats.actual_hz > 0.0
