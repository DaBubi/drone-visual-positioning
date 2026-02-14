"""Tests for benchmark module (verifies benchmarks run, not perf targets)."""

import numpy as np
import pytest

from onboard.benchmark import (
    BenchmarkResult,
    benchmark_homography,
    benchmark_msp_encoding,
    benchmark_nmea_encoding,
    benchmark_orb_extraction,
    benchmark_orb_matching,
    run_all_benchmarks,
)


class TestBenchmarkResult:
    def test_stats_calculation(self):
        r = BenchmarkResult(name="test", iterations=5, times_ms=[1.0, 2.0, 3.0, 4.0, 5.0])
        assert r.mean_ms == pytest.approx(3.0)
        assert r.median_ms == pytest.approx(3.0)
        assert r.min_ms == 1.0
        assert r.max_ms == 5.0

    def test_summary_string(self):
        r = BenchmarkResult(name="test", iterations=3, times_ms=[1.0, 2.0, 3.0])
        s = r.summary()
        assert "test" in s
        assert "mean=" in s

    def test_empty_times(self):
        r = BenchmarkResult(name="empty", iterations=0, times_ms=[])
        assert r.mean_ms == 0.0


class TestBenchmarks:
    def test_orb_extraction(self):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        r = benchmark_orb_extraction(img, iterations=5)
        assert r.iterations == 5
        assert len(r.times_ms) == 5
        assert r.mean_ms > 0

    def test_orb_matching(self):
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        r = benchmark_orb_matching(img1, img2, iterations=5)
        assert r.mean_ms >= 0

    def test_homography(self):
        r = benchmark_homography(iterations=10)
        assert r.iterations == 10
        assert r.mean_ms > 0

    def test_nmea_encoding(self):
        r = benchmark_nmea_encoding(iterations=100)
        assert r.mean_ms > 0
        assert r.mean_ms < 1.0  # should be sub-ms

    def test_msp_encoding(self):
        r = benchmark_msp_encoding(iterations=100)
        assert r.mean_ms > 0
        assert r.mean_ms < 1.0  # should be sub-ms

    def test_run_all(self):
        results = run_all_benchmarks(image_size=128)
        assert len(results) == 5
        for r in results:
            assert r.mean_ms >= 0
