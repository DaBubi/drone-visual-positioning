"""Onboard performance benchmarking.

Measures key pipeline stage latencies to help tune the system
for the target 2-5 Hz update rate on RPi CM4/5.

Benchmarks:
- Feature extraction (ORB / SuperPoint ONNX)
- Feature matching
- FAISS retrieval
- Homography estimation
- NMEA/MSP encoding
- Full pipeline end-to-end
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkResult:
    """Timing results for a benchmark run."""
    name: str
    iterations: int
    times_ms: list[float]
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def __post_init__(self):
        if self.times_ms:
            self.mean_ms = statistics.mean(self.times_ms)
            self.median_ms = statistics.median(self.times_ms)
            sorted_t = sorted(self.times_ms)
            idx95 = min(len(sorted_t) - 1, int(0.95 * len(sorted_t)))
            idx99 = min(len(sorted_t) - 1, int(0.99 * len(sorted_t)))
            self.p95_ms = sorted_t[idx95]
            self.p99_ms = sorted_t[idx99]
            self.min_ms = sorted_t[0]
            self.max_ms = sorted_t[-1]

    def summary(self) -> str:
        return (
            f"{self.name}: mean={self.mean_ms:.1f}ms "
            f"median={self.median_ms:.1f}ms p95={self.p95_ms:.1f}ms "
            f"p99={self.p99_ms:.1f}ms min={self.min_ms:.1f}ms max={self.max_ms:.1f}ms "
            f"(n={self.iterations})"
        )


def _time_fn(fn, iterations: int, warmup: int = 3) -> list[float]:
    """Time a function over multiple iterations with warmup."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def benchmark_orb_extraction(
    image: np.ndarray,
    iterations: int = 100,
    max_features: int = 1000,
) -> BenchmarkResult:
    """Benchmark ORB feature extraction."""
    orb = cv2.ORB_create(nfeatures=max_features)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    def fn():
        orb.detectAndCompute(gray, None)

    times = _time_fn(fn, iterations)
    return BenchmarkResult(name="ORB extraction", iterations=iterations, times_ms=times)


def benchmark_orb_matching(
    image1: np.ndarray,
    image2: np.ndarray,
    iterations: int = 100,
    max_features: int = 1000,
) -> BenchmarkResult:
    """Benchmark ORB feature matching between two images."""
    orb = cv2.ORB_create(nfeatures=max_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    g1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    g2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2

    _, d1 = orb.detectAndCompute(g1, None)
    _, d2 = orb.detectAndCompute(g2, None)

    if d1 is None or d2 is None:
        return BenchmarkResult(name="ORB matching", iterations=0, times_ms=[])

    def fn():
        bf.knnMatch(d1, d2, k=2)

    times = _time_fn(fn, iterations)
    return BenchmarkResult(name="ORB matching", iterations=iterations, times_ms=times)


def benchmark_homography(
    n_points: int = 50,
    iterations: int = 1000,
) -> BenchmarkResult:
    """Benchmark RANSAC homography estimation."""
    rng = np.random.RandomState(42)
    pts1 = rng.rand(n_points, 2).astype(np.float32) * 256
    H_true = np.array([[1.1, 0.05, 10], [-0.05, 0.95, 15], [0, 0, 1]], dtype=np.float32)
    pts2_h = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), H_true).reshape(-1, 2)
    # Add noise
    pts2 = pts2_h + rng.randn(n_points, 2).astype(np.float32) * 2

    def fn():
        cv2.findHomography(pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2), cv2.RANSAC, 5.0)

    times = _time_fn(fn, iterations)
    return BenchmarkResult(name="RANSAC homography", iterations=iterations, times_ms=times)


def benchmark_nmea_encoding(iterations: int = 10000) -> BenchmarkResult:
    """Benchmark NMEA sentence generation."""
    from onboard.nmea import PositionFix, format_gga, format_rmc

    fix = PositionFix(lat=52.5200, lon=13.4050, hdop=1.2, speed_knots=5.0)

    def fn():
        format_gga(fix)
        format_rmc(fix)

    times = _time_fn(fn, iterations, warmup=10)
    return BenchmarkResult(name="NMEA encoding", iterations=iterations, times_ms=times)


def benchmark_msp_encoding(iterations: int = 10000) -> BenchmarkResult:
    """Benchmark MSP frame encoding."""
    from onboard.msp import MSPGPSData, encode_set_raw_gps

    gps = MSPGPSData.from_position(lat=52.52, lon=13.405, speed_mps=10.0)

    def fn():
        encode_set_raw_gps(gps)

    times = _time_fn(fn, iterations, warmup=10)
    return BenchmarkResult(name="MSP encoding", iterations=iterations, times_ms=times)


def run_all_benchmarks(image_size: int = 640) -> list[BenchmarkResult]:
    """Run all benchmarks with synthetic images."""
    img1 = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    results = [
        benchmark_orb_extraction(img1, iterations=50),
        benchmark_orb_matching(img1, img2, iterations=50),
        benchmark_homography(iterations=200),
        benchmark_nmea_encoding(iterations=5000),
        benchmark_msp_encoding(iterations=5000),
    ]

    for r in results:
        logger.info(r.summary())

    return results
