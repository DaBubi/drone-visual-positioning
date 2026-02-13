"""Telemetry logging for flight data recording and post-analysis.

Logs every frame's result (fix or miss) to a CSV file for later analysis
of positioning accuracy, match quality, and system performance.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FIELDS = [
    "timestamp",
    "frame_num",
    "fix",             # 1=fix, 0=miss
    "lat",
    "lon",
    "hdop",
    "inlier_ratio",
    "num_matches",
    "tile_z",
    "tile_x",
    "tile_y",
    "retrieval_ms",
    "match_ms",
    "total_ms",
    "ekf_lat",
    "ekf_lon",
    "ekf_vlat",
    "ekf_vlon",
    "ekf_speed_mps",
    "ekf_gate",
    "ekf_accepted",
]


@dataclass(slots=True)
class FrameRecord:
    """Telemetry for a single processed frame."""
    timestamp: float = 0.0
    frame_num: int = 0
    fix: bool = False
    lat: float = 0.0
    lon: float = 0.0
    hdop: float = 0.0
    inlier_ratio: float = 0.0
    num_matches: int = 0
    tile_z: int = 0
    tile_x: int = 0
    tile_y: int = 0
    retrieval_ms: float = 0.0
    match_ms: float = 0.0
    total_ms: float = 0.0
    ekf_lat: float = 0.0
    ekf_lon: float = 0.0
    ekf_vlat: float = 0.0
    ekf_vlon: float = 0.0
    ekf_speed_mps: float = 0.0
    ekf_gate: float = 0.0
    ekf_accepted: bool = False

    def to_row(self) -> list:
        return [
            f"{self.timestamp:.3f}",
            self.frame_num,
            1 if self.fix else 0,
            f"{self.lat:.8f}" if self.fix else "",
            f"{self.lon:.8f}" if self.fix else "",
            f"{self.hdop:.2f}" if self.fix else "",
            f"{self.inlier_ratio:.3f}" if self.fix else "",
            self.num_matches,
            self.tile_z if self.fix else "",
            self.tile_x if self.fix else "",
            self.tile_y if self.fix else "",
            f"{self.retrieval_ms:.1f}",
            f"{self.match_ms:.1f}",
            f"{self.total_ms:.1f}",
            f"{self.ekf_lat:.8f}",
            f"{self.ekf_lon:.8f}",
            f"{self.ekf_vlat:.10f}",
            f"{self.ekf_vlon:.10f}",
            f"{self.ekf_speed_mps:.2f}",
            f"{self.ekf_gate:.2f}",
            1 if self.ekf_accepted else 0,
        ]


class TelemetryLogger:
    """Writes frame-by-frame telemetry to a CSV log file."""

    def __init__(self, log_dir: Path, prefix: str = "vps"):
        self._log_dir = log_dir
        self._prefix = prefix
        self._writer: csv.writer | None = None
        self._file = None
        self._frame_count = 0

    def start(self) -> Path:
        """Open a new log file. Returns the file path."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._log_dir / f"{self._prefix}_{ts}.csv"
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(FIELDS)
        self._frame_count = 0
        logger.info("Telemetry logging to %s", path)
        return path

    def log(self, record: FrameRecord) -> None:
        """Write a frame record."""
        if self._writer is None:
            return
        self._writer.writerow(record.to_row())
        self._frame_count += 1
        # Flush every 100 frames
        if self._frame_count % 100 == 0 and self._file:
            self._file.flush()

    def stop(self) -> None:
        """Close the log file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            logger.info("Telemetry stopped. %d frames logged.", self._frame_count)
