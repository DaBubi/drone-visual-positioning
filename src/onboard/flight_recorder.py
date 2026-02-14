"""Binary flight recorder for post-flight analysis.

Records compact binary frames with position, velocity, match quality,
and timing data. Much smaller than CSV telemetry for long flights.

Format: fixed-size records with a simple header.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path


# Record format: timestamp(d), lat(d), lon(d), vn(f), ve(f),
#                hdop(f), speed(f), heading(f), fix_quality(B),
#                source(B), match_count(H), inlier_ratio(f),
#                latency_ms(H), flags(H)
RECORD_FMT = "<3d5fBBHfHH"
RECORD_SIZE = struct.calcsize(RECORD_FMT)  # 58 bytes

HEADER_MAGIC = b"VPSF"  # VPS Flight recorder
HEADER_VERSION = 2
HEADER_FMT = "<4sHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 8 bytes

# Source codes
SOURCE_NONE = 0
SOURCE_VISUAL = 1
SOURCE_EKF_PREDICT = 2
SOURCE_DEAD_RECKONING = 3

SOURCE_MAP = {
    "none": SOURCE_NONE,
    "visual": SOURCE_VISUAL,
    "ekf_predict": SOURCE_EKF_PREDICT,
    "dead_reckoning": SOURCE_DEAD_RECKONING,
}

# Flag bits
FLAG_GEOFENCE_OK = 0x01
FLAG_EKF_ACCEPTED = 0x02
FLAG_BLUR_SKIP = 0x04


@dataclass(slots=True)
class FlightRecord:
    """Single frame record."""
    timestamp: float
    lat: float
    lon: float
    vn_mps: float
    ve_mps: float
    hdop: float
    speed_mps: float
    heading_deg: float
    fix_quality: int
    source: int
    match_count: int
    inlier_ratio: float
    latency_ms: int
    flags: int

    def pack(self) -> bytes:
        return struct.pack(
            RECORD_FMT,
            self.timestamp, self.lat, self.lon,
            self.vn_mps, self.ve_mps, self.hdop,
            self.speed_mps, self.heading_deg,
            self.fix_quality, self.source,
            self.match_count, self.inlier_ratio,
            self.latency_ms, self.flags,
        )

    @classmethod
    def unpack(cls, data: bytes) -> FlightRecord:
        vals = struct.unpack(RECORD_FMT, data)
        return cls(
            timestamp=vals[0], lat=vals[1], lon=vals[2],
            vn_mps=vals[3], ve_mps=vals[4], hdop=vals[5],
            speed_mps=vals[6], heading_deg=vals[7],
            fix_quality=vals[8], source=vals[9],
            match_count=vals[10], inlier_ratio=vals[11],
            latency_ms=vals[12], flags=vals[13],
        )


class FlightRecorder:
    """Binary flight data recorder.

    Usage:
        rec = FlightRecorder(Path("flight.vpsf"))
        rec.start()
        rec.record(FlightRecord(...))
        rec.stop()

        # Read back:
        records = FlightRecorder.read(Path("flight.vpsf"))
    """

    def __init__(self, path: Path):
        self._path = path
        self._file = None
        self._count = 0

    @property
    def record_count(self) -> int:
        return self._count

    @property
    def is_recording(self) -> bool:
        return self._file is not None

    def start(self) -> None:
        """Open file and write header."""
        self._file = open(self._path, "wb")
        header = struct.pack(HEADER_FMT, HEADER_MAGIC, HEADER_VERSION, RECORD_SIZE)
        self._file.write(header)
        self._count = 0

    def record(self, rec: FlightRecord) -> None:
        """Write one record."""
        if self._file is None:
            return
        self._file.write(rec.pack())
        self._count += 1
        if self._count % 100 == 0:
            self._file.flush()

    def stop(self) -> None:
        """Close the file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    @staticmethod
    def read(path: Path) -> list[FlightRecord]:
        """Read all records from a flight file."""
        records = []
        with open(path, "rb") as f:
            header = f.read(HEADER_SIZE)
            if len(header) < HEADER_SIZE:
                return records

            magic, version, rec_size = struct.unpack(HEADER_FMT, header)
            if magic != HEADER_MAGIC:
                raise ValueError(f"Invalid file magic: {magic!r}")
            if version > HEADER_VERSION:
                raise ValueError(f"Unsupported version: {version}")

            while True:
                data = f.read(rec_size)
                if len(data) < rec_size:
                    break
                records.append(FlightRecord.unpack(data))

        return records

    @staticmethod
    def file_info(path: Path) -> dict:
        """Get metadata about a flight file without reading all records."""
        with open(path, "rb") as f:
            header = f.read(HEADER_SIZE)
            magic, version, rec_size = struct.unpack(HEADER_FMT, header)

            f.seek(0, 2)  # end
            file_size = f.tell()
            data_size = file_size - HEADER_SIZE
            num_records = data_size // rec_size if rec_size > 0 else 0

        return {
            "version": version,
            "record_size": rec_size,
            "record_count": num_records,
            "file_size_bytes": file_size,
        }
