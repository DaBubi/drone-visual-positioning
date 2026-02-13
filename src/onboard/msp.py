"""MSP (MultiWii Serial Protocol) GPS output for INAV/Betaflight.

Alternative to NMEA for feeding position data to flight controllers.
MSP is more efficient (binary protocol) and supports richer data
including velocity and heading.

Implements MSP_SET_RAW_GPS (command 201) which allows injecting
GPS data directly into the FC's GPS module.

MSP frame format:
  $M< len cmd payload checksum
  - Header: '$M<' (3 bytes)
  - Length: payload size (1 byte)
  - Command: MSP command ID (1 byte)
  - Payload: command data (N bytes)
  - Checksum: XOR of length, command, and all payload bytes
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import serial

# MSP Command IDs
MSP_SET_RAW_GPS = 201
MSP_SET_WP = 209  # Set waypoint


@dataclass(slots=True)
class MSPGPSData:
    """GPS data for MSP_SET_RAW_GPS.

    Fields match INAV/Betaflight MSP_SET_RAW_GPS:
    - fix_type: 0=no fix, 1=2D, 2=3D
    - num_sats: number of satellites
    - lat: latitude in 1e-7 degrees (int32)
    - lon: longitude in 1e-7 degrees (int32)
    - alt: altitude in cm (int16)
    - speed: ground speed in cm/s (uint16)
    - ground_course: heading in 0.1 degrees (uint16)
    - hdop: HDOP * 100 (uint16)
    """
    fix_type: int = 2
    num_sats: int = 10
    lat: int = 0          # 1e-7 degrees
    lon: int = 0          # 1e-7 degrees
    alt: int = 0          # cm
    speed: int = 0        # cm/s
    ground_course: int = 0  # 0.1 degrees
    hdop: int = 100       # HDOP * 100

    @classmethod
    def from_position(
        cls,
        lat: float,
        lon: float,
        alt_m: float = 0.0,
        speed_mps: float = 0.0,
        heading_deg: float = 0.0,
        hdop: float = 1.0,
        num_sats: int = 10,
        fix_type: int = 2,
    ) -> MSPGPSData:
        """Create MSP GPS data from float values."""
        return cls(
            fix_type=fix_type,
            num_sats=num_sats,
            lat=int(lat * 1e7),
            lon=int(lon * 1e7),
            alt=int(alt_m * 100),
            speed=int(speed_mps * 100),
            ground_course=int(heading_deg * 10) % 3600,
            hdop=int(hdop * 100),
        )


def msp_checksum(data: bytes) -> int:
    """Compute MSP checksum (XOR of all bytes)."""
    cs = 0
    for b in data:
        cs ^= b
    return cs


def encode_set_raw_gps(gps: MSPGPSData) -> bytes:
    """Encode MSP_SET_RAW_GPS frame.

    Payload format (18 bytes):
        uint8  fix_type
        uint8  num_sats
        int32  lat (1e-7 deg)
        int32  lon (1e-7 deg)
        int16  alt (cm)
        uint16 speed (cm/s)
        uint16 ground_course (0.1 deg)
        uint16 hdop (*100)
    """
    payload = struct.pack(
        "<BBiiHHHH",
        gps.fix_type,
        gps.num_sats,
        gps.lat,
        gps.lon,
        gps.alt & 0xFFFF,  # alt as signed->unsigned conversion
        gps.speed,
        gps.ground_course,
        gps.hdop,
    )

    length = len(payload)
    cmd = MSP_SET_RAW_GPS

    # Header
    frame = b"$M<"
    # Length + command + payload
    data = struct.pack("BB", length, cmd) + payload
    cs = msp_checksum(data)
    frame += data + struct.pack("B", cs)

    return frame


class MSPSender:
    """Sends MSP GPS frames over UART to a flight controller."""

    def __init__(self, port: str = "/dev/ttyAMA0", baudrate: int = 115200):
        self._port = port
        self._baudrate = baudrate
        self._serial: serial.Serial | None = None

    def open(self) -> None:
        self._serial = serial.Serial(
            self._port,
            self._baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
        )

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None

    def send_gps(self, gps: MSPGPSData) -> None:
        """Send a single GPS update via MSP."""
        if self._serial is None:
            raise RuntimeError("MSP not open. Call open() first.")
        frame = encode_set_raw_gps(gps)
        self._serial.write(frame)

    def send_position(
        self,
        lat: float,
        lon: float,
        alt_m: float = 0.0,
        speed_mps: float = 0.0,
        heading_deg: float = 0.0,
        hdop: float = 1.0,
    ) -> None:
        """Convenience: send position from float values."""
        gps = MSPGPSData.from_position(
            lat, lon, alt_m, speed_mps, heading_deg, hdop,
        )
        self.send_gps(gps)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
