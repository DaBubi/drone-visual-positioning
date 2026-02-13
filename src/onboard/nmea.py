"""NMEA 0183 sentence generation and UART output.

Generates $GPGGA and $GPRMC sentences for feeding position estimates
to a Betaflight/INAV flight controller as a fake GPS source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import serial


@dataclass(slots=True)
class PositionFix:
    """A single position estimate from the visual matching pipeline."""
    lat: float          # degrees, positive = N
    lon: float          # degrees, positive = E
    altitude: float = 0.0   # meters above MSL
    hdop: float = 1.0       # horizontal dilution of precision
    num_sats: int = 8       # simulated satellite count
    speed_knots: float = 0.0
    course_deg: float = 0.0
    fix_quality: int = 1    # 0=no fix, 1=GPS fix


def nmea_checksum(sentence: str) -> str:
    """Compute NMEA XOR checksum for content between $ and *."""
    cs = 0
    for ch in sentence:
        cs ^= ord(ch)
    return f"{cs:02X}"


def _format_lat(lat: float) -> tuple[str, str]:
    """Format latitude as NMEA ddmm.mmmmm,N/S."""
    hemisphere = "N" if lat >= 0 else "S"
    lat = abs(lat)
    degrees = int(lat)
    minutes = (lat - degrees) * 60.0
    return f"{degrees:02d}{minutes:08.5f}", hemisphere


def _format_lon(lon: float) -> tuple[str, str]:
    """Format longitude as NMEA dddmm.mmmmm,E/W."""
    hemisphere = "E" if lon >= 0 else "W"
    lon = abs(lon)
    degrees = int(lon)
    minutes = (lon - degrees) * 60.0
    return f"{degrees:03d}{minutes:08.5f}", hemisphere


def _utc_time_str(dt: datetime | None = None) -> str:
    """Format UTC time as hhmmss.ss."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.{dt.microsecond // 10000:02d}"


def _utc_date_str(dt: datetime | None = None) -> str:
    """Format UTC date as ddmmyy."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return f"{dt.day:02d}{dt.month:02d}{dt.year % 100:02d}"


def format_gga(fix: PositionFix, dt: datetime | None = None) -> str:
    """Generate a $GPGGA sentence.

    Format: $GPGGA,time,lat,N/S,lon,E/W,quality,numSV,HDOP,alt,M,sep,M,age,stn*cs
    """
    time_str = _utc_time_str(dt)
    lat_str, lat_ns = _format_lat(fix.lat)
    lon_str, lon_ew = _format_lon(fix.lon)

    body = (
        f"GPGGA,{time_str},{lat_str},{lat_ns},{lon_str},{lon_ew},"
        f"{fix.fix_quality},{fix.num_sats:02d},{fix.hdop:.1f},"
        f"{fix.altitude:.1f},M,0.0,M,,"
    )
    cs = nmea_checksum(body)
    return f"${body}*{cs}\r\n"


def format_rmc(fix: PositionFix, dt: datetime | None = None) -> str:
    """Generate a $GPRMC sentence.

    Format: $GPRMC,time,status,lat,N/S,lon,E/W,speed,course,date,magvar,vardir,mode*cs
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    time_str = _utc_time_str(dt)
    date_str = _utc_date_str(dt)
    lat_str, lat_ns = _format_lat(fix.lat)
    lon_str, lon_ew = _format_lon(fix.lon)
    status = "A" if fix.fix_quality > 0 else "V"

    body = (
        f"GPRMC,{time_str},{status},{lat_str},{lat_ns},{lon_str},{lon_ew},"
        f"{fix.speed_knots:.1f},{fix.course_deg:.1f},{date_str},,,A"
    )
    cs = nmea_checksum(body)
    return f"${body}*{cs}\r\n"


class UartSender:
    """Sends NMEA sentences over a serial UART port."""

    def __init__(self, port: str = "/dev/ttyAMA0", baudrate: int = 9600):
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
            timeout=1,
        )

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._serial = None

    def send_fix(self, fix: PositionFix) -> None:
        """Send GGA + RMC sentences for a position fix."""
        if self._serial is None:
            raise RuntimeError("UART not open. Call open() first.")
        dt = datetime.now(timezone.utc)
        gga = format_gga(fix, dt)
        rmc = format_rmc(fix, dt)
        self._serial.write(gga.encode("ascii"))
        self._serial.write(rmc.encode("ascii"))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
