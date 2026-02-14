"""UART connection manager with reconnection and health monitoring.

Handles serial port lifecycle: open, write, close, reconnect on failure.
Supports both NMEA and MSP output protocols with automatic retry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Protocol(Enum):
    NMEA = "nmea"
    MSP = "msp"


@dataclass(slots=True)
class UARTStats:
    """UART connection statistics."""
    bytes_sent: int = 0
    messages_sent: int = 0
    errors: int = 0
    reconnects: int = 0
    last_send_t: float = 0.0
    last_error_t: float = 0.0
    connected: bool = False


class UARTManager:
    """Manages serial port connection with automatic reconnection.

    Usage:
        uart = UARTManager("/dev/ttyAMA0", baudrate=115200)
        uart.open()
        uart.send(nmea_bytes)
        uart.close()
    """

    def __init__(
        self,
        port: str = "/dev/ttyAMA0",
        baudrate: int = 115200,
        timeout: float = 1.0,
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
    ):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay_s
        self._serial = None
        self._stats = UARTStats()

    @property
    def stats(self) -> UARTStats:
        return self._stats

    @property
    def is_connected(self) -> bool:
        return self._stats.connected

    def open(self) -> bool:
        """Open serial port. Returns True on success."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=self._timeout,
            )
            self._stats.connected = True
            logger.info("UART opened: %s @ %d", self._port, self._baudrate)
            return True
        except Exception as e:
            logger.error("Failed to open UART %s: %s", self._port, e)
            self._stats.connected = False
            return False

    def close(self) -> None:
        """Close serial port."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._stats.connected = False

    def send(self, data: bytes, t: float | None = None) -> bool:
        """Send bytes over UART. Returns True on success.

        Attempts reconnection on failure, up to max_retries.
        """
        if t is None:
            t = time.monotonic()

        for attempt in range(self._max_retries + 1):
            if self._serial is None or not self._stats.connected:
                if not self._reconnect():
                    return False

            try:
                self._serial.write(data)
                self._serial.flush()
                self._stats.bytes_sent += len(data)
                self._stats.messages_sent += 1
                self._stats.last_send_t = t
                return True
            except Exception as e:
                self._stats.errors += 1
                self._stats.last_error_t = t
                self._stats.connected = False
                logger.warning(
                    "UART send failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, e,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        return False

    def send_nmea(self, *sentences: str) -> bool:
        """Send one or more NMEA sentences (adds \\r\\n if needed)."""
        parts = []
        for s in sentences:
            line = s if s.endswith("\r\n") else s + "\r\n"
            parts.append(line.encode("ascii"))
        return self.send(b"".join(parts))

    def send_msp(self, frame: bytes) -> bool:
        """Send an MSP binary frame."""
        return self.send(frame)

    def _reconnect(self) -> bool:
        """Attempt to reconnect."""
        self.close()
        self._stats.reconnects += 1
        logger.info("Reconnecting UART %s (attempt #%d)", self._port, self._stats.reconnects)
        return self.open()

    def summary(self) -> str:
        """Human-readable connection summary."""
        s = self._stats
        return (
            f"UART {self._port}: "
            f"{'connected' if s.connected else 'disconnected'}, "
            f"{s.messages_sent} msgs, {s.bytes_sent} bytes, "
            f"{s.errors} errors, {s.reconnects} reconnects"
        )
