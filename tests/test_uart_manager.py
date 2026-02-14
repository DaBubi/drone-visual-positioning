"""Tests for UART manager."""

from unittest.mock import MagicMock, patch

import pytest

from onboard.uart_manager import UARTManager, UARTStats, Protocol


class TestUARTStats:
    def test_defaults(self):
        s = UARTStats()
        assert s.bytes_sent == 0
        assert s.messages_sent == 0
        assert s.errors == 0
        assert not s.connected

    def test_protocol_enum(self):
        assert Protocol.NMEA.value == "nmea"
        assert Protocol.MSP.value == "msp"


class _MockSerial:
    """Mock serial.Serial for testing without pyserial."""
    def __init__(self, **kwargs):
        self.port = kwargs.get("port", "")
        self.baudrate = kwargs.get("baudrate", 9600)
        self._written = []
        self._fail_write = False

    def write(self, data):
        if self._fail_write:
            raise OSError("Write failed")
        self._written.append(data)

    def flush(self):
        pass

    def close(self):
        pass


class TestUARTManager:
    def test_init(self):
        uart = UARTManager("/dev/ttyUSB0", baudrate=9600)
        assert not uart.is_connected
        assert uart.stats.bytes_sent == 0

    def test_open_success(self):
        uart = UARTManager("/dev/ttyAMA0")
        mock_ser = _MockSerial(port="/dev/ttyAMA0")
        with patch.dict("sys.modules", {"serial": MagicMock()}):
            import sys
            sys.modules["serial"].Serial.return_value = mock_ser
            assert uart.open()
            assert uart.is_connected

    def test_open_failure(self):
        uart = UARTManager("/dev/ttyNONE")
        with patch.dict("sys.modules", {"serial": MagicMock()}):
            import sys
            sys.modules["serial"].Serial.side_effect = OSError("Port not found")
            assert not uart.open()
            assert not uart.is_connected

    def _open_with_mock(self, uart):
        """Helper: inject mock serial and open."""
        mock_ser = _MockSerial()
        uart._serial = mock_ser
        uart._stats.connected = True
        return mock_ser

    def test_send_success(self):
        uart = UARTManager()
        self._open_with_mock(uart)

        assert uart.send(b"hello", t=1.0)
        assert uart.stats.bytes_sent == 5
        assert uart.stats.messages_sent == 1
        assert uart.stats.last_send_t == 1.0

    def test_send_failure_retries(self):
        uart = UARTManager(max_retries=0, retry_delay_s=0.0)
        mock_ser = self._open_with_mock(uart)
        mock_ser._fail_write = True

        assert not uart.send(b"fail", t=1.0)
        assert uart.stats.errors >= 1

    def test_send_nmea(self):
        uart = UARTManager()
        mock_ser = self._open_with_mock(uart)

        assert uart.send_nmea("$GPGGA,...")
        data = mock_ser._written[0]
        assert data.endswith(b"\r\n")

    def test_send_nmea_no_double_crlf(self):
        uart = UARTManager()
        mock_ser = self._open_with_mock(uart)

        uart.send_nmea("$GPGGA,...\r\n")
        data = mock_ser._written[0]
        assert not data.endswith(b"\r\n\r\n")

    def test_send_msp(self):
        uart = UARTManager()
        mock_ser = self._open_with_mock(uart)

        frame = b"\x24\x4d\x3c\x12\xc9\x00"
        assert uart.send_msp(frame)
        assert mock_ser._written[0] == frame

    def test_close(self):
        uart = UARTManager()
        self._open_with_mock(uart)
        uart.close()
        assert not uart.is_connected

    def test_summary(self):
        uart = UARTManager("/dev/ttyAMA0")
        self._open_with_mock(uart)
        s = uart.summary()
        assert "ttyAMA0" in s
        assert "connected" in s

    def test_multiple_sends(self):
        uart = UARTManager()
        self._open_with_mock(uart)

        for i in range(5):
            uart.send(b"x" * 10, t=float(i))

        assert uart.stats.messages_sent == 5
        assert uart.stats.bytes_sent == 50

    def test_send_without_open(self):
        uart = UARTManager(max_retries=0, retry_delay_s=0.0)
        assert not uart.send(b"test")

    def test_multiple_nmea_sentences(self):
        uart = UARTManager()
        mock_ser = self._open_with_mock(uart)

        assert uart.send_nmea("$GPGGA,...", "$GPRMC,...")
        data = mock_ser._written[0]
        assert data.count(b"\r\n") == 2
