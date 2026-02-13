"""Tests for NMEA sentence generation."""

from datetime import datetime, timezone

import pytest

from onboard.nmea import (
    PositionFix,
    format_gga,
    format_rmc,
    nmea_checksum,
    _format_lat,
    _format_lon,
)


class TestChecksum:
    def test_known_checksum(self):
        # Known NMEA sentence checksum
        body = "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,"
        cs = nmea_checksum(body)
        assert len(cs) == 2
        assert all(c in "0123456789ABCDEF" for c in cs)

    def test_all_same_chars(self):
        cs = nmea_checksum("AAA")
        # A ^ A ^ A = A (0x41)
        assert cs == "41"


class TestFormatCoordinates:
    def test_lat_north(self):
        val, hem = _format_lat(47.6205)
        assert hem == "N"
        # 47 degrees, 37.23 minutes
        assert val.startswith("47")

    def test_lat_south(self):
        val, hem = _format_lat(-33.8688)
        assert hem == "S"
        assert val.startswith("33")

    def test_lon_east(self):
        val, hem = _format_lon(139.6503)
        assert hem == "E"
        assert val.startswith("139")

    def test_lon_west(self):
        val, hem = _format_lon(-122.3493)
        assert hem == "W"
        assert val.startswith("122")

    def test_lat_zero(self):
        val, hem = _format_lat(0.0)
        assert hem == "N"
        assert val.startswith("00")


class TestFormatGGA:
    def test_basic_format(self):
        fix = PositionFix(lat=47.6205, lon=-122.3493, altitude=100.0)
        dt = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        gga = format_gga(fix, dt)

        assert gga.startswith("$GPGGA,")
        assert gga.endswith("\r\n")
        assert "*" in gga
        # Check time field
        assert "123045" in gga
        # Check it has a valid checksum
        body = gga[1:gga.index("*")]
        expected_cs = nmea_checksum(body)
        actual_cs = gga[gga.index("*") + 1:gga.index("*") + 3]
        assert actual_cs == expected_cs

    def test_no_fix(self):
        fix = PositionFix(lat=0, lon=0, fix_quality=0)
        gga = format_gga(fix)
        # Fix quality should be 0
        fields = gga.split(",")
        assert fields[6] == "0"

    def test_field_count(self):
        fix = PositionFix(lat=47.6205, lon=-122.3493)
        gga = format_gga(fix)
        # GGA has 15 fields (including the one with checksum)
        body = gga[1:gga.index("*")]
        fields = body.split(",")
        assert len(fields) == 15


class TestFormatRMC:
    def test_basic_format(self):
        fix = PositionFix(lat=47.6205, lon=-122.3493, speed_knots=5.2, course_deg=270.0)
        dt = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        rmc = format_rmc(fix, dt)

        assert rmc.startswith("$GPRMC,")
        assert rmc.endswith("\r\n")
        assert "123045" in rmc
        assert "150624" in rmc  # date ddmmyy

    def test_status_active(self):
        fix = PositionFix(lat=47.6205, lon=-122.3493, fix_quality=1)
        rmc = format_rmc(fix)
        fields = rmc.split(",")
        assert fields[2] == "A"  # Active

    def test_status_void(self):
        fix = PositionFix(lat=0, lon=0, fix_quality=0)
        rmc = format_rmc(fix)
        fields = rmc.split(",")
        assert fields[2] == "V"  # Void

    def test_checksum_valid(self):
        fix = PositionFix(lat=51.5074, lon=-0.1278)
        rmc = format_rmc(fix)
        body = rmc[1:rmc.index("*")]
        expected = nmea_checksum(body)
        actual = rmc[rmc.index("*") + 1:rmc.index("*") + 3]
        assert actual == expected
