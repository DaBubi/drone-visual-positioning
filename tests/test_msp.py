"""Tests for MSP protocol GPS output."""

import struct

import pytest

from onboard.msp import (
    MSPGPSData,
    MSP_SET_RAW_GPS,
    encode_set_raw_gps,
    msp_checksum,
)


class TestMSPChecksum:
    def test_empty(self):
        assert msp_checksum(b"") == 0

    def test_single_byte(self):
        assert msp_checksum(b"\x42") == 0x42

    def test_two_bytes(self):
        assert msp_checksum(b"\x42\x42") == 0  # XOR cancels

    def test_known(self):
        assert msp_checksum(b"\x01\x02\x03") == 0x01 ^ 0x02 ^ 0x03


class TestMSPGPSData:
    def test_from_position(self):
        gps = MSPGPSData.from_position(
            lat=52.5200, lon=13.4050,
            alt_m=100.0, speed_mps=15.0,
            heading_deg=90.0, hdop=1.2,
        )
        assert gps.lat == 525200000  # 52.52 * 1e7
        assert gps.lon == 134050000  # 13.405 * 1e7
        assert gps.alt == 10000      # 100m * 100
        assert gps.speed == 1500     # 15 m/s * 100
        assert gps.ground_course == 900  # 90 * 10
        assert gps.hdop == 120       # 1.2 * 100
        assert gps.fix_type == 2
        assert gps.num_sats == 10

    def test_from_position_no_fix(self):
        gps = MSPGPSData.from_position(lat=0, lon=0, fix_type=0, num_sats=0)
        assert gps.fix_type == 0
        assert gps.num_sats == 0

    def test_heading_wraps(self):
        gps = MSPGPSData.from_position(lat=0, lon=0, heading_deg=370.0)
        assert gps.ground_course == 100  # 370 * 10 = 3700, mod 3600 = 100


class TestEncodeMSP:
    def test_frame_header(self):
        gps = MSPGPSData()
        frame = encode_set_raw_gps(gps)
        assert frame[:3] == b"$M<"

    def test_frame_length(self):
        gps = MSPGPSData()
        frame = encode_set_raw_gps(gps)
        length = frame[3]
        assert length == 18  # payload size for SET_RAW_GPS

    def test_frame_command(self):
        gps = MSPGPSData()
        frame = encode_set_raw_gps(gps)
        cmd = frame[4]
        assert cmd == MSP_SET_RAW_GPS  # 201

    def test_checksum_valid(self):
        gps = MSPGPSData.from_position(lat=52.52, lon=13.405)
        frame = encode_set_raw_gps(gps)
        # Checksum covers length + cmd + payload (everything after header, before checksum)
        data = frame[3:-1]
        expected_cs = msp_checksum(data)
        actual_cs = frame[-1]
        assert actual_cs == expected_cs

    def test_payload_decode(self):
        gps = MSPGPSData.from_position(
            lat=52.5200, lon=13.4050,
            alt_m=50.0, speed_mps=10.0,
            heading_deg=180.0, hdop=2.0,
        )
        frame = encode_set_raw_gps(gps)
        # Extract payload (after header + length + cmd)
        payload = frame[5:-1]
        fix_type, num_sats, lat, lon, alt, speed, course, hdop = struct.unpack(
            "<BBiiHHHH", payload,
        )
        assert fix_type == 2
        assert num_sats == 10
        assert lat == 525200000
        assert lon == 134050000
        assert alt == 5000
        assert speed == 1000
        assert course == 1800
        assert hdop == 200

    def test_total_frame_size(self):
        gps = MSPGPSData()
        frame = encode_set_raw_gps(gps)
        # $M< (3) + length (1) + cmd (1) + payload (18) + checksum (1) = 24
        assert len(frame) == 24

    def test_negative_coordinates(self):
        gps = MSPGPSData.from_position(lat=-33.8688, lon=-70.6693)
        frame = encode_set_raw_gps(gps)
        payload = frame[5:-1]
        _, _, lat, lon, _, _, _, _ = struct.unpack("<BBiiHHHH", payload)
        assert lat == -338688000
        assert lon == -706693000
