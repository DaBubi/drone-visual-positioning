"""Tests for binary flight recorder."""

import struct

import pytest

from onboard.flight_recorder import (
    FlightRecord, FlightRecorder, RECORD_SIZE, HEADER_SIZE,
    HEADER_MAGIC, HEADER_VERSION, SOURCE_VISUAL, SOURCE_NONE,
    FLAG_GEOFENCE_OK, FLAG_EKF_ACCEPTED, SOURCE_MAP,
)


def _sample_record(t=1.0, lat=52.52, lon=13.405) -> FlightRecord:
    return FlightRecord(
        timestamp=t, lat=lat, lon=lon,
        vn_mps=5.0, ve_mps=3.0, hdop=1.5,
        speed_mps=5.83, heading_deg=31.0,
        fix_quality=1, source=SOURCE_VISUAL,
        match_count=45, inlier_ratio=0.65,
        latency_ms=150, flags=FLAG_GEOFENCE_OK | FLAG_EKF_ACCEPTED,
    )


class TestFlightRecord:
    def test_pack_size(self):
        rec = _sample_record()
        data = rec.pack()
        assert len(data) == RECORD_SIZE

    def test_roundtrip(self):
        rec = _sample_record()
        data = rec.pack()
        rec2 = FlightRecord.unpack(data)
        assert rec2.timestamp == pytest.approx(rec.timestamp)
        assert rec2.lat == pytest.approx(rec.lat)
        assert rec2.lon == pytest.approx(rec.lon)
        assert rec2.vn_mps == pytest.approx(rec.vn_mps, abs=0.01)
        assert rec2.fix_quality == rec.fix_quality
        assert rec2.source == rec.source
        assert rec2.match_count == rec.match_count
        assert rec2.flags == rec.flags

    def test_source_map(self):
        assert SOURCE_MAP["visual"] == SOURCE_VISUAL
        assert SOURCE_MAP["none"] == SOURCE_NONE

    def test_flags(self):
        assert FLAG_GEOFENCE_OK == 0x01
        assert FLAG_EKF_ACCEPTED == 0x02

    def test_zero_record(self):
        rec = FlightRecord(
            timestamp=0, lat=0, lon=0,
            vn_mps=0, ve_mps=0, hdop=0,
            speed_mps=0, heading_deg=0,
            fix_quality=0, source=0,
            match_count=0, inlier_ratio=0,
            latency_ms=0, flags=0,
        )
        data = rec.pack()
        rec2 = FlightRecord.unpack(data)
        assert rec2.lat == 0.0


class TestFlightRecorder:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "test.vpsf"
        rec = FlightRecorder(path)
        rec.start()
        for i in range(10):
            rec.record(_sample_record(t=float(i), lat=52.52 + i * 0.0001))
        rec.stop()

        records = FlightRecorder.read(path)
        assert len(records) == 10
        assert records[0].lat == pytest.approx(52.52)
        assert records[9].lat == pytest.approx(52.521, abs=0.001)

    def test_record_count(self, tmp_path):
        path = tmp_path / "test.vpsf"
        rec = FlightRecorder(path)
        rec.start()
        for i in range(5):
            rec.record(_sample_record(t=float(i)))
        assert rec.record_count == 5
        rec.stop()

    def test_is_recording(self, tmp_path):
        path = tmp_path / "test.vpsf"
        rec = FlightRecorder(path)
        assert not rec.is_recording
        rec.start()
        assert rec.is_recording
        rec.stop()
        assert not rec.is_recording

    def test_file_info(self, tmp_path):
        path = tmp_path / "test.vpsf"
        rec = FlightRecorder(path)
        rec.start()
        for i in range(20):
            rec.record(_sample_record(t=float(i)))
        rec.stop()

        info = FlightRecorder.file_info(path)
        assert info["version"] == HEADER_VERSION
        assert info["record_count"] == 20
        assert info["record_size"] == RECORD_SIZE

    def test_empty_file(self, tmp_path):
        path = tmp_path / "test.vpsf"
        rec = FlightRecorder(path)
        rec.start()
        rec.stop()

        records = FlightRecorder.read(path)
        assert len(records) == 0

    def test_invalid_magic(self, tmp_path):
        path = tmp_path / "bad.vpsf"
        with open(path, "wb") as f:
            f.write(b"BAAD\x02\x00\x3a\x00")

        with pytest.raises(ValueError, match="Invalid file magic"):
            FlightRecorder.read(path)

    def test_header_size(self):
        assert HEADER_SIZE == 8

    def test_record_without_start(self, tmp_path):
        path = tmp_path / "test.vpsf"
        rec = FlightRecorder(path)
        rec.record(_sample_record())  # should not crash
        assert rec.record_count == 0

    def test_file_size(self, tmp_path):
        path = tmp_path / "test.vpsf"
        rec = FlightRecorder(path)
        rec.start()
        for i in range(100):
            rec.record(_sample_record(t=float(i)))
        rec.stop()

        expected = HEADER_SIZE + 100 * RECORD_SIZE
        assert path.stat().st_size == expected
