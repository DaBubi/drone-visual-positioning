"""Tests for the telemetry logging module."""

import csv
import time
from pathlib import Path

import pytest

from onboard.telemetry import FrameRecord, TelemetryLogger


class TestFrameRecord:
    def test_fix_record_to_row(self):
        rec = FrameRecord(
            timestamp=1000.0,
            frame_num=42,
            fix=True,
            lat=52.5200,
            lon=13.4050,
            hdop=1.2,
            inlier_ratio=0.85,
            num_matches=45,
            tile_z=17,
            tile_x=70406,
            tile_y=42987,
            retrieval_ms=5.3,
            match_ms=78.2,
            total_ms=90.1,
            ekf_lat=52.52001,
            ekf_lon=13.40499,
            ekf_vlat=0.0000012,
            ekf_vlon=-0.0000003,
            ekf_speed_mps=0.15,
            ekf_gate=1.23,
            ekf_accepted=True,
        )
        row = rec.to_row()
        assert len(row) == 21
        assert row[0] == "1000.000"
        assert row[1] == 42
        assert row[2] == 1  # fix=True
        assert "52.52" in row[3]
        assert row[7] == 45  # num_matches

    def test_miss_record_to_row(self):
        rec = FrameRecord(
            timestamp=1001.0,
            frame_num=43,
            fix=False,
            retrieval_ms=4.1,
            match_ms=120.0,
            total_ms=130.0,
        )
        row = rec.to_row()
        assert row[2] == 0  # fix=False
        assert row[3] == ""  # no lat
        assert row[4] == ""  # no lon


class TestTelemetryLogger:
    def test_creates_csv_file(self, tmp_path):
        logger = TelemetryLogger(tmp_path, prefix="test")
        path = logger.start()
        assert path.exists()
        assert path.name.startswith("test_")
        assert path.suffix == ".csv"
        logger.stop()

    def test_writes_header(self, tmp_path):
        logger = TelemetryLogger(tmp_path)
        path = logger.start()
        logger.stop()

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "timestamp" in header
        assert "ekf_lat" in header
        assert len(header) == 21

    def test_writes_records(self, tmp_path):
        logger = TelemetryLogger(tmp_path)
        path = logger.start()

        for i in range(10):
            rec = FrameRecord(
                timestamp=float(i),
                frame_num=i,
                fix=i % 2 == 0,
                lat=52.52 if i % 2 == 0 else 0.0,
                lon=13.40 if i % 2 == 0 else 0.0,
                retrieval_ms=5.0,
                match_ms=80.0,
                total_ms=90.0,
            )
            logger.log(rec)

        logger.stop()

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 11  # header + 10 records

    def test_no_write_before_start(self, tmp_path):
        logger = TelemetryLogger(tmp_path)
        rec = FrameRecord(timestamp=0.0, frame_num=0)
        # Should not raise
        logger.log(rec)

    def test_stop_without_start(self, tmp_path):
        logger = TelemetryLogger(tmp_path)
        # Should not raise
        logger.stop()
