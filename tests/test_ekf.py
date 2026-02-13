"""Tests for the Extended Kalman Filter position smoother."""

import math

import numpy as np
import pytest

from onboard.ekf import EKFConfig, PositionEKF
from shared.tile_math import GeoPoint, haversine_km


class TestEKFInitialization:
    def test_starts_uninitialized(self):
        ekf = PositionEKF()
        state = ekf.state
        assert not state.initialized
        assert state.lat == 0.0
        assert state.lon == 0.0

    def test_first_measurement_initializes(self):
        ekf = PositionEKF()
        accepted = ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.0)
        assert accepted
        assert ekf.state.initialized
        assert abs(ekf.position.lat - 52.5200) < 1e-6
        assert abs(ekf.position.lon - 13.4050) < 1e-6

    def test_reset_clears_state(self):
        ekf = PositionEKF()
        ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.0)
        ekf.reset()
        assert not ekf.state.initialized


class TestEKFPrediction:
    def test_predict_without_init_returns_zero(self):
        ekf = PositionEKF()
        pos = ekf.predict(1.0)
        assert pos.lat == 0.0
        assert pos.lon == 0.0

    def test_predict_stationary(self):
        ekf = PositionEKF()
        ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.0)
        # After one measurement, velocity is zero
        pos = ekf.predict(1.0)
        assert abs(pos.lat - 52.5200) < 1e-6
        assert abs(pos.lon - 13.4050) < 1e-6

    def test_predict_moving(self):
        ekf = PositionEKF()
        # Two measurements 1 second apart, moving north
        ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.0)
        ekf.update(GeoPoint(lat=52.5201, lon=13.4050), hdop=1.0, t=1.0)
        # Predict 1 second ahead — should continue northward
        pos = ekf.predict(2.0)
        assert pos.lat > 52.5201


class TestEKFSmoothing:
    def test_smooths_noisy_stationary(self):
        """Stationary target with noisy measurements should converge."""
        ekf = PositionEKF()
        true_lat, true_lon = 52.5200, 13.4050
        rng = np.random.RandomState(42)

        positions = []
        for i in range(50):
            noise_lat = rng.normal(0, 1e-5)
            noise_lon = rng.normal(0, 1e-5)
            ekf.update(
                GeoPoint(lat=true_lat + noise_lat, lon=true_lon + noise_lon),
                hdop=1.0,
                t=float(i) * 0.3,
            )
            positions.append(ekf.position)

        # Final position should be closer to truth than raw measurements
        final = positions[-1]
        error_m = haversine_km(GeoPoint(true_lat, true_lon), final) * 1000
        assert error_m < 2.0, f"EKF error {error_m:.1f}m, expected <2m"

    def test_smooths_linear_trajectory(self):
        """Moving target with noise — EKF should track smoothly."""
        ekf = PositionEKF()
        rng = np.random.RandomState(123)

        # Simulate drone moving north at ~10 m/s
        speed_deg_per_sec = 10.0 / 111320.0  # ~10 m/s northward

        errors = []
        for i in range(100):
            t = float(i) * 0.3
            true_lat = 52.5200 + speed_deg_per_sec * t
            true_lon = 13.4050

            noise_lat = rng.normal(0, 2e-5)  # ~2m noise
            noise_lon = rng.normal(0, 2e-5)

            ekf.update(
                GeoPoint(lat=true_lat + noise_lat, lon=true_lon + noise_lon),
                hdop=1.0,
                t=t,
            )

            pos = ekf.position
            err = haversine_km(GeoPoint(true_lat, true_lon), pos) * 1000
            errors.append(err)

        # After convergence, errors should be small
        late_errors = errors[20:]
        mean_err = np.mean(late_errors)
        assert mean_err < 5.0, f"Mean tracking error {mean_err:.1f}m, expected <5m"

    def test_velocity_estimation(self):
        """EKF should estimate velocity after a few updates."""
        ekf = PositionEKF()
        speed_deg_per_sec = 15.0 / 111320.0  # 15 m/s north

        for i in range(30):
            t = float(i) * 0.3
            lat = 52.5200 + speed_deg_per_sec * t
            ekf.update(GeoPoint(lat=lat, lon=13.4050), hdop=1.0, t=t)

        vn, ve = ekf.velocity_mps
        assert abs(vn - 15.0) < 3.0, f"North velocity {vn:.1f} m/s, expected ~15"
        assert abs(ve) < 2.0, f"East velocity {ve:.1f} m/s, expected ~0"
        assert abs(ekf.speed_mps - 15.0) < 3.0


class TestEKFOutlierRejection:
    def test_rejects_large_outlier(self):
        """A measurement far from the current state should be rejected."""
        ekf = PositionEKF()
        # Establish state with 10 good measurements
        for i in range(10):
            ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=float(i) * 0.3)

        # Inject a 1km outlier
        accepted = ekf.update(
            GeoPoint(lat=52.5300, lon=13.4050),  # ~1.1 km away
            hdop=1.0,
            t=3.5,
        )
        assert not accepted, "Outlier should be rejected"

        # Position should still be near the original
        err = haversine_km(GeoPoint(52.5200, 13.4050), ekf.position) * 1000
        assert err < 50, f"Position drifted {err:.0f}m after outlier"

    def test_accepts_normal_measurement(self):
        ekf = PositionEKF()
        ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.0)
        # Small step
        accepted = ekf.update(
            GeoPoint(lat=52.52001, lon=13.40501),
            hdop=1.0,
            t=0.3,
        )
        assert accepted

    def test_gate_value_recorded(self):
        ekf = PositionEKF()
        ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.0)
        ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.3)
        assert ekf.state.innovation_gate >= 0.0


class TestEKFEdgeCases:
    def test_long_gap_resets(self):
        """If too much time passes, EKF should reset."""
        ekf = PositionEKF(EKFConfig(max_dt=2.0))
        ekf.update(GeoPoint(lat=52.5200, lon=13.4050), hdop=1.0, t=0.0)
        # 10 second gap (> max_dt)
        ekf.update(GeoPoint(lat=48.8566, lon=2.3522), hdop=1.0, t=10.0)
        # Should have reset to the new position (Paris)
        assert abs(ekf.position.lat - 48.8566) < 1e-4

    def test_high_hdop_increases_uncertainty(self):
        """Measurements with high HDOP should have less influence."""
        # Use wide-gate config so nothing gets rejected
        cfg = EKFConfig(gate_threshold=1e6)
        ekf1 = PositionEKF(cfg)
        ekf2 = PositionEKF(cfg)

        origin = GeoPoint(lat=52.5200, lon=13.4050)
        # Establish baseline with several measurements
        for i in range(5):
            ekf1.update(origin, hdop=1.0, t=float(i) * 0.3)
            ekf2.update(origin, hdop=1.0, t=float(i) * 0.3)

        # Noisy measurement ~110m away
        noisy = GeoPoint(lat=52.5210, lon=13.4060)
        t_noisy = 5 * 0.3
        ekf1.update(noisy, hdop=1.0, t=t_noisy)
        ekf2.update(noisy, hdop=50.0, t=t_noisy)

        # ekf2 should be pulled less toward the noisy measurement
        err1 = haversine_km(origin, ekf1.position)
        err2 = haversine_km(origin, ekf2.position)
        assert err2 < err1, f"High HDOP should reduce influence: err1={err1:.6f} err2={err2:.6f}"
