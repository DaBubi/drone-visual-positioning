"""Extended Kalman Filter for position smoothing and velocity estimation.

State vector: [lat, lon, vlat, vlon]
- lat, lon: position in degrees
- vlat, vlon: velocity in degrees/second

The EKF fuses noisy visual position fixes with a constant-velocity motion
model to produce smooth position estimates and reject outliers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from shared.tile_math import GeoPoint


@dataclass(slots=True)
class EKFState:
    """Current filter state."""
    lat: float
    lon: float
    vlat: float  # degrees/sec
    vlon: float  # degrees/sec
    covariance: np.ndarray  # 4x4
    timestamp: float
    initialized: bool = False
    innovation_gate: float = 0.0  # last Mahalanobis distance


@dataclass(slots=True)
class EKFConfig:
    """EKF tuning parameters."""
    process_noise_pos: float = 1e-9    # position process noise (deg^2/s)
    process_noise_vel: float = 1e-7    # velocity process noise ((deg/s)^2/s)
    measurement_noise: float = 1e-8    # position measurement noise (deg^2)
    gate_threshold: float = 9.0        # chi-squared gate (3 sigma for 2 DoF)
    max_dt: float = 5.0                # max time gap before reset (seconds)


class PositionEKF:
    """Extended Kalman Filter for GPS-denied drone positioning.

    Fuses visual position fixes with constant-velocity prediction to:
    - Smooth noisy position estimates
    - Estimate velocity
    - Reject outlier measurements (Mahalanobis gating)
    - Interpolate between fixes for higher update rate
    """

    def __init__(self, config: EKFConfig | None = None):
        self._config = config or EKFConfig()
        # State: [lat, lon, vlat, vlon]
        self._x = np.zeros(4, dtype=np.float64)
        # Covariance
        self._P = np.eye(4, dtype=np.float64) * 1e-4
        self._last_t: float = 0.0
        self._initialized = False
        self._last_innovation_gate = 0.0

    @property
    def state(self) -> EKFState:
        return EKFState(
            lat=self._x[0],
            lon=self._x[1],
            vlat=self._x[2],
            vlon=self._x[3],
            covariance=self._P.copy(),
            timestamp=self._last_t,
            initialized=self._initialized,
            innovation_gate=self._last_innovation_gate,
        )

    @property
    def position(self) -> GeoPoint:
        return GeoPoint(lat=self._x[0], lon=self._x[1])

    @property
    def velocity_mps(self) -> tuple[float, float]:
        """Velocity in meters/second (north, east)."""
        # Convert deg/s to m/s
        lat_rad = math.radians(self._x[0])
        vn = self._x[2] * 111320.0  # 1 deg lat ≈ 111320 m
        ve = self._x[3] * 111320.0 * math.cos(lat_rad)
        return vn, ve

    @property
    def speed_mps(self) -> float:
        vn, ve = self.velocity_mps
        return math.sqrt(vn * vn + ve * ve)

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 1e-4
        self._initialized = False
        self._last_t = 0.0

    def predict(self, t: float) -> GeoPoint:
        """Predict state forward to time t.

        Call this between measurements to get interpolated position.
        Does NOT modify internal state — use for read-only prediction.
        """
        if not self._initialized:
            return GeoPoint(lat=0.0, lon=0.0)

        dt = t - self._last_t
        if dt <= 0:
            return self.position

        x_pred = self._x.copy()
        x_pred[0] += x_pred[2] * dt
        x_pred[1] += x_pred[3] * dt
        return GeoPoint(lat=x_pred[0], lon=x_pred[1])

    def update(self, measurement: GeoPoint, hdop: float, t: float | None = None) -> bool:
        """Incorporate a new visual position measurement.

        Args:
            measurement: measured GPS position from visual matching
            hdop: horizontal dilution of precision (higher = less precise)
            t: measurement timestamp (monotonic seconds). Uses time.monotonic() if None.

        Returns:
            True if measurement was accepted, False if rejected by gating.
        """
        if t is None:
            t = time.monotonic()

        z = np.array([measurement.lat, measurement.lon], dtype=np.float64)

        if not self._initialized:
            self._x[0] = z[0]
            self._x[1] = z[1]
            self._x[2] = 0.0
            self._x[3] = 0.0
            self._P = np.diag([
                self._config.measurement_noise * hdop,
                self._config.measurement_noise * hdop,
                self._config.process_noise_vel * 10,
                self._config.process_noise_vel * 10,
            ])
            self._last_t = t
            self._initialized = True
            self._last_innovation_gate = 0.0
            return True

        dt = t - self._last_t
        if dt > self._config.max_dt:
            # Too long since last fix — reset
            self.reset()
            return self.update(measurement, hdop, t)

        if dt <= 0:
            dt = 1e-3  # avoid division by zero

        # === Prediction step ===
        # State transition: constant velocity
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        x_pred = F @ self._x

        # Process noise
        Q = np.diag([
            self._config.process_noise_pos * dt,
            self._config.process_noise_pos * dt,
            self._config.process_noise_vel * dt,
            self._config.process_noise_vel * dt,
        ])

        P_pred = F @ self._P @ F.T + Q

        # === Update step ===
        # Measurement matrix: we observe [lat, lon]
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Measurement noise scaled by HDOP
        R = np.eye(2, dtype=np.float64) * self._config.measurement_noise * max(1.0, hdop)

        # Innovation
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R

        # Mahalanobis gating — reject outliers
        S_inv = np.linalg.inv(S)
        mahal_sq = float(y.T @ S_inv @ y)
        self._last_innovation_gate = mahal_sq

        if mahal_sq > self._config.gate_threshold:
            # Outlier — don't update
            # Still advance time with prediction only
            self._x = x_pred
            self._P = P_pred
            self._last_t = t
            return False

        # Kalman gain
        K = P_pred @ H.T @ S_inv

        # State update
        self._x = x_pred + K @ y
        self._P = (np.eye(4) - K @ H) @ P_pred

        self._last_t = t
        return True
