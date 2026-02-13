"""Runtime configuration for the onboard VPS module."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class CameraConfig(BaseModel):
    """Camera capture settings."""
    device: int | str = 0       # OpenCV device index or /dev/videoN path
    width: int = 640
    height: int = 640
    fps: int = 10
    use_picamera2: bool = False  # Use picamera2 (RPi CSI) instead of OpenCV


class UartConfig(BaseModel):
    """UART output settings for flight controller."""
    port: str = "/dev/ttyAMA0"
    baudrate: int = 9600
    enabled: bool = True


class MatcherConfig(BaseModel):
    """Feature matching pipeline settings."""
    superpoint_onnx: Path = Path("models/superpoint.onnx")
    lightglue_onnx: Path = Path("models/lightglue.onnx")
    min_matches: int = 15       # minimum inlier matches to accept a fix
    confidence_threshold: float = 0.3  # minimum inlier ratio
    max_candidates: int = 5     # top-k tiles from retrieval
    use_orb_fallback: bool = False  # fall back to ORB if ONNX unavailable


class VPSConfig(BaseModel):
    """Top-level configuration."""
    map_pack: Path = Path("/opt/vps/maps/map_pack")
    camera: CameraConfig = CameraConfig()
    uart: UartConfig = UartConfig()
    matcher: MatcherConfig = MatcherConfig()
    target_hz: float = 3.0      # target position update rate
    log_level: str = "INFO"

    # EKF settings
    ekf_measurement_noise: float = 1e-8  # position measurement noise (deg^2)
    ekf_gate_threshold: float = 9.0      # Mahalanobis gate (chi-sq, 2 DoF)

    # Telemetry
    telemetry_dir: Path | None = None    # set to enable CSV flight logging
