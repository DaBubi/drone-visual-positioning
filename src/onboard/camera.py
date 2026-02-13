"""Frame capture abstraction for drone camera.

Supports OpenCV VideoCapture (USB cameras) and picamera2 (RPi CSI).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from onboard.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraBase(ABC):
    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def grab(self) -> np.ndarray | None:
        """Capture a single frame. Returns BGR image or None on failure."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


class OpenCVCamera(CameraBase):
    """USB camera via OpenCV VideoCapture."""

    def __init__(self, config: CameraConfig):
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        dev = self._config.device
        if isinstance(dev, str):
            self._cap = cv2.VideoCapture(dev)
        else:
            self._cap = cv2.VideoCapture(dev)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {dev}")
        logger.info("Camera opened: %s (%dx%d @ %dfps)",
                     dev, self._config.width, self._config.height, self._config.fps)

    def grab(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        # Resize to configured dimensions if capture returns different size
        h, w = frame.shape[:2]
        if w != self._config.width or h != self._config.height:
            frame = cv2.resize(frame, (self._config.width, self._config.height))
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class PiCamera2Camera(CameraBase):
    """Raspberry Pi CSI camera via picamera2."""

    def __init__(self, config: CameraConfig):
        self._config = config
        self._picam = None

    def open(self) -> None:
        from picamera2 import Picamera2
        self._picam = Picamera2()
        cam_config = self._picam.create_still_configuration(
            main={"size": (self._config.width, self._config.height), "format": "BGR888"}
        )
        self._picam.configure(cam_config)
        self._picam.start()
        logger.info("picamera2 started (%dx%d)", self._config.width, self._config.height)

    def grab(self) -> np.ndarray | None:
        if self._picam is None:
            return None
        return self._picam.capture_array()

    def close(self) -> None:
        if self._picam is not None:
            self._picam.stop()
            self._picam = None


def create_camera(config: CameraConfig) -> CameraBase:
    """Factory: create appropriate camera backend."""
    if config.use_picamera2:
        return PiCamera2Camera(config)
    return OpenCVCamera(config)
