"""Threaded frame producers.

`FrameSource` is an abstract base that runs a background thread, pumping
frames from a hardware source (Pi camera or USB webcam) into a thread-safe
slot that the rest of the pipeline reads via `get_frame()`. Both concrete
backends emit BGR numpy arrays shaped (height, width, 3) so the downstream
vision / streamer code doesn't need to know which one is active."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..config import CameraConfig
from ..logging_setup import get_logger

log = get_logger(__name__)


class FrameSource(ABC):
    """Abstract base: threaded BGR-frame producer with black-frame fallback."""

    #: subclass override: set True if the backend's capture call already
    #: rate-limits to the camera's native fps (e.g. blocking reads from
    #: cv2.VideoCapture). Prevents us from double-throttling.
    _internal_rate_limits: bool = False

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._opened = False

    def start(self) -> None:
        self._opened = self._open()
        self._thread = threading.Thread(
            target=self._run, name=type(self).__name__, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    # --- subclass API ------------------------------------------------------

    @abstractmethod
    def _open(self) -> bool:
        """Open the underlying hardware. Return True on success."""

    @abstractmethod
    def _close(self) -> None:
        """Release the underlying hardware."""

    @abstractmethod
    def _capture_one(self, black: np.ndarray) -> np.ndarray:
        """Return the next BGR frame (or `black` on failure)."""

    # --- main loop --------------------------------------------------------

    def _run(self) -> None:
        period = 1.0 / max(1, self.cfg.framerate)
        black = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        while not self._stop.is_set():
            frame = self._capture_one(black)
            with self._lock:
                self._latest = frame
            if not self._internal_rate_limits:
                self._stop.wait(period)


# ---------------------------------------------------------------------------
# picamera2 backend
# ---------------------------------------------------------------------------


class PiCameraFrameSource(FrameSource):
    """Reads from the Raspberry Pi Camera via picamera2 / libcamera. Applies
    hardware 180° rotation and tuning controls (EV bias, gain, brightness,
    contrast, saturation) on open."""

    _internal_rate_limits = False      # capture_array() is non-blocking

    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        self._picam = None

    def _open(self) -> bool:
        try:
            from picamera2 import Picamera2
            from libcamera import Transform

            cam = Picamera2()
            transform = Transform(hflip=1, vflip=1) if self.cfg.rotate_180 else Transform()
            frame_us = int(1_000_000 / max(1, self.cfg.framerate))
            config = cam.create_video_configuration(
                # BGR888 matches OpenCV's native channel order.
                main={"size": (self.cfg.width, self.cfg.height), "format": "BGR888"},
                transform=transform,
                controls={"FrameDurationLimits": (frame_us, frame_us)},
            )
            cam.configure(config)
            cam.start()

            tuning = self._build_tuning_controls()
            if tuning:
                try:
                    cam.set_controls(tuning)
                except Exception as e:
                    log.warning("set_controls failed (%s); continuing with defaults", e)

            log.info(
                "picamera2 started at %dx%d fps=%d rotate_180=%s tuning=%s",
                self.cfg.width, self.cfg.height, self.cfg.framerate,
                self.cfg.rotate_180, tuning,
            )
            self._picam = cam
            return True
        except Exception as e:
            log.warning("picamera2 unavailable (%s); falling back to black frames", e)
            return False

    def _close(self) -> None:
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass
            self._picam = None

    def _capture_one(self, black: np.ndarray) -> np.ndarray:
        if self._picam is None:
            return black
        try:
            return self._picam.capture_array()
        except Exception as e:
            log.warning("picamera capture failed: %s", e)
            return black

    def _build_tuning_controls(self) -> dict:
        ctrls: dict = {}
        if self.cfg.exposure_value is not None:
            ctrls["ExposureValue"] = float(self.cfg.exposure_value)
        if self.cfg.analogue_gain is not None:
            ctrls["AnalogueGain"] = float(self.cfg.analogue_gain)
        if self.cfg.brightness is not None:
            ctrls["Brightness"] = float(self.cfg.brightness)
        if self.cfg.contrast is not None:
            ctrls["Contrast"] = float(self.cfg.contrast)
        if self.cfg.saturation is not None:
            ctrls["Saturation"] = float(self.cfg.saturation)
        return ctrls
