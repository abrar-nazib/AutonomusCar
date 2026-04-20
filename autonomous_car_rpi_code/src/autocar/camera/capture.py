from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from ..config import CameraConfig
from ..logging_setup import get_logger

log = get_logger(__name__)


class FrameSource:
    """Background frame producer. Uses picamera2 when available, otherwise
    emits black frames so the rest of the pipeline keeps running."""

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._picam = None

    def start(self) -> None:
        self._picam = self._try_open_picamera()
        self._thread = threading.Thread(target=self._run, name="FrameSource", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def _try_open_picamera(self):
        try:
            from picamera2 import Picamera2

            cam = Picamera2()
            config = cam.create_video_configuration(
                main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            log.info("picamera2 started at %dx%d", self.cfg.width, self.cfg.height)
            return cam
        except Exception as e:
            log.warning("picamera2 unavailable (%s); falling back to black frames", e)
            return None

    def _run(self) -> None:
        period = 1.0 / max(1, self.cfg.framerate)
        black = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        while not self._stop.is_set():
            frame = self._capture_one(black)
            with self._lock:
                self._latest = frame
            self._stop.wait(period)

    def _capture_one(self, black: np.ndarray) -> np.ndarray:
        if self._picam is None:
            return black
        try:
            return self._picam.capture_array()
        except Exception as e:
            log.warning("frame capture failed (%s); emitting black frame", e)
            return black
