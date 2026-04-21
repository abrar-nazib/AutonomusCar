"""USB webcam frame source (V4L2 / cv2.VideoCapture).

Matches the `FrameSource` interface so the rest of the pipeline is unchanged.
Best for USB wide-angle cameras connected to the Pi — the picamera2 backend
is not used in this case."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import CameraConfig
from ..logging_setup import get_logger
from .capture import FrameSource

log = get_logger(__name__)


class WebcamFrameSource(FrameSource):
    """Opens /dev/videoN via OpenCV's V4L2 backend. Applies `rotate_180` in
    software via `cv2.rotate` (the Pi camera does it in hardware)."""

    #: `cap.read()` blocks at the camera's native fps, so don't also sleep.
    _internal_rate_limits = True

    def __init__(self, cfg: CameraConfig):
        super().__init__(cfg)
        self._cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> bool:
        index = int(self.cfg.webcam_index)
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            # Fall back to auto-detect backend (useful on non-Linux dev boxes).
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            log.warning("webcam index %d could not be opened; falling back to black frames", index)
            return False

        # Prefer MJPG for higher resolutions on USB bus; many wide-angle webcams
        # only hit their advertised fps under MJPG.
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.framerate)
        # Keep the driver buffer short so we always render the freshest frame.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Explicitly enable auto-exposure. The picamera tuning fields
        # (brightness/contrast/exposure_value/...) are not forwarded here
        # because their ranges are picamera-native ([-1, 1], EV stops) and do
        # not map to V4L2 property ranges (typically 0..255 or driver-specific).
        # Webcam-specific tuning should be done via `v4l2-ctl` or a dedicated
        # config field — not implemented yet; the camera's auto settings are
        # usually sane for indoor use.
        try:
            # 3 = auto (Linux UVC convention); 1 = manual.
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        except Exception:
            pass

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = cap.get(cv2.CAP_PROP_FPS)
        log.info(
            "webcam[%d] opened: requested %dx%d@%d, got %dx%d@%.1f, rotate_180=%s",
            index, self.cfg.width, self.cfg.height, self.cfg.framerate,
            aw, ah, afps, self.cfg.rotate_180,
        )
        self._cap = cap
        return True

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _capture_one(self, black: np.ndarray) -> np.ndarray:
        if self._cap is None:
            return black
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return black
        if self.cfg.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        # If the webcam delivered a different size than requested, resize so
        # the downstream pipeline (perspective transform etc.) sees the
        # dimensions it was calibrated for.
        h, w = frame.shape[:2]
        if w != self.cfg.width or h != self.cfg.height:
            frame = cv2.resize(frame, (self.cfg.width, self.cfg.height))
        return frame

