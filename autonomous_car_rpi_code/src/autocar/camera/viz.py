from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class AnnotatedFrameProvider:
    """Duck-types as FrameSource for the MJPEG streamer. The control loop pushes
    annotated frames via `set_annotated`; viewers can either poll via
    `get_frame()` (snapshot endpoints) or block on `wait_next(last_id)` to be
    woken exactly when a fresh annotated frame is available — eliminating the
    encode-the-same-frame-twice spin in the streamer."""

    def __init__(self, source):
        self._source = source
        self._cv = threading.Condition()
        self._annotated: Optional[np.ndarray] = None
        self._frame_id: int = 0

    def set_annotated(self, frame: Optional[np.ndarray]) -> None:
        with self._cv:
            self._annotated = frame
            self._frame_id += 1
            self._cv.notify_all()

    def wait_next(self, after_id: int, timeout: float = 1.0):
        """Block until a frame with id > `after_id` is available (or timeout).
        Returns (frame_id, frame_copy). On timeout returns (after_id, None)."""
        with self._cv:
            self._cv.wait_for(lambda: self._frame_id > after_id, timeout=timeout)
            if self._frame_id <= after_id or self._annotated is None:
                return after_id, None
            return self._frame_id, self._annotated.copy()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._cv:
            if self._annotated is not None:
                return self._annotated.copy()
        return self._source.get_frame()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Always return a raw (un-annotated) frame from the wrapped source."""
        return self._source.get_frame()
