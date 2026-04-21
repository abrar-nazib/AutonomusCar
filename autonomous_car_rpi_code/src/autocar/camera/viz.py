from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class AnnotatedFrameProvider:
    """Duck-types as FrameSource for the MJPEG streamer. The control loop pushes
    annotated frames via `set_annotated`; the streamer always calls `get_frame`
    and gets either the latest annotated frame (preferred) or a raw frame from
    the wrapped source as fallback."""

    def __init__(self, source):
        self._source = source
        self._lock = threading.Lock()
        self._annotated: Optional[np.ndarray] = None

    def set_annotated(self, frame: Optional[np.ndarray]) -> None:
        with self._lock:
            self._annotated = frame

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._annotated is not None:
                return self._annotated.copy()
        return self._source.get_frame()

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Always return a raw (un-annotated) frame from the wrapped source."""
        return self._source.get_frame()
