from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import VisionConfig


@dataclass
class TrackDetection:
    found: bool
    center_x: int          # absolute pixel x of the track centroid
    frame_width: int
    offset: float          # normalized [-1, 1]; 0 is centered, + is right

    @property
    def centered(self) -> bool:
        return self.found and abs(self.offset) < 0.05


class TrackDetector:
    """Finds the track centroid in the lower ROI of the frame using a simple
    threshold + largest-contour heuristic. Good starting point for a dark line
    on a light surface; tune `binarize_threshold` and invert if needed."""

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg

    def detect(self, frame: np.ndarray) -> TrackDetection:
        h, w = frame.shape[:2]
        y0 = int(h * self.cfg.roi_top)
        y1 = int(h * self.cfg.roi_bottom)
        roi = frame[y0:y1, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            gray, self.cfg.binarize_threshold, 255, cv2.THRESH_BINARY_INV
        )

        contour = self._largest_contour(binary)
        if contour is None:
            return TrackDetection(found=False, center_x=w // 2, frame_width=w, offset=0.0)

        M = cv2.moments(contour)
        if M["m00"] == 0:
            return TrackDetection(found=False, center_x=w // 2, frame_width=w, offset=0.0)

        cx = int(M["m10"] / M["m00"])
        offset = (cx - w / 2.0) / (w / 2.0)
        return TrackDetection(found=True, center_x=cx, frame_width=w, offset=offset)

    def _largest_contour(self, binary: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= self.cfg.min_contour_area]
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
