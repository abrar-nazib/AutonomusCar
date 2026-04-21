"""Bird's-eye-view perspective transform.

Classical lane-detection preprocessing: warp the road-surface trapezoid
(seen in perspective) into a rectangle (seen from directly above), so lanes
become near-vertical and can be tracked with sliding windows + polynomial
fits. After fitting, sample points are transformed back to image
coordinates with the inverse matrix for visualisation.

References: Udacity Self-Driving Car Nanodegree "Advanced Lane Finding",
OpenCV `getPerspectiveTransform` / `warpPerspective` tutorials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PerspectiveTransform:
    src: np.ndarray                    # 4 source pts in image coords (TL, TR, BR, BL)
    dst: np.ndarray                    # 4 destination pts in bird's-eye coords
    M: np.ndarray                      # 3x3 forward warp matrix
    Minv: np.ndarray                   # 3x3 inverse warp matrix
    out_size: Tuple[int, int]          # (w, h) of the bird's-eye canvas

    @classmethod
    def build(cls, src_pts, dst_pts, out_size: Tuple[int, int]) -> "PerspectiveTransform":
        src = np.asarray(src_pts, dtype=np.float32)
        dst = np.asarray(dst_pts, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return cls(src=src, dst=dst, M=M, Minv=Minv, out_size=out_size)

    def warp(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, self.out_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
        return cv2.warpPerspective(img, self.Minv, frame_size, flags=cv2.INTER_LINEAR)

    def unwarp_points(self, pts: np.ndarray) -> np.ndarray:
        """pts: Nx2 (x, y) in bird's-eye coords. Returns Nx2 in image coords."""
        if pts.size == 0:
            return pts.astype(np.int32).reshape(-1, 2)
        a = pts.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(a, self.Minv)
        return out.reshape(-1, 2)

    def warp_points(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts.astype(np.int32).reshape(-1, 2)
        a = pts.astype(np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(a, self.M)
        return out.reshape(-1, 2)


def build_from_config(cfg, frame_w: int, frame_h: int) -> PerspectiveTransform:
    """Build a PerspectiveTransform from a VisionConfig.

    Supports four independent corner fractions so the trapezoid can hug an
    asymmetric / curving track. When the per-corner fracs are missing (old
    config), falls back to the legacy symmetric `persp_top_width_frac` /
    `persp_bottom_width_frac` pair.

    The dst rectangle uses the full bird's-eye canvas at frame_size with a
    side margin so there's room for lane curvature."""
    top_y = int(frame_h * cfg.persp_top_y_frac)
    bot_y = int(frame_h * cfg.persp_bottom_y_frac)

    has_corners = (
        getattr(cfg, "persp_top_left_frac", None) is not None
        and getattr(cfg, "persp_top_right_frac", None) is not None
        and getattr(cfg, "persp_bottom_left_frac", None) is not None
        and getattr(cfg, "persp_bottom_right_frac", None) is not None
    )
    if has_corners:
        tl_x = int(frame_w * cfg.persp_top_left_frac)
        tr_x = int(frame_w * cfg.persp_top_right_frac)
        bl_x = int(frame_w * cfg.persp_bottom_left_frac)
        br_x = int(frame_w * cfg.persp_bottom_right_frac)
    else:
        cx = frame_w // 2
        top_half = int(frame_w * cfg.persp_top_width_frac / 2)
        bot_half = int(frame_w * cfg.persp_bottom_width_frac / 2)
        tl_x, tr_x = cx - top_half, cx + top_half
        bl_x, br_x = cx - bot_half, cx + bot_half

    src = [
        (tl_x, top_y),   # TL
        (tr_x, top_y),   # TR
        (br_x, bot_y),   # BR
        (bl_x, bot_y),   # BL
    ]
    margin = int(frame_w * cfg.persp_dst_margin_frac)
    dst = [
        (margin,              0),
        (frame_w - margin,    0),
        (frame_w - margin,    frame_h),
        (margin,              frame_h),
    ]
    return PerspectiveTransform.build(src, dst, (frame_w, frame_h))
