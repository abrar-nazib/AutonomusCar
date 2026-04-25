"""Classical lane detector modeled on the Udacity "Advanced Lane Finding" pipeline:

  1. Warp the forward road surface to a bird's-eye (top-down) view.
     Lanes become near-vertical, easy to search column-wise.
  2. Adaptive threshold (robust to shadow/distance) produces a tape mask.
  3. Histogram the bottom half of the warp → up to 3 starting x-peaks
     (left-edge, center-divider, right-edge).
  4. Slide a fixed-height window up from each peak, recentering on the
     detected pixels as we go. Windows with too few hits are skipped; after
     N consecutive misses the search stops.
  5. Quadratic polyfit on each lane's collected pixels (in bird's-eye coords,
     where lanes are close to vertical so x = f(y) is well-conditioned).
  6. Sample the fits densely, then unwarp the samples + the raw segmentation
     pixels back to image coords for overlay rendering.

The car's trajectory center is a calibrated column (`car_center_x_frac` of
frame width), *not* the geometric image center — that's what the PID offset
is measured against."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..config import VisionConfig
from . import perspective as persp


@dataclass
class LaneDetection:
    found: bool
    frame_size: Tuple[int, int]              # (w, h) in image coords
    # Bird's-eye sampled polylines, unwarped back to image coords.
    left_polyline: Optional[np.ndarray] = None
    center_polyline: Optional[np.ndarray] = None
    right_polyline: Optional[np.ndarray] = None
    # Segmentation pixels in image coords (for painting onto the frame).
    left_pts: Optional[np.ndarray] = None
    center_pts: Optional[np.ndarray] = None
    right_pts: Optional[np.ndarray] = None
    # Point counts (informational).
    left_points: int = 0
    center_points: int = 0
    right_points: int = 0
    # Bird's-eye-space context, kept for offset calculations.
    bird_size: Tuple[int, int] = (0, 0)      # (w, h) of warp
    car_center_x_bird: int = 0               # where the car-trajectory column lands in bird's-eye
    car_center_x: int = 0                    # same column in original image coords (for overlay)
    left_fit_bird: Optional[np.ndarray] = None
    center_fit_bird: Optional[np.ndarray] = None
    right_fit_bird: Optional[np.ndarray] = None
    left_y_range_bird: Optional[Tuple[int, int]] = None
    center_y_range_bird: Optional[Tuple[int, int]] = None
    right_y_range_bird: Optional[Tuple[int, int]] = None

    # --- API used by the app / overlay --------------------------------------

    def lane_center_offset(self, target_lane: str) -> Optional[float]:
        """Normalized [-1, 1] offset of the target lane's midline from the
        car's trajectory column, sampled at a *look-ahead* point along the
        fit (not at the car). Falls back gracefully:
          - (L, R) pair if the divider is missing
          - single-lane mode if only one edge is visible (uses an assumed
            lane half-width along the lane normal)
        Positive = lane midline is to the right of the car.
        """
        bw, bh = self.bird_size
        if bw == 0:
            return None
        lane = target_lane.upper()
        L = (self.left_fit_bird, self.left_y_range_bird)
        C = (self.center_fit_bird, self.center_y_range_bird)
        R = (self.right_fit_bird, self.right_y_range_bird)

        # Look-ahead sampling: 0 = at car (max y in bird's-eye), 1 = far end
        # of the fit. At the car, a curving lane has barely begun to bend, so
        # the offset reads "lane is right here" — the controller doesn't
        # commit to the upcoming curve. Sampling further up means the offset
        # already reflects where the lane is going.
        look = float(getattr(self, "_look_ahead_frac", 0.6))

        def y_eval_pair(yr_a, yr_b):
            y_hi = min(yr_a[1], yr_b[1])
            y_lo = max(yr_a[0], yr_b[0])
            if y_hi <= y_lo:
                return None
            return float(y_hi - look * (y_hi - y_lo))

        def y_eval_one(yr):
            return float(yr[1] - look * (yr[1] - yr[0]))

        def mid_from_pair(pa, pb):
            if pa[0] is None or pb[0] is None:
                return None
            ye = y_eval_pair(pa[1], pb[1])
            if ye is None:
                return None
            a = float(np.polyval(pa[0], ye))
            b = float(np.polyval(pb[0], ye))
            return 0.5 * (a + b)

        mid: Optional[float] = None
        if lane == "L":
            mid = mid_from_pair(L, C) or mid_from_pair(L, R)
        elif lane == "R":
            mid = mid_from_pair(C, R) or mid_from_pair(L, R)
        else:
            mid = mid_from_pair(L, R)

        # Single-lane fallback: only one edge visible (typical at sharp
        # curves). Step `qw` along the lane *normal* — perpendicular to the
        # tangent of x = f(y) at y_eval — not horizontally. The horizontal
        # component of that step is qw / sqrt(1 + f'(y)²); using qw raw
        # over-shoots on steep curves. Sign per lane: for Lane L the midline
        # is to the RIGHT of the left edge and to the LEFT of divider / right.
        if mid is None:
            qw = self._lane_half_width_bird()
            for pol, y_range, sign_for_L, sign_for_R in [
                (L[0], L[1], +1, +1),   # only L: both lanes are to the right
                (C[0], C[1], -1, +1),   # only C: Lane L to left, Lane R to right
                (R[0], R[1], -1, -1),   # only R: both lanes are to the left
            ]:
                if pol is None:
                    continue
                ye = y_eval_one(y_range)
                edge_x = float(np.polyval(pol, ye))
                slope = float(np.polyval(np.polyder(pol), ye))
                step_x = qw / float(np.sqrt(1.0 + slope * slope))
                sign = sign_for_L if lane == "L" else sign_for_R
                mid = edge_x + sign * step_x
                break

        if mid is None:
            return None
        half = bw * 0.5
        return (mid - self.car_center_x_bird) / max(1.0, half)

    def _lane_half_width_bird(self) -> float:
        """Half of one lane's width in bird's-eye pixels. Used by the
        single-lane offset fallback. Pulled from VisionConfig when this
        LaneDetection was produced — cached on the instance."""
        return float(getattr(self, "_lane_half_width_bird_px", 100.0))


class LaneDetector:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self._persp: Optional[persp.PerspectiveTransform] = None
        self._frame_size: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ main

    def detect(self, frame: np.ndarray) -> LaneDetection:
        h, w = frame.shape[:2]
        self._ensure_persp(w, h)

        empty = LaneDetection(found=False, frame_size=(w, h),
                              bird_size=self._persp.out_size,
                              car_center_x_bird=self._car_center_x_bird(w),
                              car_center_x=int(w * self.cfg.car_center_x_frac))
        if self._persp is None:
            return empty

        # 1. Warp to bird's-eye.
        bird = self._persp.warp(frame)

        gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        # CLAHE normalises local brightness so the sheet threshold and the
        # adaptive tape threshold both work under uneven lighting (half the
        # sheet in shadow, half lit). Without this, the shadow half of the
        # sheet falls below `sheet_threshold` and the sheet mask excludes
        # anything over there — which wipes out the outer tape detection.
        if self.cfg.clahe_enabled:
            clahe = cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip,
                tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
            )
            gray = clahe.apply(gray)

        # 2. Sheet mask: the track sheet is the largest bright region in
        #    bird's-eye. Everything off the sheet (floor/curtain/furniture
        #    that the trapezoid caught) gets excluded from tape detection.
        _, sheet = cv2.threshold(
            gray, self.cfg.sheet_threshold, 255, cv2.THRESH_BINARY
        )
        # Close pinholes (dashes, wrinkles) so the sheet is a single blob.
        closek = max(1, self.cfg.sheet_close_kernel)
        ckernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closek, closek))
        sheet = cv2.morphologyEx(sheet, cv2.MORPH_CLOSE, ckernel)
        # Pick the largest bright component.
        nn, lbl, stt, _ = cv2.connectedComponentsWithStats(sheet, connectivity=8)
        if nn > 1:
            biggest = 1 + int(np.argmax(stt[1:, cv2.CC_STAT_AREA]))
            sheet_mask = (lbl == biggest).astype(np.uint8) * 255
            # Erode slightly so the sheet's own boundary (where brightness
            # falls off) doesn't read as tape.
            ek = max(1, self.cfg.sheet_erode_kernel)
            ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
            sheet_mask = cv2.erode(sheet_mask, ekernel)
        else:
            sheet_mask = np.zeros_like(gray)

        # 3. Adaptive threshold for dark tape, then AND with the sheet mask.
        block = self.cfg.adaptive_block_size
        if block % 2 == 0:
            block += 1
        tape_raw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
            block, self.cfg.adaptive_c,
        )
        binary = cv2.bitwise_and(tape_raw, sheet_mask)

        # 4. Small reconnecting dilation.
        dk = max(1, self.cfg.dilate_kernel)
        if dk > 1:
            kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
            binary = cv2.dilate(binary, kd)

        # 5. Morphological open to kill salt-and-pepper noise but keep thin
        #    tape strokes intact.
        bh, bw = binary.shape
        ok = max(1, self.cfg.open_kernel)
        if ok > 1:
            ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ke)

        # 6. Find every connected component; cluster them by x-centroid. A
        #    dashed lane shows up as many little components, but they all
        #    sit at the same x, so they collapse into one cluster. Solid
        #    tape is already one component and forms its own cluster.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 1:
            return empty

        pieces: List[dict] = []
        for i in range(1, num):
            _, _, _, _, area = stats[i]
            if area < self.cfg.piece_min_area:
                continue
            comp_mask = (labels == i)
            ys_c, xs_c = np.where(comp_mask)
            pieces.append({
                "xc": float(np.mean(xs_c)),
                "xs": xs_c,
                "ys": ys_c,
            })
        if not pieces:
            return empty

        pieces.sort(key=lambda p: p["xc"])
        clusters: List[List[dict]] = []
        gap = self.cfg.cluster_gap_px_bird
        for p in pieces:
            if not clusters or (p["xc"] - clusters[-1][-1]["xc"]) > gap:
                clusters.append([p])
            else:
                clusters[-1].append(p)

        # At a sharp curve the tape forms a C-shape in bird's-eye and
        # x = f(y) is multi-valued up top. Fit only the NEAR-CAR portion
        # (bottom fraction of bird's-eye) where the tape is still near-
        # vertical even when the track curves hard ahead. The full-set
        # pixels are kept separately so the overlay can paint the entire
        # segmented tape (including the curved part above the horizon).
        near_y_min = int(bh * (1.0 - self.cfg.near_car_fit_frac))

        # Each lane candidate carries: (start_x, xs_fit, ys_fit, xs_all, ys_all).
        candidates: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for cl in clusters:
            xs_all = np.concatenate([p["xs"] for p in cl])
            ys_all = np.concatenate([p["ys"] for p in cl])
            if len(xs_all) < self.cfg.min_lane_points:
                continue
            y_span = int(ys_all.max() - ys_all.min())
            if y_span < int(bh * self.cfg.min_lane_height_frac_bird):
                continue
            near = ys_all >= near_y_min
            if int(np.sum(near)) >= self.cfg.min_lane_points // 2:
                xs_fit = xs_all[near]
                ys_fit = ys_all[near]
            else:
                xs_fit = xs_all
                ys_fit = ys_all
            y_bot = int(ys_all.max())
            band = ys_all >= (y_bot - max(20, y_span // 5))
            start_x = int(np.mean(xs_all[band])) if np.any(band) else int(np.mean(xs_all))
            candidates.append((start_x, xs_fit, ys_fit, xs_all, ys_all))

        if not candidates:
            return empty

        candidates.sort(key=lambda t: len(t[3]), reverse=True)  # sort by full-pixel count
        candidates = candidates[:3]
        candidates.sort(key=lambda t: t[0])
        tracked = candidates

        # Fit the near-car slice, but sample + draw across the full lane
        # y-range so the polyline stretches along the segmented tape.
        polylines: List[Optional[np.ndarray]] = []
        segs: List[Optional[np.ndarray]] = []
        fits_bird: List[Optional[np.ndarray]] = []
        y_ranges_bird: List[Optional[Tuple[int, int]]] = []
        counts: List[int] = []

        deg = max(1, self.cfg.fit_degree)
        for _, xs_fit, ys_fit, xs_all, ys_all in tracked:
            if len(np.unique(ys_fit)) < deg + 1:
                polylines.append(None); segs.append(None)
                fits_bird.append(None); y_ranges_bird.append(None); counts.append(0)
                continue
            fit = np.polyfit(ys_fit, xs_fit, deg)
            # Polyline drawn over the FIT y-range (where the curve is valid).
            yf0 = int(ys_fit.min()); yf1 = int(ys_fit.max())
            y_samples = np.linspace(yf0, yf1, 48).astype(np.int32)
            x_samples = np.clip(np.polyval(fit, y_samples), 0, bw - 1).astype(np.int32)
            bird_poly = np.column_stack([x_samples, y_samples])
            img_poly = self._persp.unwarp_points(bird_poly).astype(np.int32)
            # Segmentation uses the FULL tape pixels so the curved-top arc is
            # still visible even though the fit only covers the near portion.
            bird_seg = np.column_stack([xs_all, ys_all]).astype(np.int32)
            img_seg = self._persp.unwarp_points(bird_seg).astype(np.int32)
            polylines.append(img_poly)
            segs.append(img_seg)
            fits_bird.append(fit)
            y_ranges_bird.append((yf0, yf1))
            counts.append(int(len(xs_all)))

        # 6. Slot into L/C/R.
        left_poly = center_poly = right_poly = None
        left_seg = center_seg = right_seg = None
        left_fit = center_fit = right_fit = None
        left_y = center_y = right_y = None
        lp = cp = rp = 0

        n = sum(1 for f in fits_bird if f is not None)
        if n == 1:
            i = next(k for k, f in enumerate(fits_bird) if f is not None)
            sx = tracked[i][0]
            if sx < bw / 2:
                left_poly, left_seg, left_fit, left_y, lp = (
                    polylines[i], segs[i], fits_bird[i], y_ranges_bird[i], counts[i]
                )
            else:
                right_poly, right_seg, right_fit, right_y, rp = (
                    polylines[i], segs[i], fits_bird[i], y_ranges_bird[i], counts[i]
                )
        elif n == 2:
            idxs = [k for k, f in enumerate(fits_bird) if f is not None]
            left_poly, left_seg, left_fit, left_y, lp = (
                polylines[idxs[0]], segs[idxs[0]], fits_bird[idxs[0]],
                y_ranges_bird[idxs[0]], counts[idxs[0]]
            )
            right_poly, right_seg, right_fit, right_y, rp = (
                polylines[idxs[1]], segs[idxs[1]], fits_bird[idxs[1]],
                y_ranges_bird[idxs[1]], counts[idxs[1]]
            )
        else:
            left_poly, left_seg, left_fit, left_y, lp = (
                polylines[0], segs[0], fits_bird[0], y_ranges_bird[0], counts[0]
            )
            center_poly, center_seg, center_fit, center_y, cp = (
                polylines[1], segs[1], fits_bird[1], y_ranges_bird[1], counts[1]
            )
            right_poly, right_seg, right_fit, right_y, rp = (
                polylines[2], segs[2], fits_bird[2], y_ranges_bird[2], counts[2]
            )

        # "found" = at least one lane locked. The app can still steer from a
        # single lane via the fallback in lane_center_offset().
        any_lane = (left_fit is not None) or (center_fit is not None) or (right_fit is not None)
        lane_half_px = float(self.cfg.lane_half_width_bird)
        det_out = LaneDetection(
            found=any_lane,
            frame_size=(w, h),
            left_polyline=left_poly, center_polyline=center_poly, right_polyline=right_poly,
            left_pts=left_seg, center_pts=center_seg, right_pts=right_seg,
            left_points=lp, center_points=cp, right_points=rp,
            bird_size=self._persp.out_size,
            car_center_x_bird=self._car_center_x_bird(w),
            car_center_x=int(w * self.cfg.car_center_x_frac),
            left_fit_bird=left_fit, center_fit_bird=center_fit, right_fit_bird=right_fit,
            left_y_range_bird=left_y, center_y_range_bird=center_y, right_y_range_bird=right_y,
        )
        # Stash for the offset sampling in LaneDetection.lane_center_offset.
        det_out._lane_half_width_bird_px = lane_half_px
        det_out._look_ahead_frac = float(self.cfg.look_ahead_frac)
        return det_out

    # ------------------------------------------------------------------ helpers

    def _ensure_persp(self, w: int, h: int) -> None:
        if self._persp is not None and self._frame_size == (w, h):
            return
        self._persp = persp.build_from_config(self.cfg, w, h)
        self._frame_size = (w, h)

    def _car_center_x_bird(self, frame_w: int) -> int:
        """Map the car's calibrated trajectory column (in image space) forward
        into bird's-eye x at the very bottom of the warp."""
        if self._persp is None:
            return frame_w // 2
        car_x_img = int(frame_w * self.cfg.car_center_x_frac)
        bottom_y_img = int(self._frame_size[1] - 1) if self._frame_size else 479
        pts = np.array([[car_x_img, bottom_y_img]], dtype=np.float32)
        warped = self._persp.warp_points(pts)
        return int(warped[0, 0])

    def _pick_peaks(self, hist: np.ndarray, frame_w: int) -> List[int]:
        k = 9
        kernel = np.ones(k, dtype=np.float32) / k
        smooth = np.convolve(hist, kernel, mode="same")
        # Mask bird's-eye horizontal edges — warped curtain / chassis / floor
        # often produces long vertical blobs there that out-vote the real tape.
        edge = max(1, int(frame_w * self.cfg.peak_edge_mask_frac))
        smooth[:edge] = 0
        smooth[-edge:] = 0
        min_count = max(self.cfg.window_minpix, 10)
        min_sep = self.cfg.min_peak_separation

        order = np.argsort(smooth)[::-1]
        chosen: List[int] = []
        for i in order:
            if smooth[i] < min_count:
                break
            if all(abs(int(i) - p) >= min_sep for p in chosen):
                chosen.append(int(i))
            if len(chosen) >= 3:
                break
        return sorted(chosen)

    def _track(
        self, binary: np.ndarray, start_x: int, win_h: int, margin: int, minpix: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        bh, bw = binary.shape
        current_x = int(start_x)
        pxs: List[np.ndarray] = []
        pys: List[np.ndarray] = []
        misses = 0
        max_misses = 4

        for w_idx in range(self.cfg.nwindows):
            y_high = bh - w_idx * win_h
            y_low = max(0, y_high - win_h)
            if y_high <= y_low:
                break
            x_low = max(0, current_x - margin)
            x_high = min(bw, current_x + margin)

            sub = binary[y_low:y_high, x_low:x_high]
            ys_w, xs_w = np.where(sub > 0)
            if len(xs_w) >= minpix:
                current_x = int(np.mean(xs_w)) + x_low
                pxs.append(xs_w + x_low)
                pys.append(ys_w + y_low)
                misses = 0
            else:
                misses += 1
                if misses >= max_misses:
                    break

        if not pxs:
            return None, None
        return np.concatenate(pxs), np.concatenate(pys)
