"""AV-style HUD overlay for the new (polyline-based) LaneDetection.

Draws:
  - the ROI trapezoid (source of the bird's-eye warp)
  - each lane's segmented tape pixels in its lane color
  - each lane's unwarped polyline with an L / C / R tag
  - a translucent drivable region between the pair of lanes that bound the
    *target* lane, plus a dashed yellow centerline inside that lane
  - a status panel (lane, offset, steer, detection dots)
  - a car-trajectory crosshair at the calibrated car-center column
  - a heading arrow that swings with the PID steering output

All colors are BGR."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .lane_detector import LaneDetection

# Palette (BGR)
LEFT_COLOR = (60, 180, 255)
CENTER_COLOR = (220, 120, 255)
RIGHT_COLOR = (255, 180, 60)
DRIVABLE_COLOR = (60, 210, 80)
TARGET_CENTER_COLOR = (0, 255, 255)
TRAJECTORY_COLOR = (80, 255, 160)
HUD_OK = (80, 255, 160)
HUD_WARN = (80, 150, 255)
HUD_TEXT = (235, 235, 235)
HUD_DIM = (160, 160, 160)


def draw(
    frame: np.ndarray,
    det: LaneDetection,
    steering: Optional[float] = None,
    fps: Optional[float] = None,
    target_lane: str = "R",
    perspective=None,
    lidar_scan: Optional[np.ndarray] = None,
    lidar_max_range_mm: float = 6000.0,
    lidar_inset_size_px: int = 220,
    lidar_inset_margin_px: int = 12,
) -> np.ndarray:
    out = frame.copy()
    if perspective is not None:
        _draw_perspective_src(out, perspective)
    # Draw whatever is present — segmentation and polylines are per-lane, so
    # a single-lane lock is still useful to see. Drivable region needs two
    # edges; it skips itself internally when that pair isn't available.
    _draw_segmentation(out, det)
    if det.found:
        _draw_drivable(out, det, target_lane, perspective)
    _draw_polylines(out, det)

    _draw_trajectory_marker(out, det)
    _draw_hud_panel(out, det, steering=steering, fps=fps, target_lane=target_lane)
    _draw_heading(out, steering, det.car_center_x)
    if lidar_scan is not None and len(lidar_scan) > 0:
        _draw_lidar_inset(
            out, lidar_scan,
            max_range_mm=lidar_max_range_mm,
            size_px=lidar_inset_size_px,
            margin_px=lidar_inset_margin_px,
        )
    return out


# --- perspective source trapezoid -----------------------------------------

def _draw_perspective_src(out: np.ndarray, perspective) -> None:
    pts = perspective.src.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, HUD_OK, 1, cv2.LINE_AA)


# --- segmentation ----------------------------------------------------------

def _paint_pts(out: np.ndarray, pts: Optional[np.ndarray], color: tuple) -> None:
    if pts is None or len(pts) == 0:
        return
    h, w = out.shape[:2]
    xs = pts[:, 0]
    ys = pts[:, 1]
    m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[m], ys[m]
    out[ys, xs] = color
    xs2 = np.clip(xs + 1, 0, w - 1)
    out[ys, xs2] = color


def _draw_segmentation(out: np.ndarray, det: LaneDetection) -> None:
    _paint_pts(out, det.left_pts, LEFT_COLOR)
    _paint_pts(out, det.center_pts, CENTER_COLOR)
    _paint_pts(out, det.right_pts, RIGHT_COLOR)


# --- polylines + labels ---------------------------------------------------

def _draw_polylines(out: np.ndarray, det: LaneDetection) -> None:
    _draw_one(out, det.left_polyline, LEFT_COLOR, "L")
    _draw_one(out, det.center_polyline, CENTER_COLOR, "C")
    _draw_one(out, det.right_polyline, RIGHT_COLOR, "R")


def _draw_one(out: np.ndarray, poly: Optional[np.ndarray], color: tuple, label: str) -> None:
    if poly is None or len(poly) < 2:
        return
    h, w = out.shape[:2]
    clip = poly.copy().astype(np.int32)
    clip[:, 0] = np.clip(clip[:, 0], 0, w - 1)
    clip[:, 1] = np.clip(clip[:, 1], 0, h - 1)
    cv2.polylines(out, [clip], False, color, 2, cv2.LINE_AA)
    # Place the label at the bird's-eye-top end of the polyline, which after
    # unwarp is the *image-top* end (smallest y).
    anchor = clip[int(np.argmin(clip[:, 1]))]
    _draw_label(out, (int(anchor[0]), int(anchor[1])), label, color)


def _draw_label(out: np.ndarray, anchor: tuple, text: str, color: tuple) -> None:
    x, y = anchor
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    bx0 = max(0, x - tw // 2 - 5)
    bx1 = bx0 + tw + 10
    by1 = max(th + 6, y - 4)
    by0 = by1 - th - 8
    cv2.rectangle(out, (bx0, by0), (bx1, by1), (20, 20, 20), -1)
    cv2.rectangle(out, (bx0, by0), (bx1, by1), color, 1, cv2.LINE_AA)
    cv2.putText(out, text, (bx0 + 5, by1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# --- drivable region + target centerline -----------------------------------

def _pick_target(det: LaneDetection, target_lane: str):
    lane = target_lane.upper()
    if lane == "L":
        a = (det.left_polyline, det.left_fit_bird, det.left_y_range_bird)
        b = (det.center_polyline, det.center_fit_bird, det.center_y_range_bird)
        if b[0] is None:
            b = (det.right_polyline, det.right_fit_bird, det.right_y_range_bird)
    else:
        a = (det.center_polyline, det.center_fit_bird, det.center_y_range_bird)
        if a[0] is None:
            a = (det.left_polyline, det.left_fit_bird, det.left_y_range_bird)
        b = (det.right_polyline, det.right_fit_bird, det.right_y_range_bird)
    return a, b


def _draw_drivable(out: np.ndarray, det: LaneDetection, target_lane: str, perspective) -> None:
    a, b = _pick_target(det, target_lane)
    if a[0] is None or b[0] is None:
        return

    # Build the drivable band in bird's-eye, then unwarp so it lies on the road
    # in the original image (curved, not a flat trapezoid).
    if perspective is not None and a[1] is not None and b[1] is not None:
        y_lo = max(a[2][0], b[2][0])
        y_hi = min(a[2][1], b[2][1])
        if y_hi <= y_lo:
            return
        ys = np.linspace(y_lo, y_hi, 48).astype(np.int32)
        ax = np.polyval(a[1], ys)
        bx = np.polyval(b[1], ys)
        bird_poly = np.concatenate([
            np.column_stack([ax, ys]),
            np.column_stack([bx[::-1], ys[::-1]]),
        ]).astype(np.float32)
        img_poly = perspective.unwarp_points(bird_poly).astype(np.int32)
        layer = out.copy()
        cv2.fillPoly(layer, [img_poly], DRIVABLE_COLOR)
        cv2.addWeighted(layer, 0.22, out, 0.78, 0, out)
        # Centerline inside the target lane, dashed.
        cx = ((ax + bx) / 2.0).astype(np.float32)
        bird_center = np.column_stack([cx, ys])
        img_center = perspective.unwarp_points(bird_center).astype(np.int32)
        for i in range(0, len(img_center) - 1, 2):
            cv2.line(out, tuple(img_center[i]), tuple(img_center[i + 1]),
                     TARGET_CENTER_COLOR, 2, cv2.LINE_AA)


# --- car trajectory marker -------------------------------------------------

def _draw_trajectory_marker(out: np.ndarray, det: LaneDetection) -> None:
    """Dashed vertical marker at the car's calibrated trajectory column.
    This is the reference point the PID offset is measured against — it
    moves off the geometric image center whenever `car_center_x_frac` is
    non-0.5 (the camera is not perfectly aligned with the chassis)."""
    h, w = out.shape[:2]
    cx = det.car_center_x if det.car_center_x else w // 2
    cx = int(max(0, min(w - 1, cx)))
    for y0 in range(h - 60, h - 12, 6):
        cv2.line(out, (cx, y0), (cx, y0 + 3), TRAJECTORY_COLOR, 1, cv2.LINE_AA)


# --- HUD panel -------------------------------------------------------------

def _draw_hud_panel(
    out: np.ndarray,
    det: LaneDetection,
    steering: Optional[float],
    fps: Optional[float],
    target_lane: str,
) -> None:
    x0, y0 = 10, 10
    x1, y1 = 230, 138

    roi = out[y0:y1, x0:x1]
    blk = np.zeros_like(roi)
    cv2.addWeighted(blk, 0.55, roi, 0.45, 0, roi)

    border = HUD_OK if det.found else HUD_WARN
    cv2.rectangle(out, (x0, y0), (x1, y1), border, 1, cv2.LINE_AA)

    status = "LANES LOCKED" if det.found else "SEARCHING"
    cv2.putText(out, status, (x0 + 10, y0 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, border, 1, cv2.LINE_AA)

    cv2.putText(out, f"LANE  {target_lane.upper()}", (x0 + 10, y0 + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, TARGET_CENTER_COLOR, 1, cv2.LINE_AA)

    off = det.lane_center_offset(target_lane)
    off_txt = f"OFFSET {off:+.2f}" if off is not None else "OFFSET   --"
    cv2.putText(out, off_txt, (x0 + 10, y0 + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, HUD_TEXT, 1, cv2.LINE_AA)

    s_txt = f"STEER  {steering:+.2f}" if steering is not None else "STEER   --"
    cv2.putText(out, s_txt, (x0 + 10, y0 + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, HUD_TEXT, 1, cv2.LINE_AA)

    base_x, base_y = x0 + 10, y0 + 112
    _feature_dot(out, (base_x + 0, base_y), "L", LEFT_COLOR, det.left_fit_bird is not None)
    _feature_dot(out, (base_x + 60, base_y), "C", CENTER_COLOR, det.center_fit_bird is not None)
    _feature_dot(out, (base_x + 120, base_y), "R", RIGHT_COLOR, det.right_fit_bird is not None)

    if fps is not None:
        cv2.putText(out, f"{fps:4.1f} FPS", (out.shape[1] - 90, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, HUD_TEXT, 1, cv2.LINE_AA)


def _feature_dot(out: np.ndarray, pos: tuple, label: str, color: tuple, on: bool) -> None:
    x, y = pos
    if on:
        cv2.circle(out, (x, y), 6, color, -1, cv2.LINE_AA)
    cv2.circle(out, (x, y), 6, color, 1, cv2.LINE_AA)
    cv2.putText(out, label, (x + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                color if on else HUD_DIM, 1, cv2.LINE_AA)


# --- heading arrow --------------------------------------------------------

# --- LIDAR radar inset -----------------------------------------------------

LIDAR_BG_COLOR    = (15, 15, 15)
LIDAR_BORDER      = (90, 200, 255)
LIDAR_RING        = (50, 80, 90)
LIDAR_FORWARD     = (60, 180, 255)
LIDAR_POINT_NEAR  = (80, 255, 180)
LIDAR_POINT_FAR   = (180, 130, 80)


def _draw_lidar_inset(
    out: np.ndarray,
    scan: np.ndarray,           # Nx2 (angle_deg, distance_mm)
    max_range_mm: float,
    size_px: int,
    margin_px: int,
) -> None:
    """Top-down 2-D radar of the latest lidar revolution. Cheap: cached
    background blit + vectorized point projection (numpy), bottom-right."""
    h, w = out.shape[:2]
    diam = int(size_px)
    r = diam // 2
    cx = w - margin_px - r
    cy = h - margin_px - r

    # Translucent background panel.
    x0, y0 = cx - r - 4, cy - r - 4
    x1, y1 = cx + r + 4, cy + r + 4
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return
    roi = out[y0:y1, x0:x1]
    blk = np.zeros_like(roi)
    cv2.addWeighted(blk, 0.55, roi, 0.45, 0, roi)

    # Range rings (1/3, 2/3, full).
    for k in (1, 2, 3):
        cv2.circle(out, (cx, cy), int(r * k / 3), LIDAR_RING, 1, cv2.LINE_AA)
    # Crosshair + forward axis (lidar 0° = forward by default; user's mount
    # may differ — this is just a reference axis).
    cv2.line(out, (cx, cy - r), (cx, cy + r), LIDAR_RING, 1, cv2.LINE_AA)
    cv2.line(out, (cx - r, cy), (cx + r, cy), LIDAR_RING, 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy), (cx, cy - r + 4), LIDAR_FORWARD, 2, cv2.LINE_AA)

    # Vectorized projection: angle_deg → image coords. Lidar 0° points up,
    # angle increases clockwise (top-down view). dx = sin(a), dy = -cos(a).
    angles = np.deg2rad(scan[:, 0].astype(np.float32))
    dists  = scan[:, 1].astype(np.float32)
    mask   = (dists > 0) & (dists <= float(max_range_mm))
    if not np.any(mask):
        cv2.circle(out, (cx, cy), r, LIDAR_BORDER, 1, cv2.LINE_AA)
        return
    angles = angles[mask]
    dists  = dists[mask]
    norm   = dists / float(max_range_mm)
    xs = (cx + norm * r * np.sin(angles)).astype(np.int32)
    ys = (cy - norm * r * np.cos(angles)).astype(np.int32)

    # Color by range (near=greenish, far=blueish). Single 2-color ramp.
    blend = norm[:, None]                      # (N,1)
    near = np.array(LIDAR_POINT_NEAR, dtype=np.float32)
    far  = np.array(LIDAR_POINT_FAR,  dtype=np.float32)
    cols = (1.0 - blend) * near + blend * far
    cols = cols.astype(np.uint8)

    # Direct pixel write — clip to inset bounding box first.
    in_box = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    xs = xs[in_box]; ys = ys[in_box]; cols = cols[in_box]
    out[ys, xs] = cols
    # 1-px halo so points read clearly against the dark panel.
    out[np.clip(ys + 1, 0, h - 1), xs] = cols
    out[ys, np.clip(xs + 1, 0, w - 1)] = cols

    # Border + count label.
    cv2.circle(out, (cx, cy), r, LIDAR_BORDER, 1, cv2.LINE_AA)
    cv2.putText(out, f"LIDAR {len(scan)}", (x0 + 6, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, LIDAR_BORDER, 1, cv2.LINE_AA)


def _draw_heading(out: np.ndarray, steering: Optional[float], car_center_x: int) -> None:
    h, w = out.shape[:2]
    cx = int(max(0, min(w - 1, car_center_x if car_center_x else w // 2)))
    base_y = h - 28
    tip_y = h - 72
    s = 0.0 if steering is None else float(np.clip(steering, -1.0, 1.0))
    tip_x = cx + int(s * 110)
    cv2.arrowedLine(out, (cx, base_y), (tip_x, tip_y),
                    TARGET_CENTER_COLOR, 3, cv2.LINE_AA, tipLength=0.35)
    cv2.circle(out, (cx, base_y), 5, TARGET_CENTER_COLOR, -1, cv2.LINE_AA)
    cv2.circle(out, (cx, base_y), 9, TARGET_CENTER_COLOR, 1, cv2.LINE_AA)
