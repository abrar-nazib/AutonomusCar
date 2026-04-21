"""Local-iteration tool for the bird's-eye lane detector.

Runs the full pipeline on a raw frame pulled from the Pi and dumps every
intermediate stage into scripts/debug_out/ so you can eyeball what the
detector is seeing without redeploying.

Usage (from project root):

    ./scripts/pull_raw.sh                  # grab latest raw frame
    ./venv/bin/python scripts/local_dev.py # regenerate all debug outputs

Outputs under scripts/debug_out/:
  raw.jpg         — the frame pulled from the Pi
  persp_src.jpg   — raw frame with the source trapezoid drawn on it
  birds_eye.jpg   — perspective-warped bird's-eye image
  bird_binary.jpg — adaptive-threshold mask in bird's-eye
  bird_dilated.jpg— after dilation
  bird_windows.jpg— sliding windows visualised on bird's-eye
  annotated.jpg   — final HUD overlay on the original image
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from autocar.config import AppConfig, DEFAULT_CONFIG_PATH
from autocar.vision import LaneDetector, overlay
from autocar.vision import perspective as persp_mod


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    debug_dir = project_root / "scripts" / "debug_out"
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=debug_dir / "raw.jpg")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--target-lane", default=None)
    p.add_argument("--out-dir", type=Path, default=debug_dir)
    args = p.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    target_lane = args.target_lane or cfg.control.target_lane
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(args.input))
    if frame is None:
        raise SystemExit(f"could not read {args.input}")
    h, w = frame.shape[:2]
    print(f"loaded {args.input}  {w}x{h}")

    # --- run the detector ---------------------------------------------------
    detector = LaneDetector(cfg.vision)
    det = detector.detect(frame)
    persp = detector._persp  # exposed deliberately for the debug script

    print(
        f"found={det.found}  "
        f"L:pts={det.left_points}  C:pts={det.center_points}  R:pts={det.right_points}  "
        f"bird_size={det.bird_size}  car_center_x_bird={det.car_center_x_bird}"
    )
    for name, yr in [
        ("L", det.left_y_range_bird),
        ("C", det.center_y_range_bird),
        ("R", det.right_y_range_bird),
    ]:
        if yr is not None:
            print(f"  {name} y_range_bird={yr}")
    print(f"  lane_center_offset  L={det.lane_center_offset('L')}  R={det.lane_center_offset('R')}")

    # --- dump intermediates -------------------------------------------------
    persp_src = frame.copy()
    cv2.polylines(persp_src, [persp.src.astype(np.int32)], True, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / "persp_src.jpg"), persp_src)

    bird = persp.warp(frame)
    cv2.imwrite(str(args.out_dir / "birds_eye.jpg"), bird)

    gray = cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    if cfg.vision.clahe_enabled:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.vision.clahe_clip,
            tileGridSize=(cfg.vision.clahe_tile, cfg.vision.clahe_tile),
        )
        gray = clahe.apply(gray)

    _, sheet_raw = cv2.threshold(gray, cfg.vision.sheet_threshold, 255, cv2.THRESH_BINARY)
    ck = max(1, cfg.vision.sheet_close_kernel)
    sheet_raw = cv2.morphologyEx(
        sheet_raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    )
    nn, lbl, stt, _ = cv2.connectedComponentsWithStats(sheet_raw, connectivity=8)
    if nn > 1:
        biggest = 1 + int(np.argmax(stt[1:, cv2.CC_STAT_AREA]))
        sheet_mask = (lbl == biggest).astype(np.uint8) * 255
        ek = max(1, cfg.vision.sheet_erode_kernel)
        sheet_mask = cv2.erode(
            sheet_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        )
    else:
        sheet_mask = np.zeros_like(gray)
    cv2.imwrite(str(args.out_dir / "bird_sheet.jpg"), sheet_mask)

    block = cfg.vision.adaptive_block_size
    if block % 2 == 0:
        block += 1
    tape_raw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        block, cfg.vision.adaptive_c,
    )
    binary = cv2.bitwise_and(tape_raw, sheet_mask)
    cv2.imwrite(str(args.out_dir / "bird_binary.jpg"), binary)

    dk = max(1, cfg.vision.dilate_kernel)
    dil = binary
    if dk > 1:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
        dil = cv2.dilate(binary, kd)
    cv2.imwrite(str(args.out_dir / "bird_dilated.jpg"), dil)

    # --- connected components in bird's-eye binary -----------------------
    dkv = max(1, cfg.vision.dash_join_kernel)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dkv))
    joined = cv2.dilate(dil, vkernel)
    nccc, lbls, stts, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    bh, bw = joined.shape
    min_h = int(bh * cfg.vision.min_lane_height_frac_bird)
    min_aspect = cfg.vision.min_lane_aspect_bird
    max_width = int(bw * cfg.vision.max_lane_width_frac_bird)
    print(f"  bird CCs (min_h={min_h}, min_aspect={min_aspect}, max_w={max_width}): {nccc-1}")
    for i in range(1, nccc):
        cxb, cyb, cwb, chb, area = stts[i]
        ar = chb / max(1, cwb)
        keep = (chb >= min_h) and (cwb <= max_width) and (ar >= min_aspect)
        print(f"    #{i:2d} bbox=({cxb},{cyb},{cwb},{chb}) area={area} aspect={ar:.2f}"
              + (" KEEP" if keep else ""))

    # --- visualise sliding windows -----------------------------------------
    win_vis = cv2.cvtColor(dil, cv2.COLOR_GRAY2BGR)
    bh, bw = dil.shape
    histo = (dil[bh // 2:, :] > 0).sum(axis=0)
    starts = detector._pick_peaks(histo, bw)
    print(f"  starting peaks in bird's-eye: {starts}")
    win_h = max(1, bh // cfg.vision.nwindows)
    colors = [(60, 180, 255), (220, 120, 255), (255, 180, 60)]
    for ci, sx in enumerate(starts):
        color = colors[ci % 3]
        cur = int(sx)
        for wi in range(cfg.vision.nwindows):
            y_high = bh - wi * win_h
            y_low = max(0, y_high - win_h)
            x_low = max(0, cur - cfg.vision.window_margin)
            x_high = min(bw, cur + cfg.vision.window_margin)
            cv2.rectangle(win_vis, (x_low, y_low), (x_high, y_high), color, 1)
            sub = dil[y_low:y_high, x_low:x_high]
            ys_w, xs_w = np.where(sub > 0)
            if len(xs_w) >= cfg.vision.window_minpix:
                cur = int(np.mean(xs_w)) + x_low
    # Draw the car's trajectory column in bird's-eye as a cyan dashed line.
    for y in range(0, bh, 12):
        cv2.line(win_vis, (det.car_center_x_bird, y),
                 (det.car_center_x_bird, min(bh, y + 6)),
                 (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.out_dir / "bird_windows.jpg"), win_vis)

    # --- final annotated frame ----------------------------------------------
    annotated = overlay.draw(frame, det,
                             steering=0.0, fps=30.0,
                             target_lane=target_lane, perspective=persp)
    cv2.imwrite(str(args.out_dir / "annotated.jpg"), annotated)

    print(f"wrote {args.out_dir} — persp_src, birds_eye, bird_binary, bird_dilated, bird_windows, annotated")


if __name__ == "__main__":
    main()
