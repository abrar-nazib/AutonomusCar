"""Live calibration tool.

Pulls the Pi's raw frame every ~100 ms, runs the full autocar detection
pipeline on your dev machine with the current slider values, and shows the
whole pipeline (raw + source trapezoid / bird's-eye / sheet mask / tape
binary / windows / annotated) in a single grid window. Tweak the sliders in
the "Controls" window and the preview updates immediately.

Usage (from project root, with autocar venv active):

    ./venv/bin/python scripts/calibrate.py
    # optional:
    ./venv/bin/python scripts/calibrate.py --url http://raspberrypi.local:8000
    ./venv/bin/python scripts/calibrate.py --input scripts/debug_out/raw.jpg   # offline, one file

Keys:
    s     dump the current yaml-ready config values to stdout AND write
          scripts/debug_out/calibrated.yaml
    r     reload config from disk (revert sliders to saved yaml)
    p     pause / resume frame fetching
    ESC   quit

Sliders are scaled so opencv's int-only trackbar can represent floats:
    <name>*100  → divide by 100
    odd kernels → clamped to odd below in code

Only the knobs that users tune after a lighting / camera change are exposed;
perspective corners etc. go through the yaml (edit + reload with `r`)."""

from __future__ import annotations

import argparse
import dataclasses
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from autocar.config import AppConfig, DEFAULT_CONFIG_PATH
from autocar.vision import LaneDetector, overlay
from autocar.vision import perspective as persp_mod


CONTROLS = "Controls"
PIPELINE = "Pipeline"
ANNOTATED = "Annotated"


class Slider:
    """One int-backed OpenCV slider with a scale factor so it can carry
    fractional / odd-only values without the caller caring."""

    __slots__ = ("name", "field", "max_val", "scale", "odd_only", "clamp_min")

    def __init__(self, name: str, field: str, max_val: int, scale: float = 1.0,
                 odd_only: bool = False, clamp_min: Optional[int] = None):
        self.name = name
        self.field = field
        self.max_val = max_val
        self.scale = scale
        self.odd_only = odd_only
        self.clamp_min = clamp_min

    def read(self, cfg) -> float:
        """Pull the current slider value back into a scaled, clamped number."""
        v = cv2.getTrackbarPos(self.name, CONTROLS)
        if self.odd_only and v % 2 == 0:
            v = max(1, v + 1)
        if self.clamp_min is not None and v < self.clamp_min:
            v = self.clamp_min
        return v * self.scale


# Which VisionConfig / ControlConfig knobs get a slider.
# (slider name, dotted cfg path, max, scale, odd_only, clamp_min)
SLIDER_SPEC = [
    # --- Thresholds / masks ---
    Slider("sheet_threshold",   "vision.sheet_threshold",  255, 1.0),
    Slider("adaptive_c",        "vision.adaptive_c",        60, 1.0),
    Slider("adaptive_block",    "vision.adaptive_block_size", 101, 1.0, odd_only=True, clamp_min=3),
    Slider("clahe_clip*10",     "vision.clahe_clip",       100, 0.1),
    Slider("dilate_kernel",     "vision.dilate_kernel",     15, 1.0, clamp_min=1),
    Slider("sheet_close",       "vision.sheet_close_kernel", 60, 1.0, clamp_min=1),
    Slider("sheet_erode",       "vision.sheet_erode_kernel", 30, 1.0, clamp_min=1),
    # --- Clustering / filtering ---
    Slider("piece_min_area",    "vision.piece_min_area",   200, 1.0, clamp_min=1),
    Slider("cluster_gap",       "vision.cluster_gap_px_bird", 80, 1.0, clamp_min=1),
    Slider("min_lane_points",   "vision.min_lane_points",  400, 1.0, clamp_min=10),
    Slider("near_car_fit*100",  "vision.near_car_fit_frac", 100, 0.01, clamp_min=10),
    Slider("lane_half_w_bird",  "vision.lane_half_width_bird", 250, 1.0, clamp_min=10),
    # --- Car center calibration ---
    Slider("car_center*100",    "vision.car_center_x_frac", 100, 0.01, clamp_min=1),
    # --- PID gain (sent to Arduino on handshake) ---
    Slider("kp*10",             "control.pid.kp",          100, 0.1),
]


def _setattr_path(obj, path: str, value):
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    last = parts[-1]
    current = getattr(obj, last)
    if isinstance(current, int) and not isinstance(current, bool):
        value = int(round(value))
    setattr(obj, last, value)


def _getattr_path(obj, path: str):
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


# ---------------------------------------------------------------------------


def fetch_raw(url: str, timeout: float = 1.5) -> Optional[np.ndarray]:
    try:
        with urllib.request.urlopen(url + "/raw.jpg", timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] fetch error: {e}")
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def make_grid(tiles, cols: int, tile_w: int, tile_h: int,
              bg: tuple = (30, 30, 30)) -> np.ndarray:
    """Arrange labelled tiles into a grid. Each tile is (title, image)."""
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * tile_h, cols * tile_w, 3), bg, dtype=np.uint8)
    for i, (title, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(img, (tile_w, tile_h))
        # title bar
        cv2.rectangle(tile, (0, 0), (tile_w, 22), (0, 0, 0), -1)
        cv2.putText(tile, title, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)
        out[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
    return out


# ---------------------------------------------------------------------------
# Pipeline stages — mirror LaneDetector internals so we can show each.
# ---------------------------------------------------------------------------


def run_pipeline(cfg: AppConfig, frame: np.ndarray):
    """Return a dict of intermediate images + the LaneDetection object."""
    h, w = frame.shape[:2]
    persp = persp_mod.build_from_config(cfg.vision, w, h)

    # Copy of raw with source trapezoid drawn.
    persp_src = frame.copy()
    cv2.polylines(persp_src, [persp.src.astype(np.int32)],
                  True, (0, 255, 120), 2, cv2.LINE_AA)

    bird = persp.warp(frame)
    gray = cv2.medianBlur(cv2.cvtColor(bird, cv2.COLOR_BGR2GRAY), 3)
    if cfg.vision.clahe_enabled:
        clahe = cv2.createCLAHE(clipLimit=cfg.vision.clahe_clip,
                                tileGridSize=(cfg.vision.clahe_tile,) * 2)
        gray = clahe.apply(gray)

    # Sheet mask
    _, sheet_raw = cv2.threshold(gray, cfg.vision.sheet_threshold, 255, cv2.THRESH_BINARY)
    ck = max(1, cfg.vision.sheet_close_kernel)
    sheet_raw = cv2.morphologyEx(sheet_raw, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    nn, lbl, stt, _ = cv2.connectedComponentsWithStats(sheet_raw, connectivity=8)
    if nn > 1:
        biggest = 1 + int(np.argmax(stt[1:, cv2.CC_STAT_AREA]))
        sheet_mask = (lbl == biggest).astype(np.uint8) * 255
        ek = max(1, cfg.vision.sheet_erode_kernel)
        sheet_mask = cv2.erode(
            sheet_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek)),
        )
    else:
        sheet_mask = np.zeros_like(gray)

    # Tape binary inside sheet
    block = cfg.vision.adaptive_block_size
    if block % 2 == 0:
        block += 1
    tape_raw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        block, cfg.vision.adaptive_c,
    )
    binary = cv2.bitwise_and(tape_raw, sheet_mask)
    dk = max(1, cfg.vision.dilate_kernel)
    if dk > 1:
        binary = cv2.dilate(
            binary,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk)),
        )

    # Real detector (uses the same cfg)
    detector = LaneDetector(cfg.vision)
    det = detector.detect(frame)

    # Annotated full output
    annotated = overlay.draw(frame, det,
                             steering=0.0, fps=None,
                             target_lane=cfg.control.target_lane,
                             perspective=persp)

    return {
        "persp_src": persp_src,
        "bird": bird,
        "gray_clahe": gray,
        "sheet_mask": sheet_mask,
        "tape_binary": binary,
        "annotated": annotated,
        "det": det,
    }


# ---------------------------------------------------------------------------


def dump_config(cfg: AppConfig) -> str:
    """Print only the tunable values in a shape that's easy to paste back
    into config/default.yaml."""
    lines = ["# autocalib dump — paste the keys you want into config/default.yaml"]
    lines.append("vision:")
    for key in (
        "car_center_x_frac",
        "sheet_threshold", "sheet_close_kernel", "sheet_erode_kernel",
        "clahe_enabled", "clahe_clip", "clahe_tile",
        "adaptive_block_size", "adaptive_c",
        "dilate_kernel",
        "piece_min_area", "cluster_gap_px_bird",
        "min_lane_points", "near_car_fit_frac", "lane_half_width_bird",
        "fit_degree",
    ):
        lines.append(f"  {key}: {getattr(cfg.vision, key)}")
    lines.append("control:")
    lines.append(f"  pid:")
    lines.append(f"    kp: {cfg.control.pid.kp:.4f}")
    lines.append(f"    ki: {cfg.control.pid.ki:.4f}")
    lines.append(f"    kd: {cfg.control.pid.kd:.4f}")
    lines.append(f"  target_lane: \"{cfg.control.target_lane}\"")
    lines.append(f"  arduino_pwm_min: {cfg.control.arduino_pwm_min}")
    lines.append(f"  arduino_pwm_max: {cfg.control.arduino_pwm_max}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://raspberrypi.local:8000")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--input", type=Path, default=None,
                    help="Offline mode: run against a single raw frame file instead of the Pi")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent / "debug_out")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig.from_yaml(args.config)

    cv2.namedWindow(CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS, 420, 560)
    cv2.namedWindow(PIPELINE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PIPELINE, 1200, 720)
    cv2.namedWindow(ANNOTATED, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ANNOTATED, 640, 480)

    # Initialise sliders from the loaded config.
    def rebuild_sliders(from_cfg: AppConfig) -> None:
        for s in SLIDER_SPEC:
            current = _getattr_path(from_cfg, s.field)
            raw = int(round(float(current) / s.scale))
            raw = max(0, min(s.max_val, raw))
            # createTrackbar can't be deleted; recreating replaces the callback.
            cv2.createTrackbar(s.name, CONTROLS, raw, s.max_val, lambda _v: None)

    rebuild_sliders(cfg)

    paused = False
    last_frame: Optional[np.ndarray] = None

    print("keys: s=save  r=reload  p=pause  ESC=quit")

    while True:
        # Apply slider values into the mutable cfg.
        for s in SLIDER_SPEC:
            _setattr_path(cfg, s.field, s.read(cfg))

        if args.input is not None:
            if last_frame is None:
                last_frame = cv2.imread(str(args.input))
                if last_frame is None:
                    print(f"could not read {args.input}")
                    return
            frame = last_frame
        elif not paused:
            frame = fetch_raw(args.url)
            if frame is not None:
                last_frame = frame
            else:
                frame = last_frame
        else:
            frame = last_frame

        if frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "waiting for frame...", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(PIPELINE, blank)
            cv2.imshow(ANNOTATED, blank)
        else:
            try:
                st = run_pipeline(cfg, frame)
            except Exception as e:
                print(f"pipeline error: {e}")
                st = None

            if st is not None:
                det = st["det"]
                off = det.lane_center_offset(cfg.control.target_lane)
                hud = frame.copy()
                text = (
                    f"LANE {cfg.control.target_lane}  "
                    f"OFFSET {'---' if off is None else f'{off:+.2f}'}  "
                    f"L={det.left_points}  C={det.center_points}  R={det.right_points}  "
                    f"{'PAUSED' if paused else ''}"
                )
                cv2.rectangle(hud, (0, 0), (640, 22), (0, 0, 0), -1)
                cv2.putText(hud, text, (6, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)
                tiles = [
                    ("raw + trap", st["persp_src"]),
                    ("bird + CLAHE gray", st["gray_clahe"]),
                    ("sheet mask", st["sheet_mask"]),
                    ("bird", st["bird"]),
                    ("tape binary", st["tape_binary"]),
                    ("annotated", st["annotated"]),
                ]
                grid = make_grid(tiles, cols=3, tile_w=400, tile_h=300)
                cv2.imshow(PIPELINE, grid)
                cv2.imshow(ANNOTATED, st["annotated"])
                cv2.imshow(CONTROLS, hud)   # tiny preview on the Controls window too

        k = cv2.waitKey(30) & 0xFF
        if k == 27:   # ESC
            break
        elif k == ord("p"):
            paused = not paused
            print(f"paused = {paused}")
        elif k == ord("r"):
            cfg = AppConfig.from_yaml(args.config)
            rebuild_sliders(cfg)
            print("reloaded config from disk")
        elif k == ord("s"):
            dump = dump_config(cfg)
            out_path = args.out_dir / "calibrated.yaml"
            out_path.write_text(dump + "\n")
            print("----- calibrated values -----")
            print(dump)
            print(f"----- also saved to {out_path} -----")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
