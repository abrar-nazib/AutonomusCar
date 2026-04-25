"""Shared spec of the runtime-tunable knobs.

Used by:
  * the browser tuner served from the Pi's streamer (`camera/streamer.py`)
  * the dev-machine OpenCV calibrator (`scripts/calibrate.py`)

Each `Knob` describes a single AppConfig field that should be exposed to
the live tuner: how to find it, what range to show in the UI, and whether
any special clamping applies (odd-only integers for kernels, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .config import AppConfig


@dataclass
class Knob:
    label: str                  # human-readable slider label
    path: str                   # dotted path into AppConfig, e.g. "vision.sheet_threshold"
    min_val: float              # UI min
    max_val: float              # UI max
    step: float = 1.0           # UI step size
    kind: str = "int"           # "int" or "float"
    odd_only: bool = False      # clamp to odd integers (for kernel sizes)
    clamp_min: Optional[float] = None


# Single source of truth. Add new rows here and they show up in both UIs.
KNOBS: List[Knob] = [
    # --- Perspective / car center -------------------------------------------
    Knob("car_center_x_frac",   "vision.car_center_x_frac",    0.00, 1.00, 0.01, "float", clamp_min=0.01),
    # --- Sheet mask ---------------------------------------------------------
    Knob("sheet_threshold",     "vision.sheet_threshold",         0,  255,    1, "int"),
    Knob("sheet_close_kernel",  "vision.sheet_close_kernel",      1,   60,    2, "int",  odd_only=False, clamp_min=1),
    Knob("sheet_erode_kernel",  "vision.sheet_erode_kernel",      1,   30,    1, "int",  clamp_min=1),
    # --- CLAHE --------------------------------------------------------------
    Knob("clahe_clip",          "vision.clahe_clip",           0.50, 10.0, 0.10, "float"),
    Knob("clahe_tile",          "vision.clahe_tile",              2,   32,    1, "int",  clamp_min=2),
    # --- Tape adaptive threshold -------------------------------------------
    Knob("adaptive_block_size", "vision.adaptive_block_size",     3,  101,    2, "int",  odd_only=True, clamp_min=3),
    Knob("adaptive_c",          "vision.adaptive_c",             -5,   60,    1, "int"),
    Knob("dilate_kernel",       "vision.dilate_kernel",           1,   15,    1, "int",  clamp_min=1),
    # --- Clustering + filters ----------------------------------------------
    Knob("piece_min_area",      "vision.piece_min_area",          1,  300,    1, "int",  clamp_min=1),
    Knob("cluster_gap_px_bird", "vision.cluster_gap_px_bird",     1,  100,    1, "int",  clamp_min=1),
    Knob("min_lane_points",     "vision.min_lane_points",        10,  600,   10, "int",  clamp_min=10),
    Knob("near_car_fit_frac",   "vision.near_car_fit_frac",    0.10, 1.00, 0.01, "float"),
    Knob("lane_half_width_bird","vision.lane_half_width_bird",   10,  260,    1, "int",  clamp_min=10),
    Knob("look_ahead_frac",     "vision.look_ahead_frac",      0.00, 1.00, 0.05, "float"),
    # --- PID ----------------------------------------------------------------
    Knob("kp",                  "control.pid.kp",              0.00, 10.0, 0.05, "float"),
    Knob("ki",                  "control.pid.ki",              0.00,  2.0, 0.01, "float"),
    Knob("kd",                  "control.pid.kd",              0.00,  2.0, 0.01, "float"),
    # --- Arduino PWM / lane selection --------------------------------------
    Knob("arduino_pwm_min",     "control.arduino_pwm_min",        0,  150,    1, "int"),
    Knob("arduino_pwm_max",     "control.arduino_pwm_max",        5,  200,    5, "int"),
]


def get_value(cfg: AppConfig, path: str) -> Any:
    obj = cfg
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def set_value(cfg: AppConfig, path: str, value: Any) -> Any:
    """Apply `value` to the knob at `path`, with type/parity clamping based
    on KNOBS. Returns the actually-applied value (after clamping)."""
    knob = next((k for k in KNOBS if k.path == path), None)
    if knob is None:
        raise KeyError(path)
    v = _clamp_for_knob(knob, value)
    parts = path.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], v)
    return v


def _clamp_for_knob(knob: Knob, value: Any) -> Any:
    v = float(value)
    if v < knob.min_val:
        v = knob.min_val
    if v > knob.max_val:
        v = knob.max_val
    if knob.clamp_min is not None and v < knob.clamp_min:
        v = knob.clamp_min
    if knob.kind == "int":
        iv = int(round(v))
        if knob.odd_only and iv % 2 == 0:
            iv += 1
        return iv
    return float(v)


def dump_snapshot(cfg: AppConfig) -> dict:
    """Return the current values of every knob as {path: value}."""
    return {k.path: get_value(cfg, k.path) for k in KNOBS}
