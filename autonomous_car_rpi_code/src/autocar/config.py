from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CameraConfig:
    source: str = "picamera"     # "picamera" (picamera2/libcamera) or "webcam" (cv2.VideoCapture)
    webcam_index: int = 0        # /dev/videoN when source=="webcam"
    width: int = 640
    height: int = 480
    framerate: int = 30
    stream_host: str = "0.0.0.0"
    stream_port: int = 8000
    stream_enabled: bool = True
    jpeg_quality: int = 80
    rotate_180: bool = False     # picamera: libcamera Transform(hflip,vflip). webcam: cv2.rotate(ROT_180)
    # Image tuning applied via libcamera controls at camera-open time. Unset /
    # null means "leave on auto / sensor default". Framerate caps exposure via
    # FrameDurationLimits so AE can never drag the control loop rate down.
    exposure_value: Optional[float] = None   # EV bias [-8.0, 8.0]; +1.0 ≈ 2× target
    analogue_gain: Optional[float] = None    # None = auto; typical [1.0, 16.0]
    brightness: Optional[float] = None       # [-1.0, 1.0], post-ISP
    contrast: Optional[float] = None         # 1.0 = default
    saturation: Optional[float] = None       # 1.0 = default


@dataclass
class VisionConfig:
    # --- Bird's-eye source trapezoid (road surface in the image) ---
    # Asymmetric corners so the trapezoid can hug a curving / offset track.
    # Each *_frac is that corner's x or y as a fraction of frame size.
    persp_top_y_frac: float = 0.18
    persp_bottom_y_frac: float = 0.88
    persp_top_left_frac: float = 0.15        # (x_frac, top_y_frac)
    persp_top_right_frac: float = 0.72
    persp_bottom_left_frac: float = -0.05    # can go < 0 so far-out tapes still warp
    persp_bottom_right_frac: float = 1.00
    persp_dst_margin_frac: float = 0.10      # side margin in the bird's-eye canvas
    # Legacy symmetric fields — used as a fallback only when *_left_frac /
    # *_right_frac are None. Retained so older YAMLs still load.
    persp_top_width_frac: float = 0.55
    persp_bottom_width_frac: float = 1.05
    # --- Car trajectory reference (NOT frame center) ---
    car_center_x_frac: float = 0.50          # calibrated column where the car will actually drive
    # --- Adaptive threshold in bird's-eye ---
    adaptive_block_size: int = 31            # must be odd
    adaptive_c: int = 10
    dilate_kernel: int = 3                   # reconnect fragmented tape strokes
    # --- Sheet mask (stage 2) — isolates the track sheet from floor/curtains ---
    sheet_threshold: int = 110               # pixels brighter than this = track sheet
    sheet_close_kernel: int = 25             # close pinholes (dashes/wrinkles) in the sheet mask
    sheet_erode_kernel: int = 9              # shrink the sheet mask to avoid its own bright boundary
    # CLAHE — normalises local brightness before thresholding so shadow/lit
    # halves of the sheet look similar.
    clahe_enabled: bool = True
    clahe_clip: float = 2.5
    clahe_tile: int = 8
    # --- Tape-shaped connected-component filter (bird's-eye) ---
    open_kernel: int = 3                     # morphological open to drop salt-and-pepper speckle
    piece_min_area: int = 20                 # reject any component smaller than this before clustering
    cluster_gap_px_bird: int = 55            # merge components whose x-centroids are within this gap
    dash_join_kernel: int = 45               # retained for back-compat (unused by clustering path)
    min_lane_height_frac_bird: float = 0.30  # cluster's y-span must be at least this of bird height
    min_lane_aspect_bird: float = 2.0        # retained for back-compat
    max_lane_width_frac_bird: float = 0.12   # retained for back-compat
    min_lane_points: int = 120               # original-tape pixels required per lane
    near_car_fit_frac: float = 0.60          # fit the polynomial on the bottom N%% of each lane's pixels — avoids multi-valued fits when the far-ahead tape curves hard
    lane_half_width_bird: float = 100.0      # assumed half-lane width (bird's-eye px) — used by the single-lane offset fallback at sharp curves
    look_ahead_frac: float = 0.6             # 0 = sample offset at car (max y), 1 = at far end of fit. Higher = react to upcoming curve sooner.
    peak_edge_mask_frac: float = 0.05        # retained for back-compat (unused by CC path)
    # --- Sliding-window knobs (retained for back-compat; unused by CC path) ---
    nwindows: int = 12
    window_margin: int = 55
    window_minpix: int = 40
    min_peak_separation: int = 90
    fit_degree: int = 2

    # Back-compat fields retained so older YAMLs still load. Unused by this
    # detector path. Do not rely on these.
    roi_top: float = 0.0
    roi_bottom: float = 1.0
    binarize_threshold: int = 55
    min_contour_area: int = 500
    roi_left_trim: float = 0.0
    roi_right_trim: float = 0.0
    dash_join_kernel_h: int = 21
    dash_join_kernel_w: int = 3
    opening_kernel: int = 1
    piece_min_area: int = 8
    min_lane_area: int = 60
    max_lane_area_frac: float = 0.15
    min_lane_height_frac: float = 0.08
    min_aspect_ratio: float = 1.2
    max_lane_width_px: int = 40
    cluster_max_gap_px: int = 45
    trapezoid_top_width_frac: float = 0.32
    trapezoid_bottom_width_frac: float = 0.95


@dataclass
class PIDConfig:
    kp: float = 0.6
    ki: float = 0.0
    kd: float = 0.15


@dataclass
class ControlConfig:
    pid: PIDConfig = field(default_factory=PIDConfig)
    base_speed: int = 120
    max_speed: int = 200
    loop_hz: int = 30
    target_lane: str = "R"            # 'L' = left lane, 'R' = right lane
    # PWM clamp sent to the Arduino at boot-time handshake. Keep both small
    # for a slow steering test; the Arduino runs its own PID on the error the
    # Pi forwards via `E <offset>`.
    arduino_pwm_min: int = 10
    arduino_pwm_max: int = 50


@dataclass
class CommsConfig:
    uart_port: str = "/dev/serial0"
    uart_baud: int = 115200
    command_timeout_s: float = 0.2


@dataclass
class LidarConfig:
    enabled: bool = True
    port: str = "/dev/ttyUSB0"
    baud: int = 256000               # this unit; A2 standard is 115200
    pwm: int = 1023                  # this unit will not start below ~1000
    max_range_mm: float = 6000.0     # range clip for the on-screen radar
    overlay_enabled: bool = True
    overlay_size_px: int = 220       # diameter of the radar inset
    overlay_margin_px: int = 12


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    comms: CommsConfig = field(default_factory=CommsConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            vision=VisionConfig(**data.get("vision", {})),
            control=ControlConfig(
                pid=PIDConfig(**data.get("control", {}).get("pid", {})),
                **{k: v for k, v in data.get("control", {}).items() if k != "pid"},
            ),
            comms=CommsConfig(**data.get("comms", {})),
            lidar=LidarConfig(**data.get("lidar", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
