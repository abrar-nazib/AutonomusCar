from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    framerate: int = 30
    stream_host: str = "0.0.0.0"
    stream_port: int = 8000
    stream_enabled: bool = True
    jpeg_quality: int = 80


@dataclass
class VisionConfig:
    roi_top: float = 0.55
    roi_bottom: float = 0.95
    binarize_threshold: int = 80
    min_contour_area: int = 500


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


@dataclass
class CommsConfig:
    uart_port: str = "/dev/serial0"
    uart_baud: int = 115200
    command_timeout_s: float = 0.2


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
            logging=LoggingConfig(**data.get("logging", {})),
        )


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
