from ..config import CameraConfig
from .capture import FrameSource, PiCameraFrameSource
from .streamer import MJPEGStreamer
from .viz import AnnotatedFrameProvider
from .webcam import WebcamFrameSource

__all__ = [
    "AnnotatedFrameProvider",
    "FrameSource",
    "MJPEGStreamer",
    "PiCameraFrameSource",
    "WebcamFrameSource",
    "create_frame_source",
]


def create_frame_source(cfg: CameraConfig) -> FrameSource:
    """Pick the frame-source backend based on `camera.source` in the config.
    `"picamera"` (default) uses picamera2/libcamera; `"webcam"` uses an
    OpenCV V4L2 capture over /dev/videoN."""
    src = (cfg.source or "picamera").strip().lower()
    if src in ("webcam", "usb", "v4l2", "usb_webcam"):
        return WebcamFrameSource(cfg)
    return PiCameraFrameSource(cfg)
