from __future__ import annotations

import signal
import time
from typing import Optional

from .camera import AnnotatedFrameProvider, MJPEGStreamer, create_frame_source
from .comms import UARTLink
from .config import AppConfig
from .control import PID, DifferentialMixer
from .logging_setup import get_logger
from .vision import LaneDetector, overlay

log = get_logger(__name__)


class AutonomousCarApp:
    """Wires capture → vision → control → comms in a single control loop.
    Publishes an annotated (HUD) frame to the MJPEG stream each iteration."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.frames = create_frame_source(config.camera)
        self.viz = AnnotatedFrameProvider(self.frames)
        self.streamer = MJPEGStreamer(self.viz, config.camera)
        self.detector = LaneDetector(config.vision)
        self.pid = PID.from_config(config.control.pid, output_limit=1.0)
        self.mixer = DifferentialMixer(
            base_speed=config.control.base_speed,
            max_speed=config.control.max_speed,
        )
        self.uart = UARTLink(config.comms)
        self._stop = False
        self._fps_ema = 0.0

    def run(self) -> None:
        self._install_signal_handlers()
        period = 1.0 / max(1, self.config.control.loop_hz)

        self.frames.start()
        self.streamer.start()
        # UARTLink runs its own background thread: opens/reopens the port as
        # the Arduino comes and goes, auto-answers CFG? with the control
        # config, and gates send_offset() on the link state. The main loop
        # below only has to call send_offset() — no blocking handshake.
        self.uart.open(ctrl=self.config.control)
        log.info("control loop starting at %d Hz", self.config.control.loop_hz)

        last_t: Optional[float] = None
        try:
            while not self._stop:
                loop_start = time.monotonic()
                dt = period if last_t is None else loop_start - last_t
                last_t = loop_start

                frame = self.frames.get_frame()
                if frame is None:
                    time.sleep(period)
                    continue

                detection = self.detector.detect(frame)
                target = self.config.control.target_lane
                offset = detection.lane_center_offset(target) if detection.found else None

                steering: Optional[float] = None
                if offset is not None:
                    # Keep the local PID running so the HUD reflects what the
                    # Pi *would* command; actual motor control lives on the
                    # Arduino, which runs its own PID on the offset we forward.
                    steering = self.pid.update(offset, dt)
                    self.uart.send_offset(offset)
                else:
                    self.pid.reset()
                    # Don't spam stop — the Arduino's PID falls back to 0
                    # automatically when offsets go stale (500 ms).

                self._update_fps(dt)
                annotated = overlay.draw(
                    frame, detection,
                    steering=steering, fps=self._fps_ema, target_lane=target,
                    perspective=getattr(self.detector, "_persp", None),
                )
                self.viz.set_annotated(annotated)

                elapsed = time.monotonic() - loop_start
                remaining = period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self._shutdown()

    def _update_fps(self, dt: float) -> None:
        if dt <= 0:
            return
        inst = 1.0 / dt
        self._fps_ema = inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self._request_stop())

    def _request_stop(self) -> None:
        log.info("stop requested")
        self._stop = True

    def _shutdown(self) -> None:
        log.info("shutting down")
        try:
            self.uart.send_stop()
        finally:
            self.uart.close()
            self.streamer.stop()
            self.frames.stop()
