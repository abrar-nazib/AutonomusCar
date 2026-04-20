from __future__ import annotations

import signal
import time
from typing import Optional

from .camera import FrameSource, MJPEGStreamer
from .comms import UARTLink
from .config import AppConfig
from .control import PID, DifferentialMixer
from .logging_setup import get_logger
from .vision import TrackDetector

log = get_logger(__name__)


class AutonomousCarApp:
    """Wires capture → vision → control → comms in a single control loop."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.frames = FrameSource(config.camera)
        self.streamer = MJPEGStreamer(self.frames, config.camera)
        self.detector = TrackDetector(config.vision)
        self.pid = PID.from_config(config.control.pid, output_limit=1.0)
        self.mixer = DifferentialMixer(
            base_speed=config.control.base_speed,
            max_speed=config.control.max_speed,
        )
        self.uart = UARTLink(config.comms)
        self._stop = False

    def run(self) -> None:
        self._install_signal_handlers()
        period = 1.0 / max(1, self.config.control.loop_hz)

        self.frames.start()
        self.streamer.start()
        self.uart.open()
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
                if not detection.found:
                    self.pid.reset()
                    self.uart.send_stop()
                else:
                    steering = self.pid.update(detection.offset, dt)
                    cmd = self.mixer.mix(throttle=1.0, steering=steering)
                    self.uart.send(cmd)

                elapsed = time.monotonic() - loop_start
                remaining = period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self._shutdown()

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
