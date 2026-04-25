from __future__ import annotations

import signal
import time
from typing import Optional

from pathlib import Path

from .camera import AnnotatedFrameProvider, MJPEGStreamer, create_frame_source
from .camera.streamer import TunerHooks
from .comms import UARTLink
from .config import AppConfig
from .control import PID, DifferentialMixer
from .lidar import RPLidarReader
from .logging_setup import get_logger
from .vision import LaneDetector, overlay

log = get_logger(__name__)


class AutonomousCarApp:
    """Wires capture → vision → control → comms in a single control loop.
    Publishes an annotated (HUD) frame to the MJPEG stream each iteration."""

    def __init__(self, config: AppConfig, config_path: Optional[Path] = None):
        self.config = config
        self.config_path = config_path
        self.frames = create_frame_source(config.camera)
        self.viz = AnnotatedFrameProvider(self.frames)
        self.tuner = TunerHooks(
            config=config,
            config_path=config_path,
            on_apply=self._on_tune,
            set_running=self._set_running,
            get_running=lambda: self._running,
        )
        self.streamer = MJPEGStreamer(self.viz, config.camera, tuner_hooks=self.tuner)
        self.detector = LaneDetector(config.vision)
        self.pid = PID.from_config(config.control.pid, output_limit=1.0)
        self.mixer = DifferentialMixer(
            base_speed=config.control.base_speed,
            max_speed=config.control.max_speed,
        )
        self.uart = UARTLink(config.comms)
        self.lidar = RPLidarReader(config.lidar) if config.lidar.enabled else None
        self._stop = False
        # Web-UI emergency stop flag. When False the loop keeps running (frames,
        # vision, HUD) but skips send_offset and resets the PID — Arduino's
        # stale-offset watchdog (500 ms) holds the motors at zero. We also
        # send an explicit stop on the running→paused transition so the wheels
        # halt within one loop tick instead of after the watchdog window.
        self._running = True
        self._fps_ema = 0.0

    def run(self) -> None:
        self._install_signal_handlers()
        period = 1.0 / max(1, self.config.control.loop_hz)

        self.frames.start()
        self.streamer.start()
        if self.lidar is not None:
            self.lidar.start()
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
                if not self._running:
                    # Emergency-stop via web UI. Don't steer, don't forward
                    # offsets. PID resets so it doesn't wind up while paused.
                    self.pid.reset()
                elif offset is not None:
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
                lidar_scan = self.lidar.latest_scan() if self.lidar is not None else None
                annotated = overlay.draw(
                    frame, detection,
                    steering=steering, fps=self._fps_ema, target_lane=target,
                    perspective=getattr(self.detector, "_persp", None),
                    lidar_scan=lidar_scan if self.config.lidar.overlay_enabled else None,
                    lidar_max_range_mm=self.config.lidar.max_range_mm,
                    lidar_inset_size_px=self.config.lidar.overlay_size_px,
                    lidar_inset_margin_px=self.config.lidar.overlay_margin_px,
                )
                self.viz.set_annotated(annotated)

                elapsed = time.monotonic() - loop_start
                remaining = period - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self._shutdown()

    def _on_tune(self, applied: dict) -> None:
        """Called by the web tuner whenever slider values change. Vision
        knobs take effect immediately (the detector holds a reference to
        `self.config.vision`). For PID gains and PWM limits we refresh the
        local PID object and push a fresh config line to the Arduino."""
        pid_keys = {"control.pid.kp", "control.pid.ki", "control.pid.kd"}
        arduino_keys = {"control.arduino_pwm_min", "control.arduino_pwm_max"}
        if applied.keys() & pid_keys:
            self.pid = PID.from_config(self.config.control.pid, output_limit=1.0)
        if applied.keys() & (pid_keys | arduino_keys):
            # Re-send config so Arduino's PID/PWM clamp picks up the change.
            try:
                self.uart.set_config(self.config.control)
            except Exception as e:
                log.warning("tuner: could not refresh arduino config: %s", e)

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

    def _set_running(self, running: bool) -> None:
        running = bool(running)
        if running == self._running:
            return
        self._running = running
        if not running:
            # Pause: drop PID state and tell Arduino to halt RIGHT NOW so the
            # car stops within a UART-roundtrip instead of waiting for the
            # 500 ms stale-offset watchdog to kick in.
            self.pid.reset()
            try:
                self.uart.send_stop()
            except Exception as e:
                log.warning("e-stop: send_stop failed: %s", e)
            log.warning("E-STOP via web UI")
        else:
            # Resume: send G so the Arduino re-arms its drivers and accepts
            # offsets again. Without this, send_offset() stays a no-op
            # because the link state is still PAUSED.
            self.pid.reset()
            try:
                self.uart.send_resume()
            except Exception as e:
                log.warning("resume: send_resume failed: %s", e)
            log.info("RESUME via web UI")

    def _shutdown(self) -> None:
        log.info("shutting down")
        try:
            self.uart.send_stop()
        finally:
            self.uart.close()
            self.streamer.stop()
            if self.lidar is not None:
                self.lidar.stop()
            self.frames.stop()
