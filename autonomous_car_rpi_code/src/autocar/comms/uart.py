"""UART link to the Arduino.

A dedicated background thread owns the serial port. It:
  * opens `/dev/ttyACM0` (or whatever `comms.uart_port` is) and reopens it
    if the port goes away (Arduino unplugged and replugged).
  * reads every line the Arduino emits and drives a small state machine.
  * automatically (re-)sends the configuration line whenever the Arduino
    asks `CFG?` — boot handshake and post-reset re-handshake use the same
    code path.
  * updates `self._state` so the main control loop, which only ever calls
    `send_offset()`, is decoupled from the handshake lifecycle.

State transitions driven by Arduino chatter:
    CFG?            → send C line,  state = NEED_CFG
    CFG_OK          →               state = RUNNING
    CFG_ERR <why>   →               state = HALTED
    PAUSED          →               state = PAUSED
    RUNNING         →               state = RUNNING (boot or after G)
    EXPIRED|HALTED  →               state = HALTED  (legacy firmware only)
    READY           →               (informational)

`send_offset()` is a no-op unless state is RUNNING, so the main loop can
call it on every tick without knowing anything about the handshake."""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..config import CommsConfig, ControlConfig
from ..control.mixer import MotorCommand
from ..logging_setup import get_logger

log = get_logger(__name__)


def encode_command(cmd: MotorCommand) -> bytes:
    """Legacy wire format: ASCII line 'M <left> <right>\\n'."""
    left = max(-255, min(255, int(cmd.left)))
    right = max(-255, min(255, int(cmd.right)))
    return f"M {left} {right}\n".encode("ascii")


STATE_UNKNOWN  = "UNKNOWN"       # port not open yet, or Arduino silent
STATE_NEED_CFG = "NEED_CFG"      # sent config, waiting for CFG_OK
STATE_RUNNING  = "RUNNING"       # Arduino is armed and accepting E commands
STATE_PAUSED   = "PAUSED"        # Arduino received S; ignoring E until G
STATE_HALTED   = "HALTED"        # Arduino sent CFG_ERR or legacy HALTED


class UARTLink:
    def __init__(self, cfg: CommsConfig):
        self.cfg = cfg
        self._serial = None
        self._state = STATE_UNKNOWN
        self._last_ctrl: Optional[ControlConfig] = None
        self._rx_buf = bytearray()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._write_lock = threading.Lock()  # serialise concurrent writes

    @property
    def state(self) -> str:
        return self._state

    # -- lifecycle ----------------------------------------------------------

    def open(self, ctrl: Optional[ControlConfig] = None) -> None:
        """Remember the control config for future handshakes and kick off
        the background reader/reconnector thread. Non-blocking."""
        if ctrl is not None:
            self._last_ctrl = ctrl
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._reader_loop, name="UARTLink", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._write_lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None

    # -- write API (main thread) -------------------------------------------

    def send_offset(self, offset: float) -> None:
        """Forward one lane-center error to the Arduino. Dropped silently if
        the Arduino isn't currently RUNNING (boot handshake, halted window,
        port offline — all look the same to the caller)."""
        if self._state != STATE_RUNNING:
            return
        self._write(f"E {offset:.4f}\n".encode("ascii"), "offset")

    def send_stop(self) -> None:
        """Emergency stop (resumable). Safe to call regardless of state."""
        self._write(b"S\n", "stop")

    def send_resume(self) -> None:
        """Bring the Arduino out of its PAUSED state. Safe to call always —
        firmware ignores G when not paused."""
        self._write(b"G\n", "resume")

    def send(self, cmd: MotorCommand) -> None:
        """Legacy raw-PWM command (no state gating, for the old protocol)."""
        self._write(encode_command(cmd), "M")

    def set_config(self, ctrl: ControlConfig) -> None:
        """Update the config the link will send on the next `CFG?` prompt."""
        self._last_ctrl = ctrl

    # -- background reader/reconnector -------------------------------------

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._serial is None:
                self._try_open()
                if self._serial is None:
                    self._stop_event.wait(1.0)   # port missing — retry in 1 s
                    continue
            try:
                self._pump_once()
            except OSError as e:
                # Arduino was unplugged mid-session; drop the handle and let
                # the outer loop re-open once the device reappears.
                log.warning("uart reader lost the port (%s); will reopen", e)
                self._reset_connection()
                self._stop_event.wait(0.5)
                continue
            except Exception as e:
                log.debug("uart reader error: %s", e)
            self._stop_event.wait(0.01)          # 10 ms poll

    def _try_open(self) -> None:
        try:
            import serial  # pyserial

            s = serial.Serial(
                port=self.cfg.uart_port,
                baudrate=self.cfg.uart_baud,
                timeout=self.cfg.command_timeout_s,
                write_timeout=self.cfg.command_timeout_s,
            )
        except Exception as e:
            log.debug("uart open retry: %s", e)
            return
        # Success.
        with self._write_lock:
            self._serial = s
            self._rx_buf.clear()
            self._state = STATE_UNKNOWN
        log.info("uart opened on %s @ %d baud", self.cfg.uart_port, self.cfg.uart_baud)

    def _reset_connection(self) -> None:
        with self._write_lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
            self._rx_buf.clear()
            self._state = STATE_UNKNOWN

    def _pump_once(self) -> None:
        s = self._serial
        if s is None:
            return
        try:
            n = s.in_waiting
        except OSError:
            raise
        if n:
            chunk = s.read(n)
            self._rx_buf.extend(chunk)
        while True:
            nl = self._rx_buf.find(b"\n")
            if nl < 0:
                return
            raw = bytes(self._rx_buf[:nl])
            del self._rx_buf[: nl + 1]
            line = raw.decode("ascii", errors="ignore").strip()
            if line:
                self._on_arduino_line(line)

    def _on_arduino_line(self, line: str) -> None:
        prev = self._state
        log.info("arduino: %s", line)

        if line == "CFG?":
            # (Re-)handshake — boot or post-reset.
            if self._last_ctrl is not None:
                self._write_config(self._last_ctrl)
                self._state = STATE_NEED_CFG
            else:
                log.warning("arduino asked CFG? but no config has been set yet")
        elif line == "CFG_OK":
            self._state = STATE_RUNNING
        elif line.startswith("CFG_ERR"):
            log.error("arduino rejected config: %s", line)
            self._state = STATE_HALTED
        elif line == "PAUSED":
            self._state = STATE_PAUSED
        elif line == "RUNNING":
            # Sent on boot (after CFG_OK) and again after every G (resume).
            self._state = STATE_RUNNING
        elif line in ("EXPIRED", "HALTED"):
            # Legacy firmware that halts forever — pre-pauseDrive build.
            self._state = STATE_HALTED
        # READY is informational.

        if self._state != prev:
            log.info("uart link state %s → %s", prev, self._state)

    # -- internals ---------------------------------------------------------

    def _write(self, data: bytes, tag: str) -> None:
        with self._write_lock:
            if self._serial is None:
                return
            try:
                self._serial.write(data)
            except Exception as e:
                log.warning("uart %s write failed: %s", tag, e)

    def _write_config(self, ctrl: ControlConfig) -> None:
        line = (
            f"C {ctrl.pid.kp:.4f} {ctrl.pid.ki:.4f} {ctrl.pid.kd:.4f} "
            f"{int(ctrl.arduino_pwm_min)} {int(ctrl.arduino_pwm_max)}\n"
        )
        log.info("uart sending config → %s", line.strip())
        self._write(line.encode("ascii"), "config")

    def __enter__(self) -> "UARTLink":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.send_stop()
        self.close()
