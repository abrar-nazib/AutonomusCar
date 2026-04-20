from __future__ import annotations

from typing import Optional

from ..config import CommsConfig
from ..control.mixer import MotorCommand
from ..logging_setup import get_logger

log = get_logger(__name__)


def encode_command(cmd: MotorCommand) -> bytes:
    """Wire format: ASCII line 'M <left> <right>\\n', with left/right in
    [-255, 255]. Simple, easy to debug, easy to parse on the Arduino."""
    left = max(-255, min(255, int(cmd.left)))
    right = max(-255, min(255, int(cmd.right)))
    return f"M {left} {right}\n".encode("ascii")


class UARTLink:
    """Thin wrapper over pyserial. Opens lazily so unit tests / dev runs don't
    need a real serial port. `send` is a no-op when the port cannot be opened."""

    def __init__(self, cfg: CommsConfig):
        self.cfg = cfg
        self._serial = None

    def open(self) -> None:
        try:
            import serial  # pyserial

            self._serial = serial.Serial(
                port=self.cfg.uart_port,
                baudrate=self.cfg.uart_baud,
                timeout=self.cfg.command_timeout_s,
                write_timeout=self.cfg.command_timeout_s,
            )
            log.info("uart opened on %s @ %d baud", self.cfg.uart_port, self.cfg.uart_baud)
        except Exception as e:
            log.warning("uart open failed (%s); commands will be dropped", e)
            self._serial = None

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

    def send(self, cmd: MotorCommand) -> None:
        if self._serial is None:
            return
        try:
            self._serial.write(encode_command(cmd))
        except Exception as e:
            log.warning("uart write failed: %s", e)

    def send_stop(self) -> None:
        self.send(MotorCommand(left=0, right=0))

    def __enter__(self) -> "UARTLink":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.send_stop()
        self.close()
