"""Background-thread RPLIDAR reader.

Spins the motor (SetMotorPWM), starts a continuous standard scan, parses the
5-byte node format, and snapshots the most recent full revolution into a
thread-safe slot. The autocar control loop never blocks on the lidar — it
just calls `latest_scan()` to get whatever revolution finished most recently.

Verified against this unit: model 0x2c, FW 1.32, 256000 baud, PWM=1023.

Standard scan node (5 bytes, after the A5 5A scan descriptor):
    byte0  bits 7..2 = quality, bit1 = !S, bit0 = S (start of new revolution)
    byte1  bits 7..1 = angle_low, bit0 = check bit (must be 1)
    byte2  bits 7..0 = angle_high
                       angle_q6 = (byte1 >> 1) | (byte2 << 7);
                       angle_deg = angle_q6 / 64
    byte3  distance_low
    byte4  distance_high
                       distance_q2 = byte3 | (byte4 << 8);
                       distance_mm = distance_q2 / 4
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Optional

import numpy as np
import serial

from ..config import LidarConfig
from ..logging_setup import get_logger

log = get_logger(__name__)

_CMD_SYNC = 0xA5
_CMD_STOP = 0x25
_CMD_SCAN = 0x20
_CMD_SET_MOTOR_PWM = 0xF0


class RPLidarReader:
    def __init__(self, cfg: LidarConfig):
        self.cfg = cfg
        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        # Last completed revolution: Nx2 float32 array (angle_deg, distance_mm).
        self._latest: Optional[np.ndarray] = None
        self._scan_count: int = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public API --------------------------------------------------------

    def latest_scan(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    @property
    def scan_count(self) -> int:
        return self._scan_count

    def start(self) -> None:
        if not self.cfg.enabled:
            log.info("lidar disabled by config")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="RPLidarReader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            s = self._serial
            self._serial = None
        if s is not None:
            try:
                self._send_cmd_on(s, _CMD_STOP)
                time.sleep(0.02)
                self._send_cmd_on(s, _CMD_SET_MOTOR_PWM, struct.pack("<H", 0))
            except Exception:
                pass
            try: s.close()
            except Exception: pass

    # -- run loop ----------------------------------------------------------

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                s = serial.Serial(self.cfg.port, self.cfg.baud, timeout=0.5)
            except Exception as e:
                log.warning("lidar open %s failed: %s; retrying in %.1fs",
                            self.cfg.port, e, backoff)
                self._stop_event.wait(backoff)
                backoff = min(backoff * 1.5, 5.0)
                continue
            backoff = 1.0
            with self._lock:
                self._serial = s
            try:
                s.dtr = False
                self._send_cmd_on(s, _CMD_STOP)
                time.sleep(0.1)
                self._send_cmd_on(s, _CMD_SET_MOTOR_PWM, struct.pack("<H", self.cfg.pwm))
                time.sleep(0.6)             # let the motor come up to speed
                s.reset_input_buffer()
                self._send_cmd_on(s, _CMD_SCAN)
                desc = s.read(7)
                if len(desc) != 7 or desc[0] != 0xA5 or desc[1] != 0x5A:
                    raise serial.SerialException(
                        f"bad scan descriptor: {desc.hex()}"
                    )
                log.info("lidar streaming (PWM=%d, %d baud)", self.cfg.pwm, self.cfg.baud)
                self._scan_loop(s)
            except serial.SerialException as e:
                log.warning("lidar reader: %s; reopening", e)
            except Exception as e:
                log.exception("lidar reader unexpected: %s", e)
            finally:
                with self._lock:
                    self._serial = None
                try: s.close()
                except Exception: pass
                self._stop_event.wait(1.0)

    def _scan_loop(self, s: serial.Serial) -> None:
        buf = bytearray()
        accum: list = []
        while not self._stop_event.is_set():
            try:
                chunk = s.read(512)
            except serial.SerialException:
                raise
            if not chunk:
                continue
            buf.extend(chunk)
            # Parse as many 5-byte nodes as possible. Resync byte-by-byte
            # if either consistency bit is wrong.
            i = 0
            n = len(buf)
            while n - i >= 5:
                b0 = buf[i]; b1 = buf[i + 1]
                if (b0 & 1) == ((b0 >> 1) & 1) or (b1 & 1) != 1:
                    i += 1
                    continue
                b2 = buf[i + 2]; b3 = buf[i + 3]; b4 = buf[i + 4]
                start_bit = b0 & 1
                quality = b0 >> 2
                angle_q6 = (b1 >> 1) | (b2 << 7)
                distance_q2 = b3 | (b4 << 8)
                angle = angle_q6 / 64.0
                distance = distance_q2 / 4.0
                if start_bit:
                    if accum:
                        arr = np.asarray(accum, dtype=np.float32)
                        with self._lock:
                            self._latest = arr
                            self._scan_count += 1
                    accum = []
                if quality > 0 and distance > 0.0:
                    accum.append((angle, distance))
                i += 5
            if i:
                del buf[:i]

    # -- write helpers -----------------------------------------------------

    @staticmethod
    def _send_cmd_on(s: serial.Serial, cmd: int, payload: bytes = b"") -> None:
        if payload:
            cmd_b = cmd | 0x80
            body = bytes([_CMD_SYNC, cmd_b, len(payload)]) + payload
            chk = 0
            for b in body:
                chk ^= b
            s.write(body + bytes([chk]))
        else:
            s.write(bytes([_CMD_SYNC, cmd]))
        s.flush()
