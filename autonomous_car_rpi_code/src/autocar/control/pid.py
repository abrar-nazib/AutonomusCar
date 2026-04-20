from __future__ import annotations

from dataclasses import dataclass

from ..config import PIDConfig


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    output_limit: float = 1.0

    _integral: float = 0.0
    _prev_error: float = 0.0
    _initialized: bool = False

    @classmethod
    def from_config(cls, cfg: PIDConfig, output_limit: float = 1.0) -> "PID":
        return cls(kp=cfg.kp, ki=cfg.ki, kd=cfg.kd, output_limit=output_limit)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0

        derivative = 0.0 if not self._initialized else (error - self._prev_error) / dt
        self._integral += error * dt
        out = self.kp * error + self.ki * self._integral + self.kd * derivative

        if out > self.output_limit:
            out = self.output_limit
            self._integral -= error * dt  # anti-windup
        elif out < -self.output_limit:
            out = -self.output_limit
            self._integral -= error * dt

        self._prev_error = error
        self._initialized = True
        return out
