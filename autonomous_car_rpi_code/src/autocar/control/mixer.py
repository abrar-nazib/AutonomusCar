from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MotorCommand:
    left: int       # signed PWM, -255..255
    right: int


class DifferentialMixer:
    """Turns (throttle, steering) in [-1, 1] into per-side signed PWM commands."""

    def __init__(self, base_speed: int, max_speed: int):
        self.base_speed = base_speed
        self.max_speed = max_speed

    def mix(self, throttle: float, steering: float) -> MotorCommand:
        throttle = _clamp(throttle, -1.0, 1.0)
        steering = _clamp(steering, -1.0, 1.0)

        base = throttle * self.base_speed
        left = base - steering * self.base_speed
        right = base + steering * self.base_speed

        limit = self.max_speed
        return MotorCommand(
            left=int(_clamp(left, -limit, limit)),
            right=int(_clamp(right, -limit, limit)),
        )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
