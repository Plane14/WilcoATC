"""Flight control laws: PID, cascaded rate/attitude, feedforward, TECS.

Pure math, no simulator: given a target and the current state, each
controller returns a control command in -1..1 (or 0..1 for throttle).
The autonomous flight code applies these through the FlightDirector.

No SimConnect, no aircraft model, no dependency on anything but the
standard library.
"""

from __future__ import annotations

from dataclasses import dataclass


class PID:
    """Proportional-integral-derivative controller with clamping.

    ``error`` is the signed difference target minus measured. Output is
    clamped to [output_min, output_max]; the integral term is clamped to
    [integrator_min, integrator_max] for anti-windup. Call with dt > 0;
    a zero or negative dt returns 0 and does not integrate.
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 output_min: float = -1.0, output_max: float = 1.0,
                 integrator_min: float | None = None,
                 integrator_max: float | None = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integrator_min = (integrator_min if integrator_min is not None
                               else output_min)
        self.integrator_max = (integrator_max if integrator_max is not None
                               else output_max)
        self._integral = 0.0
        self._prev_error: float | None = None
        self._prev_time: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self._integral += error * dt
        self._integral = max(self.integrator_min, min(self.integrator_max,
                                                      self._integral))
        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(self.output_min, min(self.output_max, output))


class RatePID:
    """Inner loop: turns a rate error (deg/s) into a control output."""

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0,
                 output_min: float = -1.0, output_max: float = 1.0):
        self.pid = PID(kp, ki, kd, output_min, output_max)

    def reset(self) -> None:
        self.pid.reset()

    def update(self, rate_error: float, dt: float) -> float:
        return self.pid.update(rate_error, dt)


class AttitudePID:
    """Outer loop: turns an attitude error (deg) into a rate command, then
    cascades it through a RatePID to produce a control output.

    ``max_rate`` limits the rate command so a large attitude error does not
    demand an instantaneous snap roll or pitch.
    """

    def __init__(self, kp: float, rate_pid: RatePID,
                 max_rate: float = 15.0):
        self.kp = kp
        self.rate_pid = rate_pid
        self.max_rate = max_rate

    def reset(self) -> None:
        self.rate_pid.reset()

    def update(self, attitude_error: float, rate: float, dt: float) -> float:
        rate_command = self.kp * attitude_error
        rate_command = max(-self.max_rate, min(self.max_rate, rate_command))
        rate_error = rate_command - rate
        return self.rate_pid.update(rate_error, dt)


class FeedForward:
    """Adds an anticipated control input based on the commanded target.

    For example, when commanding a pitch-up, add nose-up elevator
    immediately rather than waiting for the error to build.
    """

    def __init__(self, gain: float = 1.0,
                 output_min: float = -1.0, output_max: float = 1.0):
        self.gain = gain
        self.output_min = output_min
        self.output_max = output_max

    def update(self, target: float) -> float:
        return max(self.output_min, min(self.output_max,
                                        self.gain * target))


@dataclass
class TECSOutput:
    throttle: float = 0.0   # 0..1 lever position
    pitch: float = 0.0      # -1..1 elevator command


class TECS:
    """Total Energy Control System.

    Separates speed and altitude errors into a total-energy error (handled
    by throttle) and an energy-distribution error (handled by pitch).
    Kinetic energy changes are fast (throttle), potential energy changes
    are slow (pitch), so each axis has its own PI loop.

    The feet-to-knots factor converts an altitude error into an
    equivalent speed error for the energy sum; it is a tuning parameter,
    not a physical constant.
    """

    def __init__(self, kp_throttle: float = 0.05, ki_throttle: float = 0.01,
                 kp_pitch: float = 0.05, ki_pitch: float = 0.01,
                 ft_per_kt: float = 10.0,
                 throttle_min: float = 0.0, throttle_max: float = 1.0,
                 pitch_min: float = -1.0, pitch_max: float = 1.0):
        self.throttle_pid = PID(kp_throttle, ki_throttle, 0.0,
                                throttle_min, throttle_max)
        self.pitch_pid = PID(kp_pitch, ki_pitch, 0.0,
                             pitch_min, pitch_max)
        self.ft_per_kt = ft_per_kt

    def reset(self) -> None:
        self.throttle_pid.reset()
        self.pitch_pid.reset()

    def update(self, speed_error: float, altitude_error: float,
               dt: float) -> TECSOutput:
        """Return (throttle, pitch) commands from speed/altitude errors.

        ``speed_error`` = target_kt - current_kt, in knots.
        ``altitude_error`` = target_ft - current_ft, in feet.
        """
        energy_error = speed_error + altitude_error / self.ft_per_kt
        throttle = self.throttle_pid.update(energy_error, dt)
        pitch = self.pitch_pid.update(altitude_error, dt)
        return TECSOutput(throttle=throttle, pitch=pitch)


__all__ = ["PID", "RatePID", "AttitudePID", "FeedForward", "TECS",
           "TECSOutput"]
