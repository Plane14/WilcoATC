"""Tests for wilcoatc.atc.control: pure control laws, no simulator."""

from __future__ import annotations

import pytest

from wilcoatc.atc.control import (
    PID,
    TECS,
    AttitudePID,
    FeedForward,
    RatePID,
    TECSOutput,
)

# ----------------------------------------------------------------------
# PID

def test_pid_proportional_only():
    p = PID(kp=2.0, output_min=-10.0, output_max=10.0)
    assert p.update(1.0, 0.1) == pytest.approx(2.0)
    assert p.update(-0.5, 0.1) == pytest.approx(-1.0)


def test_pid_clamping():
    p = PID(kp=10.0, output_min=-0.5, output_max=0.5)
    assert p.update(1.0, 0.1) == pytest.approx(0.5)
    assert p.update(-1.0, 0.1) == pytest.approx(-0.5)


def test_pid_integral_accumulates():
    p = PID(kp=0.0, ki=10.0, kd=0.0,
            integrator_min=-1.0, integrator_max=1.0)
    p.update(0.1, 1.0)
    p.update(0.1, 1.0)
    # Integral = 0.2, ki * integral = 2.0, clamped to 1.0
    assert p.update(0.0, 1.0) == pytest.approx(1.0)


def test_pid_derivative():
    p = PID(kp=0.0, ki=0.0, kd=10.0, output_min=-100.0, output_max=100.0)
    p.update(0.0, 1.0)
    assert p.update(1.0, 1.0) == pytest.approx(10.0)
    assert p.update(1.0, 1.0) == pytest.approx(0.0)


def test_pid_zero_dt():
    p = PID(kp=2.0)
    p.update(1.0, 1.0)
    assert p.update(1.0, 0.0) == 0.0


def test_pid_reset():
    p = PID(kp=0.0, ki=10.0)
    p.update(0.1, 1.0)
    p.reset()
    assert p.update(0.1, 1.0) == pytest.approx(1.0)  # integral starts fresh


# ----------------------------------------------------------------------
# RatePID

def test_rate_pid_delegates():
    inner = RatePID(kp=2.0, output_min=-10.0, output_max=10.0)
    assert inner.update(1.0, 0.1) == pytest.approx(2.0)
    inner.reset()
    assert inner.update(1.0, 0.1) == pytest.approx(2.0)


# ----------------------------------------------------------------------
# AttitudePID

def test_attitude_pid_cascades():
    inner = RatePID(kp=1.0)
    outer = AttitudePID(kp=2.0, rate_pid=inner, max_rate=30.0)
    # 10 deg error -> rate command 20 deg/s, rate error 20 - 0 = 20,
    # output = 1.0 * 20 = 20, clamped to 1.0
    assert outer.update(10.0, 0.0, 0.1) == pytest.approx(1.0)


def test_attitude_pid_rate_limited():
    inner = RatePID(kp=0.1)
    outer = AttitudePID(kp=2.0, rate_pid=inner, max_rate=5.0)
    # 100 deg error -> rate command 200, clamped to 5,
    # rate error 5 - 0 = 5, output = 0.1 * 5 = 0.5
    assert outer.update(100.0, 0.0, 0.1) == pytest.approx(0.5)


# ----------------------------------------------------------------------
# FeedForward

def test_feed_forward():
    ff = FeedForward(gain=0.5)
    assert ff.update(2.0) == pytest.approx(1.0)
    assert ff.update(-2.0) == pytest.approx(-1.0)
    assert ff.update(10.0) == pytest.approx(1.0)  # clamped


# ----------------------------------------------------------------------
# TECS

def test_tecs_throttle_for_speed_error():
    t = TECS(kp_throttle=0.1, ki_throttle=0.0,
             kp_pitch=0.0, ki_pitch=0.0, ft_per_kt=10.0)
    out = t.update(speed_error=10.0, altitude_error=0.0, dt=1.0)
    assert out.throttle == pytest.approx(1.0)  # 0.1 * 10 = 1.0
    assert out.pitch == 0.0


def test_tecs_pitch_for_altitude_error():
    t = TECS(kp_throttle=0.0, ki_throttle=0.0,
             kp_pitch=0.1, ki_pitch=0.0, ft_per_kt=10.0)
    out = t.update(speed_error=0.0, altitude_error=100.0, dt=1.0)
    # Throttle PID has no gain: throttle stays at 0.0.
    assert out.throttle == pytest.approx(0.0)
    # Pitch: kp * altitude_error = 0.1 * 100.0 = 10.0, clamped to 1.0.
    assert out.pitch == pytest.approx(1.0)


def test_tecs_output_range():
    t = TECS(kp_throttle=1.0, ki_throttle=0.0,
             kp_pitch=1.0, ki_pitch=0.0, ft_per_kt=10.0)
    out = t.update(speed_error=100.0, altitude_error=1000.0, dt=1.0)
    assert 0.0 <= out.throttle <= 1.0
    assert -1.0 <= out.pitch <= 1.0


def test_tecs_reset():
    t = TECS(kp_throttle=0.0, ki_throttle=1.0,
             kp_pitch=0.0, ki_pitch=0.0, ft_per_kt=10.0)
    t.update(speed_error=1.0, altitude_error=0.0, dt=1.0)
    t.update(speed_error=1.0, altitude_error=0.0, dt=1.0)
    t.reset()
    out = t.update(speed_error=1.0, altitude_error=0.0, dt=1.0)
    assert out.throttle == pytest.approx(1.0)  # integral starts fresh


def test_tecs_dataclass_defaults():
    out = TECSOutput()
    assert out.throttle == 0.0
    assert out.pitch == 0.0
    out = TECSOutput(throttle=0.5, pitch=-0.3)
    assert out.throttle == 0.5
    assert out.pitch == -0.3
