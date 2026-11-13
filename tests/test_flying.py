"""Golden tests for the pure flight model in wilcoatc.atc.flying.

No simulator, no session: a target, a performance and a number of seconds
move an aeroplane, and the result is checked.
"""

from __future__ import annotations

import math

import pytest

from wilcoatc.atc.flying import (
    AIRLINER, LIGHT, RATE_ONE, TURBOPROP, Command, FlightConfig, FlightModel,
    FlightState, Performance, Target, flight_config_for, glide_altitude,
    offset_from_track, performance_for, step, turn_towards,
)


class _Flight:
    """A minimal stand-in carrying the six numbers step() moves."""

    def __init__(self, *, lat=50.0, lon=1.0, altitude_ft=180.0,
                 heading=90.0, ground_speed_kt=100.0,
                 vertical_rate_fpm=0.0, on_ground=False):
        self.latitude = lat
        self.longitude = lon
        self.altitude_ft = altitude_ft
        self.heading = heading
        self.ground_speed_kt = ground_speed_kt
        self.vertical_rate_fpm = vertical_rate_fpm
        self.on_ground = on_ground


# ----------------------------------------------------------------------
# turn_towards

def test_turn_towards_short_way():
    h = turn_towards(10.0, 350.0, 3.0, 10.0)
    assert h == pytest.approx(350.0)


def test_turn_towards_rate_limited():
    h = turn_towards(10.0, 100.0, 3.0, 1.0)
    assert h == pytest.approx(13.0)


def test_turn_towards_left_forced():
    h = turn_towards(10.0, 350.0, 3.0, 1.0, direction="left")
    assert h == pytest.approx(7.0)


def test_turn_towards_right_forced():
    h = turn_towards(350.0, 10.0, 3.0, 1.0, direction="right")
    assert h == pytest.approx(353.0)


def test_turn_towards_wraps():
    h = turn_towards(350.0, 10.0, 40.0, 1.0)
    assert h == pytest.approx(10.0)


# ----------------------------------------------------------------------
# step: hold

def test_step_hold_stops():
    f = _Flight(ground_speed_kt=20.0, on_ground=True)
    t = Target(hold=True)
    step(f, t, AIRLINER, 5.0)
    assert f.ground_speed_kt == pytest.approx(0.0)
    assert f.vertical_rate_fpm == 0.0


def test_step_hold_slows_gradually():
    f = _Flight(ground_speed_kt=20.0, on_ground=True)
    t = Target(hold=True)
    step(f, t, AIRLINER, 1.0)
    assert f.ground_speed_kt == pytest.approx(20.0 - 4.5)


# ----------------------------------------------------------------------
# step: speed

def test_step_speed_accelerates():
    f = _Flight(ground_speed_kt=100.0)
    t = Target(speed_kt=140.0)
    step(f, t, AIRLINER, 10.0)
    assert f.ground_speed_kt == pytest.approx(130.0)


def test_step_speed_decelerates_in_air():
    f = _Flight(ground_speed_kt=200.0)
    t = Target(speed_kt=140.0)
    step(f, t, AIRLINER, 10.0)
    assert f.ground_speed_kt == pytest.approx(180.0)


def test_step_speed_accel_override():
    f = _Flight(ground_speed_kt=0.0, on_ground=True)
    t = Target(speed_kt=140.0, accel_kt_s=10.0)
    step(f, t, AIRLINER, 10.0)
    assert f.ground_speed_kt == pytest.approx(100.0)


def test_step_speed_max_descent():
    f = _Flight(altitude_ft=3000.0)
    t = Target(altitude_ft=180.0, max_descent_fpm=300.0)
    step(f, t, AIRLINER, 60.0)
    # 300 fpm for 60 seconds = 300 ft descent: 3000 - 300 = 2700
    assert f.altitude_ft == pytest.approx(2700.0)


# ----------------------------------------------------------------------
# step: heading

def test_step_heading_turns():
    f = _Flight(heading=90.0)
    t = Target(heading=180.0)
    step(f, t, AIRLINER, 10.0)
    assert f.heading == pytest.approx(120.0)


def test_step_heading_ground_steers_faster():
    f = _Flight(heading=90.0, ground_speed_kt=20.0, on_ground=True)
    t = Target(heading=180.0)
    step(f, t, AIRLINER, 10.0)
    # On ground below 40 kt: turn_rate * 3 = 9 deg/s * 10 s = 90 deg → 180.
    assert f.heading == pytest.approx(180.0)


# ----------------------------------------------------------------------
# step: altitude

def test_step_altitude_climbs():
    f = _Flight(altitude_ft=1800.0, vertical_rate_fpm=1000.0)
    t = Target(altitude_ft=3000.0)
    step(f, t, AIRLINER, 10.0)
    climb = min(AIRLINER.climb_fpm, 1000.0 + 450.0 * 10.0) / 60.0 * 10.0
    assert f.altitude_ft == pytest.approx(1800.0 + climb)
    assert f.vertical_rate_fpm == pytest.approx(climb / 10.0 * 60.0)


def test_step_altitude_never_below_ground():
    f = _Flight(altitude_ft=300.0)
    t = Target(altitude_ft=0.0)
    step(f, t, AIRLINER, 60.0, ground_elevation_ft=180.0)
    assert f.altitude_ft >= 180.0


# ----------------------------------------------------------------------
# step: moves along heading

def test_step_moves_along_heading():
    f = _Flight(lat=50.0, lon=1.0, heading=90.0, ground_speed_kt=180.0)
    step(f, Target(), AIRLINER, 60.0)
    # 180 kt for one minute = 3 nm due east
    assert f.longitude > 1.0
    assert f.latitude == pytest.approx(50.0)
    delta_lon = f.longitude - 1.0
    expected = 3.0 * 1852.0 / (111320.0 * math.cos(math.radians(50.0)))
    assert delta_lon == pytest.approx(expected, rel=1e-6)


# ----------------------------------------------------------------------
# step: zero dt

def test_step_zero_dt_no_op():
    f = _Flight(heading=90.0, ground_speed_kt=100.0)
    before = (f.latitude, f.longitude, f.heading, f.ground_speed_kt,
              f.altitude_ft)
    step(f, Target(speed_kt=140.0), AIRLINER, 0.0)
    after = (f.latitude, f.longitude, f.heading, f.ground_speed_kt,
             f.altitude_ft)
    assert before == after


# ----------------------------------------------------------------------
# glide_altitude

def test_glide_altitude():
    assert glide_altitude(180.0, 5.0) == pytest.approx(180.0 + 5.0 * 318.0)


def test_glide_altitude_negative_distance_clamped():
    assert glide_altitude(180.0, -1.0) == 180.0


# ----------------------------------------------------------------------
# offset_from_track

def test_offset_from_track_on_track():
    along, across = offset_from_track(50.0, 1.0, 50.0, 1.0, 90.0)
    assert along == 0.0
    assert across == 0.0


def test_offset_from_track_abeam():
    # One mile due north of a point, track east: 1 nm across to the left.
    along, across = offset_from_track(50.0 + 1.0 / 60.0, 1.0, 50.0, 1.0, 90.0)
    assert across == pytest.approx(-1.0 * 1852.0, abs=1.0)


# ----------------------------------------------------------------------
# performance_for

def test_performance_for_light_type():
    p = performance_for("C172")
    assert p is not LIGHT
    assert p.rotate_kt == pytest.approx(55.0)


def test_performance_for_airliner_default():
    p = performance_for("A320")
    assert p.rotate_kt == pytest.approx(140.0)
    assert p.climb_fpm == pytest.approx(2200.0)


def test_performance_for_turboprop():
    p = performance_for("AT76")
    assert p.climb_fpm == pytest.approx(1500.0)
    assert p.approach_kt == pytest.approx(200.0)


def test_performance_for_unknown_is_airliner():
    p = performance_for("ZZZZ")
    assert p is AIRLINER or p.rotate_kt == pytest.approx(140.0)


# ----------------------------------------------------------------------
# Performance / Target dataclasses

def test_performance_defaults():
    p = Performance()
    assert p.turn_rate == 3.0
    assert p.climb_fpm == 2200.0
    assert p.approach_kt == 140.0


def test_target_clear():
    t = Target(heading=90.0, altitude_ft=3000.0, speed_kt=140.0,
               turn="left", hold=True)
    t.clear()
    assert t.heading is None
    assert t.altitude_ft is None
    assert t.speed_kt is None
    assert t.turn == ""
    assert t.hold is False


def test_command_detail():
    c = Command(kind="heading", values={"heading": 270})
    assert c.detail == "270"
    c = Command(kind="altitude", values={"altitude_ft": 3000})
    assert c.detail == "3,000"
    c = Command(kind="speed", values={"speed_kt": 140})
    assert c.detail == "140"
    c = Command(kind="takeoff")
    assert c.detail == ""


# ----------------------------------------------------------------------
# FlightState / FlightConfig

def test_flight_state_defaults():
    s = FlightState()
    assert s.latitude == 0.0
    assert s.longitude == 0.0
    assert s.altitude_ft == 0.0
    assert s.heading == 0.0
    assert s.ground_speed_kt == 0.0
    assert s.vertical_rate_fpm == 0.0
    assert s.on_ground is True


def test_flight_config_defaults():
    c = FlightConfig()
    assert c.turn_rate == RATE_ONE
    assert c.climb_fpm == 2200.0
    assert c.approach_kt == 140.0


def test_flight_config_is_frozen():
    c = FlightConfig()
    with pytest.raises(Exception):
        c.turn_rate = 5.0


# ----------------------------------------------------------------------
# FlightModel

def test_flight_model_hold_stops():
    model = FlightModel(
        state=FlightState(ground_speed_kt=20.0, on_ground=True),
        config=FlightConfig(braking_kt_s=4.5),
    )
    model.step(Target(hold=True), 5.0)
    assert model.state.ground_speed_kt == 0.0
    assert model.state.vertical_rate_fpm == 0.0


def test_flight_model_accelerates():
    model = FlightModel(
        state=FlightState(ground_speed_kt=100.0),
        config=FlightConfig(accelerate_kt_s=3.0, decelerate_kt_s=2.0),
    )
    model.step(Target(speed_kt=140.0), 10.0)
    assert model.state.ground_speed_kt == pytest.approx(130.0)


def test_flight_model_turns():
    model = FlightModel(
        state=FlightState(heading=90.0, on_ground=False),
        config=FlightConfig(turn_rate=3.0),
    )
    model.step(Target(heading=180.0), 10.0)
    assert model.state.heading == pytest.approx(120.0)


def test_flight_model_climbs():
    model = FlightModel(
        state=FlightState(altitude_ft=1800.0, vertical_rate_fpm=1000.0),
        config=FlightConfig(climb_fpm=2200.0),
    )
    model.step(Target(altitude_ft=3000.0), 10.0)
    climb = min(2200.0, 1000.0 + 450.0 * 10.0) / 60.0 * 10.0
    assert model.state.altitude_ft == pytest.approx(1800.0 + climb)


def test_flight_model_moves_along_heading():
    model = FlightModel(
        state=FlightState(lat=50.0, lon=1.0, heading=90.0,
                          ground_speed_kt=180.0),
        config=FlightConfig(),
    )
    model.step(Target(), 60.0)
    assert model.state.latitude == pytest.approx(50.0)
    assert model.state.longitude > 1.0


def test_flight_model_zero_dt_no_op():
    model = FlightModel(
        state=FlightState(heading=90.0, ground_speed_kt=100.0),
        config=FlightConfig(),
    )
    before = (model.state.latitude, model.state.longitude, model.state.heading,
              model.state.ground_speed_kt, model.state.altitude_ft)
    model.step(Target(speed_kt=140.0), 0.0)
    after = (model.state.latitude, model.state.longitude, model.state.heading,
             model.state.ground_speed_kt, model.state.altitude_ft)
    assert before == after


def test_flight_model_step_matches_module_step():
    """The module-level step() shim must produce the same result as
    FlightModel.step() given the same inputs."""
    from wilcoatc.atc.traffic import Flight as TrafficFlight

    flight = TrafficFlight(
        written="TEST", type_code="A320", arriving=False,
        latitude=50.9, longitude=1.4, heading=90.0,
        ground_speed_kt=100.0, altitude_ft=1800.0,
        vertical_rate_fpm=500.0, on_ground=False,
        performance=AIRLINER,
    )
    target = Target(heading=180.0, altitude_ft=3000.0, speed_kt=140.0)
    step(flight, target, AIRLINER, 10.0)

    model = FlightModel(
        state=FlightState(latitude=50.9, longitude=1.4, heading=90.0,
                          ground_speed_kt=100.0, altitude_ft=1800.0,
                          vertical_rate_fpm=500.0, on_ground=False),
        config=FlightConfig(turn_rate=AIRLINER.turn_rate,
                            climb_fpm=AIRLINER.climb_fpm,
                            descend_fpm=AIRLINER.descend_fpm,
                            accelerate_kt_s=AIRLINER.accelerate_kt_s,
                            decelerate_kt_s=AIRLINER.decelerate_kt_s,
                            takeoff_kt_s=AIRLINER.takeoff_kt_s,
                            taxi_kt=AIRLINER.taxi_kt,
                            rotate_kt=AIRLINER.rotate_kt,
                            climb_kt=AIRLINER.climb_kt,
                            cruise_kt=AIRLINER.cruise_kt,
                            approach_kt=AIRLINER.approach_kt,
                            braking_kt_s=AIRLINER.braking_kt_s,
                            pattern_agl_ft=AIRLINER.pattern_agl_ft,
                            range_nm=AIRLINER.range_nm,
                            ceiling_ft=AIRLINER.ceiling_ft),
    )
    model.step(target, 10.0)

    assert flight.latitude == pytest.approx(model.state.latitude)
    assert flight.longitude == pytest.approx(model.state.longitude)
    assert flight.altitude_ft == pytest.approx(model.state.altitude_ft)
    assert flight.heading == pytest.approx(model.state.heading)
    assert flight.ground_speed_kt == pytest.approx(model.state.ground_speed_kt)
    assert flight.vertical_rate_fpm == pytest.approx(model.state.vertical_rate_fpm)
    assert flight.on_ground == model.state.on_ground


# ----------------------------------------------------------------------
# flight_config_for

def test_flight_config_for_known_type():
    config = flight_config_for("A320")
    assert config.climb_fpm == pytest.approx(2500.0)
    assert config.climb_kt == pytest.approx(175.0)
    assert config.cruise_kt == pytest.approx(290.0)
    assert config.approach_kt == pytest.approx(137.0)
    assert config.range_nm == pytest.approx(2700.0)
    assert config.ceiling_ft == pytest.approx(39000.0)


def test_flight_config_for_unknown_type():
    config = flight_config_for("ZZZZ")
    assert config.rotate_kt == pytest.approx(140.0)  # airliner default
    assert config.climb_fpm == pytest.approx(2200.0)


def test_flight_config_for_light_type():
    config = flight_config_for("C172", light=True)
    assert config.rotate_kt == pytest.approx(55.0)  # from vspeeds table


def test_flight_config_for_turboprop():
    config = flight_config_for("AT76")
    assert config.climb_fpm == pytest.approx(1500.0)
    assert config.approach_kt == pytest.approx(200.0)

