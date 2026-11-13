"""Sim-free harness tests for wilcoatc.atc.traffic.TrafficInjector.

A fake navdb with one airport and one runway stands in for the real
database. Flights are put on the roster by hand and advanced with
``advance`` / ``_fly`` so the flight model's movement is exercised without
a simulator. Nothing here touches SimConnect.
"""

from __future__ import annotations

import pytest

from wilcoatc.atc.flying import (
    Performance, Target, traffic_performance,
)
from wilcoatc.atc.traffic import Flight

from conftest import (make_injector, snapshot_of)


# ----------------------------------------------------------------------
# construction and field

def test_injector_has_fake_field():
    inject = make_injector()
    assert inject.field.ident == "EGTEST"
    assert inject.field.latitude == 50.9
    assert inject.field.longitude == 1.4
    assert inject.requires_sim is False
    assert inject.can_invent is True


def test_set_field_controlled():
    inject = make_injector(controlled=True)
    assert "TWR" in inject.field.stations
    assert inject.field.stations["TWR"].is_controller


# ----------------------------------------------------------------------
# _fly moves a takeoff flight

def _takeoff_flight(written="TEST1", lat=50.9, lon=1.4,
                    heading=90.0, gs=50.0, alt=180.0,
                    type_code="A320",
                    perf: Performance | None = None) -> Flight:
    f = Flight(
        written=written, type_code=type_code,
        arriving=False, runway="09L",
        latitude=lat, longitude=lon, heading=heading,
        ground_speed_kt=gs, altitude_ft=alt, on_ground=True,
        performance=perf or traffic_performance(type_code),
    )
    f.airborne_model = True
    f.phase = "takeoff"
    f.target = Target()
    f.next_at = 0.0
    return f


def test_fly_accelerates_takeoff():
    inject = make_injector()
    now = 1000.0
    f = _takeoff_flight(gs=0.0)
    inject.flights.append(f)
    # First call: elapsed is 0, nothing moves.
    inject._fly(now)
    assert inject._flown_at == now
    # Second call: 10 seconds later, the roll accelerates from 0
    # (_start_flying resets speed on the first real tick).
    inject._fly(now + 10.0)
    assert f.ground_speed_kt > 0.0
    assert f.on_ground is True  # not rotated yet


def test_fly_rotates_and_lifts_off():
    inject = make_injector()
    now = 1000.0
    f = _takeoff_flight(gs=0.0)
    inject.flights.append(f)
    inject._fly(now)
    # Roll to rotate speed (~140 kt for A320 at 4.2 kt/s).
    # _start_flying resets speed to 0 on the first real tick, so this
    # accelerates from a standstill.
    for _ in range(40):
        if f.ground_speed_kt >= f.performance.rotate_kt:
            break
        if not f.on_ground:
            break
        inject._fly(now)
        now += 5.0
    assert f.ground_speed_kt >= f.performance.rotate_kt
    # Rotate, then liftoff after LIFTOFF_AFTER_S (~3.2 s).
    for _ in range(20):
        inject._fly(now)
        now += 1.0
    assert f.on_ground is False


def test_fly_climbs_after_takeoff():
    inject = make_injector()
    now = 1000.0
    f = _takeoff_flight(gs=140.0, alt=180.0)
    f.on_ground = False
    inject.flights.append(f)
    inject._fly(now)
    inject._fly(now + 60.0)
    assert f.altitude_ft > 180.0
    assert f.vertical_rate_fpm > 0.0


# ----------------------------------------------------------------------
# physics == "model" path

def test_fly_model_uses_physics_model():
    inject = make_injector()
    inject.physics = "model"
    now = 1000.0
    f = _takeoff_flight(gs=0.0)
    inject.flights.append(f)
    inject._fly(now)
    # First call creates the physics_model.
    assert f.physics_model is not None
    # Second call uses the existing one (not recreated).
    model_before = f.physics_model
    inject._fly(now + 10.0)
    assert f.physics_model is model_before
    assert f.ground_speed_kt > 0.0


def test_fly_model_accelerates_and_lifts_off():
    inject = make_injector()
    inject.physics = "model"
    now = 1000.0
    f = _takeoff_flight(gs=0.0)
    inject.flights.append(f)
    inject._fly(now)
    # Roll to rotate speed.
    for _ in range(40):
        if f.ground_speed_kt >= f.performance.rotate_kt:
            break
        if not f.on_ground:
            break
        inject._fly(now)
        now += 5.0
    assert f.ground_speed_kt >= f.performance.rotate_kt
    for _ in range(20):
        inject._fly(now)
        now += 1.0
    assert f.on_ground is False


def test_fly_model_climbs():
    inject = make_injector()
    inject.physics = "model"
    now = 1000.0
    f = _takeoff_flight(gs=140.0, alt=180.0)
    f.on_ground = False
    inject.flights.append(f)
    inject._fly(now)
    inject._fly(now + 60.0)
    assert f.altitude_ft > 180.0
    assert f.vertical_rate_fpm > 0.0


def test_fly_model_uses_perftable_config():
    inject = make_injector()
    inject.physics = "model"
    now = 1000.0
    f = _takeoff_flight(gs=0.0, type_code="C172")
    f.performance = traffic_performance("C172", light=True)
    inject.flights.append(f)
    inject._fly(now)
    assert f.physics_model is not None
    # C172 rotate speed from vspeeds table is 55 kt.
    assert f.physics_model.config.rotate_kt == pytest.approx(55.0)


# ----------------------------------------------------------------------
# advance

def test_advance_runs_without_spawning():
    inject = make_injector(density=0.0)
    now = 1000.0
    f = _takeoff_flight(gs=0.0)
    inject.flights.append(f)
    inject.advance(now)
    inject.advance(now + 10.0)
    # _start_flying resets speed to 0 on the first real tick, then the roll
    # accelerates. Just check it is moving.
    assert f.ground_speed_kt > 0.0


def test_advance_removes_done_flights():
    inject = make_injector(density=0.0)
    now = 1000.0
    f = _takeoff_flight(gs=100.0)
    f.done = True
    inject.flights.append(f)
    inject.advance(now)
    assert f not in inject.flights


# ----------------------------------------------------------------------
# positions

def test_positions_returns_snapshot():
    inject = make_injector(density=0.0)
    now = 1000.0
    f = _takeoff_flight(gs=100.0, lat=50.9, lon=1.4)
    inject.flights.append(f)
    inject.advance(now)
    snaps = inject.positions(invented_only=True)
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap["callsign"] == "TEST1"
    assert snap["lat"] == pytest.approx(50.9)
    assert snap["lon"] == pytest.approx(1.4)
    assert snap["phase"] == "takeoff"
    assert snap["on_ground"] is True


def test_positions_excludes_done():
    inject = make_injector(density=0.0)
    f = _takeoff_flight(gs=100.0)
    f.done = True
    inject.flights.append(f)
    snaps = inject.positions(invented_only=True)
    assert snaps == []


# ----------------------------------------------------------------------
# traffic_ahead

def test_traffic_ahead_arrivals():
    inject = make_injector(density=0.0)
    f = _takeoff_flight()
    f.arriving = True
    f.phase = "final"
    f.distance_nm = 5.0
    inject.flights.append(f)
    ahead = inject.traffic_ahead(arriving=True)
    assert len(ahead) == 1
    assert ahead[0].written == "TEST1"


def test_traffic_ahead_departures():
    inject = make_injector(density=0.0)
    f = _takeoff_flight()
    f.arriving = False
    f.phase = "lineup"
    f.sequence = 0
    inject.flights.append(f)
    ahead = inject.traffic_ahead(arriving=False)
    assert len(ahead) == 1


# ----------------------------------------------------------------------
# runway_is_busy

def test_runway_is_busy_false_initially():
    inject = make_injector(density=0.0)
    assert inject.runway_is_busy(1000.0) is False


def test_runway_is_busy_after_takeoff_clearance():
    inject = make_injector(density=0.0)
    now = 1000.0
    inject.runway_free_at = now + 60.0
    assert inject.runway_is_busy(now) is True
    assert inject.runway_is_busy(now + 120.0) is False


# ----------------------------------------------------------------------
# snapshot_of helper

def test_snapshot_of_shape():
    inject = make_injector(density=0.0)
    f = _takeoff_flight(gs=100.0, lat=50.9, lon=1.4)
    inject.flights.append(f)
    inject.advance(1000.0)
    snap = snapshot_of(inject)
    assert "TEST1" in snap
    entry = snap["TEST1"]
    assert "phase" in entry
    assert "lat" in entry
    assert "lon" in entry
    assert "altitude_ft" in entry
    assert "ground_speed_kt" in entry
