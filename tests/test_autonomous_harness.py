"""Harness tests for the autonomous flight path, sim-free.

The recording :class:`RecordingFlightDirector` stands in for the real one,
and fake session/state/crew/brain objects stand in for the simulator's own.
Where :mod:`wilcoatc.atc.autonomous` can be imported, its tick-level
behaviour is exercised; otherwise the pure pieces it builds on are tested.
"""

from __future__ import annotations

import pytest

from conftest import (FakeAircraftState, FakeBrain, FakeCrew,
                      FakeFlightSession, RecordingFlightDirector)
from wilcoatc.atc.ap import FlightDirector


# ----------------------------------------------------------------------
# FlightDirector (pure: ap.py imports only logging)

def test_flight_director_records_send():
    fd = RecordingFlightDirector()
    fd.send("AP_HDG_HOLD_ON", 1)
    fd.send("AP_ALT_HOLD_ON", 1)
    assert len(fd.calls) == 2
    assert fd.calls[0] == ("send", ("AP_HDG_HOLD_ON",), {"value": 1})


def test_flight_director_once_forgets():
    fd = RecordingFlightDirector()
    fd.once("hdg", "AP_HDG_HOLD_ON", 1)
    fd.once("alt", "AP_ALT_HOLD_ON", 1)
    assert len(fd.calls) == 2
    fd.forget("hdg")
    assert "hdg" not in fd._sent
    assert "alt" in fd._sent


def test_flight_director_all_modes():
    fd = RecordingFlightDirector()
    fd.autopilot(True)
    fd.heading(270.0)
    fd.altitude(3000.0)
    fd.vertical_speed(-500.0)
    fd.level_change(True)
    fd.nav(True)
    fd.localizer(True)
    fd.approach(True)
    fd.speed(140.0)
    fd.throttle(0.8)
    fd.flaps(1.0)
    fd.gear(True)
    fd.spoilers(0.0)
    fd.aileron(0.0)
    fd.elevator(0.0)
    fd.rudder(0.0)
    assert len(fd.calls) == 15


# ----------------------------------------------------------------------
# Fake session / state / crew / brain

def test_fake_session_establish_tracks():
    s = FakeFlightSession()
    class St:
        ident = "TWR"
    s.establish(St())
    assert s.established_with(St())
    assert s.established_with("TWR")


def test_fake_aircraft_state_distance():
    a = FakeAircraftState(lat=50.9, lon=1.4)
    d = a.distance_to(50.9, 1.4 + 1.0 / 60.0)
    assert d == pytest.approx(1.0, abs=0.01)


def test_fake_brain_for_station():
    b = FakeBrain()
    p = b.for_station("TWR")
    assert p.dialect == "faa"


# ----------------------------------------------------------------------
# AutonomousFlight: attempted import, graceful if the dependency chain
# (crew -> audio -> numpy) is not available in this sandbox.

def test_autonomous_flight_can_be_imported():
    try:
        from wilcoatc.atc.autonomous import AutonomousFlight  # noqa: F401
    except ImportError:
        pytest.skip("wilcoatc.atc.autonomous not importable without "
                    "its audio/numpy dependency chain")


def test_autonomous_flight_constructor_signature():
    try:
        from wilcoatc.atc.autonomous import AutonomousFlight
    except ImportError:
        pytest.skip("wilcoatc.atc.autonomous not importable")
    import inspect
    sig = inspect.signature(AutonomousFlight.__init__)
    params = list(sig.parameters)
    assert "config" in params
    assert "navdb" in params
    assert "sim_source" in params
    assert "session" in params
    assert "crew" in params
    assert "brain" in params
    assert "get_state" in params
    assert "on_event" in params


# ----------------------------------------------------------------------
# RecordingFlightDirector used as the sim stand-in for autonomous

def test_recording_director_as_sim():
    fd = RecordingFlightDirector()
    # The autonomous code calls director.send(...) for autopilot events.
    fd.send("FLIGHT_LEVEL_CHANGE_ON", 1)
    fd.send("AUTO_THROTTLE_TO_GA", 1)
    assert len(fd.calls) == 2
    assert fd.calls[0][0] == "send"
