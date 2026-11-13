"""Golden tests for wilcoatc.atc.perftable: per-type traffic performance.

traffic_performance, type_perf and range_for are pure and need no simulator.
"""

from __future__ import annotations

import pytest

from wilcoatc.atc.flying import AIRLINER
from wilcoatc.atc.perftable import (
    HELI_TYPES,
    TABLE,
    TypePerf,
    range_for,
    traffic_performance,
    type_perf,
)

# ----------------------------------------------------------------------
# traffic_performance

def test_traffic_performance_known_type():
    p = traffic_performance("A320")
    assert p.climb_fpm == pytest.approx(2500.0)
    assert p.climb_kt == pytest.approx(175.0)
    assert p.cruise_kt == pytest.approx(290.0)
    assert p.approach_kt == pytest.approx(137.0)
    assert p.range_nm == pytest.approx(2700.0)
    assert p.ceiling_ft == pytest.approx(39000.0)


def test_traffic_performance_unknown_type_is_class():
    p = traffic_performance("ZZZZ")
    assert p.rotate_kt == pytest.approx(AIRLINER.rotate_kt)


def test_traffic_performance_light_type():
    p = traffic_performance("C172", light=True)
    assert p is not AIRLINER


def test_traffic_performance_takeoff_accel():
    p = traffic_performance("A320")
    assert p.takeoff_kt_s > 0.0


def test_traffic_performance_heli_uses_approach_ias():
    p = traffic_performance("H125")
    assert p.approach_kt == pytest.approx(60.0)  # from perftable approach_ias_kt


# ----------------------------------------------------------------------
# type_perf

def test_type_perf_known():
    perf = type_perf("A320")
    assert isinstance(perf, TypePerf)
    assert perf.db == "A320"
    assert perf.v2_kt == 145.0
    assert perf.takeoff_m == 2190.0


def test_type_perf_unknown():
    assert type_perf("ZZZZ") is None


# ----------------------------------------------------------------------
# range_for

def test_range_for_known_type():
    assert range_for("A320") == pytest.approx(2700.0)


def test_range_for_unknown_type():
    r = range_for("ZZZZ")
    assert r > 0.0


def test_range_for_heli():
    r = range_for("H125")
    assert r == pytest.approx(324.0)  # from perftable range_nm


# ----------------------------------------------------------------------
# HELI_TYPES / TABLE sanity

def test_heli_types_contains_known_rotorcraft():
    assert "H125" in HELI_TYPES
    assert "R44" in HELI_TYPES


def test_table_has_types():
    assert "A320" in TABLE
    assert "B738" in TABLE
    assert len(TABLE) > 50
