"""Golden tests for wilcoatc.atc.vspeeds: the speeds table.

table_speeds, speeds_for, rotate_kt and the Speeds/Stated dataclasses are
pure and need no simulator.
"""

from __future__ import annotations

from wilcoatc.atc.vspeeds import (
    AIRLINER,
    CLASS_FIGURES,
    TRAFFIC_VR,
    Speeds,
    Stated,
    rotate_kt,
    speeds_for,
    table_speeds,
)

# ----------------------------------------------------------------------
# Speeds / Stated dataclasses

def test_speeds_defaults():
    s = Speeds(vr=60.0)
    assert s.vr == 60.0
    assert s.v1 is None
    assert s.v2 is None
    assert s.vapp is None
    assert s.source == ""
    assert s.unverified is False


def test_stated_defaults():
    s = Stated()
    assert s.v1 is None
    assert s.vr is None
    assert s.v2 is None
    assert s.vapp is None


# ----------------------------------------------------------------------
# table_speeds

def test_table_speeds_known_type():
    s = table_speeds("C172")
    assert s.vr == 55.0
    assert s.vapp == 65.0
    assert s.source.startswith("Cessna 172S")


def test_table_speeds_unknown_type_falls_back_to_class():
    s = table_speeds("ZZZZ")
    assert s.vr == CLASS_FIGURES[AIRLINER].vr
    assert s.source == CLASS_FIGURES[AIRLINER].source


def test_table_speeds_alias():
    assert table_speeds("P28A") is table_speeds("PA28")


# ----------------------------------------------------------------------
# speeds_for

def test_speeds_for_table_only():
    s = speeds_for("C172", AIRLINER)
    assert s.vr == 55.0
    assert s.vapp == 65.0


def test_speeds_for_sim_vr_overrides():
    s = speeds_for("A320", AIRLINER, min_rotation_kt=150.0)
    assert s.vr == 150.0
    assert s.unverified is False
    assert "simulator" in s.source


def test_speeds_for_stated_overrides():
    s = speeds_for("A320", AIRLINER,
                   stated=Stated(v1=130.0, vr=135.0, v2=145.0, vapp=140.0))
    assert s.v1 == 130.0
    assert s.vr == 135.0
    assert s.v2 == 145.0
    assert s.vapp == 140.0


def test_speeds_for_stated_partial():
    s = speeds_for("A320", AIRLINER, stated=Stated(vr=150.0))
    assert s.vr == 150.0
    assert s.v1 is None  # not stated, table has none


def test_speeds_for_design_takeoff():
    s = speeds_for("B738", AIRLINER, design_takeoff_kt=150.0)
    assert s.vr == 150.0
    assert s.unverified is False


# ----------------------------------------------------------------------
# rotate_kt

def test_rotate_kt_from_table():
    assert rotate_kt("C172") == 55.0


def test_rotate_kt_from_traffic_vr():
    assert rotate_kt("B738") == 145.0


def test_rotate_kt_none_for_unknown():
    assert rotate_kt("ZZZZ") is None


def test_traffic_vr_contains_b738():
    assert "B738" in TRAFFIC_VR
