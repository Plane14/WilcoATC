"""Golden tests for wilcoatc.atc.approach_flight: the approach recipe.

profile_alt, legs, sink_fpm, frame_point and the ApproachRecipe dataclass
are pure geometry and need no simulator or session.
"""

from __future__ import annotations

import math

import pytest

from wilcoatc.atc.approach_flight import (
    ApproachRecipe, circle_legs, frame_point, legs, profile_alt, sink_fpm,
)


def _recipe(**overrides):
    data = dict(
        ident="EGTEST", name="ILS Z", family="ils",
        approach_runway="09L", landing_runway="09L", line="",
        minima_ft=200.0, circle_minima_ft=0.0,
        final_course=90.0,
        threshold_lat=50.9, threshold_lon=1.4,
        field_elev_ft=180.0,
        faf_lat=50.9, faf_lon=1.3, faf_alt_ft=1770.0,
        missed_alt_ft=2500.0, navaid=None,
        circling=False, back_course=False, why="",
    )
    data.update(overrides)
    return ApproachRecipe(**data)


# ----------------------------------------------------------------------
# profile_alt

def test_profile_alt_three_degree():
    r = _recipe()
    assert profile_alt(r, 5.0) == pytest.approx(180.0 + 5.0 * 318.0)


def test_profile_alt_at_threshold():
    r = _recipe()
    assert profile_alt(r, 0.0) == pytest.approx(180.0)


def test_profile_alt_negative_distance_clamped():
    r = _recipe()
    assert profile_alt(r, -3.0) == pytest.approx(180.0)


# ----------------------------------------------------------------------
# legs

def test_legs_three_points():
    r = _recipe()
    points = legs(r)
    assert len(points) == 3
    gate, faf, end = points
    # The gate is GATE_NM behind the FAF.
    gate_dist = math.hypot(
        (gate[0] - r.faf_lat) * 111320.0,
        (gate[1] - r.faf_lon) * 111320.0 * math.cos(math.radians(50.9)),
    ) / 1852.0
    assert gate_dist == pytest.approx(3.0, abs=0.1)
    # The end is PAST_THRESHOLD_NM past the threshold on course.
    end_dist = math.hypot(
        (end[0] - r.threshold_lat) * 111320.0,
        (end[1] - r.threshold_lon) * 111320.0 * math.cos(math.radians(50.9)),
    ) / 1852.0
    assert end_dist == pytest.approx(0.3, abs=0.05)


# ----------------------------------------------------------------------
# sink_fpm

def test_sink_fpm_three_degree():
    # At 140 kt, a three-degree path is 140 * 318 / 60 fpm.
    assert sink_fpm(140.0) == pytest.approx(140.0 * 318.0 / 60.0)


def test_sink_fpm_clamped():
    assert sink_fpm(999.0) == pytest.approx(900.0)
    assert sink_fpm(10.0) == pytest.approx(250.0)


# ----------------------------------------------------------------------
# frame_point

def test_frame_point_out_and_side():
    lat, lon = frame_point(50.9, 1.4, 90.0, 5.0, 1.0)
    # Five miles out on the extended centreline (east of the threshold on a
    # 09 course), then one mile right (south).
    assert lat < 50.9
    assert lon > 1.4


def test_frame_point_negative_out_behind():
    lat, lon = frame_point(50.9, 1.4, 90.0, -2.0, 0.0)
    # Behind the threshold: west of it on a 09 course (course 270).
    assert lon < 1.4


# ----------------------------------------------------------------------
# ApproachRecipe dataclass

def test_approach_recipe_defaults():
    r = ApproachRecipe(ident="EG", name="VOR-A", family="vor",
                       approach_runway="09", landing_runway="27",
                       final_course=90.0, threshold_lat=50.0, threshold_lon=1.0,
                       field_elev_ft=200.0)
    assert r.minima_ft == 200.0
    assert r.circle_minima_ft == 0.0
    assert r.faf_alt_ft == 0.0
    assert r.navaid is None
    assert r.has_navaid is False


def test_approach_recipe_has_navaid():
    from wilcoatc.atc.approach_flight import Navaid
    r = _recipe(navaid=Navaid(ident="I-YY", freq_mhz=110.5,
                              lat=50.9, lon=1.4, kind="LOC"))
    assert r.has_navaid is True
    r = _recipe(navaid=None)
    assert r.has_navaid is False
    r = _recipe(navaid=Navaid(ident="X", freq_mhz=0.0, lat=0.0, lon=0.0))
    assert r.has_navaid is False


# ----------------------------------------------------------------------
# circle_legs (needs a final_for stub; skipped without a real navdb)

def test_circle_legs_needs_navdb():
    r = _recipe(circling=True)
    class FakeNavDB:
        def runways(self, ident):
            return []
    assert circle_legs(r, FakeNavDB(), None) is None
