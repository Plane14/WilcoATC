"""Golden tests for wilcoatc.atc.auto_nav: pure geometry, no simulator.

path_steering, runway_end, glide and align answer where to point from a
position and a route, and each answer is checked against a hand-computed
expectation.
"""

from __future__ import annotations

import math

import pytest

from wilcoatc.atc.auto_nav import (
    TAXI_LOOKAHEAD_M, align, glide, path_steering, runway_end,
)


def _rw(le_ident="09L", he_ident="27R", le_heading=90.0, he_heading=270.0,
        le_lat=50.9, le_lon=1.4, he_lat=50.9, he_lon=1.42,
        le_elev_ft=180.0, he_elev_ft=190.0, length_ft=3500.0,
        le_ils=None, he_ils=None, le_ils_kind=None, he_ils_kind=None,
        le_closed=False, he_closed=False, width_ft=45.0):
    class R:
        pass
    rw = R()
    rw.le_ident = le_ident
    rw.he_ident = he_ident
    rw.le_heading = le_heading
    rw.he_heading = he_heading
    rw.le_lat = le_lat
    rw.le_lon = le_lon
    rw.he_lat = he_lat
    rw.he_lon = he_lon
    rw.le_elev_ft = le_elev_ft
    rw.he_elev_ft = he_elev_ft
    rw.length_ft = length_ft
    rw.le_ils = le_ils
    rw.he_ils = he_ils
    rw.le_ils_kind = le_ils_kind
    rw.he_ils_kind = he_ils_kind
    rw.le_closed = le_closed
    rw.he_closed = he_closed
    rw.width_ft = width_ft

    def ends():
        return [(le_ident, le_heading, le_lat, le_lon),
                (he_ident, he_heading, he_lat, he_lon)]

    rw.ends = ends
    return rw


# ----------------------------------------------------------------------
# path_steering

def test_path_steering_empty():
    result = path_steering(50.0, 1.0, [])
    assert result["done"] is True
    assert result["heading"] is None
    assert result["cross_track_m"] == 0.0
    assert result["remaining_m"] == 0.0
    assert result["nearest"] == 0


def test_path_steering_single_point():
    result = path_steering(50.0, 1.0, [(50.01, 1.01)])
    assert result["done"] is False
    assert result["heading"] is not None
    assert result["cross_track_m"] == 0.0
    assert result["nearest"] == 0


def test_path_steering_on_path():
    # Two points a mile apart on an east-west line; the aeroplane is on the
    # line halfway between them.
    path = [(50.0, 1.0), (50.0, 1.0 + 1.0 / 60.0)]
    result = path_steering(50.0, 1.0 + 0.5 / 60.0, path)
    assert result["done"] is False
    assert abs(result["cross_track_m"]) < 1.0
    assert result["heading"] is not None
    assert result["remaining_m"] > 0.0


def test_path_steering_off_path():
    path = [(50.0, 1.0), (50.0, 1.0 + 1.0 / 60.0)]
    result = path_steering(50.0 + 0.001, 1.0 + 0.5 / 60.0, path)
    assert result["cross_track_m"] != 0.0
    # North of an eastbound track is to the left, so negative.
    assert result["cross_track_m"] < 0.0


def test_path_steering_done_at_end():
    path = [(50.0, 1.0), (50.0, 1.0 + 0.001 / 60.0)]
    result = path_steering(50.0, 1.0 + 0.002 / 60.0, path)
    assert result["done"] is True


# ----------------------------------------------------------------------
# runway_end

def test_runway_end_by_designation():
    runways = [_rw(le_ident="09L", he_ident="27R")]
    found = runway_end(runways, "09L")
    assert found is not None
    lat, lon, heading, length = found
    assert lat == pytest.approx(50.9)
    assert lon == pytest.approx(1.4)
    assert heading == pytest.approx(90.0)
    assert length == pytest.approx(3500.0)


def test_runway_end_other_end():
    runways = [_rw(le_ident="09L", he_ident="27R")]
    found = runway_end(runways, "27R")
    assert found is not None
    _, _, heading, _ = found
    assert heading == pytest.approx(270.0)


def test_runway_end_by_digits():
    runways = [_rw(le_ident="09L", he_ident="27R")]
    found = runway_end(runways, "09")
    assert found is not None
    _, _, heading, _ = found
    assert heading == pytest.approx(90.0)


def test_runway_end_missing():
    runways = [_rw()]
    assert runway_end(runways, "18") is None
    assert runway_end(runways, "") is None


def test_runway_end_no_coords():
    class R:
        le_ident = "09"
        he_ident = "27"
        le_lat = None
        le_lon = None
        he_lat = None
        he_lon = None
        length_ft = 3500.0
    assert runway_end([R()], "09") is None


# ----------------------------------------------------------------------
# glide

def test_glide():
    assert glide(50.0, 1.0, 180.0, 5.0) == pytest.approx(180.0 + 5.0 * 318.0)


def test_glide_negative_distance():
    assert glide(50.0, 1.0, 180.0, -1.0) == 180.0


# ----------------------------------------------------------------------
# align

def test_align_on_centreline():
    # One mile out on the centreline of runway 09 (heading 90).
    lat, lon = 50.9, 1.4 + 1.0 / 60.0
    result = align(lat, lon, 50.9, 1.4, 90.0)
    assert abs(result["offset_m"]) < 1.0
    assert abs(result["correction_deg"]) < 1.0
    assert result["along_m"] > 0.0


def test_align_right_of_centreline():
    # One mile out, a bit to the right (south) of the eastbound centreline.
    lat, lon = 50.9 - 0.001, 1.4 + 1.0 / 60.0
    result = align(lat, lon, 50.9, 1.4, 90.0)
    assert result["offset_m"] > 0.0
    # Right of track: correction is negative (come left).
    assert result["correction_deg"] < 0.0


def test_align_behind_threshold():
    # Behind the threshold there is no correction.
    lat, lon = 50.9, 1.4 - 1.0 / 60.0
    result = align(lat, lon, 50.9, 1.4, 90.0)
    assert result["correction_deg"] == 0.0
    assert result["along_m"] <= 0.0


def test_align_defaults():
    assert TAXI_LOOKAHEAD_M == 120.0
