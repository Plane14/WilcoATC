"""Record baseline traces of pure-module outputs for regression detection.

Run this before a refactor to capture expected values, then after to diff
against. Nothing here touches the simulator; every value is computed from
pure functions with hand-built inputs.

Usage (when Python is available):

    python tests/record_traces.py > tests/baseline_traces.txt
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wilcoatc.atc.approach_flight import ApproachRecipe, legs, profile_alt
from wilcoatc.atc.auto_nav import align, glide, path_steering, runway_end
from wilcoatc.atc.flying import (AIRLINER, LIGHT, TURBOPROP, offset_from_track,
                                 performance_for, step, turn_towards)
from wilcoatc.atc.perftable import range_for, traffic_performance, type_perf
from wilcoatc.atc.vspeeds import rotate_kt, speeds_for, table_speeds


class _Flight:
    def __init__(self, lat=50.0, lon=1.0, altitude_ft=180.0, heading=90.0,
                 ground_speed_kt=100.0, vertical_rate_fpm=0.0,
                 on_ground=False):
        self.latitude = lat
        self.longitude = lon
        self.altitude_ft = altitude_ft
        self.heading = heading
        self.ground_speed_kt = ground_speed_kt
        self.vertical_rate_fpm = vertical_rate_fpm
        self.on_ground = on_ground


def _rw(**kw):
    class R:
        pass
    defaults = dict(le_ident="09L", he_ident="27R", le_heading=90.0,
                    he_heading=270.0, le_lat=50.9, le_lon=1.4,
                    he_lat=50.9, he_lon=1.42, le_elev_ft=180.0,
                    he_elev_ft=190.0, length_ft=3500.0)
    defaults.update(kw)
    rw = R()
    for k, v in defaults.items():
        setattr(rw, k, v)

    def ends():
        return [(rw.le_ident, rw.le_heading, rw.le_lat, rw.le_lon),
                (rw.he_ident, rw.he_heading, rw.he_lat, rw.he_lon)]

    rw.ends = ends
    return rw


def trace(name, value):
    print(json.dumps({"name": name, "value": value}))


def main():
    # --- turn_towards ---
    trace("turn_towards_short", turn_towards(10.0, 350.0, 3.0, 1.0))
    trace("turn_towards_rate_limited", turn_towards(10.0, 100.0, 3.0, 1.0))
    trace("turn_towards_left", turn_towards(10.0, 350.0, 3.0, 1.0, "left"))
    trace("turn_towards_right", turn_towards(350.0, 10.0, 3.0, 1.0, "right"))

    # --- step: hold ---
    f = _Flight(ground_speed_kt=20.0, on_ground=True)
    step(f, type("T", (), {"hold": True, "turn": "", "accel_kt_s": None,
                           "max_descent_fpm": None,
                           "heading": None, "altitude_ft": None,
                           "speed_kt": None, "intent": ""})(),
         AIRLINER, 5.0)
    trace("step_hold_speed", f.ground_speed_kt)

    # --- step: speed ---
    f = _Flight(ground_speed_kt=100.0)
    step(f, type("T", (), {"hold": False, "turn": "", "accel_kt_s": None,
                           "max_descent_fpm": None,
                           "heading": None, "altitude_ft": None,
                           "speed_kt": 140.0, "intent": ""})(),
         AIRLINER, 10.0)
    trace("step_speed_accelerate", f.ground_speed_kt)

    f = _Flight(ground_speed_kt=200.0)
    step(f, type("T", (), {"hold": False, "turn": "", "accel_kt_s": None,
                           "max_descent_fpm": None,
                           "heading": None, "altitude_ft": None,
                           "speed_kt": 140.0, "intent": ""})(),
         AIRLINER, 10.0)
    trace("step_speed_decelerate", f.ground_speed_kt)

    # --- step: heading ---
    f = _Flight(heading=90.0)
    step(f, type("T", (), {"hold": False, "turn": "", "accel_kt_s": None,
                           "max_descent_fpm": None,
                           "heading": 180.0, "altitude_ft": None,
                           "speed_kt": None, "intent": ""})(),
         AIRLINER, 10.0)
    trace("step_heading", f.heading)

    # --- step: altitude ---
    f = _Flight(altitude_ft=1800.0, vertical_rate_fpm=1000.0)
    step(f, type("T", (), {"hold": False, "turn": "", "accel_kt_s": None,
                           "max_descent_fpm": None,
                           "heading": None, "altitude_ft": 3000.0,
                           "speed_kt": None, "intent": ""})(),
         AIRLINER, 10.0)
    trace("step_climb", f.altitude_ft)

    f = _Flight(altitude_ft=300.0)
    step(f, type("T", (), {"hold": False, "turn": "", "accel_kt_s": None,
                           "max_descent_fpm": None,
                           "heading": None, "altitude_ft": 0.0,
                           "speed_kt": None, "intent": ""})(),
         AIRLINER, 60.0, ground_elevation_ft=180.0)
    trace("step_descent_floor", f.altitude_ft)

    # --- step: moves ---
    f = _Flight(lat=50.0, lon=1.0, heading=90.0, ground_speed_kt=180.0)
    step(f, type("T", (), {"hold": False, "turn": "", "accel_kt_s": None,
                           "max_descent_fpm": None,
                           "heading": None, "altitude_ft": None,
                           "speed_kt": None, "intent": ""})(),
         AIRLINER, 60.0)
    trace("step_move_lat", f.latitude)
    trace("step_move_lon", f.longitude)

    # --- glide_altitude ---
    trace("glide_altitude", 318.0 * 5.0 + 180.0)

    # --- offset_from_track ---
    trace("offset_on_track", offset_from_track(50.0, 1.0, 50.0, 1.0, 90.0))

    # --- performance_for ---
    for code in ("C172", "A320", "AT76", "ZZZZ"):
        p = performance_for(code)
        trace(f"perf_{code}", {
            "turn_rate": p.turn_rate, "climb_fpm": p.climb_fpm,
            "descend_fpm": p.descend_fpm, "rotate_kt": p.rotate_kt,
            "climb_kt": p.climb_kt, "cruise_kt": p.cruise_kt,
            "approach_kt": p.approach_kt,
        })

    # --- path_steering ---
    path = [(50.0, 1.0), (50.0, 1.0 + 1.0 / 60.0)]
    r = path_steering(50.0, 1.0 + 0.5 / 60.0, path)
    trace("path_steering_on_path", r)
    r = path_steering(50.0, 1.0, [])
    trace("path_steering_empty", r)

    # --- runway_end ---
    r = runway_end([_rw()], "09L")
    trace("runway_end_09L", r)
    r = runway_end([_rw()], "27R")
    trace("runway_end_27R", r)
    r = runway_end([_rw()], "18")
    trace("runway_end_missing", r)

    # --- glide / align ---
    trace("glide", glide(50.0, 1.0, 180.0, 5.0))
    trace("align_on_centreline", align(50.9, 1.4 + 1.0 / 60.0, 50.9, 1.4, 90.0))
    trace("align_right", align(50.9 - 0.001, 1.4 + 1.0 / 60.0, 50.9, 1.4, 90.0))

    # --- profile_alt / legs ---
    recipe = ApproachRecipe(
        ident="EGTEST", name="ILS Z", family="ils",
        approach_runway="09L", landing_runway="09L",
        final_course=90.0, threshold_lat=50.9, threshold_lon=1.4,
        field_elev_ft=180.0, faf_lat=50.9, faf_lon=1.3,
        faf_alt_ft=1770.0, missed_alt_ft=2500.0,
    )
    trace("profile_alt", profile_alt(recipe, 5.0))
    trace("legs", legs(recipe))

    # --- traffic_performance ---
    for code in ("A320", "B738", "ZZZZ"):
        p = traffic_performance(code)
        trace(f"traf_perf_{code}", {
            "climb_fpm": p.climb_fpm, "climb_kt": p.climb_kt,
            "cruise_kt": p.cruise_kt, "approach_kt": p.approach_kt,
            "range_nm": p.range_nm, "ceiling_ft": p.ceiling_ft,
            "takeoff_kt_s": p.takeoff_kt_s,
        })

    # --- type_perf / range_for ---
    for code in ("A320", "ZZZZ"):
        tp = type_perf(code)
        trace(f"type_perf_{code}", None if tp is None else {
            "db": tp.db, "v2_kt": tp.v2_kt, "takeoff_m": tp.takeoff_m,
            "climb_roc_fpm": tp.climb_roc_fpm,
            "cruise_tas_kt": tp.cruise_tas_kt, "ceiling_ft": tp.ceiling_ft,
            "range_nm": tp.range_nm,
        })
    for code in ("A320", "ZZZZ", "H125"):
        trace(f"range_for_{code}", range_for(code))

    # --- vspeeds ---
    for code in ("C172", "A320", "ZZZZ"):
        trace(f"table_speeds_{code}", {
            "vr": table_speeds(code).vr,
            "vapp": table_speeds(code).vapp,
            "source": table_speeds(code).source[:40],
            "unverified": table_speeds(code).unverified,
        })
    s = speeds_for("A320", "airliner",
                   stated=type("S", (), {"v1": 130.0, "vr": 135.0,
                                         "v2": 145.0, "vapp": 140.0})())
    trace("speeds_for_stated", {"v1": s.v1, "vr": s.vr, "v2": s.v2,
                                "vapp": s.vapp})
    trace("rotate_kt_C172", rotate_kt("C172"))
    trace("rotate_kt_B738", rotate_kt("B738"))
    trace("rotate_kt_unknown", rotate_kt("ZZZZ"))


if __name__ == "__main__":
    main()
