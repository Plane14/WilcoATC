# -*- coding: utf-8 -*-
"""Where the aeroplane should be pointing, as pure geometry.

No simulator, no session, no events: given a position and a route, these
functions say where to go next. The taxi follower and the approach work
in :mod:`wilcoatc.atc.autonomous` are the callers; keeping the math here
means it can be checked without a flight.

Three problems, three functions:

* :func:`path_steering` -- follow a polyline of (latitude, longitude)
  points, which is what ``session.taxi_path`` already is: the route the
  tower planned on the airport's own graph. It answers the heading to
  fly, how far off the line the aeroplane is, and how much path is left.

* :func:`runway_end` -- which end of a runway a departure lines up at
  and an arrival touches down on, with the runway's own heading.

* :func:`glide` -- where a three-degree path is, that far from the
  threshold, and how far the aeroplane is above or below it.
"""

from __future__ import annotations

import math
import re

from ..navdata.build import bearing_deg, haversine_nm
from ..navdata.db import normalize_runway

# How far ahead along the path the taxi steers: far enough to smooth the
# graph's own corners, near enough that a junction is not cut.
TAXI_LOOKAHEAD_M = 120.0
# Within this of a path point the aeroplane counts as having reached it,
# and the follower moves on. A junction's width, roughly.
PATH_REACHED_M = 25.0
# Below this groundspeed the aeroplane is stopped, and a path that will
# not advance is a problem to report rather than to steer around.
STOPPED_KT = 1.0


def _point_segment_distance(lat: float, lon: float,
                            a: tuple[float, float],
                            b: tuple[float, float]) -> tuple[float, float]:
    """(metres along the segment, metres off it) for the closest point."""
    # Equirectangular metres: fine at the tens-of-metres scale a taxi leg
    # is, and the same approximation the ground layout uses.
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians((a[0] + b[0]) / 2.0))
    ax, az = a[1] * m_per_deg_lon, a[0] * m_per_deg_lat
    bx, bz = b[1] * m_per_deg_lon, b[0] * m_per_deg_lat
    px, pz = lon * m_per_deg_lon, lat * m_per_deg_lat
    dx, dz = bx - ax, bz - az
    length2 = dx * dx + dz * dz
    t = 0.0 if length2 == 0 else max(
        0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length2))
    cx, cz = ax + t * dx, az + t * dz
    along = math.hypot(cx - ax, cz - az)
    across = math.hypot(px - cx, pz - cz)
    # Which side: positive to the right of the direction of travel.
    cross = dx * (pz - az) - dz * (px - ax)
    side = across if cross >= 0 else -across
    return along, side


def path_steering(lat: float, lon: float, path: list[tuple[float, float]],
                  lookahead_m: float = TAXI_LOOKAHEAD_M) -> dict | None:
    """Where on a path the aeroplane is, and where to point next.

    ``path`` is a list of (latitude, longitude) -- ``session.taxi_path``.
    The answer is a dict with ``heading`` (degrees true to fly),
    ``cross_track_m`` (signed, positive right of the path), ``remaining_m``
    (path left to the end), ``nearest`` (index of the path point the
    aeroplane is closest to) and ``done`` (the end is reached or the path
    is empty).

    The aeroplane steers at a point ``lookahead_m`` further along the
    path, which is what keeps it on the line through corners instead of
    aiming at the next vertex and overshooting every junction.
    """
    if not path:
        return {"heading": None, "cross_track_m": 0.0, "remaining_m": 0.0,
                "nearest": 0, "done": True}
    if len(path) == 1:
        return {"heading": bearing_deg(lat, lon, path[0][0], path[0][1]),
                "cross_track_m": 0.0,
                "remaining_m": haversine_nm(lat, lon,
                                            path[0][0], path[0][1]) * 1852.0,
                "nearest": 0, "done": False}

    # The segment the aeroplane is closest to, and how far along it.
    best = None
    for i, (a, b) in enumerate(zip(path, path[1:])):
        along, side = _point_segment_distance(lat, lon, a, b)
        # Distance to the segment itself, for picking the nearest one.
        seg_len_m = haversine_nm(a[0], a[1], b[0], b[1]) * 1852.0
        off = abs(side)
        if best is None or off < best[0]:
            best = (off, i, along, side, seg_len_m)
    if best is None:
        return {"heading": None, "cross_track_m": 0.0, "remaining_m": 0.0,
                "nearest": 0, "done": True}
    _, index, along_m, side_m, seg_len_m = best

    # Walk forward from the closest point until the lookahead is reached.
    target = None
    walked = lookahead_m
    i = index
    frac_along = min(1.0, along_m / max(seg_len_m, 1e-6))
    while i < len(path) - 1:
        a, b = path[i], path[i + 1]
        seg_m = haversine_nm(a[0], a[1], b[0], b[1]) * 1852.0
        # Where on this segment the walk starts: the projection point on
        # the first segment, the start point on every later one.
        start_frac = frac_along if i == index else 0.0
        available = seg_m * (1.0 - start_frac)
        if walked <= available and seg_m > 0:
            t = start_frac + (walked / seg_m)
            t = min(1.0, t)
            target = (a[0] + (b[0] - a[0]) * t,
                      a[1] + (b[1] - a[1]) * t)
            break
        walked -= available
        i += 1
    if target is None:
        target = path[-1]

    # How much path is left: the rest of this segment, then every segment
    # after it.
    remaining = seg_len_m * (1.0 - frac_along)
    for j in range(index + 1, len(path) - 1):
        remaining += haversine_nm(path[j][0], path[j][1],
                                   path[j + 1][0], path[j + 1][1]) * 1852.0
    done = (index >= len(path) - 2
            and haversine_nm(lat, lon, path[-1][0], path[-1][1]) * 1852.0
            < PATH_REACHED_M * 2)

    return {"heading": bearing_deg(lat, lon, target[0], target[1]),
            "cross_track_m": side_m,
            "remaining_m": remaining,
            "nearest": index,
            "done": done}


def runway_end(runways, runway: str):
    """The threshold and heading for a runway designation.

    ``runways`` are the navdb's :class:`Runway` rows; ``runway`` the
    spoken designation ("24", "24L"). Returns ``(lat, lon, heading_deg,
    length_ft)`` for the end the designation names -- the threshold a
    departure lines up at and an arrival crosses first -- or None when
    the field does not have it or the row lacks coordinates.
    """
    wanted = normalize_runway(runway or "")
    if not wanted:
        return None
    candidates = []
    for rw in runways:
        coords = (getattr(rw, "le_lat", None), getattr(rw, "le_lon", None),
                  getattr(rw, "he_lat", None), getattr(rw, "he_lon", None))
        if None in coords:
            continue
        le, he = (normalize_runway(getattr(rw, "le_ident", "") or ""),
                  normalize_runway(getattr(rw, "he_ident", "") or ""))
        length = float(getattr(rw, "length_ft", 0.0) or 0.0)
        # Which end is the named threshold: the end whose own designation
        # matches first, the magnetic heading as the check after it.
        for ident, lat, lon, other_lat, other_lon in (
                (le, coords[0], coords[1], coords[2], coords[3]),
                (he, coords[2], coords[3], coords[0], coords[1])):
            heading = bearing_deg(lat, lon, other_lat, other_lon)
            candidates.append((ident == wanted, ident, lat, lon,
                               heading, length))
    exact = [c for c in candidates if c[0]]
    if exact:
        _, _, lat, lon, heading, length = exact[0]
        return lat, lon, heading, length
    # No end named it: the designation's own heading picks the end whose
    # outbound track is closest to it.
    digits = re.sub(r"[^0-9]", "", wanted or "")
    if len(digits) < 2:
        return None
    want_hdg = float(digits[:2]) * 10.0
    best = None
    for _, _, lat, lon, heading, length in candidates:
        diff = abs((heading - want_hdg + 540.0) % 360.0 - 180.0)
        if best is None or diff < best[0]:
            best = (diff, lat, lon, heading, length)
    if best is None:
        return None
    _, lat, lon, heading, length = best
    return lat, lon, heading, length


def glide(threshold_lat: float, threshold_lon: float,
          threshold_elev_ft: float, distance_nm: float,
          ft_per_nm: float = 318.0) -> float:
    """Where a three-degree path is, that far from the threshold."""
    return threshold_elev_ft + max(0.0, distance_nm) * ft_per_nm


def align(lat: float, lon: float, threshold_lat: float, threshold_lon: float,
          runway_heading: float) -> dict:
    """How far the aeroplane is from the extended centreline, in metres,
    and the correction to fly to put it on it. Positive offset is right
    of the runway heading."""
    distance = haversine_nm(threshold_lat, threshold_lon, lat, lon)
    track = bearing_deg(threshold_lat, threshold_lon, lat, lon)
    rel = math.radians(track - runway_heading)
    across_nm = distance * math.sin(rel)
    along_nm = distance * math.cos(rel)
    # The heading to the threshold is the runway heading only on the
    # centreline; the correction is the difference. Behind the threshold
    # there is no "on the numbers" to intercept -- the aeroplane is on the
    # ground short of it or on a departure leg -- so the correction is nil.
    if along_nm <= 0.0:
        correction = 0.0
    else:
        # Off to the right, the nose comes left: the correction is the
        # negative of the offset's angle.
        correction = -math.degrees(math.atan2(across_nm * 1852.0,
                                              along_nm * 1852.0))
    return {"offset_m": across_nm * 1852.0,
            "along_m": along_nm * 1852.0,
            "correction_deg": (correction + 540.0) % 360.0 - 180.0}


__all__ = ["path_steering", "runway_end", "glide", "align",
           "TAXI_LOOKAHEAD_M", "PATH_REACHED_M", "STOPPED_KT"]
