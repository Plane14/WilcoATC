# -*- coding: utf-8 -*-
"""SID/STAR legs as the simulator's facility data gives them, and how the
First Officer flies them.

The navigation database has no procedures; the simulator does. This module
is the canonical model the rest of WilcoATC uses: the decoded departure or
arrival, the transition that was cleared, and a navigator that turns the
leg list into the heading, level and speed the autopilot is given.

The leg record is ARINC 424's path terminator plus the fields MSFS exposes
for it (``SimConnect_AddToFacilityDefinition``, ``APPROACH_LEG``). Not every
terminator is flown as charted -- a DME arc needs the station's DME, a
manual terminator waits for a vector -- but the decoded fields are kept so
the guidance can say what it did rather than guessing.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..navdata.build import bearing_deg, haversine_nm
from ..navdata.magvar import to_true, variation
from . import auto_nav
from .procedures import designator, namesake

log = logging.getLogger(__name__)

# SIMCONNECT_FACILITY_DATA_TYPE members this model is built from.
FACILITY_DEPARTURE = 10
FACILITY_ARRIVAL = 11
FACILITY_RUNWAY_TRANSITION = 12
FACILITY_ENROUTE_TRANSITION = 13
FACILITY_APPROACH_LEG = 7

# APPROACH_LEG.TYPE, the ARINC path terminators MSFS names.
PATH_TYPES = {
    1: "AF", 2: "CA", 3: "CD", 4: "CF", 5: "CI", 6: "CR", 7: "DF",
    8: "FA", 9: "FC", 10: "FD", 11: "FM", 12: "HA", 13: "HF",
    14: "HM", 15: "IF", 16: "PI", 17: "RF", 18: "TF", 19: "VA",
    20: "VD", 21: "VI", 22: "VM", 23: "VR",
}

# APPROACH_ALT_DESC.
ALT_NOT_USED, ALT_AT, ALT_AT_OR_ABOVE, ALT_AT_OR_BELOW, ALT_IN_BETWEEN = \
    range(5)
ALT_DESC = {0: "", 1: "AT", 2: "AT OR ABOVE", 3: "AT OR BELOW",
            4: "IN BETWEEN"}

TURN_NONE, TURN_LEFT, TURN_RIGHT, TURN_EITHER = range(4)
TURN_NAMES = {0: "", 1: "L", 2: "R", 3: "E"}

FT_PER_M = 3.28084
M_PER_NM = 1852.0
# Enroute path steering looks much further ahead than a taxiway does.
PROCEDURE_LOOKAHEAD_M = 9000.0


@dataclass(frozen=True)
class ProcedureLeg:
    """One decoded ``APPROACH_LEG`` record, in the units the FO flies."""

    index: int
    path: str                     # AF, CA, ...; "" when unknown
    path_type: int = 0
    fix_ident: str = ""
    fix_region: str = ""
    fix_kind: str = ""
    lat: float = 0.0
    lon: float = 0.0
    fix_alt_ft: float = 0.0
    fly_over: bool = False
    distance_minute: bool = False
    true_degree: bool = False
    turn: str = ""                 # L, R, E or ""
    origin_ident: str = ""
    origin_region: str = ""
    origin_kind: str = ""
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    origin_alt_ft: float = 0.0
    theta: float = 0.0
    rho_m: float = 0.0
    course: float = 0.0
    route_distance: float = 0.0   # metres, or minutes when distance_minute
    alt_desc: int = 0
    alt1_ft: float = 0.0
    alt2_ft: float = 0.0
    speed_limit_kt: float = 0.0
    vertical_angle: float = 0.0
    arc_ident: str = ""
    arc_region: str = ""
    arc_kind: str = ""
    arc_lat: float = 0.0
    arc_lon: float = 0.0
    arc_alt_ft: float = 0.0
    radius_m: float = 0.0
    rnp_m: float = 0.0
    speed_desc: int = 0
    is_iaf: bool = False
    is_if: bool = False
    is_faf: bool = False
    is_map: bool = False
    is_atc_reporting: bool = False
    is_essential: bool = False
    is_idf: bool = False
    is_stepdown: bool = False

    @property
    def has_fix(self) -> bool:
        return bool(self.fix_ident) and not (self.lat == 0.0 and self.lon == 0.0)

    @property
    def altitude_constraint(self) -> float:
        """The level this leg asks for, feet; 0 when it asks for none."""
        if self.alt_desc == ALT_NOT_USED:
            return 0.0
        return self.alt1_ft or self.alt2_ft


@dataclass
class ProcedureTransition:
    """The cleared part of a SID or STAR: one runway or enroute transition."""

    name: str = ""
    runway: str = ""
    legs: tuple[ProcedureLeg, ...] = ()

    @property
    def first_fix(self) -> ProcedureLeg | None:
        return self.legs[0] if self.legs else None

    @property
    def last_fix(self) -> ProcedureLeg | None:
        return self.legs[-1] if self.legs else None


@dataclass
class TerminalProcedure:
    """A named SID or STAR at one airport."""

    name: str
    kind: str                          # "sid" or "star"
    transitions: list[ProcedureTransition] = field(default_factory=list)
    direct_legs: tuple[ProcedureLeg, ...] = ()


@dataclass
class AirportProcedures:
    """Everything one airport publishes, decoded once."""

    ident: str
    departures: dict[str, TerminalProcedure] = field(default_factory=dict)
    arrivals: dict[str, TerminalProcedure] = field(default_factory=dict)
    lat: float = 0.0
    lon: float = 0.0
    alt_ft: float = 0.0

    def departure(self, name: str) -> TerminalProcedure | None:
        return _find(self.departures, name)

    def arrival(self, name: str) -> TerminalProcedure | None:
        return _find(self.arrivals, name)


def _norm(name: str) -> str:
    return (name or "").strip().upper().replace(" ", "")


def _find(table: dict[str, TerminalProcedure], name: str
          ) -> TerminalProcedure | None:
    wanted = _norm(name)
    if not wanted:
        return None
    if wanted in table:
        return table[wanted]
    parts = designator(wanted)
    for key, proc in table.items():
        if _norm(key) == wanted:
            return proc
        other = designator(key)
        if parts and other and parts == other:
            return proc
        # The spoken form ("ROUSY 1L") against the stored six characters.
        if namesake(key) and namesake(key) in wanted:
            digit = parts[1] if parts else ""
            if digit and digit in wanted:
                return proc
    return None


def _runway_key(runway: str) -> str:
    return (runway or "").strip().upper().replace(" ", "")


def _transition_for(procedure: TerminalProcedure, kind: str,
                    runway: str = "", preferred: str = "",
                    route_fix_idents: Iterable[str] = ()) -> \
        ProcedureTransition | None:
    """The transition to fly: the named one, the runway's, or the one the
    route's own fixes say is next."""
    transitions = [t for t in procedure.transitions if t.legs]
    if not transitions:
        if procedure.direct_legs:
            return ProcedureTransition(legs=procedure.direct_legs)
        return None
    wanted = _norm(preferred)
    if wanted:
        for trans in transitions:
            if _norm(trans.name) == wanted or _norm(trans.runway) == wanted:
                return trans
    if kind == "sid":
        key = _runway_key(runway)
        if key:
            for trans in transitions:
                if _runway_key(trans.runway) == key:
                    return trans
    else:
        key = _runway_key(runway)
        if key:
            for trans in transitions:
                if _runway_key(trans.runway) == key:
                    return trans
    # The route's fixes name the transition's fixes better than a guess:
    # the one whose fixes the route reaches first wins.
    order = [i.upper() for i in route_fix_idents if i]
    idents = set(order)
    best, best_at = None, len(order) + 1
    for trans in transitions:
        names = {leg.fix_ident.upper() for leg in trans.legs
                 if leg.fix_ident}
        hits = names & idents
        if hits:
            earliest = min(idents_order(order, hits))
            if earliest < best_at:
                best, best_at = trans, earliest
    if best is not None:
        return best
    return transitions[0] if len(transitions) == 1 else None


def idents_order(order: list[str], hits) -> list[int]:
    """Order ``hits`` by where they sit in the route's fix list."""
    position = {ident: i for i, ident in enumerate(order)}
    return sorted(position.get(h, len(order)) for h in hits)


def choose_sid(procedures: AirportProcedures | None, name: str,
               runway: str = "", preferred: str = "",
               route_fix_idents: Iterable[str] = ()) -> \
        tuple[TerminalProcedure | None, ProcedureTransition | None]:
    if procedures is None:
        return None, None
    proc = procedures.departure(name)
    if proc is None:
        return None, None
    return proc, _transition_for(proc, "sid", runway, preferred,
                                 route_fix_idents)


def choose_star(procedures: AirportProcedures | None, name: str,
                runway: str = "", preferred: str = "",
                route_fix_idents: Iterable[str] = ()) -> \
        tuple[TerminalProcedure | None, ProcedureTransition | None]:
    if procedures is None:
        return None, None
    proc = procedures.arrival(name)
    if proc is None:
        return None, None
    return proc, _transition_for(proc, "star", runway, preferred,
                                 route_fix_idents)


# ----------------------------------------------------------------------
# flying it
# ----------------------------------------------------------------------


def _point(lat: float, lon: float, alt_ft: float = 0.0) -> tuple[float, float]:
    return lat, lon


def _offset(lat: float, lon: float, bearing: float, distance_nm: float
            ) -> tuple[float, float]:
    """A point ``distance_nm`` along ``bearing`` from a position."""
    if distance_nm <= 0:
        return lat, lon
    angular = distance_nm / 60.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brg = math.radians(bearing)
    lat2 = math.asin(math.sin(lat1) * math.cos(angular)
                     + math.cos(lat1) * math.sin(angular) * math.cos(brg))
    lon2 = lon1 + math.atan2(math.sin(brg) * math.sin(angular)
                             * math.cos(lat1),
                             math.cos(angular) - math.sin(lat1)
                             * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def _leg_end(leg: ProcedureLeg, from_lat: float, from_lon: float
             ) -> tuple[float, float] | None:
    """Where a leg's lateral path ends, best available."""
    if leg.has_fix:
        return leg.lat, leg.lon
    # Course/heading terminators carry their own direction and distance.
    course = leg.course or leg.theta
    if course > 0 and not leg.true_degree:
        course = to_true(course, variation(from_lat, from_lon))
    if course <= 0:
        return None
    distance_nm = 0.0
    if leg.route_distance:
        distance_nm = (leg.route_distance / 60.0 if leg.distance_minute
                       else leg.route_distance / M_PER_NM)
    elif leg.rho_m:
        distance_nm = leg.rho_m / M_PER_NM
    if distance_nm <= 0:
        return None
    return _offset(from_lat, from_lon, course, distance_nm)


def _arc_points(leg: ProcedureLeg, from_lat: float, from_lon: float
                ) -> list[tuple[float, float]] | None:
    """Points along an RF arc, when the centre and radius are known."""
    if not leg.has_fix or leg.arc_lat == 0.0 and leg.arc_lon == 0.0:
        return None
    centre_lat, centre_lon = leg.arc_lat, leg.arc_lon
    radius_nm = leg.radius_m / M_PER_NM if leg.radius_m > 0 else \
        haversine_nm(centre_lat, centre_lon, leg.lat, leg.lon)
    if radius_nm < 0.5:
        return None
    start_brg = bearing_deg(centre_lat, centre_lon, from_lat, from_lon)
    end_brg = bearing_deg(centre_lat, centre_lon, leg.lat, leg.lon)
    turn = leg.turn
    if turn == "L":
        sweep = (end_brg - start_brg) % 360.0
        if sweep > 180.0:
            sweep -= 360.0
    elif turn == "R":
        sweep = (start_brg - end_brg) % 360.0
        if sweep > 180.0:
            sweep -= 360.0
        sweep = -sweep
    else:
        sweep = ((end_brg - start_brg + 540.0) % 360.0) - 180.0
    steps = max(2, int(abs(sweep) / 15.0) + 1)
    points = []
    for i in range(1, steps + 1):
        brg = start_brg + sweep * i / steps
        # A point on the circle: offset from the centre by the radius.
        lat, lon = _offset(centre_lat, centre_lon, brg, radius_nm)
        points.append((lat, lon))
    points[-1] = (leg.lat, leg.lon)
    return points


def _turn_points(leg: ProcedureLeg, from_lat: float, from_lon: float
                 ) -> list[tuple[float, float]]:
    """A procedure turn or racetrack, approximated as outbound then back."""
    course = leg.course or leg.theta
    if course > 0 and not leg.true_degree:
        course = to_true(course, variation(from_lat, from_lon))
    outbound = leg.route_distance
    if leg.distance_minute:
        outbound *= M_PER_NM / 60.0 * 4.0  # ~4 NM a minute, a jet's guess
    else:
        outbound /= M_PER_NM
    if course > 0 and outbound > 0:
        mid_lat, mid_lon = _offset(from_lat, from_lon, course, outbound)
    else:
        mid_lat, mid_lon = from_lat, from_lon
    end = _leg_end(leg, mid_lat, mid_lon)
    points = [(mid_lat, mid_lon)]
    if end is not None:
        points.append(end)
    return points


def build_path(transition: ProcedureTransition,
               start: tuple[float, float] | None = None
               ) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    """The procedure as a path the autopilot can steer, with each point's
    level and speed limit.

    Returns ``(points, notes)`` where a point is ``(lat, lon, alt_ft,
    speed_kt)``. Unsupported terminators are noted and flown as a direct
    leg to the fix the record names, which is what the chart's own text
    falls back to when the geometry cannot be built.
    """
    points: list[tuple[float, float, float, float]] = []
    notes: list[str] = []
    if not transition.legs:
        return points, ["the procedure has no legs"]
    cursor = start or (transition.legs[0].lat, transition.legs[0].lon)
    for leg in transition.legs:
        alt = leg.altitude_constraint or leg.fix_alt_ft
        speed = leg.speed_limit_kt
        path = leg.path
        generated: list[tuple[float, float]] = []
        if path == "RF":
            generated = _arc_points(leg, cursor[0], cursor[1]) or []
            if generated:
                notes.append(f"{leg.fix_ident or leg.path}: RF arc approximated")
            else:
                notes.append(f"{leg.fix_ident or leg.path}: RF arc unavailable")
        elif path in ("PI", "HA", "HF", "HM"):
            generated = _turn_points(leg, cursor[0], cursor[1])
            notes.append(f"{leg.fix_ident or leg.path}: {path} approximated")
        elif path in ("VA", "VD", "VI", "VM", "VR"):
            notes.append(f"{leg.fix_ident or leg.path}: {path} flown direct")
        end = _leg_end(leg, cursor[0], cursor[1])
        if not generated and end is not None:
            generated = [end]
        if not generated:
            notes.append(f"{leg.fix_ident or leg.path}: no geometry")
            continue
        for lat, lon in generated:
            points.append((lat, lon, alt, speed))
        cursor = generated[-1]
    return points, notes


@dataclass
class ProcedureGuidance:
    """What the navigator says this tick."""

    heading: float | None = None       # true degrees
    altitude_ft: float = 0.0
    vs_fpm: float = 0.0
    speed_kt: float = 0.0
    done: bool = False
    note: str = ""


class ProcedureNavigator:
    """Flies one chosen transition: lateral path plus leg constraints."""

    def __init__(self, procedure: TerminalProcedure,
                 transition: ProcedureTransition,
                 start: tuple[float, float] | None = None,
                 fallback_altitude_ft: float = 0.0,
                 fallback_speed_kt: float = 0.0):
        self.procedure = procedure
        self.transition = transition
        leg_points, self.notes = build_path(transition, start)
        # The path the aeroplane steers starts where the procedure does:
        # the runway end for a SID, the position it was picked up on for a
        # STAR. Without that first point the steering can aim past the
        # first fix while the constraints still say the second leg's.
        if start is not None and leg_points:
            first = leg_points[0]
            self.points = [(start[0], start[1], first[2], first[3]),
                           *leg_points]
        else:
            self.points = leg_points
        self.fallback_altitude_ft = fallback_altitude_ft
        self.fallback_speed_kt = fallback_speed_kt
        self._bare = [(lat, lon) for lat, lon, _a, _s in self.points]

    @property
    def active(self) -> bool:
        return bool(self._bare)

    def guidance(self, state, climb_fpm: float = 1500.0,
                 descend_fpm: float = 1800.0) -> ProcedureGuidance:
        if not self._bare:
            return ProcedureGuidance(done=True, note="no path")
        lat, lon = state.latitude, state.longitude
        steer = auto_nav.path_steering(
            lat, lon, self._bare, lookahead_m=PROCEDURE_LOOKAHEAD_M)
        if steer is None:
            return ProcedureGuidance(done=True, note="no steering")
        index = min(max(steer.get("nearest", 0) + 1, 0),
                    len(self.points) - 1)
        _lat, _lon, alt, speed = self.points[index]
        current_alt = state.altitude_ft or 0.0
        # A SID's levels are floors the climb makes through; a STAR's are
        # step-downs the descent holds to. The fallback is the clearance's
        # own level for the SID and the arrival floor for the STAR.
        if self.procedure.kind == "sid":
            # A SID's levels are floors: never level below the leg's own
            # altitude, and keep climbing to the clearance's target.
            altitude = max(self.fallback_altitude_ft, alt or 0.0)
            vs = climb_fpm if altitude and current_alt < altitude - 250 \
                else 0.0
        else:
            altitude = alt or self.fallback_altitude_ft
            if altitude and current_alt > altitude + 250:
                vs = -descend_fpm
            elif altitude and current_alt < altitude - 250:
                # Already under the step-down: hold here rather than let
                # the altitude bug climb back to a constraint passed.
                altitude = current_alt
                vs = 0.0
            else:
                vs = 0.0
        target_speed = speed or self.fallback_speed_kt
        # A procedure's speed limit is a limit, not a target: only slow to
        # it when the assigned speed is above it.
        if speed and self.fallback_speed_kt and \
                self.fallback_speed_kt > speed:
            target_speed = speed
        return ProcedureGuidance(
            heading=steer.get("heading"), altitude_ft=altitude, vs_fpm=vs,
            speed_kt=target_speed,
            done=bool(steer.get("done")), note=self.notes[0]
            if self.notes else "")


__all__ = [
    "AirportProcedures",
    "ProcedureLeg",
    "ProcedureNavigator",
    "ProcedureTransition",
    "TerminalProcedure",
    "build_path",
    "choose_sid",
    "choose_star",
]
