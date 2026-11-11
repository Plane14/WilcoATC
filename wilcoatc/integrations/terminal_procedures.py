# -*- coding: utf-8 -*-
"""Asking the simulator for an airport's SIDs and STARs.

The navigation database has runways and frequencies and no procedures, so a
filed SID or STAR was a name without a path. The simulator's facility data
has the whole hierarchy -- airport, departure or arrival, runway or enroute
transition, and the ARINC legs under it -- and this reads it the way
:mod:`wilcoatc.integrations.approaches` reads the approaches: one session
of its own, one request, every block checked against the shape it was
written for and the airport left unknown rather than read wrongly when a
simulator numbers it differently.

The fields are the SDK's (``SimConnect_AddToFacilityDefinition``): the
procedure's ``NAME`` and transition counts, the transition's runway or
name, and every ``APPROACH_LEG`` field the path terminators need -- fix,
origin, course, distance, altitude descriptor and limits, speed, vertical
angle, arc centre and the IAF/IF/FAF/MAP flags. Altitudes come back in
metres and are kept in feet here, the unit the flight phases work in.
"""

from __future__ import annotations

import logging
import struct

from ..atc.approaches import runway_from
from ..atc.procedure_legs import (
    FACILITY_APPROACH_LEG,
    FACILITY_ARRIVAL,
    FACILITY_DEPARTURE,
    FACILITY_ENROUTE_TRANSITION,
    FACILITY_RUNWAY_TRANSITION,
    AirportProcedures,
    ProcedureLeg,
    ProcedureTransition,
    TerminalProcedure,
)
from .taxiways import ask_facility_records

log = logging.getLogger(__name__)

_FACILITY_AIRPORT = 0

# The leg fields, in the order the definition asks for them; the order is
# the layout of every leg block. The SDK's names are the first element,
# the decoded key the third.
_LEG_FIELDS = (
    ("TYPE", "i", "path_type"), ("FIX_ICAO", "8s", "fix_icao"),
    ("FIX_REGION", "8s", "fix_region"), ("FIX_TYPE", "i", "fix_type"),
    ("FIX_LATITUDE", "d", "fix_lat"), ("FIX_LONGITUDE", "d", "fix_lon"),
    ("FIX_ALTITUDE", "d", "fix_alt"), ("FLY_OVER", "i", "fly_over"),
    ("DISTANCE_MINUTE", "i", "distance_minute"),
    ("TRUE_DEGREE", "i", "true_degree"),
    ("TURN_DIRECTION", "i", "turn_direction"),
    ("ORIGIN_ICAO", "8s", "origin_icao"),
    ("ORIGIN_REGION", "8s", "origin_region"),
    ("ORIGIN_TYPE", "i", "origin_type"),
    ("ORIGIN_LATITUDE", "d", "origin_lat"),
    ("ORIGIN_LONGITUDE", "d", "origin_lon"),
    ("ORIGIN_ALTITUDE", "d", "origin_alt"),
    ("THETA", "f", "theta"), ("RHO", "f", "rho"), ("COURSE", "f", "course"),
    ("ROUTE_DISTANCE", "f", "route_distance"),
    ("APPROACH_ALT_DESC", "i", "alt_desc"), ("ALTITUDE1", "f", "alt1"),
    ("ALTITUDE2", "f", "alt2"), ("SPEED_LIMIT", "f", "speed_limit"),
    ("VERTICAL_ANGLE", "f", "vertical_angle"),
    ("ARC_CENTER_FIX_ICAO", "8s", "arc_icao"),
    ("ARC_CENTER_FIX_REGION", "8s", "arc_region"),
    ("ARC_CENTER_FIX_TYPE", "i", "arc_type"),
    ("ARC_CENTER_FIX_LATITUDE", "d", "arc_lat"),
    ("ARC_CENTER_FIX_LONGITUDE", "d", "arc_lon"),
    ("ARC_CENTER_FIX_ALTITUDE", "d", "arc_alt"), ("RADIUS", "f", "radius"),
    ("IS_IAF", "i", "is_iaf"), ("IS_IF", "i", "is_if"),
    ("IS_FAF", "i", "is_faf"), ("IS_MAP", "i", "is_map"),
    ("IS_ATC_REPORTING", "i", "is_atc_reporting"),
    ("IS_ESSENTIAL", "i", "is_essential"), ("IS_IDF", "i", "is_idf"),
    ("IS_STEPDOWN", "i", "is_stepdown"),
    ("REQUIRED_NAVIGATION_PERFORMANCE", "f", "rnp"),
    ("APPROACH_SPEED_DESC", "i", "speed_desc"),
)
_LEG_NAMES = tuple(key for _field, _fmt, key in _LEG_FIELDS)
_LEG_REQUEST = tuple(field for field, _fmt, _key in _LEG_FIELDS)
_LEG_SHAPE = "<" + "".join(fmt for _field, fmt, _key in _LEG_FIELDS)
_LEG_SIZE = struct.calcsize(_LEG_SHAPE)

_FIX_KINDS = {65: "apt", 86: "vor", 78: "ndb", 87: "wpt"}
_FT_PER_M = 3.28084
_M_PER_NM = 1852.0


def _leg_fields() -> tuple[bytes, ...]:
    return (b"OPEN APPROACH_LEG",) + tuple(
        name.encode("ascii") for name in _LEG_REQUEST) + \
        (b"CLOSE APPROACH_LEG",)


def _procedure_fields(kind: bytes) -> tuple[bytes, ...]:
    """One departure or arrival with both kinds of transition and any
    legs the procedure carries itself."""
    return (
        b"OPEN " + kind,
        b"NAME", b"N_RUNWAY_TRANSITIONS", b"N_ENROUTE_TRANSITIONS",
        b"N_APPROACH_LEGS",
        b"OPEN RUNWAY_TRANSITION", b"RUNWAY_NUMBER", b"RUNWAY_DESIGNATOR",
        b"N_APPROACH_LEGS",
        *_leg_fields(),
        b"CLOSE RUNWAY_TRANSITION",
        b"OPEN ENROUTE_TRANSITION", b"NAME", b"N_APPROACH_LEGS",
        *_leg_fields(),
        b"CLOSE ENROUTE_TRANSITION",
        *_leg_fields(),
        b"CLOSE " + kind,
    )


_FIELDS = (
    b"OPEN AIRPORT", b"LATITUDE", b"LONGITUDE", b"ALTITUDE",
    *_procedure_fields(b"DEPARTURE"),
    *_procedure_fields(b"ARRIVAL"),
    b"CLOSE AIRPORT",
)

_AIRPORT_SHAPE = "<ddd"
_PROCEDURE_SHAPE = "<8siii"
_RUNWAY_TRANSITION_SHAPE = "<iii"
_ENROUTE_TRANSITION_SHAPE = "<8si"


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace").strip().upper()


def _kind(code: int) -> str:
    return _FIX_KINDS.get(code, "wpt")


def _leg(index: int, raw: bytes) -> ProcedureLeg | None:
    if len(raw) != _LEG_SIZE:
        return None
    try:
        values = struct.unpack(_LEG_SHAPE, raw)
    except struct.error:
        return None
    data = dict(zip(_LEG_NAMES, values))
    from ..atc.procedure_legs import PATH_TYPES, TURN_NAMES
    path_type = int(data["path_type"])
    return ProcedureLeg(
        index=index, path=PATH_TYPES.get(path_type, ""),
        path_type=path_type,
        fix_ident=_text(data["fix_icao"]), fix_region=_text(data["fix_region"]),
        fix_kind=_kind(int(data["fix_type"])),
        lat=float(data["fix_lat"]), lon=float(data["fix_lon"]),
        fix_alt_ft=float(data["fix_alt"]) * _FT_PER_M,
        fly_over=bool(data["fly_over"]),
        distance_minute=bool(data["distance_minute"]),
        true_degree=bool(data["true_degree"]),
        turn=TURN_NAMES.get(int(data["turn_direction"]), ""),
        origin_ident=_text(data["origin_icao"]),
        origin_region=_text(data["origin_region"]),
        origin_kind=_kind(int(data["origin_type"])),
        origin_lat=float(data["origin_lat"]),
        origin_lon=float(data["origin_lon"]),
        origin_alt_ft=float(data["origin_alt"]) * _FT_PER_M,
        theta=float(data["theta"]), rho_m=float(data["rho"]),
        course=float(data["course"]), route_distance=float(data["route_distance"]),
        alt_desc=int(data["alt_desc"]),
        alt1_ft=float(data["alt1"]) * _FT_PER_M,
        alt2_ft=float(data["alt2"]) * _FT_PER_M,
        speed_limit_kt=float(data["speed_limit"]),
        vertical_angle=float(data["vertical_angle"]),
        arc_ident=_text(data["arc_icao"]), arc_region=_text(data["arc_region"]),
        arc_kind=_kind(int(data["arc_type"])),
        arc_lat=float(data["arc_lat"]), arc_lon=float(data["arc_lon"]),
        arc_alt_ft=float(data["arc_alt"]) * _FT_PER_M,
        radius_m=float(data["radius"]), rnp_m=float(data["rnp"]),
        speed_desc=int(data["speed_desc"]),
        is_iaf=bool(data["is_iaf"]), is_if=bool(data["is_if"]),
        is_faf=bool(data["is_faf"]), is_map=bool(data["is_map"]),
        is_atc_reporting=bool(data["is_atc_reporting"]),
        is_essential=bool(data["is_essential"]),
        is_idf=bool(data["is_idf"]), is_stepdown=bool(data["is_stepdown"]))


def decode(records, ident: str) -> AirportProcedures | None:
    """The airport's departures and arrivals out of the raw records, or
    None if they are not the shape this was written against."""
    airport = AirportProcedures(ident)
    by_type: dict[int, list] = {}
    for record in records:
        by_type.setdefault(record.type, []).append(record)

    airport_records = by_type.get(_FACILITY_AIRPORT) or []
    if airport_records:
        raw = airport_records[0].raw
        if len(raw) < struct.calcsize(_AIRPORT_SHAPE):
            log.info("%s: a short airport block; not using procedures",
                     ident)
            return None
        airport.lat, airport.lon, alt_m = struct.unpack(
            _AIRPORT_SHAPE, raw[:struct.calcsize(_AIRPORT_SHAPE)])
        airport.alt_ft = alt_m * _FT_PER_M

    procedures: dict[int, tuple[TerminalProcedure, str]] = {}
    for kind, facility in (("sid", FACILITY_DEPARTURE),
                           ("star", FACILITY_ARRIVAL)):
        shape = _PROCEDURE_SHAPE
        size = struct.calcsize(shape)
        table = airport.departures if kind == "sid" else airport.arrivals
        for record in by_type.get(facility, []):
            if len(record.raw) != size:
                log.info("%s: a %d-byte %s block where %d were expected",
                         ident, len(record.raw), kind, size)
                return None
            name, _nr, _ne, _nl = struct.unpack(shape, record.raw)
            proc = TerminalProcedure(name=_text(name), kind=kind)
            table[proc.name] = proc
            procedures[record.unique_id] = (proc, kind)

    transitions: dict[int, tuple[ProcedureTransition, int]] = {}
    for facility, shape in ((FACILITY_RUNWAY_TRANSITION,
                             _RUNWAY_TRANSITION_SHAPE),
                            (FACILITY_ENROUTE_TRANSITION,
                             _ENROUTE_TRANSITION_SHAPE)):
        size = struct.calcsize(shape)
        for record in by_type.get(facility, []):
            if len(record.raw) != size:
                log.info("%s: a %d-byte transition block where %d expected",
                         ident, len(record.raw), size)
                return None
            if facility == FACILITY_RUNWAY_TRANSITION:
                number, designator, _n = struct.unpack(shape, record.raw)
                trans = ProcedureTransition(runway=runway_from(
                    number, designator))
            else:
                name, _n = struct.unpack(shape, record.raw)
                trans = ProcedureTransition(name=_text(name))
            transitions[record.unique_id] = (trans, record.parent_id)

    legs_by_parent: dict[int, list[tuple[int, ProcedureLeg]]] = {}
    for record in by_type.get(FACILITY_APPROACH_LEG, []):
        leg = _leg(len(legs_by_parent.get(record.parent_id, [])),
                   record.raw)
        if leg is None:
            log.info("%s: a %d-byte leg block where %d were expected",
                     ident, len(record.raw), _LEG_SIZE)
            return None
        index = (record.item_index if record.is_list_item
                 else len(legs_by_parent.get(record.parent_id, [])))
        legs_by_parent.setdefault(record.parent_id, []).append((index, leg))

    def ordered(parent: int) -> tuple[ProcedureLeg, ...]:
        return tuple(leg for _i, leg in sorted(
            legs_by_parent.get(parent, []), key=lambda item: item[0]))

    for unique_id, (trans, parent_id) in transitions.items():
        owner = procedures.get(parent_id)
        if owner is None:
            continue
        proc, _kind = owner
        trans.legs = ordered(unique_id)
        if trans.legs:
            proc.transitions.append(trans)

    for unique_id, (proc, _kind) in procedures.items():
        proc.direct_legs = ordered(unique_id)
        proc.transitions.sort(key=lambda t: (t.runway != "", t.name))

    if not airport.departures and not airport.arrivals:
        return None
    return airport


class SimProcedures:
    """The simulator's published SIDs and STARs, one airport at a time."""

    TIMEOUT_S = 20.0

    def __init__(self, dll_path: str = ""):
        self.dll_path = dll_path
        self._cache: dict[str, AirportProcedures | None] = {}

    def procedures(self, ident: str) -> AirportProcedures | None:
        ident = (ident or "").strip().upper()
        if not ident:
            return None
        if ident in self._cache:
            return self._cache[ident]
        try:
            records = ask_facility_records(
                self.dll_path, b"WilcoATC procedures", _FIELDS, ident,
                self.TIMEOUT_S)
            found = None if records is None else decode(records, ident)
        except Exception:
            log.info("asking the simulator for %s's procedures raised",
                     ident, exc_info=True)
            found = None
        self._cache[ident] = found
        return found


__all__ = ["SimProcedures", "decode"]
