"""Shared fixtures for the sim-free test harness.

The traffic module builds a field from a navdb, so the harness supplies a
minimal fake navdb with one airport and one runway, plus a fake field and
clock. Nothing here touches the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from wilcoatc.atc.traffic import TrafficInjector

FIELD_LAT = 50.9
FIELD_LON = 1.4
FIELD_ELEV_FT = 180.0


@dataclass
class FakeRunway:
    le_ident: str = "09L"
    he_ident: str = "27R"
    le_heading: float = 90.0
    he_heading: float = 270.0
    le_lat: float = FIELD_LAT
    le_lon: float = FIELD_LON - 0.02
    he_lat: float = FIELD_LAT
    he_lon: float = FIELD_LON + 0.02
    le_elev_ft: float = FIELD_ELEV_FT
    he_elev_ft: float = FIELD_ELEV_FT + 10.0
    length_ft: float = 3500.0
    le_ils: Optional[str] = None
    he_ils: Optional[str] = None
    le_ils_kind: Optional[str] = None
    he_ils_kind: Optional[str] = None
    le_closed: bool = False
    he_closed: bool = False
    width_ft: float = 45.0

    def ends(self):
        return [
            (self.le_ident, self.le_heading, self.le_lat, self.le_lon),
            (self.he_ident, self.he_heading, self.he_lat, self.he_lon),
        ]


@dataclass
class FakeAirport:
    ident: str = "EGTEST"
    spoken: str = "Egtest"
    name: str = "Egtest"
    size_rank: int = 3
    elev_ft: float = FIELD_ELEV_FT
    lat: float = FIELD_LAT
    lon: float = FIELD_LON
    scheduled: bool = True


@dataclass
class FakeStation:
    ident: str = "EGTEST"
    position: str = "TWR"
    mhz: float = 118.5
    callsign: str = "Egtest Tower"
    facility: str = "TWR"
    description: str = "Tower"
    is_controller: bool = True


class FakeNavDB:
    """A navdb with one airport, one runway, no stations and no neighbours."""

    def __init__(self, airport: Optional[FakeAirport] = None,
                 runway: Optional[FakeRunway] = None,
                 stations: Optional[dict[str, list[FakeStation]]] = None):
        self.airport_row = airport or FakeAirport()
        self.runway_row = runway or FakeRunway()
        self._stations = stations or {}

    def airport(self, ident):
        return self.airport_row if (ident or "").upper() == self.airport_row.ident else None

    def station_for(self, ident, position):
        for station in self._stations.get((ident or "").upper(), []):
            if station.position == position:
                return station
        return None

    def stations(self, ident):
        return list(self._stations.get((ident or "").upper(), []))

    def runways(self, ident):
        if (ident or "").upper() == self.airport_row.ident:
            return [self.runway_row]
        return []

    def best_runway(self, ident, wind_dir, wind_kt, variable=False, gust_kt=0.0):
        return (self.runway_row.le_ident, float(self.runway_row.le_heading))

    def nearest_airports(self, lat, lon, limit=40, max_nm=80.0,
                         min_rank=1, require_runway=True):
        return []


def make_injector(seed: int = 42, density: float = 1.0,
                  navdb: Optional[FakeNavDB] = None,
                  requires_sim: bool = False, can_invent: bool = True,
                  controlled: bool = False) -> TrafficInjector:
    """Build a TrafficInjector pointed at the fake field, sim-free."""
    inject = TrafficInjector(navdb or FakeNavDB(), seed=seed, density=density)
    inject.requires_sim = requires_sim
    inject.can_invent = can_invent
    stations = None
    if controlled:
        stations = {
            "EGTEST": [
                FakeStation(ident="EGTEST", position="TWR", mhz=118.5,
                            is_controller=True),
                FakeStation(ident="EGTEST", position="GND", mhz=121.9,
                            is_controller=True),
            ]
        }
        inject.navdb = FakeNavDB(stations=stations)
    inject.set_field("EGTEST", departure_runway="09L", arrival_runway="09L",
                     weather=None, size_rank=3)
    return inject


def snapshot_of(inject: TrafficInjector) -> dict:
    """A timestamped, callsign-keyed view of the current positions."""
    out = {}
    for snap in inject.positions(invented_only=True):
        out[snap["callsign"]] = {
            "phase": snap["phase"],
            "lat": snap["lat"],
            "lon": snap["lon"],
            "altitude_ft": snap["altitude_ft"],
            "heading": snap["heading"],
            "ground_speed_kt": snap["ground_speed_kt"],
            "vertical_rate_fpm": snap["vertical_rate_fpm"],
            "on_ground": snap["on_ground"],
            "on_runway": snap["runway"] != "",
            "sequence": snap["sequence"],
            "order": snap["order"],
            "intent": snap["intent"],
            "observed": snap["observed"],
        }
    return out


def ground_distance_m(lat1, lon1, lat2, lon2) -> float:
    north = (lat2 - lat1) * 111320.0
    east = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat1))
    return math.hypot(north, east)


class RecordingFlightDirector:
    """A stub FlightDirector that records every command instead of sending it."""

    def __init__(self, sim=None, mode: str = "managed"):
        self.sim = sim
        self.mode = mode
        self.calls: list[tuple[str, tuple, dict]] = []
        self._sent: set[str] = set()

    def reset(self) -> None:
        self._sent.clear()

    def send(self, name, value=0):
        self.calls.append(("send", (name,), {"value": value}))
        return True

    def once(self, key, name, value=0):
        self.calls.append(("once", (key, name), {"value": value}))
        self._sent.add(key)
        return True

    def forget(self, *keys):
        self.calls.append(("forget", keys, {}))
        for key in keys:
            self._sent.discard(key)

    def write(self, variable, value):
        self.calls.append(("write", (variable,), {"value": value}))
        return True

    def autopilot(self, on=True):
        self.calls.append(("autopilot", (), {"on": on}))
        return True

    def heading(self, degrees):
        self.calls.append(("heading", (), {"degrees": degrees}))
        return True

    def altitude(self, feet):
        self.calls.append(("altitude", (), {"feet": feet}))
        return True

    def vertical_speed(self, fpm):
        self.calls.append(("vertical_speed", (), {"fpm": fpm}))
        return True

    def level_change(self, on=True):
        self.calls.append(("level_change", (), {"on": on}))
        return True

    def nav(self, on=True):
        self.calls.append(("nav", (), {"on": on}))
        return True

    def localizer(self, on=True):
        self.calls.append(("localizer", (), {"on": on}))
        return True

    def approach(self, on=True):
        self.calls.append(("approach", (), {"on": on}))
        return True

    def speed(self, knots):
        self.calls.append(("speed", (), {"knots": knots}))
        return True

    def throttle(self, position):
        self.calls.append(("throttle", (), {"position": position}))
        return True

    def flaps(self, position):
        self.calls.append(("flaps", (), {"position": position}))
        return True

    def gear(self, down):
        self.calls.append(("gear", (), {"down": down}))
        return True

    def spoilers(self, position):
        self.calls.append(("spoilers", (), {"position": position}))
        return True

    def aileron(self, position):
        self.calls.append(("aileron", (), {"position": position}))
        return True

    def elevator(self, position):
        self.calls.append(("elevator", (), {"position": position}))
        return True

    def rudder(self, position):
        self.calls.append(("rudder", (), {"position": position}))
        return True


class FakeFlightSession:
    """A minimal FlightSession stand-in for the autonomous harness."""

    def __init__(self):
        self.phase = None
        self.aircraft = None
        self.destination = None
        self.departure_runway = ""
        self.arrival_runway = ""
        self.assigned_altitude_ft = None
        self.callsign = None
        self.ifr = False
        self.pending = None
        self.handoff_pending = None
        self.emergency = None
        self.emergency_details = {}
        self.established_stations = set()
        self.clearance_issued = False
        self.flight_plan_open = False
        self.pattern_work = False
        self.expected_exit = ""
        self.expected_vrp = ""
        self.traffic_called = {}
        self.approach_choice = None
        self.approach_key = ()
        self.readback_items = {}
        self.owed_items = {}

    def establish(self, station):
        self.established_stations.add(getattr(station, "ident", station))

    def established_with(self, station):
        return getattr(station, "ident", station) in self.established_stations

    def apply(self, values):
        pass

    def clear_pending(self):
        self.pending = None

    def remember(self, instruction):
        return instruction


class FakeAircraftState:
    """A minimal AircraftState stand-in."""

    def __init__(self, *, on_ground=True, altitude_ft=180.0, lat=FIELD_LAT,
                 lon=FIELD_LON, heading=90.0, gs=0.0, vs=0.0, ias=0.0,
                 on_ground_agl=None):
        self.on_ground = on_ground
        self.altitude_ft = altitude_ft
        self.latitude = lat
        self.longitude = lon
        self.heading_true = heading
        self.ground_speed_kt = gs
        self.vertical_speed_fpm = vs
        self.indicated_airspeed_kt = ias
        self.altitude_agl_ft = on_ground_agl if on_ground_agl is not None \
            else (altitude_ft - FIELD_ELEV_FT if not on_ground else 0.0)
        self.has_position = True

    def distance_to(self, lat, lon):
        return ground_distance_m(self.latitude, self.longitude, lat, lon) / 1852.0


class FakeCrew:
    """A minimal crew stand-in that records its calls."""

    def __init__(self):
        self.calls = []
        self.circuit = None
        self.frequency_for = None
        self.distance_to_field = None
        self.radio_call = None
        self.transmit = None

    def __getattr__(self, name):
        def _stub(*a, **k):
            self.calls.append((name, a, k))
            return None
        return _stub


class FakeBrain:
    """A minimal controller brain stand-in."""

    def __init__(self):
        self.traffic = None
        self.route = None
        self.route_progress = None
        self.plan_fixes = []
        self.plan_runways = {}
        self.arrivals = None

    def for_station(self, station, language="en"):
        class P:
            dialect = "faa"
            def roger(self, aircraft):
                return "roger"
        return P()
