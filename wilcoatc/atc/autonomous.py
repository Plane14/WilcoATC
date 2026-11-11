# -*- coding: utf-8 -*-
"""Autonomous flight control for the First Officer.

When enabled, the First Officer manages the entire flight from startup to
shutdown using the simulator's autopilot and aircraft systems. This includes
all phases: preflight, pushback, taxi, takeoff, climb, cruise, descent,
approach, landing, taxi-in, and shutdown.

Two autopilot management modes are supported (``autonomous_ap_mode``):
- "managed": the aeroplane's own managed modes -- flight-level change for
  the climb and descent, NAV for the route, APR for the approach, and the
  autothrottle's TO/GA where the model has one.
- "direct": headings, levels, vertical speeds and indicated airspeeds set
  explicitly, for the aeroplanes whose autopilots do not fly managed.

The controls go through :class:`~wilcoatc.atc.ap.FlightDirector`, which
tries the cleanest means first -- autopilot modes, then the ``*_SET``
control events for the moments the AP cannot fly (rotation, the flare,
the go-around's pitch) -- and the geometry through
:mod:`wilcoatc.atc.auto_nav`.

When chained flights are enabled, the First Officer will automatically
select random destinations within the aircraft's range at airports that
support the aircraft type and have ground services (fuel, parking, etc.).
"""

from __future__ import annotations

import logging
import random
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..config import Config
from ..navdata.build import haversine_nm
from ..navdata.db import Airport, NavDB
from ..sim.source import SimConnectSource
from . import approach_flight, auto_nav, vspeeds
from . import flying as fly
from .ap import FlightDirector
from .controller import ControllerBrain
from .crew import CrewDirector
from .session import FlightSession, Phase

log = logging.getLogger(__name__)


class AutonomousPhase(str, Enum):
    """Internal autonomous flight phases."""
    INIT = "init"
    PREFLIGHT = "preflight"
    REQUEST_CLEARANCE = "request_clearance"
    PUSHBACK = "pushback"
    TAXI_OUT = "taxi_out"
    HOLDING_SHORT = "holding_short"
    LINE_UP = "line_up"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    TAXI_IN = "taxi_in"
    PARKED = "parked"
    SHUTDOWN = "shutdown"
    CHAIN_NEXT = "chain_next"
    COMPLETE = "complete"
    ERROR = "error"


# The sub-steps inside the phases that are sequences rather than waits.
# They live on the state as ``sub`` and the phase methods dispatch on it.
SUB_STATES = (
    # pushback
    "engines", "after_start", "request", "wait",
    # taxi
    "taxi", "hold_point",
    # line up
    "align",
    # takeoff
    "wait_clearance", "roll", "rotate", "initial_climb",
    # approach
    "intercept", "final", "go_around",
    # landing
    "flare", "rollout",
    # shutdown
    "secure", "cooldown", "engines_off", "electrics_off",
)


@dataclass
class AutonomousState:
    """State of the autonomous flight."""
    phase: AutonomousPhase = AutonomousPhase.INIT
    previous_phase: AutonomousPhase = AutonomousPhase.INIT
    phase_entered_at: float = 0.0
    # Flight plan and route
    has_ifr_clearance: bool = False
    clearance_read_back: bool = False
    departure_runway: str = ""
    arrival_runway: str = ""
    assigned_altitude_ft: float = 0.0
    assigned_heading: float = 0.0
    assigned_speed_kt: float = 0.0
    # The sub-step inside a phase, and when it began.
    sub: str = ""
    sub_entered_at: float = 0.0
    # Pushback tracking: the simulator gives no "tug attached" variable, so
    # the phase watches the groundspeed instead -- moving means the push is
    # under way, stationary again for a while means it is done.
    push_requested: bool = False
    moved_since_push: bool = False
    still_since: float = 0.0
    # Engines: when the start was asked for, so a model that never runs
    # them is an error rather than an endless wait.
    engines_started_at: float = 0.0
    start_retried: bool = False
    push_retries: int = 0
    # Taxi: progress along session.taxi_path and where it is held.
    taxi_index: int = 0
    hold_point_m: float = 0.0
    # Line up: the runway's own numbers, once found.
    runway_lat: float = 0.0
    runway_lon: float = 0.0
    runway_heading: float = 0.0
    runway_length_ft: float = 0.0
    aligned: bool = False
    # Takeoff: when rotation began, whether the wheels are up, and the
    # rejected-takeoff watch on the roll.
    rotate_at: float = 0.0
    airborne_at: float = 0.0
    rotate_speed_kt: float = 0.0
    rejected: bool = False
    # The speeds this leg is flown at, from vspeeds.
    v1_kt: float = 0.0
    vr_kt: float = 0.0
    v2_kt: float = 0.0
    vapp_kt: float = 0.0
    climb_kt: float = 0.0
    # Climb/cruise: the top of climb, and the step the controllers are on.
    top_of_climb: bool = False
    # Descent/approach: the planned top of descent, the configuration
    # stage, the stabilized-approach verdict, the minima, and the
    # go-arounds flown.
    tod_nm: float = 0.0
    configured: str = ""
    stabilized: bool = False
    minima_ft: float = 200.0
    go_arounds: int = 0
    go_around_at: float = 0.0
    missed_alt_ft: float = 0.0
    # The cleared approach as a flyable recipe (approach_flight.resolve),
    # resolved once and kept until the field or the runway changes; whether
    # NAV1 has been tuned to its navaid yet, when that was last tried, and
    # the circling pattern's path once the circle has started.
    approach_recipe: object = None
    approach_recipe_tried: bool = False
    approach_recipe_key: tuple = ()
    approach_tuned: bool = False
    nav_tune_at: float = 0.0
    circling: bool = False
    circle_path: list = field(default_factory=list)
    # Landing: the flare's clock and the rollout's.
    flare_at: float = 0.0
    touchdown_at: float = 0.0
    rollout_at: float = 0.0
    vacated: bool = False
    # Shutdown: the stage clock.
    shutdown_at: float = 0.0
    # Emergencies: what was declared, to where, and whether the divert is
    # already set up.
    emergency: str = ""
    emergency_declared: bool = False
    emergency_destination: Optional[Airport] = None
    # Chain flights
    chain_flight_index: int = 0
    chain_flights: list[dict] = field(default_factory=list)
    # Aircraft performance for chain planning
    aircraft_range_nm: float = 0.0
    aircraft_type: str = ""
    # Error tracking
    error_message: str = ""
    error_count: int = 0
    # Autopilot bookkeeping the director keeps itself; these flags remain
    # for anything that reads them.
    ap_master: bool = False
    ap_hdg_mode: bool = False
    ap_alt_mode: bool = False
    ap_apr_mode: bool = False
    ap_nav_mode: bool = False
    ap_vs_mode: bool = False
    ap_speed_mode: bool = False
    # Checklist tracking
    checklists_completed: set[str] = field(default_factory=set)
    # Timing
    last_action_at: float = 0.0


class AutonomousFlight:
    """Manages autonomous flight from startup to shutdown."""

    def __init__(
        self,
        config: Config,
        navdb: NavDB,
        sim_source: Optional[SimConnectSource],
        session: FlightSession,
        crew: CrewDirector,
        brain: ControllerBrain,
        get_state: Callable[[], object],
        on_event: Callable[[str, str, str], None],
        use_route: Optional[Callable[[object], None]] = None,
    ):
        self.config = config
        self.navdb = navdb
        self.sim = sim_source
        self.session = session
        self.crew = crew
        self.brain = brain
        self.get_state = get_state
        self.on_event = on_event
        self.use_route = use_route

        self.state = AutonomousState()
        self._enabled = config.immersion.autonomous_flight
        # The rotorcraft controller, when the aeroplane in the sim is a
        # helicopter: it flies the whole leg itself and this class only
        # delegates (wilcoatc.atc.autonomous_heli).
        self._heli = None
        self._is_helicopter = False
        self._chained = config.immersion.autonomous_chained_flights
        self._ap_mode = config.immersion.autonomous_ap_mode
        self._chain_delay = config.immersion.autonomous_chain_delay_s
        self._request_ifr = config.immersion.autonomous_request_ifr
        self._use_sim_ai = config.immersion.autonomous_use_sim_ai
        self._dir: Optional[FlightDirector] = None
        self._perf: Optional[fly.Performance] = None
        self._speeds: Optional[vspeeds.Speeds] = None
        self._fuel_known = False
        self._ga_flag_seen = False
        self._last_state = None

        # Compute aircraft range for chain planning. The engine builds this
        # before a flight exists, so the session can be None.
        if session is not None:
            self._compute_aircraft_performance(session)

        # Timing constants
        self.ACTION_INTERVAL = 1.0  # seconds between autonomous actions
        self.CLEARANCE_TIMEOUT = 60.0
        self.ENGINE_START_TIMEOUT = 240.0
        self.PUSHBACK_TIMEOUT = 300.0
        self.TAXI_TIMEOUT = 600.0
        self.TAKEOFF_TIMEOUT = 240.0
        self.CLIMB_TIMEOUT = 3600.0
        self.DESAT_TIMEOUT = 3600.0
        self.APPROACH_TIMEOUT = 1200.0
        self.LANDING_TIMEOUT = 600.0
        self.SHUTDOWN_COOLDOWN_S = 90.0

    # -- plumbing ---------------------------------------------------------

    @property
    def director(self) -> FlightDirector:
        """The hands, made on first use so a missing sim is not an error.

        The engine swaps ``self.sim`` when the simulator reconnects, so a
        director holding the old source is remade rather than kept.
        """
        if self._dir is None or self._dir.sim is not self.sim:
            self._dir = FlightDirector(self.sim, self._ap_mode)
        return self._dir

    @property
    def performance(self) -> fly.Performance:
        if self._perf is None:
            code = getattr(getattr(self.session, "aircraft", None),
                           "type_code", "") or ""
            light = getattr(getattr(self.session, "profile", None),
                            "kind", "") in ("piston_single", "piston_twin",
                                            "light")
            self._perf = fly.performance_for(code, light=light)
        return self._perf

    def _compute_speeds(self, state=None) -> vspeeds.Speeds:
        """This leg's V-speeds, once: the model's own where it declares
        them, the table where it does not."""
        if self._speeds is not None:
            return self._speeds
        session = self.session
        code = getattr(getattr(session, "aircraft", None), "type_code", "") or ""
        if state is None:
            state = self.get_state()
        cls = getattr(self, "_aircraft_class", "") or "airliner"
        self._speeds = vspeeds.speeds_for(
            code, cls,
            min_rotation_kt=float(getattr(state, "min_rotation_kt", 0.0) or 0.0),
            design_takeoff_kt=float(getattr(state, "design_takeoff_kt", 0.0)
                                    or 0.0))
        s = self.state
        s.v1_kt = float(self._speeds.v1 or 0.0)
        s.vr_kt = float(self._speeds.vr or 0.0)
        s.v2_kt = float(self._speeds.v2 or 0.0)
        s.vapp_kt = float(self._speeds.vapp or 0.0)
        # The model's own stall speed is a floor under the approach speed:
        # a table figure below 1.25 times Vs1 is a bug in the table, not a
        # fast approach.
        vs1 = float(getattr(state, "design_vs1_kt", 0.0) or 0.0)
        if vs1 > 0 and (not s.vapp_kt or s.vapp_kt < vs1 * 1.25):
            s.vapp_kt = vs1 * 1.25
        s.climb_kt = self.performance.climb_kt
        return self._speeds

    def _goto_sub(self, sub: str) -> None:
        """A new sub-step inside the phase."""
        self.state.sub = sub
        self.state.sub_entered_at = time.monotonic()

    def _sub_age(self) -> float:
        return time.monotonic() - self.state.sub_entered_at

    # -- entry -------------------------------------------------------------

    def _detect_helicopter(self, session: FlightSession) -> bool:
        """Whether the aeroplane in the sim is a helicopter.

        The simulator's own classification first (CATEGORY / ENGINE TYPE /
        the Doc 8643 designator); the pilot's helicopter profile as the
        fallback for a session begun before the state has answered.
        """
        from . import aircraft_class as ac

        try:
            state = self.get_state()
        except Exception:
            state = None
        if state is not None and getattr(state, "connected", False):
            found = ac.of_state(state)
            self._aircraft_class = found.name
            return found.name == ac.HELICOPTER
        if getattr(session.profile, "kind", "") == "helicopter":
            self._aircraft_class = ac.HELICOPTER
            return True
        found = ac.classify(
            getattr(session.aircraft, "type_code", ""),
            airline=getattr(session.profile, "airliner", False)
        )
        self._aircraft_class = found.name
        return found.name == ac.HELICOPTER

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        self.config.immersion.autonomous_flight = True

    def disable(self) -> None:
        self._enabled = False
        self.config.immersion.autonomous_flight = False

    def begin_flight(self, session: FlightSession) -> None:
        """Start a new autonomous flight."""
        self.session = session
        self.state = AutonomousState()
        self.state.phase = AutonomousPhase.PREFLIGHT
        self.state.phase_entered_at = time.monotonic()
        self._goto_sub("systems")
        if self._dir is not None:
            self._dir.reset()
        self._perf = None
        self._speeds = None
        self._fuel_known = False
        self._heli = None
        # Compute aircraft performance for chain planning
        self._compute_aircraft_performance(session)
        self._is_helicopter = self._detect_helicopter(session)
        if self._is_helicopter:
            from .autonomous_heli import RotorcraftFlight

            self._heli = RotorcraftFlight(self)
            self._heli.begin(session)
        if self._use_sim_ai:
            # There is no SimConnect path to the simulator's own AI pilot;
            # the flag is kept for the config surface, and the flight is
            # flown on the First Officer's own autopilot regardless.
            self._emit("info", "Simulator AI assist is not available; "
                       "flying with the First Officer's own autopilot",
                       "autonomous.sim_ai.unused")
        self._emit("info", "Autonomous flight started", "autonomous.started")

    def set_chain_flights(self, flights: list[dict]) -> None:
        """Set up chained flights from a list of flight plans."""
        self.state.chain_flights = flights
        self.state.chain_flight_index = 0

    # -- the loop ------------------------------------------------------------

    def tick(self) -> None:
        """Main autonomous flight loop. Called from the situation loop."""
        if not self._enabled:
            return

        state = self.get_state()
        if state is None or not state.connected:
            return
        self._last_state = state

        # A helicopter is flown by its own controller, on every pass of the
        # loop rather than the fixed-wing action interval: the cyclic,
        # collective and pedals are closed-loop controls, not once-a-second
        # mode selections.
        if self._heli is not None:
            self._heli.tick(state)
            return

        now = time.monotonic()

        # The phases that are flown by hand -- rotation, the flare, the
        # go-around's pitch, the taxi -- want a faster hand than the 1 Hz
        # mode selections the rest of the flight needs.
        interval = self.ACTION_INTERVAL
        if self.state.phase in (AutonomousPhase.TAKEOFF,
                                AutonomousPhase.LANDING,
                                AutonomousPhase.TAXI_OUT,
                                AutonomousPhase.TAXI_IN,
                                AutonomousPhase.LINE_UP) \
                and self.state.sub in ("rotate", "flare", "taxi",
                                       "align", "roll", "rollout"):
            interval = 0.25
        if now - self.state.last_action_at < interval:
            return

        # Pushback progress, which the phase reads: a push that has moved the
        # aeroplane and then stopped is done, however the tug was attached.
        if self.state.phase == AutonomousPhase.PUSHBACK:
            if (state.ground_speed_kt or 0.0) > 2.0:
                self.state.moved_since_push = True
                self.state.still_since = now
            elif self.state.moved_since_push and self.state.still_since == 0.0:
                self.state.still_since = now

        # Emergencies outrank the phase: a MAYDAY changes what the flight
        # is for before it changes where the aeroplane is pointed.
        self._monitor_emergency(state, now)

        self._execute_phase(state, now)

        self.state.last_action_at = now

    def _execute_phase(self, state, now: float) -> None:
        """Execute logic for the current autonomous phase."""
        phase = self.state.phase

        try:
            if phase == AutonomousPhase.PREFLIGHT:
                self._phase_preflight(state, now)
            elif phase == AutonomousPhase.REQUEST_CLEARANCE:
                self._phase_request_clearance(state, now)
            elif phase == AutonomousPhase.PUSHBACK:
                self._phase_pushback(state, now)
            elif phase == AutonomousPhase.TAXI_OUT:
                self._phase_taxi_out(state, now)
            elif phase == AutonomousPhase.HOLDING_SHORT:
                self._phase_holding_short(state, now)
            elif phase == AutonomousPhase.LINE_UP:
                self._phase_line_up(state, now)
            elif phase == AutonomousPhase.TAKEOFF:
                self._phase_takeoff(state, now)
            elif phase == AutonomousPhase.CLIMB:
                self._phase_climb(state, now)
            elif phase == AutonomousPhase.CRUISE:
                self._phase_cruise(state, now)
            elif phase == AutonomousPhase.DESCENT:
                self._phase_descent(state, now)
            elif phase == AutonomousPhase.APPROACH:
                self._phase_approach(state, now)
            elif phase == AutonomousPhase.LANDING:
                self._phase_landing(state, now)
            elif phase == AutonomousPhase.TAXI_IN:
                self._phase_taxi_in(state, now)
            elif phase == AutonomousPhase.PARKED:
                self._phase_parked(state, now)
            elif phase == AutonomousPhase.SHUTDOWN:
                self._phase_shutdown(state, now)
            elif phase == AutonomousPhase.CHAIN_NEXT:
                self._phase_chain_next(state, now)
            elif phase == AutonomousPhase.ERROR:
                self._phase_error(state, now)
        except Exception as e:
            log.error("Autonomous flight error in phase %s: %s", phase, e, exc_info=True)
            self._set_error(f"Phase {phase.value}: {e}")

    # -- shared helpers ------------------------------------------------------

    def _altitude_target(self, state) -> float:
        """The level to fly now: what ATC last assigned, else the plan's."""
        return (self.session.assigned_altitude_ft
                or self.session.cruise_altitude_ft
                or self.session.initial_altitude_ft or 0.0)

    def _speed_target(self, state) -> float:
        if self.session.assigned_speed_kt:
            return self.session.assigned_speed_kt
        # In the cruise the model's own estimate of its cruise speed is
        # the best figure there is -- it is this aeroplane's, not a class's
        # -- and the class number is the fallback for a model that
        # declares none.
        if self.state.phase == AutonomousPhase.CRUISE:
            estimated = float(getattr(state, "estimated_cruise_kt", 0.0)
                              or 0.0)
            if 50.0 < estimated < 600.0:
                # The variable may be true rather than indicated speed;
                # never command past the model's own red line.
                redline = float(getattr(state, "barber_pole_kt", 0.0)
                                or 0.0)
                if redline > 0:
                    estimated = min(estimated, redline * 0.95)
                return estimated
        return self.state.climb_kt or self.performance.cruise_kt

    def _apply_flight_director(self, state, altitude: float = 0.0,
                             vs_fpm: float = 0.0, heading: float = -1.0,
                             speed_kt: float = 0.0) -> None:
        """Put the assigned numbers on the autopilot, managed or direct.

        ``heading`` < 0 means "not commanded"; ``vs_fpm`` nonzero means a
        rate is wanted (climb or descent in progress); zero means level.
        """
        d = self.director
        if not self.state.ap_master:
            d.autopilot(True)
            self.state.ap_master = True
        if altitude:
            d.altitude(altitude)
        if heading >= 0.0:
            d.heading(heading)
            self.state.ap_hdg_mode = True
        if speed_kt:
            d.speed(speed_kt)
            self.state.ap_speed_mode = True
        if vs_fpm:
            d.vertical_speed(vs_fpm)
            d.vs_hold(True)
            self.state.ap_vs_mode = True
        else:
            d.vs_hold(False)
            self.state.ap_vs_mode = False
            if altitude:
                d.alt_hold(True)
                self.state.ap_alt_mode = True
        if self._ap_mode == "managed":
            # Managed legs climb and descend on the flight-level-change
            # mode where the model has it; direct legs use the VS bug.
            if vs_fpm and self.state.phase in (
                    AutonomousPhase.CLIMB, AutonomousPhase.DESCENT):
                d.level_change(True)
            elif self.state.phase in (AutonomousPhase.CLIMB,
                                      AutonomousPhase.DESCENT) \
                    and abs(vs_fpm) < 1:
                d.level_change(False)

    def _run_checklist_for_phase(self, checklist_name: str) -> None:
        """Trigger a checklist for the current phase."""
        if checklist_name not in self.state.checklists_completed:
            self.crew.checklists.start(checklist_name)
            self.state.checklists_completed.add(checklist_name)

    def _configure_for_takeoff(self, state) -> None:
        """Set flaps, trim, lights for takeoff."""
        d = self.director
        # Takeoff flaps: the first detent, where the model has one.
        d.flaps(0.25 if state.flaps_index == 0 else 0.0)
        d.landing_light(True)
        d.strobes(True)

    def _set_takeoff_power(self) -> None:
        """Takeoff thrust: full, and TO/GA on the autothrottle where the
        model has one."""
        d = self.director
        d.takeoff_power()
        d.throttle(1.0)
        d.toga(True)

    def _steer(self, state, desired_hdg: float, target_kt: float,
               max_aileron: float = 0.35) -> None:
        """Hand-fly a heading and a speed on the ground or in the air.

        The rudder (which is the nosewheel on the ground) takes the
        heading error; the throttle takes the speed. Used for taxi,
        line-up, the takeoff roll's directional control and the rollout --
        the places an autopilot's heading mode does not reach.
        """
        d = self.director
        d.parking_brake(False)
        err = ((desired_hdg - (state.heading_true or 0.0) + 540.0)
               % 360.0 - 180.0)
        # The gain is the class's: a heavy needs more nosewheel for the
        # same error than a light single, and the performance model's turn
        # rate is the proxy it already carries.
        divisor = 18.0 if self.performance.turn_rate < fly.RATE_LIGHT \
            else 28.0
        rudder = max(-0.5, min(0.5, err / divisor))
        d.rudder(rudder)
        # Taxi and rollout speeds are throttle jobs, not AP jobs -- and
        # progressive ones: the further below target, the more asked for.
        gs = state.ground_speed_kt or 0.0
        if gs < target_kt - 2.0:
            d.throttle(max(0.05, min(0.6,
                                     0.10 + (target_kt - gs) * 0.01)))
        elif gs > target_kt + 4.0 and state.on_ground:
            d.throttle(0.0)

    def _stop(self, state, target_kt: float = 0.0) -> None:
        """Roll out to a stop: idle, controls centred."""
        d = self.director
        d.throttle(0.0)
        d.throttle_idle()
        gs = state.ground_speed_kt or 0.0
        if gs <= 1.0:
            d.parking_brake(True)
            d.rudder(0.0)
            d.center_controls()

    # -- the phase machine ----------------------------------------------------

    def _phase_preflight(self, state, now: float) -> None:
        """Preflight: aircraft configured, checks done, clearance next."""
        d = self.director
        sub = self.state.sub or "systems"

        if sub == "systems":
            self._ensure_systems_on(state)
            # The outside checks a FO runs before the before-start list:
            # transponder on with the assigned squawk, the altimeter set,
            # the lights that belong on for the phase.
            if not getattr(state, "transponder_state", -1) in (3, 4):
                d.transponder(self.session.squawk or "1200")
            if getattr(state, "kohlsman_hpa", 0.0) > 0 and self.session.altimeter:
                d.kohlsman(float(self.session.altimeter))
            d.beacon(True)
            d.nav_lights(True)
            d.seatbelts(True)
            d.parking_brake(True)
            self._goto_sub("checklist")

        elif sub == "checklist":
            self._run_checklist_for_phase("before_start")
            self._goto_sub("clearance")

        elif sub == "clearance":
            if self.session.ifr and self._request_ifr and self.session.destination:
                self._transition(AutonomousPhase.REQUEST_CLEARANCE)
            else:
                self._transition(AutonomousPhase.PUSHBACK)

    def _phase_request_clearance(self, state, now: float) -> None:
        """Request and receive IFR clearance."""
        if not self.state.has_ifr_clearance:
            # Request clearance via first officer
            self._request_clearance_via_fo()
            self.state.has_ifr_clearance = True  # Will be confirmed by ATC
        elif self.session.clearance_read_back or self.session.clearance_issued:
            # Clearance received and read back
            self._transition(AutonomousPhase.PUSHBACK)
        elif now - self.state.phase_entered_at > self.CLEARANCE_TIMEOUT:
            # No clearance came; a VFR departure is the realistic fallback
            # rather than an error, when the field allows it.
            if self.session.ifr and self._request_ifr:
                self._emit("info", "No IFR clearance; continuing VFR",
                           "autonomous.clearance.vfr_fallback")
                self.session.ifr = False
                self._transition(AutonomousPhase.PUSHBACK)
            else:
                self._set_error("Clearance timeout")

    def _phase_pushback(self, state, now: float) -> None:
        """Engine start, pushback, brakes back on."""
        d = self.director
        sub = self.state.sub or "engines"

        if sub == "engines":
            if not state.engines_running:
                if self.state.engines_started_at == 0.0:
                    self._start_engines(state)
                    self.state.engines_started_at = now
                elif now - self.state.engines_started_at > 30.0 \
                        and not self.state.start_retried:
                    self._start_engines(state)
                    self.state.start_retried = True
                elif now - self.state.engines_started_at > self.ENGINE_START_TIMEOUT:
                    self._set_error("Engine start timed out")
                    return
            else:
                self._goto_sub("after_start")

        elif sub == "after_start":
            self._run_checklist_for_phase("after_start")
            d.parking_brake(True)  # set for the push
            self._goto_sub("request")

        elif sub == "request":
            self._request_pushback()
            self.state.push_requested = True
            self._goto_sub("wait")

        elif sub == "wait":
            # The aeroplane moved under the tug, then stopped: the push is
            # done. The parking brake goes on before anything moves under
            # its own power.
            if self.state.moved_since_push and self.state.still_since \
                    and now - self.state.still_since > 5.0:
                d.parking_brake(True)
                self._goto_sub("taxi")
                self._transition(AutonomousPhase.TAXI_OUT)
                return
            if not self.state.moved_since_push:
                # The tug never moved it: ask again, a few times, then
                # give up rather than taxi out attached to nothing.
                if self._sub_age() > 60.0 and self.state.push_retries < 3:
                    self._request_pushback()
                    self.state.push_retries += 1
                    self.state.sub_entered_at = now
                if self.state.push_retries >= 3 and self._sub_age() > 60.0:
                    self._set_error("Pushback did not move the aircraft")
                    return
            if now - self.state.phase_entered_at > self.PUSHBACK_TIMEOUT:
                self._set_error("Pushback timeout")

    def _phase_taxi_out(self, state, now: float) -> None:
        """Taxi to the departure runway along the planned route."""
        self._run_checklist_for_phase("taxi")
        if self.session.phase == Phase.HOLDING_SHORT:
            self._transition(AutonomousPhase.HOLDING_SHORT)
            return

        sub = self.state.sub or "taxi"
        if sub == "taxi":
            self._taxi_along_path(state, outbound=True)
        elif sub == "hold_point":
            self._stop(state)
            if (state.ground_speed_kt or 0.0) <= 1.0:
                self._goto_sub("taxi")
                self._transition(AutonomousPhase.HOLDING_SHORT)
                return

        if now - self.state.phase_entered_at > self.TAXI_TIMEOUT:
            self._set_error("Taxi timeout")

    def _phase_holding_short(self, state, now: float) -> None:
        """Holding short of runway, run before-takeoff checklist."""
        if (state.ground_speed_kt or 0.0) > 1.0:
            self._stop(state)
            return
        self._run_checklist_for_phase("before_takeoff")

        # "Line up and wait" puts the aeroplane on the runway; the takeoff
        # clearance is what starts the roll.
        if self.session.lined_up or self.session.cleared_takeoff \
                or self.session.phase in (Phase.LINED_UP, Phase.TAKEOFF):
            self._transition(AutonomousPhase.LINE_UP)

    def _phase_line_up(self, state, now: float) -> None:
        """Line up on runway and prepare for takeoff."""
        d = self.director
        sub = self.state.sub or "align"

        if not state.on_ground:
            self._transition(AutonomousPhase.TAKEOFF)
            return

        if sub == "align":
            self._find_runway(state)
            if self.state.runway_heading <= 0:
                # No runway geometry: hold where we are and wait for the
                # clearance rather than steering blind.
                self._stop(state)
                if self.session.cleared_takeoff:
                    self._transition(AutonomousPhase.TAKEOFF)
                return
            # Taxi onto the centreline and align with the runway heading.
            # ``align`` says how far off the extended centreline the
            # aeroplane is and the heading that puts it back on; steering
            # at that heading is the FO's interception, then the runway
            # heading itself once the wings are level on the numbers.
            err = auto_nav.align(state.latitude, state.longitude,
                                 self.state.runway_lat, self.state.runway_lon,
                                 self.state.runway_heading)
            hdg_err = abs(((self.state.runway_heading
                            - (state.heading_true or 0.0) + 540.0)
                           % 360.0 - 180.0))
            steer_hdg = (self.state.runway_heading
                         + err["correction_deg"]) % 360.0
            if abs(err["offset_m"]) < 60.0:
                steer_hdg = self.state.runway_heading
            d.parking_brake(False)
            self._steer(state, steer_hdg, self.performance.taxi_kt)
            if abs(err["offset_m"]) < 25.0 and hdg_err < 8.0:
                self.state.aligned = True
                self._goto_sub("set")
                self._emit("info", "Lined up on runway",
                           "autonomous.lined_up")
            elif self._sub_age() > 120.0:
                # Could not line up: hold short rather than roll onto the
                # runway unaligned.
                self._goto_sub("taxi")
                self._transition(AutonomousPhase.HOLDING_SHORT)
                return

        elif sub == "set":
            # Aligned: brakes on, checks complete, wait for the roll.
            d.parking_brake(True)
            self._stop(state)
            self._configure_for_takeoff(state)
            if self.session.cleared_takeoff or self.session.phase == Phase.TAKEOFF:
                self._transition(AutonomousPhase.TAKEOFF)
            elif self._sub_age() > 300.0:
                self._set_error("Lined up but never cleared for takeoff")

    def _phase_takeoff(self, state, now: float) -> None:
        """Takeoff roll, rotation, initial climb."""
        d = self.director
        sub = self.state.sub or "wait_clearance"
        self._compute_speeds(state)

        if sub == "wait_clearance":
            if not state.on_ground:
                self._goto_sub("initial_climb")
                return
            d.parking_brake(True)
            self._stop(state)
            if self.session.cleared_takeoff:
                self.state.rejected = False
                self._goto_sub("roll")
            elif self._sub_age() > 300.0:
                self._set_error("Takeoff clearance never came")
                return
            return

        if sub == "roll":
            if not state.on_ground:
                self.state.airborne_at = now
                self._goto_sub("rotate")
                return
            # Brakes off, takeoff power, rudder for direction -- the roll.
            d.parking_brake(False)
            # Rejected takeoff: the roll is not accelerating. Below 60% of
            # rotation speed two thirds of the way through the expected
            # roll is an abort, not a rotate.
            vr = self.state.vr_kt or self.performance.rotate_kt
            elapsed = now - self.state.sub_entered_at
            expected = max(10.0, vr / max(self.performance.takeoff_kt_s, 0.5))
            ias = state.indicated_airspeed_kt or 0.0
            if ias < vr * 0.6 and elapsed > expected * 0.66:
                self.state.rejected = True
                self._emit("error", "Rejected takeoff: insufficient acceleration",
                           "autonomous.rejected_takeoff")
                self._goto_sub("abort")
                return
            self._set_takeoff_power()
            # Directional control with the rudder into the runway heading.
            err = ((self.state.runway_heading - (state.heading_true or 0.0)
                    + 540.0) % 360.0 - 180.0)
            if abs(err) < 2.0:
                d.rudder(0.0)
            else:
                d.rudder(max(-0.6, min(0.6, err / 10.0)))
            rotate_at = (self.state.vr_kt
                         or self.state.min_rotation_kt
                         or self.performance.rotate_kt)
            self.state.rotate_speed_kt = rotate_at
            if ias >= rotate_at:
                self._goto_sub("rotate")
                self.state.rotate_at = now
            elif self._sub_age() > self.TAKEOFF_TIMEOUT:
                self._set_error("Takeoff roll timed out")
            return

        if sub == "rotate":
            if not state.on_ground:
                self._goto_sub("initial_climb")
                return
            # Pull to the rotation attitude and hold it until the wheels
            # are up. ELEVATOR_SET is proportional, so the FO can fly the
            # rotation the way a pilot does: a firm pull, then hands off.
            elapsed = now - self.state.rotate_at
            # The pull is the class's too: a light single rotates with
            # less back pressure than a loaded airliner.
            pull_scale = (1.0 if self.performance.turn_rate < fly.RATE_LIGHT
                          else 0.8)
            pull = (0.55 if elapsed < 2.5 else 0.35) * pull_scale
            d.elevator(pull)
            d.takeoff_power()
            if elapsed > 8.0:
                # Rotated and still on the ground: the model is not
                # flying itself off; lower the nose and reject rather
                # than porpoise down the runway.
                d.elevator(0.0)
                self.state.rejected = True
                self._goto_sub("abort")
            return

        if sub == "abort":
            # Rejected takeoff: idle, maximum braking, off the runway when
            # able, and the flight is over -- a rejected takeoff is not a
            # flight to continue.
            d.throttle_cut()
            d.throttle_idle()
            d.parking_brake(True)
            self._emit("info", "Rejected takeoff complete",
                       "autonomous.rejected_done")
            self._transition(AutonomousPhase.SHUTDOWN)
            self._goto_sub("secure")
            return

        if sub == "initial_climb":
            # Positive rate: flaps and gear come up on schedule, the climb
            # speed is reached, then the AP takes the leg.
            agl = state.altitude_agl_ft or 0.0
            ias = state.indicated_airspeed_kt or 0.0
            elapsed = now - (self.state.airborne_at or now)
            # Flaps up once safely airborne and fast.
            flap_up_kt = (self.state.v2_kt or self.state.vr_kt or 0.0) + 20.0
            if state.flaps_index and ias > flap_up_kt:
                d.flaps(0.0)
            # Gear up after a positive rate, where the model has one.
            if state.gear_down and agl > 200.0:
                d.gear(False)
            # Checklist and AP engagement once stable.
            if agl > 400.0 or elapsed > 25.0:
                self._run_checklist_for_phase("after_takeoff")
                self._engage_climb(state)
                self._transition(AutonomousPhase.CLIMB)
            return

    def _phase_climb(self, state, now: float) -> None:
        """Climb to the assigned or planned level."""
        target = self._altitude_target(state)
        assigned_hdg = self.session.assigned_heading or 0.0
        if assigned_hdg:
            self._apply_flight_director(state, altitude=target,
                                        vs_fpm=self.performance.climb_fpm,
                                        heading=assigned_hdg,
                                        speed_kt=self._speed_target(state))
        else:
            self._apply_flight_director(state, altitude=target,
                                        vs_fpm=self.performance.climb_fpm,
                                        speed_kt=self._speed_target(state))

        # Level off at the top of the climb.
        if target and abs((state.altitude_ft or 0.0) - target) < 200 \
                and abs(state.vertical_speed_fpm or 0.0) < 200:
            self.state.top_of_climb = True
            self.director.alt_hold(True)
            self.director.vs_hold(False)
            self._level_off_at_cruise(state)
            self._transition(AutonomousPhase.CRUISE)
            return

        if self.session.phase == Phase.DESCENT:
            self._transition(AutonomousPhase.DESCENT)
        elif self.session.phase == Phase.APPROACH:
            self._transition(AutonomousPhase.APPROACH)
        elif now - self.state.phase_entered_at > self.CLIMB_TIMEOUT:
            self._set_error("Climb timed out")

    def _phase_cruise(self, state, now: float) -> None:
        """Cruise: NAV on the route, fuel watched, clearances followed."""
        self._run_checklist_for_phase("cruise")
        self.director.nav(True)
        self.director.airspeed_hold(True)
        target = self._altitude_target(state)
        self.director.altitude(target)
        self.director.alt_hold(True)
        self.director.speed(self._speed_target(state))

        if self.session.phase == Phase.DESCENT:
            self._transition(AutonomousPhase.DESCENT)
        elif self.session.phase == Phase.APPROACH:
            self._transition(AutonomousPhase.APPROACH)

    def _phase_descent(self, state, now: float) -> None:
        """Descend: TOD planning, configuration, approach capture."""
        self._run_checklist_for_phase("descent")
        target = self.session.assigned_altitude_ft or self._descent_floor(state)

        # Top of descent: planned once from the distance to the field,
        # ~318 ft per NM on a three-degree path plus the height to lose.
        if not self.state.tod_nm:
            self._plan_tod(state)

        distance = self._distance_to_destination_nm(state)
        agl = (state.altitude_ft or 0.0) - self._field_elevation(state)

        # Configuration schedule, by height above the field: what a crew
        # actually does, not a single "approach" switch.
        if agl < 10000 and self.state.configured != "gear":
            self.director.gear(True)
            self.state.configured = "gear"
        if agl < 5000 and self.state.configured in ("", "gear"):
            self.director.flaps(0.25)
            self.state.configured = "flaps_approach"
        if agl < 2500 or (distance is not None and distance < 8.0):
            self.director.flaps(0.5)
            self.state.configured = "flaps_final"
            self.director.spoilers_arm(True)

        # Speed: assigned, else a sensible step-down then approach speed.
        speed = self.session.assigned_speed_kt or (
            (self.state.vapp_kt or self.performance.approach_kt) + 20.0
            if agl < 3000 else self.performance.approach_kt)
        # The descent rate is the model's own where it declares one --
        # a clean turbofan comes down at three thousand, a piston at eight
        # hundred -- and the class figure where it does not. The sign is
        # taken as a magnitude: a negative or absurd report is a model
        # quirk, not a clearance to climb.
        descent_fpm = abs(float(getattr(state, "typical_descent_fpm", 0.0)
                                or 0.0))
        if not 100.0 <= descent_fpm <= 6000.0:
            descent_fpm = self.performance.descend_fpm
        # A heading ATC has given -- a vector, the intercept -- is flown
        # here as it is on the climb; without one the route's own guidance
        # (NAV in managed mode, wings level otherwise) stands.
        assigned_hdg = self.session.assigned_heading or 0.0
        self._apply_flight_director(state, altitude=target,
                                    vs_fpm=-descent_fpm,
                                    heading=assigned_hdg or -1.0,
                                    speed_kt=speed)

        # The approach's own navaid, tuned early enough that the needles
        # are alive by the time the final is flown.
        self._maybe_tune_nav(state, self._approach_recipe(state), now)

        # When the approach phase owns it, or we are close enough that the
        # approach logic should be flying.
        if self.session.phase == Phase.APPROACH or (
                distance is not None and distance < 12.0 and agl < 4000):
            self._transition(AutonomousPhase.APPROACH)
            return
        if now - self.state.phase_entered_at > self.DESAT_TIMEOUT:
            self._set_error("Descent timed out")

    def _phase_approach(self, state, now: float) -> None:
        """The approach: intercept, stabilize, land or go around."""
        self._run_checklist_for_phase("approach")
        sub = self.state.sub or "intercept"

        if sub == "go_around":
            self._fly_go_around(state, now)
            return

        self._configure_final(state)

        # The cleared approach, as a recipe: which navaid, which course,
        # which minima. Resolved once and kept until the field or the
        # runway changes; the radio is tuned to its navaid on the way in.
        recipe = self._approach_recipe(state)
        self._maybe_tune_nav(state, recipe, now)

        flown = False
        if recipe is not None and recipe.family != "visual":
            flown = self._fly_recipe(state, recipe, now)
        d = self.director
        if not flown:
            # ILS capture where the needles say it is there; otherwise fly
            # the runway heading to the threshold and let the glide path
            # be flown by reference to distance.
            has_ils = getattr(state, "nav1_has_localizer", False) and \
                getattr(state, "nav1_has_glideslope", False)
            if has_ils and self.session.ifr:
                d.approach(True)
                self.state.ap_apr_mode = True
                d.vs_hold(False)
                self.state.ap_vs_mode = False
            else:
                # Non-precision or visual: fly the final course by heading
                # and the glide path by vertical speed -- about 600 fpm on
                # a three-degree path at approach speed, less once short.
                if self.state.runway_heading <= 0:
                    self._find_runway(state)
                if self.state.runway_heading > 0:
                    agl = state.altitude_agl_ft or 0.0
                    vs = -600.0 if agl > 400.0 else -250.0
                    self._apply_flight_director(
                        state, vs_fpm=vs, heading=self.state.runway_heading,
                        speed_kt=self.state.vapp_kt
                        or self.performance.approach_kt)

        # Stabilized approach: the numbers a crew actually checks.
        self._check_stabilized(state)

        # Go-around triggers: not stabilized by minimums, sink rate or
        # speed outside limits low down, or no landing clearance when the
        # field is airborne-only.
        if self._should_go_around(state):
            self._start_go_around(state)
            return

        # Cleared and stable enough: hand over to the landing phase for
        # the flare.
        agl = state.altitude_agl_ft or 0.0
        if self.session.cleared_landing and agl < 500.0:
            self._transition(AutonomousPhase.LANDING)
            self._goto_sub("flare")
            return

        if self.session.phase == Phase.LANDING or self.session.phase == Phase.LANDED:
            self._transition(AutonomousPhase.LANDING)
            self._goto_sub("flare")
            return

        if now - self.state.phase_entered_at > self.APPROACH_TIMEOUT:
            self._set_error("Approach timed out")

    def _phase_landing(self, state, now: float) -> None:
        """Flare, touchdown, rollout, vacate."""
        d = self.director
        sub = self.state.sub or "flare"

        if sub == "flare":
            self._run_checklist_for_phase("before_landing")
            if state.on_ground:
                self.state.touchdown_at = now
                self._goto_sub("rollout")
                return
            agl = state.altitude_agl_ft or 0.0
            # Below the flare window: the autopilot's job is done -- the
            # FO flies the flare by hand, like a pilot does.
            if agl < 60.0:
                if not self.state.flare_at:
                    self.state.flare_at = now
                    d.approach(False)
                    d.autopilot(False)
                    self.state.ap_master = False
                d.throttle_idle()
                d.throttle(0.0)
                d.elevator(0.15)  # a touch of back pressure to arrest the sink
                # Still aligned with the runway.
                if self.state.runway_heading > 0:
                    err = ((self.state.runway_heading
                            - (state.heading_true or 0.0) + 540.0)
                           % 360.0 - 180.0)
                    d.rudder(max(-0.4, min(0.4, err / 15.0)))
                # Touchdown must come; a flare that floats is a go-around.
                if now - self.state.flare_at > 15.0 and agl > 10.0:
                    self._start_go_around(state)
                    return
            else:
                # On the glide: idle-ish, configured, on speed.
                d.throttle(0.05)
                d.approach(True)
                if self.state.runway_heading <= 0:
                    self._find_runway(state)
            if now - self.state.phase_entered_at > self.LANDING_TIMEOUT:
                self._start_go_around(state)
            return

        if sub == "rollout":
            # Down: spoilers if armed, brakes on, steer to the exit or the
            # taxi path, slow to taxi speed.
            d.spoilers(1.0)
            d.spoilers_arm(True)
            d.flaps(0.0)
            gs = state.ground_speed_kt or 0.0
            if gs > self.performance.taxi_kt + 5.0:
                # Speedbrakes are already out; the wheel brakes take it
                # from here, at the axis scale the BRAKES events read.
                d.parking_brake(False)
                d.send(b"BRAKES_LEFT", 16384)
                d.send(b"BRAKES_RIGHT", 16384)
            else:
                # Slow enough: brakes off, taxi in.
                d.send(b"BRAKES_LEFT", 0)
                d.send(b"BRAKES_RIGHT", 0)
                if not self.state.vacated:
                    self.state.vacated = True
                    self._run_checklist_for_phase("after_landing")
                self._transition(AutonomousPhase.TAXI_IN)
                self._goto_sub("taxi")
                return
            # Keep it on the centreline until slow.
            if self.state.runway_heading > 0:
                err = ((self.state.runway_heading
                        - (state.heading_true or 0.0) + 540.0)
                       % 360.0 - 180.0)
                d.rudder(max(-0.4, min(0.4, err / 15.0)))
            return

    def _phase_taxi_in(self, state, now: float) -> None:
        """Taxi to the stand after landing."""
        self._run_checklist_for_phase("after_landing")
        sub = self.state.sub or "taxi"
        if sub == "taxi":
            self._taxi_along_path(state, outbound=False)
        elif sub == "park":
            self._stop(state)
            if (state.ground_speed_kt or 0.0) <= 1.0:
                self.director.parking_brake(True)
                self._transition(AutonomousPhase.PARKED)

        if self.session.phase == Phase.PARKED:
            self._transition(AutonomousPhase.PARKED)
        elif now - self.state.phase_entered_at > self.TAXI_TIMEOUT:
            # Parked somewhere: a long taxi in is not an error if we are
            # stopped; only a moving never-ending one is.
            if (state.ground_speed_kt or 0.0) > 1.0:
                self._set_error("Taxi-in timed out")

    def _phase_parked(self, state, now: float) -> None:
        """At parking position: the shutdown checklist, then the sequence."""
        self.director.parking_brake(True)
        self._run_checklist_for_phase("shutdown")

        if self._chained:
            self._transition(AutonomousPhase.CHAIN_NEXT)
        else:
            self._transition(AutonomousPhase.SHUTDOWN)

    def _phase_shutdown(self, state, now: float) -> None:
        """Shutdown sequence: secure, cool, engines off, electrics off."""
        d = self.director
        sub = self.state.sub or "secure"

        if sub == "secure":
            # The walk-around items the FO does: lights off except the
            # beacon, pitot heat off, controls centred.
            d.strobes(False)
            d.landing_light(False)
            d.taxi_light(False)
            d.nav_lights(False)
            d.pitot_heat(False)
            d.center_controls()
            self.state.shutdown_at = now
            self._goto_sub("cooldown")

        elif sub == "cooldown":
            # Turbines cool at idle before shutdown; the clock is the
            # checklist's, not a guess at the model's temperatures.
            if self._is_turbine(state) and \
                    now - self.state.shutdown_at < self.SHUTDOWN_COOLDOWN_S:
                return
            self._goto_sub("engines_off")

        elif sub == "engines_off":
            if state.engines_running:
                self._shutdown_engines()
                if self._sub_age() > 60.0:
                    self._set_error("Engines would not shut down")
                    return
            else:
                self._goto_sub("electrics_off")

        elif sub == "electrics_off":
            if self._sub_age() < 5.0:
                d.avionics(False)
                d.battery(False)
                d.beacon(False)
                return
            self._emit("info", "Autonomous flight completed",
                       "autonomous.completed")
            self._transition(AutonomousPhase.COMPLETE)

    def _phase_chain_next(self, state, now: float) -> None:
        """Load and start next flight in chain with random destination."""
        if self.state.chain_flight_index >= len(self.state.chain_flights):
            self._select_next_destination()
        if self.state.chain_flight_index >= len(self.state.chain_flights):
            self._emit("info", "No suitable destination found for next flight",
                       "autonomous.chain.no_destination")
            self._transition(AutonomousPhase.SHUTDOWN)
            return

        if now - self.state.phase_entered_at < self._chain_delay:
            return

        next_flight = self.state.chain_flights[self.state.chain_flight_index]
        self.state.chain_flight_index += 1
        self._load_next_flight(next_flight)
        self._transition(AutonomousPhase.PREFLIGHT)

    def _phase_error(self, state, now: float) -> None:
        """Error state - disable autonomous flight."""
        self._emit("error", f"Autonomous flight error: {self.state.error_message}",
                   "autonomous.error")
        self.disable()

    # -- taxi -----------------------------------------------------------------

    def _taxi_along_path(self, state, outbound: bool) -> None:
        """Follow ``session.taxi_path`` at taxi speed, or hold short.

        Outbound the path ends at the holding point; inbound it ends at
        the stand. With no path (an invented route) the aeroplane steers
        for the runway's threshold outbound, and holds once there -- it
        does not invent a taxiway.
        """
        path = list(getattr(self.session, "taxi_path", None) or [])
        if not path or len(path) < 2:
            self._taxi_without_path(state, outbound)
            return

        steer = auto_nav.path_steering(state.latitude, state.longitude, path)
        if steer is None or steer.get("heading") is None:
            self._stop(state)
            return

        # Slow for the end of the route and for tight corners.
        remaining = steer["remaining_m"]
        turn = abs((steer["heading"] - (state.heading_true or 0.0)
                    + 540.0) % 360.0 - 180.0)
        target_kt = self.performance.taxi_kt
        if remaining < 60.0 or turn > 60.0:
            target_kt = max(4.0, target_kt * 0.5)
        if remaining < 25.0:
            target_kt = 0.0

        if steer["heading"] is not None:
            d = self.director
            d.parking_brake(False)
            err = ((steer["heading"] - (state.heading_true or 0.0)
                    + 540.0) % 360.0 - 180.0)
            d.rudder(max(-0.5, min(0.5, err / 25.0)))
            gs = state.ground_speed_kt or 0.0
            if target_kt <= 0.0:
                self._stop(state)
            elif gs < target_kt - 1.5:
                d.throttle(0.15)
            elif gs > target_kt + 3.0:
                d.throttle(0.0)

        # Outbound: stop short of a runway crossing the route was cleared
        # to hold at, and of the split point of a cut taxi.
        if outbound:
            if self.session.taxi_split_at is not None:
                lat, lon = self.session.taxi_split_at
                if state.distance_to(lat, lon) * 1852.0 < 60.0:
                    self._goto_sub("hold_point")
                    return
            # The path is done: at the hold point. HOLDING_SHORT runs the
            # before-takeoff checklist and waits for the line-up clearance.
            if steer.get("done") or remaining < 30.0:
                self._goto_sub("hold_point")
                self._transition(AutonomousPhase.HOLDING_SHORT)
                return
        else:
            # Inbound: the stand is the end of the path.
            if steer.get("done"):
                self._goto_sub("park")
                return

    def _taxi_without_path(self, state, outbound: bool) -> None:
        """No simulator taxiway route: fly the runway heading, hold short."""
        if outbound:
            self._find_runway(state)
            if self.state.runway_heading > 0:
                self._steer(state, self.state.runway_heading,
                            self.performance.taxi_kt)
            if self.session.cleared_takeoff or self.session.phase in (
                    Phase.HOLDING_SHORT, Phase.LINED_UP, Phase.TAKEOFF):
                self._transition(AutonomousPhase.LINE_UP)
        else:
            # Inbound with no route yet: the tower has not said where to
            # park, so the aeroplane finishes its rollout and waits --
            # stopping short on the runway is worse than rolling out.
            if self.session.phase == Phase.PARKED:
                self._stop(state)
                if (state.ground_speed_kt or 0.0) <= 1.0:
                    self._transition(AutonomousPhase.PARKED)
            elif (state.ground_speed_kt or 0.0) > self.performance.taxi_kt:
                self.director.throttle(0.0)
                self.director.throttle_idle()
                self.director.spoilers(1.0)
            else:
                # Slow with nowhere to go: stop and wait for the taxi
                # instruction rather than roll the runway's length.
                self._stop(state)

    # -- the approach's own pieces --------------------------------------------

    def _find_runway(self, state) -> None:
        """The departure/arrival runway's threshold and heading, once."""
        if self.state.runway_heading > 0:
            return
        # The field the leg is bound for: the arrival's when one is named,
        # the departure's otherwise.
        airport = (self.session.destination
                   if self.session.arrival_runway
                   else (self.session.departure or self.session.destination))
        if airport is None:
            state_now = self.get_state()
            if state_now is not None and getattr(state_now, "has_position", False):
                airport = self.navdb.home_airport(state_now.latitude,
                                                  state_now.longitude)
        if airport is None:
            return
        runways = list(self.navdb.runways(airport.ident) or [])
        name = (self.session.departure_runway or self.session.arrival_runway
                or "")
        if not name and runways:
            # The tower's choice stands in session; the longest runway is
            # the honest guess when nothing was said.
            longest = max(runways, key=lambda r: r.length_ft or 0)
            name = getattr(longest, "le_ident", "") or ""
        found = auto_nav.runway_end(runways, name)
        if found is None:
            return
        self.state.runway_lat, self.state.runway_lon, \
            self.state.runway_heading, self.state.runway_length_ft = found

    # -- the cleared approach, as a recipe ---------------------------------

    def _recipe_key(self, state) -> tuple:
        """What the recipe was resolved for: a new field, runway or
        request is a new recipe."""
        dest = getattr(self.session, "destination", None)
        return (getattr(dest, "ident", "") or "",
                getattr(self.session, "arrival_runway", "") or "",
                getattr(self.session, "requested_approach", "") or "")

    def _approach_recipe(self, state):
        """The cleared approach resolved for flying, once per leg.

        approach_flight.resolve reads the controller's own choice off the
        session -- "ILS Zulu", "RNAV Yankee", "VOR-A" -- and turns it into
        a course, a final approach fix, minima and the navaid to tune.
        """
        key = self._recipe_key(state)
        if self.state.approach_recipe_tried \
                and key == self.state.approach_recipe_key:
            return self.state.approach_recipe
        self.state.approach_recipe_tried = True
        self.state.approach_recipe_key = key
        try:
            recipe = approach_flight.resolve(
                self.session, self.brain, self.navdb, state)
        except (sqlite3.Error, AttributeError, TypeError, ValueError,
                KeyError, IndexError, RuntimeError) as exc:
            log.info("approach recipe failed: %s", exc)
            recipe = None
        self.state.approach_recipe = recipe
        if recipe is not None:
            # The approach's own minima, not the ILS's 200 ft every
            # approach used to be judged against.
            self.state.minima_ft = recipe.minima_ft
            from ..logs import decision
            decision("autonomous_approach", airport=recipe.ident,
                     approach=recipe.name, family=recipe.family,
                     line=recipe.line or None,
                     approach_runway=recipe.approach_runway,
                     landing_runway=recipe.landing_runway,
                     navaid=recipe.navaid.ident if recipe.navaid else None,
                     freq_mhz=(round(recipe.navaid.freq_mhz, 3)
                               if recipe.navaid else None),
                     minima_ft=recipe.minima_ft,
                     circle_minima_ft=recipe.circle_minima_ft or None,
                     why=recipe.why or None)
        return recipe

    def _maybe_tune_nav(self, state, recipe, now: float) -> None:
        """NAV1 to the approach's navaid, once, with readback.

        The localizer, the VOR or the NDB the procedure is flown on -- the
        needles and the station callouts are only there if the radio is
        tuned, and until now nothing ever tuned it. Tried again every ten
        seconds until the frequency reads back; an aeroplane whose radios
        the FO cannot reach simply keeps the old guidance.
        """
        nav = getattr(recipe, "navaid", None) if recipe is not None else None
        if nav is None or not nav.freq_mhz:
            return
        sim = self.sim
        if sim is None or not getattr(sim, "connected", False):
            return
        tuned = float(getattr(state, "nav1_freq", 0.0) or 0.0)
        if tuned and abs(tuned - nav.freq_mhz) < 0.005:
            if not self.state.approach_tuned:
                self.state.approach_tuned = True
                self._emit("info",
                           f"NAV1 showing {nav.ident} "
                           f"{nav.freq_mhz:.3f} MHz",
                           "autonomous.nav_tuned")
            return
        if now - self.state.nav_tune_at < 10.0:
            return
        self.state.nav_tune_at = now
        if hasattr(sim, "set_nav") and sim.set_nav(nav.freq_mhz, 1):
            self._emit("info",
                       f"Tuning NAV1 to {nav.ident} {nav.freq_mhz:.3f} MHz",
                       "autonomous.nav_tune")

    @staticmethod
    def _magnetic(true_heading: float, state) -> float:
        """A true course as the heading bug wants it: magnetic."""
        from ..navdata.magvar import to_magnetic, variation

        return to_magnetic(true_heading,
                           variation(state.latitude, state.longitude)) % 360.0

    def _fly_recipe(self, state, recipe, now: float) -> bool:
        """The cleared approach's own guidance; False for the fallback.

        What the aeroplane can actually fly decides the mode: an ILS with
        live needles gets the autopilot's approach mode; a localizer
        without a glideslope gets LOC mode and the path's vertical speed;
        everything else -- VOR, NDB, RNAV, an ILS whose needles never came
        alive -- is the computed final: the gate-FAF-threshold path flown
        by heading and a three-degree vertical speed.
        """
        d = self.director
        family = recipe.family
        has_loc = bool(getattr(state, "nav1_has_localizer", False))
        has_gs = bool(getattr(state, "nav1_has_glideslope", False))

        # A circle, once it has started, owns the aeroplane until it is
        # back on the landing runway's final.
        if recipe.circling and self._circling(state, recipe):
            return self._fly_circle(state, recipe)

        if family == "ils" and has_loc and has_gs and self.session.ifr:
            d.approach(True)
            self.state.ap_apr_mode = True
            d.vs_hold(False)
            self.state.ap_vs_mode = False
            return True
        if family == "loc" and has_loc and self.session.ifr:
            # The localizer for the lateral, the path for the vertical:
            # there is no glideslope on this approach to capture.
            d.localizer(True)
            self.state.ap_apr_mode = False
            self._fly_path_vertical(state, recipe)
            return True
        if family in ("ils", "loc", "vor", "ndb", "rnav"):
            self._fly_computed(state, recipe)
            return True
        return False

    def _path_vertical(self, state, recipe) -> float:
        """The vertical speed the three-degree path asks for here."""
        distance = state.distance_to(recipe.threshold_lat,
                                     recipe.threshold_lon) \
            if getattr(state, "has_position", False) else None
        desired = approach_flight.profile_alt(
            recipe, distance if distance is not None else 10.0)
        alt = state.altitude_ft or 0.0
        vs = approach_flight.sink_fpm(state.ground_speed_kt or 0.0)
        # Capture: well above the path it comes down harder; below it the
        # sink eases rather than deepening toward the ground.
        if alt > desired + 300.0:
            vs = min(900.0, vs * 1.5)
        elif alt < desired - 150.0:
            vs = max(200.0, vs * 0.4)
        return -vs

    def _fly_path_vertical(self, state, recipe) -> None:
        """Vertical speed and speed for a mode that owns the lateral."""
        self._apply_flight_director(
            state, vs_fpm=self._path_vertical(state, recipe),
            speed_kt=self.state.vapp_kt or self.performance.approach_kt)

    def _fly_computed(self, state, recipe) -> None:
        """The computed final: path steering for the lateral, the
        three-degree profile for the vertical. What an RNAV, a VOR, an NDB
        or an ILS whose needles never arrived all share -- the geometry is
        the same, the minima and the tuned station are what differ."""
        d = self.director
        d.localizer(False)
        if self.state.ap_apr_mode:
            # APR was flying this approach a moment ago; the needles are
            # gone or the procedure changed, and approach mode with no
            # guidance behind it is worse than none.
            d.approach(False)
            self.state.ap_apr_mode = False
        steer = auto_nav.path_steering(
            state.latitude, state.longitude, approach_flight.legs(recipe),
            lookahead_m=2500.0)
        heading = steer.get("heading")
        if heading is None:
            heading = recipe.final_course
        self._apply_flight_director(
            state, vs_fpm=self._path_vertical(state, recipe),
            heading=self._magnetic(heading, state),
            speed_kt=self.state.vapp_kt or self.performance.approach_kt)

    # -- circling -----------------------------------------------------------

    def _circling(self, state, recipe) -> bool:
        """Whether the aeroplane is in the circle, entering it once.

        The published approach is done when the aeroplane is at or below
        its minima or has passed the threshold; from there the pattern to
        the landing runway takes over -- unless it is already lined up on
        that runway's final, which the ordinary approach sees home.
        """
        if self.state.circling:
            return not self._on_landing_final(state, recipe)
        agl = state.altitude_agl_ft or 0.0
        distance = state.distance_to(recipe.threshold_lat,
                                     recipe.threshold_lon) \
            if getattr(state, "has_position", False) else None
        # The floor is the circling MDA: below the approach's own minima
        # is already too low to be circling at.
        floor = recipe.circle_minima_ft or recipe.minima_ft or 200.0
        entered = agl <= floor + 100.0 \
            or (distance is not None and distance < 1.5)
        if not entered:
            return False
        path = approach_flight.circle_legs(recipe, self.navdb, state)
        if not path:
            return False
        self.state.circling = True
        self.state.circle_path = path
        # Below this the go-around triggers are the circling minima's.
        self.state.minima_ft = recipe.circle_minima_ft or recipe.minima_ft
        self._emit("info", "Circling to land",
                   "autonomous.circling")
        return True

    def _on_landing_final(self, state, recipe) -> bool:
        """On the landing runway's final, close enough to land.

        The recipe's own course, not ``state.runway_heading``: for a
        circle that heading can still be the approach runway's.
        """
        if recipe.final_course <= 0 \
                or not getattr(state, "has_position", False):
            return False
        track = getattr(state, "track_true", None) \
            or state.heading_true or 0.0
        err = abs((recipe.final_course - track + 540.0)
                  % 360.0 - 180.0)
        distance = state.distance_to(recipe.threshold_lat,
                                     recipe.threshold_lon)
        return err < 20.0 and distance < 4.0

    def _fly_circle(self, state, recipe) -> bool:
        """The circling pattern: hold the circling MDA, fly the rectangle
        to the landing runway's final."""
        if not self.state.circle_path:
            path = approach_flight.circle_legs(recipe, self.navdb, state)
            if not path:
                self.state.circling = False
                self.state.circle_path = []
                return False
            self.state.circle_path = path
        d = self.director
        # Whatever was tracking the approach runway's needles is done with
        # them: the pattern is flown on the heading bug.
        d.localizer(False)
        if self.state.ap_apr_mode:
            d.approach(False)
            self.state.ap_apr_mode = False
        steer = auto_nav.path_steering(
            state.latitude, state.longitude, self.state.circle_path,
            lookahead_m=2500.0)
        target = recipe.field_elev_ft + \
            (recipe.circle_minima_ft or recipe.minima_ft)
        alt = state.altitude_ft or 0.0
        vs = 0.0
        if alt > target + 80.0:
            vs = -250.0
        elif alt < target - 80.0:
            vs = 150.0
        heading = steer.get("heading") or recipe.final_course
        self._apply_flight_director(
            state, altitude=int(target), vs_fpm=vs,
            heading=self._magnetic(heading, state),
            speed_kt=(self.state.vapp_kt or self.performance.approach_kt)
            + 5.0)
        return True

    def _field_elevation(self, state) -> float:
        airport = self.session.destination or self.session.departure
        if airport is not None:
            return float(getattr(airport, "elev_ft", 0.0) or 0.0)
        return 0.0

    def _distance_to_destination_nm(self, state) -> Optional[float]:
        dest = self.session.destination
        if dest is None or not getattr(state, "has_position", False):
            return None
        return state.distance_to(dest.lat, dest.lon)

    def _plan_tod(self, state) -> None:
        """Where the descent begins: the height to lose on a three-degree
        path, plus a margin for the configuration."""
        distance = self._distance_to_destination_nm(state)
        if distance is None:
            self.state.tod_nm = -1.0
            return
        agl = (state.altitude_ft or 0.0) - self._field_elevation(state)
        # 318 ft per NM is the standard path; the TOD is where that path,
        # started now, reaches the field.
        self.state.tod_nm = max(0.0, agl / 318.0)

    def _descent_floor(self, state) -> float:
        """The lowest level the descent may level at: pattern height."""
        elev = self._field_elevation(state)
        # A safe floor for the descent itself: pattern height over the
        # field. Anything lower is the approach phase's job, once it has
        # the runway in sight or the needles captured.
        return elev + 3000.0

    def _configure_final(self, state) -> None:
        """The approach configuration, by distance to the threshold."""
        d = self.director
        distance = self._distance_to_destination_nm(state)
        agl = (state.altitude_ft or 0.0) - self._field_elevation(state)

        # Gear down on final, flaps through the stages, speedbrake armed.
        if agl < 2000 or (distance is not None and distance < 10.0):
            if state.gear_down is not False:
                d.gear(True)
        if distance is not None and distance < 6.0 and agl < 2500:
            d.flaps(0.75)
            self.state.configured = "flaps_landing"
        if distance is not None and distance < 3.0:
            d.flaps(1.0)
            self.state.configured = "flaps_full"
        d.spoilers_arm(True)
        # Approach speed: Vapp where known, the type's figure otherwise.
        vapp = self.state.vapp_kt or self.performance.approach_kt
        self.director.speed(vapp)

    def _check_stabilized(self, state) -> None:
        """The stabilized-approach call, once, below a thousand feet."""
        if self.state.stabilized:
            return
        agl = state.altitude_agl_ft or 0.0
        if agl > 1000.0 or agl <= 0.0:
            return
        vs = abs(state.vertical_speed_fpm or 0.0)
        ias = state.indicated_airspeed_kt or 0.0
        target = self.state.vapp_kt or self.performance.approach_kt
        bank = abs(getattr(state, "bank_deg", 0.0) or 0.0)
        aligned = True
        if self.state.runway_heading > 0 and state.on_ground is False:
            err = abs((self.state.runway_heading
                       - (state.heading_true or 0.0) + 540.0)
                      % 360.0 - 180.0)
            aligned = err < 15.0
        if vs < 1000.0 and abs(ias - target) < 12.0 and bank < 25.0 and aligned:
            self.state.stabilized = True
            self._emit("info", "Stabilized on final",
                       "autonomous.stabilized")

    def _should_go_around(self, state) -> bool:
        """The criteria that send an approach around, conservatively."""
        # The tower called it: a missed approach is the aeroplane's, not
        # the controller's, to refuse. The flag stays up until the
        # controller sees the aeroplane back on a final, so it is honoured
        # on its rising edge only -- once per call, not once per tick.
        flag = bool(self.session.going_around)
        if flag and not self._ga_flag_seen:
            self._ga_flag_seen = True
            return True
        if not flag:
            self._ga_flag_seen = False
        agl = state.altitude_agl_ft or 0.0
        # Below minimums without the runway: a go-around, not a dive.
        if agl < (self.state.minima_ft or 200.0) and agl > 0:
            if not self.session.cleared_landing and self.session.ifr:
                return True
        # Unstable low down: sink rate, bank, or speed far off.
        if 0 < agl < 500.0:
            vs = state.vertical_speed_fpm or 0.0
            if vs < -1200.0:
                return True
            bank = abs(getattr(state, "bank_deg", 0.0) or 0.0)
            if bank > 35.0:
                return True
            ias = state.indicated_airspeed_kt or 0.0
            target = self.state.vapp_kt or self.performance.approach_kt
            if target and abs(ias - target) > 20.0:
                return True
        # Drifting toward the ground without a flare started.
        if agl < 30.0 and not self.session.cleared_landing:
            return True
        return False

    def _start_go_around(self, state) -> None:
        """The go-around: power, pitch, configuration, climb away."""
        self.state.go_arounds += 1
        self.state.circling = False
        self.state.circle_path = []
        self.state.go_around_at = time.monotonic()
        # Missed approach altitude: what was assigned, else the procedure's
        # own where the recipe knows it, else a safe pattern height.
        recipe = self.state.approach_recipe
        self.state.missed_alt_ft = (
            self.session.assigned_altitude_ft
            or (getattr(recipe, "missed_alt_ft", 0.0) or 0.0)
            or self._field_elevation(state) + 2000.0)
        # From the landing flare the go-around is an approach again: the
        # phase owns the sub-step only from APPROACH down.
        if self.state.phase == AutonomousPhase.LANDING:
            self._transition(AutonomousPhase.APPROACH)
        self._goto_sub("go_around")
        d = self.director
        d.go_around()
        d.takeoff_power()
        d.throttle(1.0)
        d.elevator(0.3)   # the pitch-up
        d.gear(False)
        # Flaps one stage up, not all the way: the go-around flap setting.
        if state.flaps_index >= 2:
            d.flaps(0.5)
        elif state.flaps_index == 1:
            d.flaps(0.25)
        self.state.stabilized = False
        self.state.configured = ""
        self.state.ap_apr_mode = False
        self._emit("info", "Going around", "autonomous.go_around")

    def _fly_go_around(self, state, now: float) -> None:
        """Climb away on the missed approach, then set up again."""
        d = self.director
        elapsed = now - self.state.go_around_at
        # Initial: pitch and power; after positive rate, clean up.
        if elapsed < 8.0:
            d.elevator(0.25)
            d.throttle(1.0)
        else:
            d.elevator(0.05)
            d.throttle(0.7)
            if state.flaps_index:
                d.flaps(0.0)
            if state.gear_down:
                d.gear(False)
        # Wings-level climb on the runway heading or the assigned heading.
        hdg = self.session.assigned_heading or self.state.runway_heading
        if hdg > 0:
            self._apply_flight_director(state, altitude=self.state.missed_alt_ft,
                                        vs_fpm=self.performance.climb_fpm,
                                        heading=hdg,
                                        speed_kt=self.state.climb_kt or 180.0)
        # Once level-ish at the missed altitude, reset for another pass.
        alt = state.altitude_ft or 0.0
        if alt >= self.state.missed_alt_ft - 150.0 \
                and abs(state.vertical_speed_fpm or 0.0) < 300:
            self.state.stabilized = False
            self.state.configured = ""
            self._goto_sub("intercept")
            self._emit("info", "Missed approach complete; setting up again",
                       "autonomous.go_around_done")
        elif elapsed > 300.0:
            self._set_error("Go-around did not climb")

    # -- emergencies ------------------------------------------------------------

    def _monitor_emergency(self, state, now: float) -> None:
        """Watch for the failures the state can actually see.

        Fuel and electrical are readable; an engine failure is only
        visible as ``engines_running`` going false or the takeoff roll
        not accelerating. What cannot be seen is not invented.
        """
        if self.state.emergency_declared or self.state.phase in (
                AutonomousPhase.PREFLIGHT, AutonomousPhase.PUSHBACK,
                AutonomousPhase.TAXI_OUT, AutonomousPhase.SHUTDOWN,
                AutonomousPhase.COMPLETE, AutonomousPhase.ERROR):
            return

        # Electrical: the main bus collapsing in the air.
        volts = getattr(state, "main_bus_volts", -1.0)
        if volts is not None and 0.0 < volts < 12.0 and not state.on_ground:
            self._declare_emergency("electrical", state)
            return

        # Fuel: nothing left, or so little the destination cannot be
        # reached. A sim that never reports fuel reads zero forever, so
        # the check only fires once a nonzero figure has been seen.
        fuel_lb = getattr(state, "fuel_total_lb", 0.0) or 0.0
        fuel_gal = getattr(state, "fuel_total_gal", 0.0) or 0.0
        if fuel_lb > 0.0 or fuel_gal > 0.0:
            self._fuel_known = True
        if self._fuel_known:
            if fuel_lb <= 0.0 and fuel_gal <= 0.0:
                self._declare_emergency("fuel", state)
                return
            if fuel_gal > 0 and fuel_gal < 8.0:
                self._declare_emergency("fuel", state)
                return
        # Engines out in the air: the sim's own flag is the only truth.
        if not state.on_ground and not getattr(state, "engines_running", True):
            self._declare_emergency("engine", state)

    def _declare_emergency(self, kind: str, state) -> None:
        """A MAYDAY, a divert, and the flight plan changed to match."""
        self.state.emergency = kind
        self.state.emergency_declared = True
        # Nearest airport that can take the aeroplane, for the divert.
        current = self._get_current_airport()
        if current is not None:
            candidates = self._find_suitable_destinations(current)
            if candidates:
                # An emergency goes to the nearest field that can take the
                # aeroplane, not the one a scheduled leg would prefer.
                self.state.emergency_destination = min(
                    candidates, key=lambda c: c[1])[0]
        dest = self.state.emergency_destination
        dest_name = dest.ident if dest is not None else "nearest airport"
        callsign = getattr(getattr(self.session.aircraft, "callsign", None),
                           "written", "") or "this aircraft"
        text = (f"Mayday mayday mayday, {callsign} "
                f"{kind} emergency, diverting to {dest_name}")
        if self.crew.transmit:
            try:
                self.crew.transmit(text)
            except Exception:
                pass
        else:
            log.warning("Emergency declared but no radio to say it: %s", text)
        self._emit("error", f"Emergency: {kind}; diverting to {dest_name}",
                   "autonomous.emergency")
        # Fly the diversion: the destination changes, the route goes
        # direct, and the phases in flight pick the new field up on their
        # own -- an aircraft on final does not climb back to cruise to
        # divert, it descends to the new one.
        if dest is not None:
            self.session.destination = dest
            self.session.ifr = False  # an emergency is flown, not filed
            if self.use_route is not None:
                try:
                    from .route import from_string, resolver
                    route = from_string(
                        f"DCT {dest.ident}", dest.ident, dest.ident,
                        resolver(self.navdb))
                    self.use_route(route)
                except Exception:
                    pass
            # An emergency descent: get down to a survivable height now,
            # the approach machinery handles the rest. On final the
            # approach is waved off for the new field, with the cached
            # runway geometry dropped so it resolves again.
            if self.state.phase in (AutonomousPhase.CLIMB,
                                    AutonomousPhase.CRUISE,
                                    AutonomousPhase.DESCENT,
                                    AutonomousPhase.APPROACH):
                self.state.runway_heading = 0.0
                self.state.aligned = False
                self.state.stabilized = False
                self.state.configured = ""
                self.state.minima_ft = 200.0
                self.state.approach_recipe = None
                self.state.approach_recipe_tried = False
                self.state.approach_recipe_key = ()
                self.state.approach_tuned = False
                self.state.circling = False
                self.state.circle_path = []
                self.state.tod_nm = 0.0
                self._transition(AutonomousPhase.DESCENT)
                self._goto_sub("")

    def _is_turbine(self, state) -> bool:
        code = (self.state.aircraft_type or "").upper()
        return code[:1] in ("A", "B", "C", "D", "E", "F", "G", "H", "L") \
            and code not in ("C152", "C172", "C182", "C208", "C25A", "PA28",
                             "PA34", "DA40", "DA42", "SR22", "BE36", "TB20",
                             "P28A", "AT3")

    # -- chain flights (unchanged behaviour) -----------------------------------

    def _select_next_destination(self) -> None:
        """Select a random destination within aircraft range with ground services."""
        current = self._get_current_airport() or self.session.departure
        if not current:
            log.warning("Cannot select next destination: no current airport")
            return

        candidates = self._find_suitable_destinations(current)
        if not candidates:
            log.warning("No suitable destinations within range of %s", current.ident)
            return

        destination = self._pick_destination(candidates)
        if not destination:
            return

        flight_plan = self._create_flight_plan(current, destination)
        self.state.chain_flights.append(flight_plan)
        log.info("Chained flight: %s -> %s (%d NM)", current.ident, destination.ident,
                 flight_plan.get("distance_nm", 0))

    def _get_current_airport(self) -> Optional[Airport]:
        """Get the airport the aircraft is currently at."""
        state = self.get_state()
        if state and state.has_position:
            return self.navdb.home_airport(state.latitude, state.longitude)
        return self.session.destination or self.session.departure

    def _find_suitable_destinations(self, origin: Airport) -> list[tuple[Airport, float]]:
        """Find airports within range that can host the aircraft with ground services."""
        candidates = []
        max_range = self.state.aircraft_range_nm * 0.8

        for airport, distance in self.navdb.nearest_airports(
                origin.lat, origin.lon, limit=5000, max_nm=max_range):
            if airport.ident == origin.ident:
                continue
            if not self._airport_suitable(airport):
                continue
            if not self._has_ground_services(airport):
                continue
            candidates.append((airport, distance))

        candidates.sort(key=lambda x: abs(x[1] - max_range * 0.5))
        return candidates[:20]

    def _airport_suitable(self, airport: Airport) -> bool:
        """Check if airport can handle the aircraft type."""
        type_code = self.state.aircraft_type

        if self._is_helicopter:
            return airport.type in ("heliport", "small_airport",
                                    "medium_airport", "large_airport")

        min_length = self._min_runway_length_for_type(type_code)
        has_long_runway = any(
            rw.length_ft >= min_length for rw in self.navdb.runways(airport.ident)
        )
        if not has_long_runway:
            return False

        if type_code in ("A388", "B744", "B748", "B77W", "B77L", "B77X", "A35K"):
            runways = self.navdb.runways(airport.ident)
            if len(runways) < 2 and not any(rw.length_ft >= 10000 for rw in runways):
                return False

        return True

    def _has_ground_services(self, airport: Airport) -> bool:
        """Check if airport has ground services (fuel, parking, handling)."""
        if self._is_helicopter:
            return airport.type in ("heliport", "small_airport",
                                    "medium_airport", "large_airport")
        if airport.type in ("large_airport", "medium_airport"):
            return True
        if self.state.aircraft_type in ("C152", "C172", "C182", "PA28", "DA40", "SR22"):
            return airport.type in ("small_airport", "medium_airport", "large_airport")
        return False

    def _min_runway_length_for_type(self, type_code: str) -> int:
        """Minimum runway length in feet for aircraft type."""
        if type_code in ("C152", "C172", "C182", "PA28", "DA40", "SR22", "BE36", "TB20"):
            return 2500
        if type_code in ("TBM9", "PC12", "C25A", "C25C", "C525", "E55P", "E50P",
                         "AT76", "DH8D", "C208"):
            return 4000
        if type_code in ("CRJ7", "CRJ9", "E170", "E175", "E190", "E195",
                         "F2TH", "H25B", "LJ45", "LJ60", "C680", "C750", "G280"):
            return 5500
        if type_code.startswith(("A3", "B73", "B38", "B39", "B75")):
            return 7000
        if type_code in ("A332", "A333", "A339", "A343", "A346", "A359", "A35K", "A388",
                         "B763", "B764", "B772", "B77W", "B77L", "B77X", "B788", "B789", "B78X",
                         "B744", "B748"):
            return 9000
        return 6000

    def _pick_destination(self, candidates: list[tuple[Airport, float]]) -> Optional[Airport]:
        """Pick a destination from candidates with some randomness."""
        if not candidates:
            return None
        max_range = self.state.aircraft_range_nm * 0.8
        weights = []
        for _, dist in candidates:
            ideal = max_range * 0.5
            diff = abs(dist - ideal) / max_range
            weight = 1.0 / (1.0 + diff * 2.0)
            weights.append(weight)
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices([c[0] for c in candidates], weights=weights, k=1)[0]

    def _create_flight_plan(self, origin: Airport, destination: Airport) -> dict:
        """Create a flight plan dictionary for the leg."""
        distance = haversine_nm(origin.lat, origin.lon,
                                destination.lat, destination.lon)

        if distance < 200:
            cruise_ft = 15000
        elif distance < 500:
            cruise_ft = 25000
        elif distance < 1000:
            cruise_ft = 35000
        else:
            cruise_ft = 39000

        if self._is_helicopter:
            cruise_ft = min(cruise_ft, 5000)
        elif self.state.aircraft_type in ("C152", "C172", "C182", "PA28", "DA40", "SR22"):
            cruise_ft = min(cruise_ft, 12000)
        elif self.state.aircraft_type in ("TBM9", "PC12", "C25A", "C25C", "AT76", "DH8D"):
            cruise_ft = min(cruise_ft, 28000)

        return {
            "origin": origin.ident,
            "destination": destination.ident,
            "origin_name": origin.name,
            "destination_name": destination.name,
            "distance_nm": round(distance),
            "cruise_altitude_ft": cruise_ft,
            "ifr": not self._is_helicopter,
            "aircraft_type": self.state.aircraft_type,
        }

    def _load_next_flight(self, flight_plan: dict) -> None:
        """Load the next flight in the chain."""
        self.config.flight.departure = flight_plan["origin"]
        self.config.flight.destination = flight_plan["destination"]
        self.config.flight.cruise_altitude_ft = flight_plan["cruise_altitude_ft"]
        self.config.flight.ifr = flight_plan.get("ifr", True)

        self.session.destination = self.navdb.airport(flight_plan["destination"])
        self.session.cruise_altitude_ft = flight_plan["cruise_altitude_ft"]
        self.session.ifr = flight_plan.get("ifr", True)
        self.session.phase = Phase.PREFLIGHT

        self.session.clearance_issued = False
        self.session.clearance_read_back = False
        self.session.assigned_altitude_ft = 0.0
        self.session.assigned_heading = None
        self.session.assigned_speed_kt = None
        self.session.cleared_takeoff = False
        self.session.cleared_landing = False
        self.session.lined_up = False
        self.session.departure_runway = ""
        self.session.arrival_runway = ""
        self.session.hold_short_of = ""
        self.session.taxi_path = []
        self.session.taxi_route = []
        self.session.taxi_nodes = []
        self.session.taxi_crossings = []
        self.session.taxi_rest = []
        self.session.taxi_split_at = None
        self.session.going_around = False
        self.state.has_ifr_clearance = False
        self.state.clearance_read_back = False
        self.state.push_requested = False
        self.state.moved_since_push = False
        self.state.still_since = 0.0
        self.state.checklists_completed.clear()
        self.state.emergency = ""
        self.state.emergency_declared = False
        self.state.emergency_destination = None
        self.state.go_arounds = 0
        self.state.stabilized = False
        self.state.configured = ""
        self.state.top_of_climb = False
        self.state.tod_nm = 0.0
        self.state.minima_ft = 200.0
        self.state.vacated = False
        self.state.sub = ""
        self.state.runway_heading = 0.0
        self.state.aligned = False
        self.state.taxi_index = 0
        self.state.airborne_at = 0.0
        self.state.rotate_at = 0.0
        self.state.touchdown_at = 0.0
        self.state.rollout_at = 0.0
        self.state.flare_at = 0.0
        self.state.rejected = False
        self.state.engines_started_at = 0.0
        self.state.start_retried = False
        self.state.shutdown_at = 0.0
        self.state.missed_alt_ft = 0.0
        self.state.go_around_at = 0.0
        self.state.approach_recipe = None
        self.state.approach_recipe_tried = False
        self.state.approach_recipe_key = ()
        self.state.approach_tuned = False
        self.state.nav_tune_at = 0.0
        self.state.circling = False
        self.state.circle_path = []
        self._ga_flag_seen = False
        if self._dir is not None:
            self._dir.reset()
        self._speeds = None
        self._perf = None

        self.session.departure = self.navdb.airport(flight_plan["origin"])

        from .route import from_string, resolver

        route = from_string(
            f"DCT {flight_plan['destination']}",
            flight_plan["origin"], flight_plan["destination"],
            resolver(self.navdb), cruise_ft=flight_plan["cruise_altitude_ft"])
        if self.use_route is not None:
            self.use_route(route)

        self._emit("info", f"Next flight: {flight_plan['origin']} -> {flight_plan['destination']}",
                   "autonomous.chain.next")

    # -- helpers kept from the original ----------------------------------------

    def _compute_aircraft_performance(self, session: FlightSession) -> None:
        """Compute aircraft range and performance for chain flight planning."""
        type_code = session.aircraft.type_code.upper()
        self.state.aircraft_type = type_code

        ranges = {
            "C152": 350, "C172": 550, "C182": 750, "C208": 800,
            "PA28": 500, "PA34": 700, "DA40": 600, "DA42": 800,
            "SR22": 700, "BE36": 800, "TB20": 700,
            "TBM9": 1500, "PC12": 1300, "AT76": 800, "DH8D": 900,
            "C25A": 1200, "C25C": 1400, "C700": 1800,
            "C525": 1500, "C560": 1800, "C680": 2200, "C750": 2500,
            "E55P": 1500, "E50P": 1800, "F2TH": 2000, "F900": 2500,
            "GLEX": 3500, "GLF5": 3500, "GLF6": 4000, "G280": 2000,
            "H25B": 1800, "LJ45": 1800, "LJ60": 2000,
            "A319": 3200, "A320": 3000, "A321": 3000, "A20N": 3500, "A21N": 3500,
            "B737": 2800, "B738": 3000, "B739": 3000, "B38M": 3500, "B39M": 3500,
            "B752": 3500, "B753": 3300,
            "A332": 5500, "A333": 5500, "A339": 6000, "A343": 6500, "A346": 7000,
            "A359": 7000, "A35K": 8000, "A388": 8000,
            "B763": 5500, "B764": 5500, "B772": 5500, "B77W": 7500, "B77L": 8500,
            "B77X": 8000, "B788": 7500, "B789": 7500, "B78X": 6500,
            "B744": 7000, "B748": 7500,
            "CRJ7": 1500, "CRJ9": 1500, "CRJX": 1500,
            "E170": 1800, "E175": 1800, "E190": 2200, "E195": 2200,
        }

        range_nm = ranges.get(type_code, 0)
        if not range_nm:
            for prefix, r in ranges.items():
                if type_code.startswith(prefix):
                    range_nm = r
                    break

        if not range_nm:
            from . import aircraft_class as ac

            found = session.profile.classify(type_code).name
            if session.profile.airliner:
                range_nm = 2500
            elif found == ac.BUSINESS_JET:
                range_nm = 2000
            elif found in (ac.TURBOPROP_SINGLE, ac.TURBOPROP_TWIN):
                range_nm = 1000
            elif found == ac.HELICOPTER:
                range_nm = 250
            else:
                range_nm = 500

        self.state.aircraft_range_nm = range_nm
        log.info("Aircraft %s: estimated range %d NM", type_code, range_nm)

    def _transition(self, new_phase: AutonomousPhase) -> None:
        """Transition to a new autonomous phase."""
        log.info("Autonomous phase: %s -> %s", self.state.phase.value, new_phase.value)
        self.state.previous_phase = self.state.phase
        self.state.phase = new_phase
        self.state.phase_entered_at = time.monotonic()
        self.state.sub = ""
        self.state.sub_entered_at = time.monotonic()
        self._emit("info", f"Autonomous: {new_phase.value}", f"autonomous.phase.{new_phase.value}")

    def _set_error(self, message: str) -> None:
        """Set error state."""
        self.state.error_message = message
        self.state.error_count += 1
        log.error("autonomous flight error in %s: %s",
                  self.state.phase.value, message)
        self._transition(AutonomousPhase.ERROR)

    def _emit(self, kind: str, text: str, key: str = "") -> None:
        """Emit an event."""
        self.on_event(kind, text, "Autonomous", key=key)

    def _ensure_systems_on(self, state) -> None:
        """Ensure battery, avionics, and essential systems are on."""
        if not getattr(state, "battery_master", False):
            self._send_event(b"MASTER_BATTERY_SET", 1)
        self.director.battery(True)
        self.director.avionics(True)

    def _start_engines(self, state) -> None:
        """Start engines using starter events."""
        if not state.engines_running:
            self._send_event(b"ENGINE_AUTO_START", 1)
            self.director.starter(True)
            self.director.all_starters(True)

    def _release_parking_brake(self) -> None:
        self._send_event(b"PARKING_BRAKE_SET", 0)
        self.director.parking_brake(False)

    def _set_parking_brake(self) -> None:
        self._send_event(b"PARKING_BRAKE_SET", 1)
        self.director.parking_brake(True)

    def _request_pushback(self) -> None:
        """Request pushback through the ramp, then the simulator's own tug."""
        # The tug cannot move the aeroplane with the parking brake set. The
        # brake goes back on only after the push has stopped the aircraft.
        self._release_parking_brake()
        ground = getattr(self.brain, "ground", None)
        if ground is not None:
            try:
                said = ground.start_pushback()
            except Exception as e:
                log.debug("ramp pushback request failed: %s", e)
            else:
                if said:
                    log.info("autonomous pushback: %s", said)
                    self._emit("info", said, "autonomous.pushback")
                    return
        if self.sim is not None and getattr(self.sim, "connected", False):
            try:
                if self.sim.pushback():
                    log.info("autonomous pushback: simulator tug")
                    return
            except Exception as e:
                log.debug("simulator pushback failed: %s", e)
        if self.director.pushback():
            log.info("autonomous pushback: director event")

    def _request_clearance_via_fo(self) -> None:
        """Request IFR clearance through the first officer."""
        if self.crew.radio_call:
            text, why = self.crew.radio_call("clearance")
            if text and self.crew.transmit:
                self.crew.transmit(text)

    def _engage_climb(self, state) -> None:
        """The climb's guidance, once the initial climb is flown."""
        target = self._altitude_target(state)
        assigned_hdg = self.session.assigned_heading or 0.0
        speed = self._speed_target(state)
        if self._ap_mode == "managed":
            d = self.director
            d.autopilot(True)
            self.state.ap_master = True
            if assigned_hdg:
                d.heading(assigned_hdg)
                self.state.ap_hdg_mode = True
            else:
                d.nav(True)
                self.state.ap_nav_mode = True
            d.speed(speed)
            d.airspeed_hold(True)
            d.altitude(target)
            d.level_change(True)
        else:
            self._apply_flight_director(state, altitude=target,
                                        vs_fpm=self.performance.climb_fpm,
                                        heading=assigned_hdg or -1.0,
                                        speed_kt=speed)

    def _level_off_at_cruise(self, state) -> None:
        """Level off at cruise altitude."""
        self.director.alt_hold(True)
        self.director.nav(True)
        self.director.airspeed_hold(True)
        self.director.speed(self._speed_target(state))
        self.director.vs_hold(False)
        self.state.ap_alt_mode = True
        self.state.ap_nav_mode = True
        self.state.ap_speed_mode = True

    def _shutdown_engines(self) -> None:
        """Shutdown engines."""
        self._send_event(b"ENGINE_AUTO_SHUTDOWN", 1)
        self.director.throttle_cut()
        self.director.throttle_idle()

    def _send_event(self, event_name: bytes, value: int) -> bool:
        """Send a SimConnect event."""
        if not self.sim or not self.sim.connected:
            return False
        try:
            return self.sim._send(event_name, value)
        except Exception as e:
            log.debug("Failed to send event %s: %s", event_name, e)
            return False


def create_autonomous_flight(
    config: Config,
    navdb: NavDB,
    sim_source: Optional[SimConnectSource],
    session: FlightSession,
    crew: CrewDirector,
    brain: ControllerBrain,
    get_state: Callable[[], object],
    on_event: Callable[[str, str, str], None],
    use_route: Optional[Callable[[object], None]] = None,
) -> Optional[AutonomousFlight]:
    """Factory function to create AutonomousFlight if enabled."""
    if config.immersion.autonomous_flight:
        return AutonomousFlight(
            config, navdb, sim_source, session, crew, brain, get_state,
            on_event, use_route
        )
    return None


__all__ = ["AutonomousFlight", "AutonomousPhase", "AutonomousState", "create_autonomous_flight"]
