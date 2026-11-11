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
import math
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..config import Config
from ..navdata.build import haversine_nm
from ..navdata.db import Airport, NavDB
from ..sim.source import SimConnectSource
from . import approach_flight, auto_nav, procedure_legs, vspeeds
from . import flying as fly
from . import interphone as ip
from .ap import FlightDirector
from .checklist import holds
from .controller import ControllerBrain
from .crew import CrewDirector
from .session import FlightSession, Phase, station_key

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
    # Filed/cleared SID and STAR guidance: the chosen transition's legs
    # walked by ProcedureNavigator, keyed so a new clearance or field is
    # a new navigator. ``procedure_suspended`` is the controller's
    # "resume own navigation" for the active procedure.
    sid_nav: object = None
    star_nav: object = None
    sid_key: tuple = ()
    star_key: tuple = ()
    procedure_suspended: str = ""
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
    # The radios, worked by the FO itself: COM1 has been put on the
    # departure field's position, which station was last checked in with,
    # and what the session's squawk/altimeter were when they were last
    # dialled in -- a new value is a new instruction to action.
    comms_tuned: bool = False
    checked_in_with: str = ""
    squawk_dialled: str = ""
    altimeter_dialled: float = 0.0
    last_radio_at: float = 0.0
    # The turnaround between chained legs: the plan being serviced for,
    # whether the engines were shut and the fuel asked for, and where the
    # diversion field is that the fuel figure was built on.
    plan_flight: Optional[dict] = None
    service_started_at: float = 0.0
    gsx_asked: bool = False
    gsx_busy: bool = False
    fuel_wrote: bool = False
    service_warned: bool = False
    alternate: Optional[Airport] = None
    # The warnings a flight gets once: overweight at the hold, the flap
    # speed the schedule asked for and could not have, minimum fuel
    # already said on the radio, and the short-of-diversion note at the
    # hold.
    mtow_warned: bool = False
    flap_gate_warned: bool = False
    minimum_fuel_said: bool = False
    depart_fuel_warned: bool = False
    # What the weather has already been reacted to: the wind that held a
    # departure, the icing the systems were turned on for, the rough air
    # the speed was pulled back through, the approach conditions that
    # sent the flight around, and how many times the weather won.
    wind_hold_said: bool = False
    wind_hold_since: float = 0.0
    convective_hold_said: bool = False
    icing_said: bool = False
    ice_warned: bool = False
    turbulence_until: float = 0.0
    turbulence_said: bool = False
    windshear_since: float = 0.0
    weather_gas: int = 0
    weather_ga_active: bool = False
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
        # The FO's own jobs while it flies: the radios, the turnaround
        # fuelling between legs, and the reserve the legs are planned to.
        self._manage_radio = getattr(
            config.immersion, "autonomous_manage_comms", True)
        self._turnaround_service = getattr(
            config.immersion, "autonomous_service_turnaround", True)
        self._reserve_min = float(getattr(
            config.immersion, "autonomous_divert_reserve_min", 45.0) or 45.0)
        # Whether the weather is reacted to at all. Off, the FO flies the
        # leg it was given whatever falls out of the sky on it.
        self._weather_reactions = getattr(
            config.immersion, "autonomous_weather", True)
        # The biggest fuel_total_gal the model has ever shown: the
        # capacity for a model that declares no FUEL TOTAL CAPACITY.
        self._fuel_capacity_seen = 0.0
        # Which checklist item the FO is working, and since when -- an
        # item that cannot be made to hold is answered after the wait,
        # not waited on for ever.
        self._checklist_item_id = 0
        self._checklist_item_since = 0.0
        # When the squawk/altimeter were last pushed into the radios, so a
        # dial that does not take is retried on a clock, not every tick.
        self._dialled_at = 0.0
        self._dir: Optional[FlightDirector] = None
        self._perf: Optional[fly.Performance] = None
        self._speeds: Optional[vspeeds.Speeds] = None
        self._fuel_known = False
        self._ga_flag_seen = False
        # The last few headwind samples on final, for the windshear
        # swing the watch works off.
        self._wind_z_hist: list = []
        self._last_state = None
        # The throttle servo's last commanded lever position, 0..1.
        self._last_lever = 0.0
        # When the aeroplane last refused to move under taxi power.
        self._ground_still_since = 0.0

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
        """Stop the FO's hands, then hand the aeroplane back.

        The flag goes first so no tick runs after the handoff has centred
        the controls; a FO that lets go leaving the elevator deflected is
        how a disengage becomes an upset.
        """
        self._enabled = False
        self.config.immersion.autonomous_flight = False
        self._handoff()

    def engage(self, session: Optional[FlightSession] = None) -> bool:
        """The panel's engage, live: start the machine fresh wherever the
        aeroplane already is. False when there is no flight to fly yet --
        the next ``begin_flight`` picks the flag up instead."""
        self.enable()
        session = session or self.session
        if session is None:
            return False
        self.begin_flight(session)
        return True

    def disengage(self) -> None:
        """The panel's stop: same as ``disable`` -- off, controls handed
        back, config flag cleared so a restart does not fly itself."""
        self.disable()

    def _handoff(self) -> None:
        """Leave the aeroplane controllable the instant the FO lets go:
        surfaces centred, and every mode *this machine* engaged back off.

        The surfaces centre unconditionally -- a deflected elevator left
        from a flare is the worst souvenir there is. The AP and AT come
        off only where the FO turned them on: an autopilot the pilot
        engaged before the FO did is the pilot's, not this machine's to
        switch.
        """
        heli = self._heli
        if heli is not None and hasattr(heli, "handoff"):
            try:
                heli.handoff()
            except Exception:
                log.debug("rotorcraft handoff raised", exc_info=True)
        d = self._dir
        if d is None:
            return
        try:
            d.elevator(0.0)
            d.aileron(0.0)
            d.rudder(0.0)
            d.steering(0.0)
            d.center_controls()
            if self.state.ap_master:
                d.autopilot(False)
            if self.state.ap_apr_mode:
                d.approach(False)
                d.localizer(False)
            if self.state.ap_vs_mode:
                d.vs_hold(False)
            if self.state.ap_alt_mode:
                d.alt_hold(False)
            if self.state.ap_speed_mode:
                d.airspeed_hold(False)
                d.autothrottle_disconnect()
            if self._ap_mode == "managed" and self.state.ap_master:
                d.level_change(False)
            state = self._last_state
            if state is None or getattr(state, "on_ground", True):
                d.throttle(0.0)
                d.throttle_idle()
                d.brakes(0.0)
            d.reset()
        except Exception:
            log.debug("autonomous handoff raised", exc_info=True)
        self._last_lever = 0.0
        self.state.ap_master = False
        self.state.ap_hdg_mode = False
        self.state.ap_nav_mode = False
        self.state.ap_alt_mode = False
        self.state.ap_vs_mode = False
        self.state.ap_speed_mode = False
        self.state.ap_apr_mode = False

    def status(self) -> dict:
        """What the panel shows about the FO: on or off, and if on, what
        the machine is doing."""
        return {
            "available": self.sim is not None,
            "enabled": self._enabled,
            "helicopter": self._is_helicopter,
            "phase": self.state.phase.value,
            "sub": self.state.sub or "",
            "error": (self.state.error_message
                      if self.state.phase == AutonomousPhase.ERROR else ""),
        }

    def _entry_phase(self, session: FlightSession):
        """Where an engage joins the flight, by where the aeroplane is.

        A ground start walks the whole machine from preflight; an engage
        in the air must not -- a FO asked to take over on climbout who
        began by running the before-start checklist would fly the
        aeroplane into whatever was in front of it.
        """
        try:
            state = self.get_state()
        except Exception:
            state = None
        if state is None or not getattr(state, "connected", False) \
                or getattr(state, "on_ground", True):
            return AutonomousPhase.PREFLIGHT, "systems"
        session_phase = getattr(session, "phase", Phase.PREFLIGHT)
        if session_phase in (Phase.APPROACH, Phase.LANDING):
            return AutonomousPhase.APPROACH, "intercept"
        if session_phase == Phase.DESCENT:
            return AutonomousPhase.DESCENT, ""
        target = self._altitude_target(state)
        altitude = float(getattr(state, "altitude_ft", 0.0) or 0.0)
        if target and altitude > target + 500.0:
            return AutonomousPhase.DESCENT, ""
        if target and abs(altitude - target) <= 500.0:
            return AutonomousPhase.CRUISE, ""
        return AutonomousPhase.CLIMB, ""

    def begin_flight(self, session: FlightSession) -> None:
        """Start a new autonomous flight."""
        self.session = session
        self.state = AutonomousState()
        self.state.phase, sub = self._entry_phase(session)
        self.state.phase_entered_at = time.monotonic()
        self._goto_sub(sub)
        if self._dir is not None:
            self._dir.reset()
        self._perf = None
        self._speeds = None
        self._fuel_known = False
        self._heli = None
        self._last_lever = 0.0
        self._ground_still_since = 0.0
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
        # The engine calls this for every new session, engaged or not --
        # only the engaged machine gets to announce itself.
        if self._enabled:
            self._emit("info", "Autonomous flight started",
                       "autonomous.started")

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

        now = time.monotonic()

        # A helicopter is flown by its own controller, on every pass of the
        # loop rather than the fixed-wing action interval: the cyclic,
        # collective and pedals are closed-loop controls, not once-a-second
        # mode selections. The radio and the checklists are a FO's jobs
        # whatever it flies, so they run from here either way.
        self._manage_comms(state, now)
        self._drive_checklist(state, now)
        # The weather watch is a FO's job whatever is flown, like the
        # radios: a helicopter ices and is blown around as an aeroplane
        # does. The go-around parts are the fixed-wing phases' own.
        self._monitor_weather(state, now)
        if self._heli is not None:
            self._heli.tick(state)
            return

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
        # The threshold is under a knot: a tug walks the aeroplane, and two
        # knots is faster than the push ever gets.
        if self.state.phase == AutonomousPhase.PUSHBACK:
            if (state.ground_speed_kt or 0.0) > 0.8:
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

    def _penetration_kt(self, state) -> float:
        """The speed rough air is ridden out at: a margin off the red
        line and off the stall, the way the books write it."""
        cruise = self.performance.cruise_kt or 200.0
        lim = self._limits(state)
        pen = cruise * 0.75
        if lim["stall_kt"] > 0.0:
            pen = max(pen, lim["stall_kt"] * 1.5)
        return max(60.0, min(pen, lim["max_ias_kt"]))

    def _speed_target(self, state) -> float:
        target = 0.0
        if self.session.assigned_speed_kt:
            target = self.session.assigned_speed_kt
        # In the cruise the model's own estimate of its cruise speed is
        # the best figure there is -- it is this aeroplane's, not a class's
        # -- and the class number is the fallback for a model that
        # declares none.
        elif self.state.phase == AutonomousPhase.CRUISE:
            estimated = float(getattr(state, "estimated_cruise_kt", 0.0)
                              or 0.0)
            if 50.0 < estimated < 600.0:
                # The variable may be true rather than indicated speed;
                # never command past the model's own red line.
                redline = float(getattr(state, "barber_pole_kt", 0.0)
                                or 0.0)
                if redline > 0:
                    estimated = min(estimated, redline * 0.95)
                target = estimated
        if not target:
            target = self.state.climb_kt or self.performance.cruise_kt
        if time.monotonic() < self.state.turbulence_until:
            # Rough air beats the assignment, the way a crew asks for a
            # slower speed rather than ride it out at it.
            target = min(target, self._penetration_kt(state))
        return target

    def _apply_flight_director(self, state, altitude: float = 0.0,
                             vs_fpm: float = 0.0, heading: float = -1.0,
                             speed_kt: float = 0.0) -> None:
        """Put the assigned numbers on the autopilot, managed or direct.

        ``heading`` < 0 means "not commanded"; ``vs_fpm`` nonzero means a
        rate is wanted (climb or descent in progress); zero means level.
        """
        d = self.director
        # The sim's own flag is the truth: a pilot who bumped the AP off
        # mid-engage gets it back rather than a machine that "remembers"
        # engaging it once and never asks again. AUTOPILOT_ON is
        # idempotent, so a model that never reports the variable just
        # gets asked every pass.
        if not getattr(state, "autopilot", False):
            d.autopilot(True)
        self.state.ap_master = True
        lim = self._limits(state)
        if altitude:
            d.altitude(self._safe_altitude(state, altitude, lim))
        if heading >= 0.0:
            d.heading(heading)
            self.state.ap_hdg_mode = True
        if speed_kt:
            speed_kt = self._safe_speed(state, speed_kt, lim)
            d.speed(speed_kt)
            self.state.ap_speed_mode = True
            self._manage_throttle(state, speed_kt)
        if vs_fpm:
            vs_fpm = max(lim["vs_dn_fpm"], min(float(vs_fpm),
                                             lim["vs_up_fpm"]))
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

    def _manage_throttle(self, state, target_kt: float) -> None:
        """The FO's own throttle, for the aeroplanes with no autothrottle.

        On a model whose autopilot cannot hold speed the lever stays
        where takeoff or the go-around left it -- full power all the way
        to the red line. This servo moves the levers only when the speed
        is genuinely off target, so a working autothrottle, which keeps
        the speed inside the deadband on its own, is never fought.
        """
        if not target_kt or state.on_ground or not state.engines_running:
            return
        # A live autothrottle owns the levers: the servo's job only exists
        # where there is none, and two hands on one lever is a porpoise.
        if getattr(state, "autothrottle", False):
            return
        if self.state.phase == AutonomousPhase.APPROACH:
            # The go-around owns the levers, and the pattern right after
            # it is still a climb to the missed altitude, not an arrival
            # at approach speed.
            if self.state.sub == "go_around":
                return
            missed = self.state.missed_alt_ft or 0.0
            if missed and (state.altitude_ft or 0.0) < missed \
                    and time.monotonic() - self.state.go_around_at < 180.0:
                return
        ias = float(getattr(state, "indicated_airspeed_kt", 0.0) or 0.0)
        if ias <= 20.0:
            return
        lever = self._lever_position(state)
        barber = float(getattr(state, "barber_pole_kt", 0.0) or 0.0)
        max_mach = float(getattr(state, "mach_max_operate", 0.0)
                         or getattr(state, "barber_pole_mach", 0.0) or 0.0)
        mach = float(getattr(state, "airspeed_mach", 0.0) or 0.0)
        err = ias - target_kt
        if (barber > 0.0 and ias >= barber * 0.97) \
                or (max_mach > 0.0 and mach >= max_mach - 0.005):
            # At the red line -- knots low down, mach high up -- the lever
            # comes down, whatever the target.
            lever = min(lever - 0.10, 0.40)
        elif err > 8.0:
            lever -= min(0.20, 0.02 + err * 0.006)
        elif err < -15.0:
            lever += min(0.15, 0.02 + (-err) * 0.006)
        else:
            return
        lever = max(0.05, min(0.97, lever))
        self._last_lever = lever
        self.director.throttle(lever)

    # -- the envelope ------------------------------------------------------

    def _limits(self, state) -> dict:
        """What the aeroplane may be asked to do, in the units it thinks in.

        Every commanded figure goes through here. The red line is the
        aeroplane's own barber pole or design VC where the model declares
        them, else the class's cruise figure bounded up -- a table cruise
        number is a cruise number, not a never-exceed. The stall floor is
        the model's VS1 with margin. A field the model never publishes is
        a conservative class default, not an exception to the envelope.
        """
        perf = self.performance
        barber = float(getattr(state, "barber_pole_kt", 0.0) or 0.0)
        design_vc = float(getattr(state, "design_vc_kt", 0.0) or 0.0)
        vs1 = float(getattr(state, "design_vs1_kt", 0.0) or 0.0)
        vs0 = float(getattr(state, "design_vs0_kt", 0.0) or 0.0)
        max_ias = perf.cruise_kt * 1.35
        if design_vc > 0.0:
            max_ias = min(max_ias, design_vc)
        if barber > 0.0:
            max_ias = min(max_ias, barber * 0.95)
        min_clean = vs1 * 1.3 if vs1 > 0.0 else 0.0
        stall = vs0 or vs1 or 0.0
        # Flap and gear speeds: the model publishes neither, so the
        # figures are the class's -- where the type's own limits are
        # higher, the lower number is still safe.
        flap_max, gear_max = self._config_limits()
        max_mach = float(getattr(state, "mach_max_operate", 0.0)
                         or getattr(state, "barber_pole_mach", 0.0) or 0.0)
        ceiling = float(getattr(perf, "ceiling_ft", 0.0) or 0.0)
        return {
            "max_ias_kt": max_ias,
            "min_clean_kt": min_clean,
            "stall_kt": stall,
            "flap_max_kt": flap_max,
            "gear_max_kt": gear_max,
            "max_mach": max_mach,
            "ceiling_ft": ceiling,
            "vs_up_fpm": min(perf.climb_fpm, 4000.0),
            "vs_dn_fpm": -min(perf.descend_fpm, 4000.0),
        }

    def _config_limits(self) -> tuple[float, float]:
        """(max speed for any flap, for the gear down), by class.

        An airliner works to roughly 250/265; a light aeroplane's manual
        says 110-ish for the flaps and about 130 for the wheels. The
        cruise figure is the proxy: it is the class's idea of how big the
        aeroplane is.
        """
        if self.performance.cruise_kt >= 230.0:
            return 250.0, 265.0
        if self.performance.cruise_kt >= 140.0:
            return 200.0, 200.0
        return 110.0, 130.0

    def _safe_speed(self, state, kt: float, lim: dict | None = None) -> float:
        """An indicated-airspeed command inside the envelope."""
        if lim is None:
            lim = self._limits(state)
        value = float(kt or 0.0)
        if value <= 0.0:
            return 0.0
        return max(45.0, min(value, lim["max_ias_kt"]))

    def _safe_altitude(self, state, ft: float, lim: dict | None = None) -> float:
        """An altitude command under the ceiling, where one is known."""
        if lim is None:
            lim = self._limits(state)
        value = float(ft or 0.0)
        if lim["ceiling_ft"] > 0.0:
            value = min(value, lim["ceiling_ft"])
        return value

    def _flaps_allowed(self, state) -> bool:
        """Whether a flap may come out now: never past the flap speed."""
        ias = float(getattr(state, "indicated_airspeed_kt", 0.0) or 0.0)
        lim = self._limits(state)
        if ias > lim["flap_max_kt"]:
            if not self.state.flap_gate_warned:
                self.state.flap_gate_warned = True
                self._emit("info",
                           f"Flaps held: {ias:.0f} kt is past the flap "
                           "speed", "autonomous.limits.flaps")
            return False
        return True

    def _set_flaps(self, state, fraction: float) -> None:
        """Move the flap lever -- an extension is gated by the flap speed.

        Retraction is always allowed: bringing them in early is a drag
        penalty, never an airframe one.
        """
        if fraction <= 0.0 or self._flaps_allowed(state):
            self.director.flaps(fraction)

    def _lever_position(self, state) -> float:
        """Where the levers are, 0..1, as the sim last reported them."""
        pct = float(getattr(state, "throttle_pct", 0.0) or 0.0)
        if pct > 0.0:
            return max(0.0, min(1.0, pct / 100.0))
        return self._last_lever

    def _run_checklist_for_phase(self, checklist_name: str) -> None:
        """Trigger a checklist for the current phase."""
        if checklist_name not in self.state.checklists_completed:
            self.crew.checklists.start(checklist_name)
            self.state.checklists_completed.add(checklist_name)

    # -- the radios ---------------------------------------------------------

    def _fo_say(self, kind: str) -> bool:
        """One transmission by the first officer, through the same speech
        and the same parsing as the pilot's. False where the call was not
        made -- nothing to say, or nobody to say it on."""
        if self.crew.radio_call is None or self.crew.transmit is None:
            return False
        try:
            text, why = self.crew.radio_call(kind)
        except Exception:
            log.debug("the first officer's %s call failed", kind,
                      exc_info=True)
            return False
        if not text:
            log.debug("first officer %s call not made: %s", kind, why)
            return False
        self.crew.transmit(text)
        self.state.last_radio_at = time.monotonic()
        return True

    def _radio_quiet(self) -> bool:
        try:
            return not (self.crew.radio_busy
                        and self.crew.radio_busy())
        except Exception:
            log.debug("could not tell whether the frequency is busy",
                      exc_info=True)
            return True

    def _manage_comms(self, state, now: float) -> None:
        """Work the radios the way the first officer flying would.

        COM1 is put on the position the phase is worked through on the
        ground -- delivery, ground, tower -- and on the station a handoff
        names in the air. The squawk and the altimeter are dialled to the
        figures the controller gave. A readback owed goes out before
        anything volunteered, and a new airborne station is checked in
        with once. Nothing is keyed while the frequency is busy, and
        nothing at all while the radio has no power.
        """
        if not self._manage_radio:
            return
        sim = self.sim
        session = self.session
        if sim is None or not getattr(sim, "connected", False) \
                or session is None:
            return
        if self.state.phase in (AutonomousPhase.INIT,
                                AutonomousPhase.COMPLETE,
                                AutonomousPhase.ERROR):
            return
        from .radio_power import decide

        powered, _why = decide(state)
        if not powered or not self._radio_quiet():
            return

        # The dials are worked whatever else the frequency is doing: they
        # are knob-turns, not transmissions.
        self._dial_radios(state, now)

        # A readback owed goes before anything the FO would volunteer --
        # a controller who has spoken is answered first.
        pending = getattr(session, "pending", None)
        if pending is not None and pending.readback_required \
                and not pending.read_back:
            if now - self.state.last_radio_at >= 4.0:
                self._fo_say("readback")
            return

        # A handoff: the number it named is tuned, and the new station is
        # checked in with below once the engine has seen the retune.
        handed = getattr(session, "handoff_pending", None)
        if handed is not None and getattr(handed, "mhz", 0.0):
            freq = float(handed.mhz)
            active = float(getattr(state, "com1_active", 0.0) or 0.0)
            if abs(active - freq) > 0.004:
                if now - self._dialled_at >= 6.0 and sim.set_com(freq, 1):
                    self._dialled_at = now
                    self._emit("info", f"COM1 to {freq:.3f}",
                               "autonomous.comms.tune")
                return

        # On the ground at the field the leg leaves from, COM1 belongs to
        # the position that works this phase. In the air the frequency is
        # only changed by a handoff.
        if state.on_ground:
            self._tune_for_phase(state, session, now)
            # The calls the flight cannot be moved without: taxi once off
            # the stand, ready-for-departure at the hold. Nobody issues
            # an instruction to a flight that never asked, which is how
            # the aeroplane sat on the ramp until the taxi watchdog bit.
            self._work_the_calls(state, session, now)

        # A new airborne station is checked in with: the level, and
        # whether it is being climbed or descended to. Ground positions
        # are talked to by the calls that ask for things, and a station
        # the flight is already established with needs nothing.
        current = getattr(session, "current_station", None)
        if current is None \
                or getattr(current, "position", "") in ("DEL", "GND") \
                or (state.on_ground and handed is None):
            return
        if session.established_with(current):
            return
        key = station_key(current)
        fresh = key != self.state.checked_in_with
        if (fresh or now - self.state.last_radio_at >= 60.0) \
                and now - self.state.last_radio_at >= 4.0 \
                and self._fo_say("check_in"):
            self.state.checked_in_with = key

    def _work_the_calls(self, state, session, now: float) -> None:
        """Ask for the thing the phase is waiting on.

        A controller only answers a call that was made: taxi is asked of
        ground once the aeroplane is off the stand, and ready-for-
        departure is asked of the tower at the hold. On a broadcast
        frequency the call is itself the announcement and nobody is
        expected to answer, but the asking is still how the flight's
        intentions get onto the frequency. A call is repeated while it
        goes unanswered -- the frequency can be too busy to have heard
        the first one -- and ``last_radio_at`` is what keeps that a
        request and not a stream.
        """
        phase = self.state.phase
        if phase == AutonomousPhase.TAXI_OUT:
            if session.phase in (Phase.TAXI_OUT, Phase.HOLDING_SHORT,
                                 Phase.LINED_UP, Phase.TAKEOFF) \
                    or getattr(session, "taxi_path", None):
                return
            kind = "taxi"
        elif phase == AutonomousPhase.HOLDING_SHORT:
            if session.lined_up or session.cleared_takeoff \
                    or session.phase in (Phase.LINED_UP, Phase.TAKEOFF):
                return
            if (state.ground_speed_kt or 0.0) > 1.0:
                # Ready-for-departure is called from a stopped aeroplane.
                return
            kind = "ready"
        else:
            return
        tried = getattr(self, "_call_tried_at", 0.0)
        if now - max(self.state.last_radio_at, tried) < 25.0:
            return
        self._call_tried_at = now
        self._fo_say(kind)

    def _tune_for_phase(self, state, session, now: float) -> None:
        """COM1 on the position the ground phase is worked through."""
        want = {"preflight": "DEL", "request_clearance": "DEL",
                "pushback": "GND", "taxi_out": "GND",
                "holding_short": "TWR", "line_up": "TWR",
                "takeoff": "TWR"}.get(self.state.phase.value)
        if want is None:
            return
        airport = session.departure or self._get_current_airport()
        if airport is None:
            return
        station = self.navdb.station_for(airport.ident, want)
        if station is None or not getattr(station, "mhz", 0.0):
            return
        active = float(getattr(state, "com1_active", 0.0) or 0.0)
        if abs(active - float(station.mhz)) <= 0.004:
            return
        if now - self._dialled_at < 8.0:
            return
        if self.sim.set_com(float(station.mhz), 1):
            self._dialled_at = now
            self._emit("info", f"COM1 to {station.callsign} "
                       f"{float(station.mhz):.3f}", "autonomous.comms.tune")

    def _dial_radios(self, state, now: float) -> None:
        """Squawk and altimeter as the controller gave them, dialled once."""
        session = self.session
        want = "".join(c for c in str(getattr(session, "squawk", "") or "")
                       if c.isdigit())[:4]
        if want and want != self.state.squawk_dialled:
            have = "".join(
                c for c in str(getattr(state, "transponder_code", ""))
                if c.isdigit())[:4]
            if have == want:
                self.state.squawk_dialled = want
            elif now - self._dialled_at >= 5.0 \
                    and self.sim.set_squawk(want):
                self._dialled_at = now
        alt = float(getattr(session, "altimeter", 0.0) or 0.0)
        if alt > 0.0 and abs(alt - self.state.altimeter_dialled) > 0.2:
            have = float(getattr(state, "kohlsman_hpa", 0.0) or 0.0)
            if have > 0.0 and abs(have - alt) <= 0.5:
                self.state.altimeter_dialled = alt
            elif now - self._dialled_at >= 5.0:
                self.director.kohlsman(alt)
                self._dialled_at = now

    # -- the checklist, answered by the FO flying ---------------------------

    def _drive_checklist(self, state, now: float) -> None:
        """Work an open checklist: do the item, then answer it.

        A pilotless aeroplane has nobody for the first officer to wait
        on, so when a run is waiting on an item the FO does the item --
        the runner's own check says whether the aeroplane agrees -- and
        answers it itself. A held run is carried on after a pause; a run
        still being read out is left to finish being said. The answer is
        the runner's ``fo_answer``: the same verification a spoken answer
        gets, so a switch the simulator says is still wrong is objected
        to, never waved through.
        """
        runner = self.crew.checklists
        if runner is None:
            return
        item, run_state = runner.pending()
        if item is None:
            self._checklist_item_id = 0
            return
        if id(item) != self._checklist_item_id:
            self._checklist_item_id = id(item)
            self._checklist_item_since = now
            return
        if run_state == "held":
            if now - self._checklist_item_since > 4.0:
                runner.hear(ip.Said(who=ip.FIRST_OFFICER, what=ip.CONTINUE))
                self._checklist_item_since = now
            return
        if run_state != "waiting":
            return
        # Do the item where the check names a variable the FO can reach;
        # the answer goes once the simulator has had a moment to show it.
        if self._perform_check_item(state, item):
            self._checklist_item_since = now
            return
        if now - self._checklist_item_since >= 2.0:
            runner.fo_answer()
            # The next pass sees the advance -- or the objection, which a
            # re-performed item then answers.
            self._checklist_item_since = now

    def _perform_check_item(self, state, item) -> bool:
        """Move the switch an item's check names, where it disagrees.

        True where a control was asked for, so the item is answered on a
        later pass once the new value shows. An item whose check names
        nothing the FO can reach, or nothing at all, is not performed --
        the answer simply reports what is.
        """
        check = getattr(item, "check", None)
        if check is None:
            return False
        runner = self.crew.checklists
        try:
            expected = runner._expected(check)
        except Exception:
            expected = check.value
        actual = getattr(state, check.var, None)
        if holds(check, actual, expected):
            return False
        return self._switch_for(state, check.var, expected)

    def _switch_for(self, state, var: str, expected) -> bool:
        """The cockpit control behind a checklist variable."""
        d = self.director
        want_on = (expected is True) or expected == "on" \
            or (isinstance(expected, (int, float))
                and not isinstance(expected, bool) and expected > 0)
        table = {
            "parking_brake": lambda: d.parking_brake(want_on),
            "gear_down": lambda: d.gear(want_on),
            "landing_light": lambda: d.landing_light(want_on),
            "taxi_light": lambda: d.taxi_light(want_on),
            "nav_light": lambda: d.nav_lights(want_on),
            "strobe_light": lambda: d.strobes(want_on),
            "beacon_light": lambda: d.beacon(want_on),
            "pitot_heat": lambda: d.pitot_heat(want_on),
            "seatbelt_sign": lambda: d.seatbelts(want_on),
            "battery_master": lambda: d.battery(want_on),
            "avionics_master": lambda: d.avionics(want_on),
            "spoilers_armed": lambda: d.spoilers_arm(want_on),
            "autopilot": lambda: d.autopilot(want_on),
            "autothrottle": lambda: (d.autothrottle_arm() if want_on
                                     else d.autothrottle_disconnect()),
            "gps_drives_nav1": lambda: d.write("GPS DRIVES NAV1",
                                               1.0 if want_on else 0.0),
            "fuel_selector": lambda: d.write("FUEL TANK SELECTOR:1",
                                             float(expected)),
            "mixture_pct": lambda: d.write("GENERAL ENG MIXTURE LEVER "
                                           "POSITION:1", float(expected)),
            "transponder_code": lambda: d.transponder(str(expected)),
            "kohlsman_hpa": lambda: d.kohlsman(float(expected)),
            "flaps_index": lambda: d.flaps(min(1.0, float(expected) / 4.0)),
            "spoilers": lambda: d.spoilers(min(1.0, float(expected))),
        }
        action = table.get(var)
        if action is None:
            return False
        try:
            action()
        except Exception:
            log.debug("checklist switch %s failed", var, exc_info=True)
            return False
        return True

    def _configure_for_takeoff(self, state) -> None:
        """Set flaps, trim, lights for takeoff."""
        d = self.director
        # Takeoff flaps: the first detent, where the model has one.
        d.flaps(0.25 if state.flaps_index == 0 else 0.0)
        d.landing_light(True)
        d.strobes(True)

    def _set_takeoff_power(self) -> None:
        """Takeoff thrust: full, and TO/GA on the autothrottle where the
        model has one. The one-shot events re-arm on each roll and each
        go-around; the lever position itself is re-asked every call."""
        d = self.director
        d.takeoff_power()
        d.throttle(1.0)
        self._last_lever = 1.0
        d.toga(True)

    def _ground_still(self, state, now: float) -> bool:
        """Whether the aeroplane has refused to move under power for a
        while -- chocked, braked, or boxed in. Taxi power against a
        stationary aeroplane is how the engine ends up at takeoff rpm on
        the ramp, so after a dozen seconds of it the levers come back to
        idle and the phase decides what to do about not moving."""
        if (state.ground_speed_kt or 0.0) > 1.0:
            self._ground_still_since = 0.0
            return False
        if self._ground_still_since == 0.0:
            self._ground_still_since = now
            return False
        stuck = now - self._ground_still_since
        if stuck <= 12.0:
            return False
        # Idle while it refuses to move, but probe with power for a few
        # seconds of every half minute: a chock coming out or a brake
        # releasing should not need a whole new flight to notice.
        return not 27.0 <= stuck % 30.0 < 30.0

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
        # The nosewheel follows the same demand where the model steers it
        # separately: a taildragger on rudder alone and a tricycle gear on
        # STEERING_SET both get their turn.
        d.steering(rudder)
        # Taxi and rollout speeds are throttle jobs, not AP jobs -- and
        # progressive ones: the further below target, the more asked for.
        gs = state.ground_speed_kt or 0.0
        if self._ground_still(state, time.monotonic()):
            d.throttle(0.0)
        elif gs < target_kt - 2.0:
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
            d.steering(0.0)
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
            # The seatbelt event is a toggle, not a set: asking for "on"
            # where it is already on would switch it off.
            if not getattr(state, "seatbelt_sign", False):
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
            # A stand the simulator says has no pushback will never send
            # a tug: rather than ask into silence, an aeroplane that can
            # drive out just does. ``pushback_available`` unread -- None
            # -- is not a refusal.
            if getattr(state, "pushback_available", None) is False \
                    and self._pushback_skippable():
                self._emit("info", "No pushback at this stand; "
                                   "taxiing out",
                           "autonomous.pushback.skipped")
                self._goto_sub("taxi")
                self._transition(AutonomousPhase.TAXI_OUT)
                return
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
                # Whether the simulator has a live request: PUSHBACK WAIT
                # while the tug is on its way, PUSHBACK ATTACHED once it
                # is on the nosewheel. The request event is a toggle, so
                # asking again while either is true cancels the push that
                # is coming -- the retry is only for a request that never
                # landed at all.
                engaged = bool(getattr(state, "pushback_wait", False)
                               or getattr(state, "pushback_attached",
                                          False))
                if not engaged:
                    if self._sub_age() > 120.0 \
                            and self.state.push_retries < 2:
                        self._request_pushback()
                        self.state.push_retries += 1
                        self.state.sub_entered_at = now
                    elif self._sub_age() > 120.0:
                        # Asked for repeatedly and nothing ever engaged.
                        # A light aeroplane on a ramp simply drives out;
                        # what cannot is still an error rather than a
                        # push through the terminal.
                        if self._pushback_skippable():
                            self._emit(
                                "info",
                                "No pushback answered at this stand; "
                                "taxiing out",
                                "autonomous.pushback.skipped")
                            self._goto_sub("taxi")
                            self._transition(AutonomousPhase.TAXI_OUT)
                            return
                        self._set_error("Pushback did not move the "
                                        "aircraft")
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
        # clearance is what starts the roll. The aeroplane only goes if it
        # is light enough and the pavement is long enough for it.
        if self.session.lined_up or self.session.cleared_takeoff \
                or self.session.phase in (Phase.LINED_UP, Phase.TAKEOFF):
            if not self._departure_possible(state):
                return
            self._transition(AutonomousPhase.LINE_UP)
            return

        # On a broadcast frequency nobody answers: the first officer's
        # "ready" call was the announcement, and the pilot approves the
        # departure themselves once it has had time to go out and any
        # final to be looked at. ``cleared_takeoff`` is the state
        # machine's name for "the runway may be taken", and on CTAF the
        # person granting that is the one flying.
        station = getattr(self.session, "current_station", None)
        if getattr(station, "position", "") in (
                "CTAF", "UNICOM", "INFO", "AFIS", "RDO") \
                and self.state.last_radio_at >= self.state.phase_entered_at \
                and now - self.state.phase_entered_at > 20.0:
            if not self._departure_possible(state):
                return
            self.session.cleared_takeoff = True
            self.session.cleared_takeoff_at = now
            self._emit("info",
                       "No tower on this frequency; departing after "
                       "self-announcement", "autonomous.depart.broadcast")
            self._transition(AutonomousPhase.LINE_UP)

    def _departure_possible(self, state) -> bool:
        """Whether the aeroplane may be taken onto the runway at all.

        The weight the sim reports against the gross the model declares,
        and the pavement against the takeoff run the model suggests. A
        model that publishes neither is taken on trust; one that does is
        held to it -- an overweight aeroplane does not get flown off.
        """
        mtow = float(getattr(state, "max_gross_weight_lb", 0.0) or 0.0)
        weight = float(getattr(state, "total_weight_lb", 0.0) or 0.0)
        if mtow > 0.0 and weight > mtow * 1.02:
            self._set_error(f"Overweight: {weight:.0f} lb for a "
                            f"{mtow:.0f} lb maximum takeoff weight")
            return False
        if self.state.runway_heading <= 0:
            self._find_runway(state)
        need = float(getattr(state, "suggested_runway_takeoff_ft", 0.0)
                     or 0.0)
        have = float(self.state.runway_length_ft or 0.0)
        if need > 0.0 and have > 0.0 and have < need * 0.9 \
                and not self.state.mtow_warned:
            # Warned rather than refused: the model's suggestion is a
            # book figure, not a measured run, and the runway data can
            # equally be the one that is off.
            self.state.mtow_warned = True
            self._emit("info",
                       f"Runway {have:.0f} ft for a suggested {need:.0f} "
                       "ft takeoff run", "autonomous.limits.runway")
        if self._weather_reactions and not self._departure_wind_ok(state):
            return False
        return self._departure_fuel_ok(state)

    def _departure_fuel_ok(self, state) -> bool:
        """Whether the tanks cover the leg with its diversion and reserve.

        What a crew signs for: enough to reach the destination, divert to
        the alternate and land with the reserve. The diversion figure is
        the aim -- under it is said once and flown with care; under the
        leg with the reserve alone is not a departure.
        """
        needed = self._fuel_needed_gal(state, estimate=True)
        if needed is None:
            return True
        fuel_gal = float(getattr(state, "fuel_total_gal", 0.0) or 0.0)
        if fuel_gal <= 0.0:
            return True
        if fuel_gal < needed["leg_gal"] + needed["reserve_gal"]:
            self._set_error(
                f"Not enough fuel: {fuel_gal:.0f} gal for a leg needing "
                f"{needed['total_gal']:.0f} with the diversion")
            return False
        if fuel_gal < needed["total_gal"] \
                and not self.state.depart_fuel_warned:
            self.state.depart_fuel_warned = True
            alt = needed.get("alternate")
            self._emit("info",
                       f"Departing short of the diversion figure: "
                       f"{fuel_gal:.0f} gal, {needed['total_gal']:.0f} "
                       f"planned via {alt.ident if alt else 'no alternate'}",
                       "autonomous.fuel.short")
        return True

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
                # The power events are one-shot: a new roll is a new
                # press of TO/GA and THROTTLE_FULL.
                self.director.forget("takeoff_power", "toga")
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
            # Directional control: the nosewheel does the early work, the
            # rudder takes over as the airflow comes alive. Both get the
            # same demand; a model that only honours one still tracks.
            err = ((self.state.runway_heading - (state.heading_true or 0.0)
                    + 540.0) % 360.0 - 180.0)
            keep = 0.0 if abs(err) < 2.0 else max(-0.6, min(0.6, err / 10.0))
            d.rudder(keep)
            d.steering(keep if (state.indicated_airspeed_kt or 0.0) < 60.0
                       else 0.0)
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
        sid = self._sid_nav(state)
        if sid is not None:
            # A filed/cleared SID owns the lateral path until its last
            # leg; the assigned heading still wins where ATC gave one.
            self._fly_procedure(state, sid)
        elif assigned_hdg:
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
        sid = self._sid_nav(state)
        if sid is not None:
            self._fly_procedure(state, sid)
        else:
            self.director.nav(True)
            self.director.airspeed_hold(True)
            target = self._altitude_target(state)
            # No assigned or planned level is not an instruction to find
            # the sea: a zero on the altitude bug is a descent to it.
            if target:
                self.director.altitude(target)
            self.director.alt_hold(True)
            self._manage_throttle(state, self._speed_target(state))
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
            self._set_flaps(state, 0.25)
            self.state.configured = "flaps_approach"
        if agl < 2500 or (distance is not None and distance < 8.0):
            self._set_flaps(state, 0.5)
            self.state.configured = "flaps_final"
            self.director.spoilers_arm(True)

        # Speed: assigned, else a sensible step-down then approach speed.
        speed = self.session.assigned_speed_kt or (
            (self._vapp(state)) + 20.0
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
        # A filed/cleared STAR owns the lateral path until its last leg;
        # an assigned heading still wins where ATC gave one.
        star = self._star_nav(state)
        if star is not None:
            self._fly_procedure(state, star)
        else:
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
                        state, vs_fpm=vs,
                        heading=self._magnetic(self.state.runway_heading,
                                               state),
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
                self._last_lever = 0.0
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
                self._last_lever = 0.05
                d.approach(True)
                if self.state.runway_heading <= 0:
                    self._find_runway(state)
            if now - self.state.phase_entered_at > self.LANDING_TIMEOUT:
                self._start_go_around(state)
            return

        if sub == "rollout":
            # Down: spoilers if armed, brakes on, steer to the exit or the
            # taxi path, slow to taxi speed. The aeroplane is on the
            # ground now -- the flight is over, and an autopilot left in
            # speed hold on the ground is what runs the levers to the
            # stops while parked.
            if self.state.ap_master or self.state.ap_speed_mode \
                    or self.state.ap_apr_mode or self.state.ap_vs_mode:
                d.autopilot(False)
                d.airspeed_hold(False)
                d.approach(False)
                d.localizer(False)
                d.vs_hold(False)
                if self._ap_mode == "managed":
                    d.level_change(False)
                self.state.ap_master = False
                self.state.ap_speed_mode = False
                self.state.ap_apr_mode = False
                self.state.ap_vs_mode = False
                self.state.ap_alt_mode = False
                self.state.ap_hdg_mode = False
                self.state.ap_nav_mode = False
            d.spoilers(1.0)
            d.spoilers_arm(True)
            d.flaps(0.0)
            gs = state.ground_speed_kt or 0.0
            if gs > self.performance.taxi_kt + 5.0:
                # Speedbrakes are already out; the wheel brakes take it
                # from here, at the axis scale the brake axes read.
                d.parking_brake(False)
                d.brakes(1.0)
            else:
                # Slow enough: brakes off, taxi in.
                d.brakes(0.0)
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
                rudder = max(-0.4, min(0.4, err / 15.0))
                d.rudder(rudder)
                d.steering(rudder)
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
            # beacon, pitot heat off, controls centred. The autopilot
            # comes off too -- a parked aeroplane has no flight director.
            d.autopilot(False)
            d.airspeed_hold(False)
            d.approach(False)
            d.vs_hold(False)
            self.state.ap_master = False
            self.state.ap_speed_mode = False
            self.state.ap_apr_mode = False
            self.state.ap_vs_mode = False
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
        """Turn the aeroplane around and load the next leg of the chain.

        ``pick`` chooses the destination, ``delay`` gives the stand its
        moment, ``service`` shuts the engines down and fuels for the leg
        with its diversion, and ``load`` files the leg and starts it.
        """
        sub = self.state.sub or "pick"

        if sub == "pick":
            if self.state.chain_flight_index >= len(self.state.chain_flights):
                self._select_next_destination()
            if self.state.chain_flight_index >= len(self.state.chain_flights):
                self._emit("info",
                           "No suitable destination found for next flight",
                           "autonomous.chain.no_destination")
                self._transition(AutonomousPhase.SHUTDOWN)
                return
            self.state.plan_flight = self.state.chain_flights[
                self.state.chain_flight_index]
            self._goto_sub("delay")
            return

        if sub == "delay":
            if now - self.state.phase_entered_at < self._chain_delay:
                return
            self._goto_sub("service")

        if sub == "service":
            if self._service_turnaround(state, now):
                self._goto_sub("load")
            return

        if sub == "load":
            self.state.chain_flight_index += 1
            self._load_next_flight(self.state.plan_flight)
            self.state.plan_flight = None
            self._transition(AutonomousPhase.PREFLIGHT)

    # -- the turnaround between legs ------------------------------------------

    def _service_turnaround(self, state, now: float) -> bool:
        """Fuel and service for the next leg; True when it may be loaded.

        The engines go off before a hose comes anywhere near the
        aeroplane. The fuel figure is the leg's own plus the diversion to
        the alternate and the reserve -- what a crew would sign for, not
        merely enough to reach the destination. GSX is asked first where
        the stand can ask; a model whose tanks are writable is fuelled
        directly where GSX cannot. A leg that cannot be fuelled for is
        not departed on: the chain ends at the stand rather than in the
        air short of fuel.
        """
        if not self._turnaround_service:
            return True
        if self.state.service_started_at == 0.0:
            self.state.service_started_at = now

        # Engines off first: nobody refuels a running aeroplane, and a
        # rotor that is still turning is worse than a jet that is.
        if getattr(state, "engines_running", False):
            self._shutdown_engines()
            if now - self.state.service_started_at < 60.0:
                return False
            self._emit("info",
                       "Turnaround: engines still running, servicing anyway",
                       "autonomous.service.engines")
        rotor = float(getattr(state, "rotor_rpm_pct", 0.0) or 0.0)
        if self._is_helicopter and rotor > 1.0:
            if now - self.state.service_started_at < 90.0:
                return False

        plan = self.state.plan_flight or {}
        dest = (self.navdb.airport(plan.get("destination", "") or "")
                or self.session.destination)
        if self.state.alternate is None:
            self.state.alternate = self._pick_alternate(state, dest)
        needed = self._fuel_needed_gal(state, destination=dest,
                                       alternate=self.state.alternate,
                                       estimate=True)
        fuel_gal = float(getattr(state, "fuel_total_gal", 0.0) or 0.0)
        self._fuel_capacity_seen = max(
            self._fuel_capacity_seen, fuel_gal,
            float(getattr(state, "fuel_capacity_gal", 0.0) or 0.0))
        if needed is not None:
            target = needed["total_gal"]
        elif self._fuel_capacity_seen > 0.0:
            # No burn figure to plan by: fill most of what it holds.
            target = self._fuel_capacity_seen * 0.9
        else:
            target = fuel_gal
        if self._fuel_capacity_seen > 0.0:
            target = min(target, self._fuel_capacity_seen)
        short = target - fuel_gal

        # What is already aboard may be enough -- tanks that were filled
        # for a longer leg cover a shorter one.
        if short <= 0.5:
            self._emit("info",
                       f"Turnaround: {fuel_gal:.0f} gal aboard covers the "
                       "leg, the diversion and the reserve",
                       "autonomous.service.enough")
            return True

        # GSX first, where the stand can ask for a truck.
        if not self.state.gsx_asked:
            self.state.gsx_asked = True
            asked = self._ask_gsx_refuel()
            if asked:
                self.state.gsx_busy = True
                self._emit("info", f"Turnaround: {asked}",
                           "autonomous.service.gsx")
                return False
            # No truck to ask: the tanks are written where the model's
            # fuel system accepts a write.
            if self._write_fuel(target):
                return True
            self._fuel_or_hold(state, needed, fuel_gal, target)
            return self.state.phase != AutonomousPhase.SHUTDOWN

        # A truck was asked: wait for GSX to say the fuel is on, then
        # check the figure the tanks now report.
        if self.state.gsx_busy:
            if self._gsx_refuel_done() \
                    or now - self.state.service_started_at > 300.0:
                self.state.gsx_busy = False
                fuel_gal = float(getattr(state, "fuel_total_gal", 0.0)
                                 or 0.0)
                if fuel_gal >= target - 0.5 or self._write_fuel(target):
                    return True
                self._fuel_or_hold(state, needed, fuel_gal, target)
                return self.state.phase != AutonomousPhase.SHUTDOWN
            return False
        return False

    def _ask_gsx_refuel(self) -> str:
        """Ask GSX for the fuel truck; the words it was asked in, or ""."""
        ground = getattr(self.brain, "ground", None)
        if ground is None or not getattr(ground, "can_ask", False):
            return ""
        try:
            return ground.request("refuel") or ""
        except Exception:
            log.debug("GSX refuel request failed", exc_info=True)
            return ""

    def _gsx_refuel_done(self) -> bool:
        """Whether GSX says the refuelling is finished."""
        ground = getattr(self.brain, "ground", None)
        gstate = getattr(ground, "state", None) if ground is not None else None
        if gstate is None or not getattr(gstate, "available", False):
            return False
        from ..integrations.gsx import COMPLETED, FINISHED
        return gstate.refuelling in (COMPLETED, FINISHED)

    def _write_fuel(self, gallons: float) -> bool:
        """The tanks written to the figure, where the model allows it."""
        if self.sim is None or not getattr(self.sim, "connected", False):
            return False
        try:
            if self.sim.set_fuel(gallons):
                self.state.fuel_wrote = True
                self._emit("info",
                           f"Turnaround: fuelled to {gallons:.0f} gal",
                           "autonomous.service.fuel")
                return True
        except Exception:
            log.debug("turnaround refuel failed", exc_info=True)
        return False

    def _fuel_or_hold(self, state, needed, fuel_gal: float,
                      target: float) -> None:
        """Nobody could bring fuel: go on what is aboard, or do not go.

        Enough for the leg and the reserve -- the diversion figure aside
        -- is departed with, said once; less than the leg itself is not
        departed on at all.
        """
        if needed is not None and fuel_gal >= needed["leg_gal"] \
                + needed["reserve_gal"]:
            self._emit("info",
                       f"Turnaround: no fuel to be had -- departing on "
                       f"{fuel_gal:.0f} gal, short of the diversion "
                       "figure", "autonomous.service.short")
            return
        if needed is None and fuel_gal > 0.0:
            self._emit("info",
                       f"Turnaround: no fuel to be had -- departing on "
                       f"{fuel_gal:.0f} gal", "autonomous.service.short")
            return
        self._emit("error",
                   f"Turnaround: cannot fuel for the leg ({fuel_gal:.0f} "
                   f"gal aboard, {target:.0f} needed) -- ending here",
                   "autonomous.service.nofuel")
        self._transition(AutonomousPhase.SHUTDOWN)

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
            turn = max(-0.5, min(0.5, err / 25.0))
            d.rudder(turn)
            d.steering(turn)
            gs = state.ground_speed_kt or 0.0
            if target_kt <= 0.0 or self._ground_still(
                    state, time.monotonic()):
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
        """No simulator taxiway route: steer at the runway, hold short."""
        if outbound:
            self._find_runway(state)
            if self.session.cleared_takeoff or self.session.phase in (
                    Phase.HOLDING_SHORT, Phase.LINED_UP, Phase.TAKEOFF):
                self._transition(AutonomousPhase.LINE_UP)
                return
            if self.state.runway_heading <= 0:
                return
            rwy_lat = self.state.runway_lat
            rwy_lon = self.state.runway_lon
            if rwy_lat and rwy_lon:
                # A heading is not a route: the runway's bearing from
                # where the aeroplane sits is what reaches it, and the
                # threshold itself is what it holds short of. ``hold_point``
                # does the stopping and the move to HOLDING_SHORT, the
                # same end a planned route arrives at.
                if state.distance_to(rwy_lat, rwy_lon) * 1852.0 <= 90.0:
                    self._goto_sub("hold_point")
                    return
                hdg = auto_nav.bearing_deg(state.latitude, state.longitude,
                                           rwy_lat, rwy_lon)
                self._steer(state, hdg, self.performance.taxi_kt)
            else:
                self._steer(state, self.state.runway_heading,
                            self.performance.taxi_kt)
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

    # -- filed/cleared SID/STAR guidance -----------------------------------

    def _route_idents(self) -> tuple:
        route = getattr(self.brain, "route", None)
        if route is None:
            return ()
        return tuple(getattr(fix, "ident", "") or ""
                     for fix in getattr(route, "fixes", ()) or ())

    def _airport_ident(self, airport, fallback: str = "") -> str:
        if airport is not None:
            ident = getattr(airport, "ident", "") or ""
            if ident:
                return ident
        return fallback

    def _runway_start(self, ident: str, runway: str,
                      procedures) -> tuple:
        """Where the SID's first leg begins: the runway end where the
        navdata knows it, the airport reference where it does not."""
        found = auto_nav.runway_end(self.navdb.runways(ident), runway)
        if found is not None:
            return found[0], found[1]
        return (getattr(procedures, "lat", 0.0) or 0.0,
                getattr(procedures, "lon", 0.0) or 0.0)

    def _sid_nav(self, state):
        """The filed/cleared departure procedure's guidance, if any.

        Only a SID the route or the clearance actually names is flown --
        an arbitrary procedure is never invented. The transition is the
        one matching the assigned or planned runway, else the only one
        the procedure has.
        """
        if self.state.procedure_suspended == "sid":
            return None
        if getattr(self.session, "assigned_heading", None):
            # A vector is off the procedure until ATC says resume own
            # navigation; the heading bug is the clearance, not the leg.
            return None
        route = getattr(self.brain, "route", None)
        ident = self._airport_ident(
            getattr(self.session, "departure", None),
            getattr(route, "origin", "") or "")
        if not ident:
            return None
        procs = getattr(self.brain, "terminal_procedures", {}).get(ident)
        if not procs:
            return None
        name = (getattr(self.session, "sid", "")
                or (getattr(route, "sid", "") if route else "") or "")
        if not name:
            return None
        runway = (getattr(self.session, "departure_runway", "")
                  or self.state.departure_runway
                  or (getattr(route, "sid_runway", "") if route else "")
                  or "")
        transition = (getattr(route, "sid_transition", "")
                      if route else "") or ""
        proc, trans = procedure_legs.choose_sid(
            procs, name, runway, transition, self._route_idents())
        if trans is None:
            return None
        key = (ident, proc.name, trans.name, trans.runway, runway)
        if self.state.sid_key != key or self.state.sid_nav is None:
            start = self._runway_start(ident, runway or trans.runway, procs)
            self.state.sid_nav = procedure_legs.ProcedureNavigator(
                proc, trans, start,
                fallback_altitude_ft=self._altitude_target(state),
                fallback_speed_kt=self._speed_target(state))
            self.state.sid_key = key
            self._emit("info", f"Flying SID {proc.name} "
                       f"{trans.name or trans.runway}".strip(),
                       "autonomous.sid")
        nav = self.state.sid_nav
        nav.fallback_altitude_ft = self._altitude_target(state)
        nav.fallback_speed_kt = self._speed_target(state)
        guidance = nav.guidance(
            state, climb_fpm=self.performance.climb_fpm,
            descend_fpm=self.performance.descend_fpm)
        if guidance.done:
            self.state.sid_nav = None
            self.state.sid_key = ()
            return None
        return guidance

    def _star_nav(self, state):
        """The filed/cleared arrival procedure's guidance, if any."""
        if self.state.procedure_suspended == "star":
            return None
        if getattr(self.session, "assigned_heading", None):
            return None
        route = getattr(self.brain, "route", None)
        ident = self._airport_ident(
            getattr(self.session, "destination", None),
            getattr(route, "destination", "") or "")
        if not ident:
            return None
        procs = getattr(self.brain, "terminal_procedures", {}).get(ident)
        if not procs:
            return None
        name = (getattr(self.session, "descend_via", "")
                or (getattr(route, "star", "") if route else "") or "")
        if not name:
            return None
        runway = (getattr(self.session, "arrival_runway", "")
                  or self.state.arrival_runway
                  or (getattr(route, "star_runway", "") if route else "")
                  or "")
        transition = (getattr(route, "star_transition", "")
                      if route else "") or ""
        proc, trans = procedure_legs.choose_star(
            procs, name, runway, transition, self._route_idents())
        if trans is None:
            return None
        key = (ident, proc.name, trans.name, trans.runway, runway)
        if self.state.star_key != key or self.state.star_nav is None:
            self.state.star_nav = procedure_legs.ProcedureNavigator(
                proc, trans,
                (state.latitude, state.longitude),
                fallback_altitude_ft=(
                    getattr(self.session, "assigned_altitude_ft", 0.0)
                    or self._descent_floor(state)),
                fallback_speed_kt=(
                    getattr(self.session, "assigned_speed_kt", 0.0)
                    or (self.state.vapp_kt
                        or self.performance.approach_kt) + 20.0))
            self.state.star_key = key
            self._emit("info", f"Descending via STAR {proc.name} "
                       f"{trans.name or trans.runway}".strip(),
                       "autonomous.star")
        nav = self.state.star_nav
        nav.fallback_altitude_ft = (
            getattr(self.session, "assigned_altitude_ft", 0.0)
            or self._descent_floor(state))
        nav.fallback_speed_kt = (
            getattr(self.session, "assigned_speed_kt", 0.0)
            or (self.state.vapp_kt
                or self.performance.approach_kt) + 20.0)
        guidance = nav.guidance(
            state, climb_fpm=self.performance.climb_fpm,
            descend_fpm=self.performance.descend_fpm)
        if guidance.done:
            self.state.star_nav = None
            self.state.star_key = ()
            return None
        return guidance

    def _fly_procedure(self, state, guidance) -> None:
        """Put a procedure leg's targets on the autopilot.

        An assigned heading, level or speed still outranks the
        procedure's own targets -- ATC's word is the clearance.
        """
        assigned_hdg = self.session.assigned_heading or 0.0
        heading = assigned_hdg or (
            self._magnetic(guidance.heading, state)
            if guidance.heading is not None else -1.0)
        assigned_alt = self.session.assigned_altitude_ft or 0.0
        altitude = assigned_alt or guidance.altitude_ft \
            or self._altitude_target(state)
        if assigned_alt:
            current = state.altitude_ft or 0.0
            if current < assigned_alt - 250.0:
                vs = self.performance.climb_fpm
            elif current > assigned_alt + 250.0:
                vs = -self.performance.descend_fpm
            else:
                vs = 0.0
        else:
            vs = guidance.vs_fpm
        speed = (self.session.assigned_speed_kt
                 or guidance.speed_kt or self._speed_target(state))
        self._apply_flight_director(
            state, altitude=altitude, vs_fpm=vs,
            heading=heading, speed_kt=speed)

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
        if family == "loc" and has_loc and self.session.ifr \
                and not getattr(recipe, "back_course", False):
            # The localizer for the lateral, the path for the vertical:
            # there is no glideslope on this approach to capture. A back
            # course is excluded: its needles sense reversed, and a
            # front-course hold on them flies away from the runway --
            # it takes the computed final like a VOR.
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
            speed_kt=self._vapp(state))

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
            speed_kt=self._vapp(state))

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
            speed_kt=(self._vapp(state))
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
            # Only a model that has said the gear is down already is not
            # asked: up or unanswered both get the command. The inverted
            # check used to ask only the aeroplane already configured.
            if state.gear_down is not True:
                d.gear(True)
        if distance is not None and distance < 6.0 and agl < 2500:
            self._set_flaps(state, 0.75)
            self.state.configured = "flaps_landing"
        if distance is not None and distance < 3.0:
            self._set_flaps(state, 1.0)
            self.state.configured = "flaps_full"
        d.spoilers_arm(True)
        # Approach speed: Vapp where known, the type's figure otherwise.
        vapp = self._vapp(state)
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
        target = self._vapp(state)
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
        # The weather's gates, below a thousand feet: the wind past the
        # limits on this runway, or the field's own report below what the
        # approach can be flown to. Counted once an approach -- the flag
        # lifts when the next one is set up.
        if self._weather_reactions and 0.0 < agl < 1000.0 \
                and not self.state.weather_ga_active:
            if self._landing_wind_bad(state) \
                    or self._weather_below_minima(state):
                self.state.weather_gas += 1
                self.state.weather_ga_active = True
                return True
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
            target = self._vapp(state)
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
        # A fresh TO/GA press: the same one-shot events the takeoff roll
        # fired have to be allowed to fire again.
        d.forget("takeoff_power", "toga")
        d.go_around()
        d.takeoff_power()
        d.throttle(1.0)
        self._last_lever = 1.0
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
        # The tower hears it: a missed approach is called, not just flown.
        if self._manage_radio:
            self._fo_say("going_around")
        self._emit("info", "Going around", "autonomous.go_around")

    def _fly_go_around(self, state, now: float) -> None:
        """Climb away on the missed approach, then set up again."""
        d = self.director
        elapsed = now - self.state.go_around_at
        # Initial: pitch and power; after positive rate, clean up.
        if elapsed < 8.0:
            d.elevator(0.25)
            d.throttle(1.0)
            self._last_lever = 1.0
        else:
            d.elevator(0.05)
            d.throttle(0.7)
            self._last_lever = 0.7
            if state.flaps_index:
                d.flaps(0.0)
            if state.gear_down:
                d.gear(False)
        # Wings-level climb on the runway heading or the assigned heading.
        # The heading bug is magnetic: the runway's is degrees true off
        # the threshold's coordinates, so it is converted like every
        # other course this hands the autopilot.
        hdg = self.session.assigned_heading
        if not hdg and self.state.runway_heading > 0:
            hdg = self._magnetic(self.state.runway_heading, state)
        if hdg > 0:
            self._apply_flight_director(state, altitude=self.state.missed_alt_ft,
                                        vs_fpm=self.performance.climb_fpm,
                                        heading=hdg,
                                        speed_kt=self.state.climb_kt or 180.0)
        # Once level-ish at the missed altitude, reset for another pass --
        # unless the weather sent this one and the one before it, in
        # which case the destination is out and the alternate gets it.
        alt = state.altitude_ft or 0.0
        if alt >= self.state.missed_alt_ft - 150.0 \
                and abs(state.vertical_speed_fpm or 0.0) < 300:
            self.state.stabilized = False
            self.state.configured = ""
            if self.state.weather_gas >= 2 and self._weather_reactions:
                self._divert_for_weather(state)
                return
            self.state.weather_ga_active = False
            self._goto_sub("intercept")
            self._emit("info", "Missed approach complete; setting up again",
                       "autonomous.go_around_done")
        elif elapsed > 300.0:
            self._set_error("Go-around did not climb")

    # -- emergencies ------------------------------------------------------------

    def _monitor_emergency(self, state, now: float) -> None:
        """Watch for the failures the state can actually see.

        Fuel, the bus and per-engine combustion are readable; what cannot
        be seen is not invented. All engines out is a MAYDAY; one out of
        several is a PAN PAN and a diversion, because a twin on one is an
        urgency, not yet a distress. Fuel is judged the way a crew plans
        it: under the leg alone is the MAYDAY; under leg + diversion +
        reserve is minimum fuel, a PAN PAN.
        """
        if self.state.emergency_declared or self.state.phase in (
                AutonomousPhase.PREFLIGHT, AutonomousPhase.PUSHBACK,
                AutonomousPhase.TAXI_OUT, AutonomousPhase.SHUTDOWN,
                AutonomousPhase.COMPLETE, AutonomousPhase.ERROR):
            return

        # Electrical: the main bus collapsing in the air.
        volts = getattr(state, "main_bus_volts", -1.0)
        if volts is not None and 0.0 < volts < 12.0 and not state.on_ground:
            self._declare_emergency("electrical", state, mayday=True)
            return

        # Engines: combustion per engine where the model reports it.
        if not state.on_ground:
            engines_out = self._engines_out(state)
            count = int(getattr(state, "engine_count", 0) or 0) or 1
            if engines_out >= count:
                self._declare_emergency("engine", state, mayday=True)
                return
            if engines_out > 0:
                self._declare_emergency("engine-out", state, mayday=False)
                return

        # Fuel: nothing left, or so little the destination cannot be
        # reached. A sim that never reports fuel reads zero forever, so
        # the check only fires once a nonzero figure has been seen.
        fuel_lb = getattr(state, "fuel_total_lb", 0.0) or 0.0
        fuel_gal = getattr(state, "fuel_total_gal", 0.0) or 0.0
        capacity = float(getattr(state, "fuel_capacity_gal", 0.0) or 0.0)
        self._fuel_capacity_seen = max(self._fuel_capacity_seen, fuel_gal,
                                       capacity)
        if fuel_lb > 0.0 or fuel_gal > 0.0:
            self._fuel_known = True
        if not self._fuel_known:
            return
        if fuel_lb <= 0.0 and fuel_gal <= 0.0:
            self._declare_emergency("fuel", state, mayday=True)
            return
        needed = self._fuel_needed_gal(state)
        if needed is None or fuel_gal <= 0.0:
            if 0.0 < fuel_gal < 8.0:
                self._declare_emergency("fuel", state, mayday=True)
            return
        if fuel_gal < needed["leg_gal"]:
            self._declare_emergency("fuel", state, mayday=True)
        elif fuel_gal < needed["total_gal"] \
                and not self.state.minimum_fuel_said:
            self.state.minimum_fuel_said = True
            self._declare_emergency("minimum fuel", state, mayday=False)

    def _engines_out(self, state) -> int:
        """How many engines are not combusting, of those the sim reports.

        ``engines_running`` is engine one's combustion; a model that
        reports no combustion variables answers nothing, and nothing is
        invented.
        """
        out = 0
        for flag in (getattr(state, "engines_running", None),
                     getattr(state, "eng2_combustion", None),
                     getattr(state, "eng3_combustion", None),
                     getattr(state, "eng4_combustion", None)):
            if flag is not None and not flag:
                out += 1
        return out

    # -- fuel planning -------------------------------------------------------

    def _is_turbine(self, state) -> bool:
        """Whether the aeroplane burns Jet-A: a jet, a turboprop or a
        turbine helicopter does; a piston -- including a piston
        helicopter -- does not.

        The simulator's ``ENGINE TYPE`` is decisive where it has
        answered; the type's Doc 8643 description answers where it has
        not; the class the session began with is the fallback, and the
        commonest class (a piston single) is what it falls back to.
        """
        from . import aircraft_class as ac

        # ``ENGINE TYPE`` is -1 until the sim answers; 0 is a real answer
        # -- a piston -- and must not be folded into "unknown".
        raw = getattr(state, "engine_type", -1) if state is not None else -1
        kind = -1 if raw is None else int(raw)
        if kind in (ac.SIM_JET, ac.SIM_HELO_TURBINE, ac.SIM_TURBOPROP):
            return True
        if kind == ac.SIM_PISTON:
            return False
        # The designator's power letter: 'J' and 'T' are Jet-A, 'P' is
        # avgas -- which is also what an H1P helicopter burns.
        code = (getattr(getattr(self.session, "aircraft", None),
                        "type_code", "") or "").strip().upper()
        description = ac.DOC_8643.get(code)
        if description:
            return description[-1] in ("T", "J")
        if self._is_helicopter:
            # The helicopter rows the table does not know are the turbine
            # ones almost everywhere; the pistons are listed above.
            return True
        name = getattr(self, "_aircraft_class", "")
        if not name:
            name = ac.classify(code).name
        return name not in (ac.PISTON_SINGLE, ac.PISTON_TWIN)

    def _lb_per_gal(self, state=None) -> float:
        """The fuel's weight per gallon: what the tanks themselves say,
        else Jet-A on a turbine and avgas else."""
        gal = float(getattr(state, "fuel_total_gal", 0.0) or 0.0)
        lb = float(getattr(state, "fuel_total_lb", 0.0) or 0.0)
        if gal > 0.5 and lb > 0.0:
            return lb / gal
        return 6.7 if self._is_turbine(state) else 6.0

    def _fuel_flow_pph(self, state, estimate: bool = False) -> float:
        """Pounds an hour the type burns in cruise, or 0 where unknown.

        The model's own ESTIMATED FUEL FLOW is the only honest answer for
        an in-flight emergency -- a guessed flow is how an aeroplane with
        plenty aboard gets a minimum-fuel call. For the turnaround's fuel
        order, where erring costs a fuller tank rather than a distress,
        the engine type's rough figure stands in.
        """
        flow = float(getattr(state, "est_fuel_flow_pph", 0.0) or 0.0)
        if flow > 0.0 or not estimate:
            return flow
        # 0 is a real answer -- a piston, eighty pounds an hour -- and
        # ``or -1`` used to fold it into "unknown", which is the airliner
        # figure: a Cessna held short for more fuel than its tanks hold.
        raw = getattr(state, "engine_type", -1)
        kind = -1 if raw is None else int(raw)
        if self._is_helicopter or kind == 3:
            return 300.0
        return {0: 80.0, 1: 4500.0, 5: 900.0}.get(kind, 1200.0)

    def _reserve_hours(self) -> float:
        """The reserve the legs are planned to, in hours."""
        return max(0.25, self._reserve_min / 60.0)

    def _pick_alternate(self, state, destination=None) -> Optional[Airport]:
        """The nearest field that can take the aeroplane, for the reserve.

        What a crew files: the closest suitable airport to the
        destination that is not the destination itself. None where there
        is no other field in reach -- an aeroplane with no alternate gets
        no alternate fuel figure, and is not penalised for a field that
        does not exist.
        """
        dest = destination or self.session.destination
        if dest is None:
            return None
        for airport, _dist in self.navdb.nearest_airports(
                dest.lat, dest.lon, limit=8, max_nm=120.0,
                require_runway=not self._is_helicopter):
            if airport.ident == dest.ident:
                continue
            if self._airport_suitable(airport):
                return airport
        return None

    def _fuel_needed_gal(self, state, destination=None,
                         alternate=None, estimate: bool = False) -> dict | None:
        """Gallons the rest of the leg needs, broken down for the caller.

        ``leg_gal`` reaches the destination; ``total_gal`` is leg +
        diversion to the alternate + the reserve, the figure a crew may
        not go below without saying so. None where the burn or the field
        is not known -- then no emergency is ever projected on a guess.
        """
        flow = self._fuel_flow_pph(state, estimate=estimate)
        dest = destination or self.session.destination
        if flow <= 0.0 or dest is None \
                or not getattr(state, "has_position", False):
            return None
        speed = max(60.0, float(getattr(state, "ground_speed_kt", 0.0)
                                or self.performance.cruise_kt or 120.0))
        leg_nm = state.distance_to(dest.lat, dest.lon)
        leg_hr = leg_nm / speed
        alt = alternate if alternate is not None \
            else self._pick_alternate(state, dest)
        divert_hr = 0.0
        if alt is not None:
            divert_hr = max(15.0, haversine_nm(
                dest.lat, dest.lon, alt.lat, alt.lon)) / speed
        per_gal = self._lb_per_gal(state)
        leg_gal = leg_hr * flow / per_gal
        divert_gal = divert_hr * flow / per_gal
        reserve_gal = self._reserve_hours() * flow / per_gal
        unusable = float(getattr(state, "fuel_unusable_gal", 0.0) or 0.0)
        return {"leg_gal": leg_gal + unusable,
                "divert_gal": divert_gal,
                "reserve_gal": reserve_gal,
                "total_gal": leg_gal + divert_gal + reserve_gal + unusable,
                "alternate": alt}

    def _declare_emergency(self, kind: str, state, mayday: bool = True) -> None:
        """The call, the squawk, a divert, and the plan changed to match.

        A distress is MAYDAY and squawked 7700; an urgency is PAN PAN --
        one engine of several, minimum fuel, an electrical problem that
        still leaves a radio. The session hears either way, because the
        transmission goes through the ordinary parser like a pilot's.
        """
        self.state.emergency = kind
        self.state.emergency_declared = True
        # An emergency ends the chain: nobody flies the next leg on one.
        self._chained = False
        if mayday:
            try:
                self.sim.set_squawk("7700")
            except Exception:
                pass
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
        prefix = "Mayday mayday mayday" if mayday else "Pan pan pan pan"
        text = (f"{prefix}, {callsign} "
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
            # An emergency descent: get down to a survivable height now,
            # the approach machinery handles the rest. On final the
            # approach is waved off for the new field, with the cached
            # runway geometry dropped so it resolves again.
            self._route_diversion()

    # -- weather ---------------------------------------------------------------

    def _monitor_weather(self, state, now: float) -> None:
        """What the weather is doing, answered the way a crew answers it.

        Nothing is invented: the icing call is made on the cloud and the
        temperature the sim is rendering, the rough air on the wind the
        airframe is actually in, the wind limits on the same METAR the
        controllers quote. The departure and approach checks live at the
        decisions they gate; what is watched here is what the airframe is
        in, which is nobody's phase to own.
        """
        if not self._weather_reactions or self.state.emergency_declared:
            return
        self._watch_icing(state, now)
        self._watch_turbulence(state, now)

    def _watch_icing(self, state, now: float) -> None:
        """Anti-ice and pitot heat on in visible moisture near freezing.

        TAT is the figure a crew watches -- it is warmer than ambient at
        speed, which is the same reason an aeroplane ices in cloud that
        is not strictly below freezing. A model that publishes no switch
        positions is asked anyway: a write that does not exist does no
        harm next to one that does.
        """
        if state.on_ground:
            return
        tat = float(getattr(state, "total_air_temperature_c", 15.0)
                    or getattr(state, "ambient_temperature_c", 15.0) or 15.0)
        ambient = float(getattr(state, "ambient_temperature_c", 15.0) or 15.0)
        moisture = (bool(getattr(state, "ambient_in_cloud", False))
                    or int(getattr(state, "ambient_precip_state", 0) or 0) > 0)
        if moisture and -30.0 < min(tat, ambient) < 6.0 \
                and not self.state.icing_said:
            self.state.icing_said = True
            d = self.director
            d.pitot_heat(True)
            for index in range(
                    1, int(getattr(state, "engine_count", 0) or 1) + 1):
                d.write(f"GENERAL ENG ANTI ICE POSITION:{index}", 1.0)
            d.write("PROP DEICE SWITCH:1", 1.0)
            d.write("STRUCTURAL DEICE SWITCH", 1.0)
            d.write("WINDSHIELD DEICE SWITCH:1", 1.0)
            self._emit("info", "Icing conditions: anti-ice and pitot "
                       "heat on", "autonomous.weather.icing")
        elif not moisture:
            self.state.icing_said = False
        # Ice that is accreting anyway is a route problem, not a switch
        # problem: it is said once so the pilot hears it.
        ice = float(getattr(state, "structural_ice_pct", 0.0) or 0.0)
        if ice >= 40.0 and not self.state.ice_warned:
            self.state.ice_warned = True
            self._emit("info",
                       f"Ice accreting on the airframe ({ice:.0f}%): "
                       "consider leaving this altitude",
                       "autonomous.weather.ice")

    def _watch_turbulence(self, state, now: float) -> None:
        """Rough air, as the airframe reports it.

        The vertical wind is the only convective evidence the sim
        publishes, and it is honest evidence: a ten-knot up or downdraft
        is a thousand feet a minute. In it the speed is pulled back to
        the penetration figure; on final a swing of it is a windshear,
        and a windshear is a go-around, not a recovery attempt.
        """
        if state.on_ground:
            return
        updraft = abs(getattr(state, "aircraft_wind_y_kt", 0.0) or 0.0)
        if updraft >= 12.0:
            self.state.turbulence_until = now + 30.0
            if not self.state.turbulence_said:
                self.state.turbulence_said = True
                self._emit("info",
                           "Rough air: pulled back toward the turbulence "
                           "penetration speed",
                           "autonomous.weather.turbulence")
        elif now > self.state.turbulence_until:
            self.state.turbulence_said = False

        # Windshear is the fixed-wing approach's problem: it is the only
        # phase with the machinery to go around.
        if self._heli is not None \
                or self.state.phase != AutonomousPhase.APPROACH:
            return
        agl = float(getattr(state, "altitude_agl_ft", 0.0) or 0.0)
        if not 0.0 < agl < 1500.0:
            self._wind_z_hist.clear()
            return
        # The headwind down final, kept for a few polls: fifteen knots
        # appearing or vanishing across it is the textbook shear figure.
        longitudinal = float(getattr(state, "aircraft_wind_z_kt", 0.0) or 0.0)
        hist = self._wind_z_hist
        hist.append(longitudinal)
        del hist[:len(hist) - 8]
        swing = (max(hist) - min(hist)) if len(hist) >= 4 else 0.0
        if (swing >= 15.0 or (updraft >= 10.0 and agl < 500.0)) \
                and now - self.state.windshear_since > 20.0:
            if self.state.sub in ("intercept", "final"):
                self.state.windshear_since = now
                self.state.weather_gas += 1
                self._emit("info", "Windshear: going around",
                           "autonomous.weather.windshear")
                self._start_go_around(state)

    def _field_weather(self, airport):
        """The weather the controllers are quoting for a field, or None.

        It is the engine's merged figure -- the sim's where the field is
        close enough to speak for itself, the observation's elsewhere --
        so the FO's decision matches the wind the tower just read out.
        """
        if airport is None:
            return None
        weather_for = getattr(self.brain, "weather_for", None)
        if weather_for is None:
            return None
        for ident in (getattr(airport, "icao", "") or "",
                      getattr(airport, "ident", "") or ""):
            if not ident:
                continue
            try:
                wx = weather_for(ident)
            except Exception:
                continue
            if wx is not None:
                return wx
        return None

    @staticmethod
    def _convective(weather) -> bool:
        """Whether the report's sky is convective: a thunderstorm, hail,
        squall or funnel in the present weather, or a CB in the clouds.
        The raw body is scanned as well as the parsed group -- ``VCTS``
        is a token the parser's present-weather list does not keep, and
        a storm in the vicinity still counts."""
        if weather is None:
            return False
        if getattr(weather, "cb", False):
            return True
        text = ((getattr(weather, "raw", "") or "")
                .split(" RMK", 1)[0].upper())
        present = getattr(weather, "present", "") or ""
        if present:
            text += " " + present.upper()
        return bool(re.search(r"(?:^|\s)[-+]?(?:VCTS|TS\w*|GR|SQ|FC)",
                              text))

    def _wind_limits(self) -> tuple[float, float]:
        """(crosswind, tailwind) in knots this class is demonstrated to.

        Nobody publishes the model's own figures, so they are the class's
        -- a transport's thirty off the side and fifteen up the tail, a
        turboprop's twenty-five and ten, a light aircraft's fifteen and
        ten. A helicopter has no runway wind that matters this way.
        """
        if self._is_helicopter:
            return 99.0, 99.0
        cruise = float(getattr(self.performance, "cruise_kt", 0.0) or 0.0)
        if cruise >= 230.0:
            return 30.0, 15.0
        if cruise >= 140.0:
            return 25.0, 10.0
        return 15.0, 10.0

    @staticmethod
    def _runway_wind(weather, heading: float) -> tuple[float, float, float]:
        """(headwind, crosswind, tailwind) in knots on a runway heading.

        The gust is counted, not reported separately: it is the number
        the airframe sees at the worst moment of the flare. A variable
        wind is nobody's crosswind -- it is worked as calm, the same way
        the runway chooser works it.
        """
        if weather is None or getattr(weather, "variable", False):
            return 0.0, 0.0, 0.0
        wind = max(float(getattr(weather, "wind_kt", 0.0) or 0.0),
                   float(getattr(weather, "gust_kt", 0.0) or 0.0))
        if wind < 0.5 or heading <= 0.0:
            return 0.0, 0.0, 0.0
        off = math.radians((float(getattr(weather, "wind_dir", 0.0) or 0.0)
                            - float(heading)) % 360.0)
        head = wind * math.cos(off)
        return max(0.0, head), abs(wind * math.sin(off)), max(0.0, -head)

    def _vapp(self, state) -> float:
        """The approach speed as a crew would fly it: the figure, plus
        half the gust, capped -- the additive a gusty day asks for, so
        the speed that is flown covers the speed that gusts."""
        base = self.state.vapp_kt or self.performance.approach_kt
        if not base or not self._weather_reactions:
            return base
        wx = self._field_weather(self.session.destination)
        gust = float(getattr(wx, "gust_kt", 0.0) or 0.0)
        steady = float(getattr(wx, "wind_kt", 0.0) or 0.0)
        if gust <= steady:
            return base
        return base + max(0.0, min((gust - steady) * 0.5, 15.0))

    def _route_diversion(self) -> None:
        """The destination changed under the flight: the plan filed again
        direct, the old field's procedures dropped, and the approach
        machinery reset so the new field is worked out fresh rather
        than on the old one's numbers."""
        self.session.sid = ""
        self.session.descend_via = ""
        self.session.star_join = ""
        self.state.sid_nav = None
        self.state.star_nav = None
        self.state.sid_key = ()
        self.state.star_key = ()
        self.state.procedure_suspended = ""
        if self.use_route is not None:
            try:
                from .route import from_string, resolver
                dest = self.session.destination
                route = from_string(
                    f"DCT {dest.ident}", dest.ident, dest.ident,
                    resolver(self.navdb))
                self.use_route(route)
            except Exception:
                pass
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
            self.state.weather_gas = 0
            self.state.weather_ga_active = False
            self._transition(AutonomousPhase.DESCENT)
            self._goto_sub("")

    def _departure_wind_ok(self, state) -> bool:
        """Whether the wind and the cells let the aeroplane go.

        A crosswind or a tailwind the class is not demonstrated for is
        held for, as a crew holds short for a gusty wind rather than
        rolling in it; a thunderstorm over the field is waited out. Both
        are held for twenty minutes, then called -- a wind that is not
        coming back is a plan that is not flying.
        """
        weather = self._field_weather(
            self.session.departure or self._get_current_airport())
        if weather is None:
            return True
        now = time.monotonic()
        if self._convective(weather):
            if not self.state.convective_hold_said:
                self.state.convective_hold_said = True
                self.state.wind_hold_since = now
                self._emit("info",
                           "Holding departure: thunderstorm at the field",
                           "autonomous.weather.hold")
            elif now - self.state.wind_hold_since > 1200.0:
                self._set_error("Thunderstorm over the field for twenty "
                                "minutes; not departing into it")
            return False
        _head, cross, tail = self._runway_wind(
            weather, self.state.runway_heading)
        xlim, tlim = self._wind_limits()
        if cross > xlim or tail > tlim:
            if not self.state.wind_hold_said:
                self.state.wind_hold_said = True
                self.state.wind_hold_since = now
                self._emit(
                    "info",
                    f"Holding departure: {cross:.0f} kt crosswind, "
                    f"{tail:.0f} kt tailwind (limits {xlim:.0f}/"
                    f"{tlim:.0f})", "autonomous.weather.hold")
            elif now - self.state.wind_hold_since > 1200.0:
                self._set_error("Departure wind beyond limits and not "
                                "improving")
            return False
        self.state.wind_hold_said = False
        return True

    def _landing_wind_bad(self, state) -> bool:
        """Whether the wind on the landing runway is past the limits."""
        if not self.state.runway_heading:
            return False
        weather = self._field_weather(self.session.destination)
        _head, cross, tail = self._runway_wind(
            weather, self.state.runway_heading)
        xlim, tlim = self._wind_limits()
        return cross > xlim or tail > tlim

    def _weather_below_minima(self, state) -> bool:
        """Whether the field's own report puts the approach out: the
        ceiling under the minima, or the visibility under what the
        approach can be flown to. A field that does not report is not a
        field below minima -- only a report can say that."""
        weather = self._field_weather(self.session.destination)
        if weather is None or not getattr(weather, "raw", ""):
            return False
        ceiling = min((h for cover, h in getattr(weather, "clouds", [])
                       if cover in ("BKN", "OVC", "VV")), default=None)
        minima = float(self.state.minima_ft or 200.0)
        if ceiling is not None and ceiling + 50.0 < minima:
            return True
        recipe = self.state.approach_recipe
        floor_sm = 1.5 if getattr(recipe, "circling", False) else (
            0.75 if getattr(recipe, "family", "") in ("ils", "rnav")
            else 1.0)
        return float(getattr(weather, "visibility_sm", 10.0)
                     or 10.0) < floor_sm

    def _divert_for_weather(self, state) -> None:
        """Two attempts at a destination the weather keeps shutting: the
        alternate gets the flight instead, said like a pilot would."""
        dest = self._pick_alternate(state)
        if dest is None:
            self.state.weather_gas = 0
            self._emit("error",
                       "Destination weather below limits and no alternate "
                       "in reach; trying the approach again",
                       "autonomous.weather.no_alternate")
            return
        self._emit("info",
                   f"Weather below limits at the destination: diverting "
                   f"to {dest.ident}", "autonomous.weather.divert")
        if self.crew.transmit:
            try:
                self.crew.transmit(
                    f"diverting to {dest.ident}, weather below minima")
            except Exception:
                pass
        self.session.destination = dest
        self.state.weather_gas = 0
        self._route_diversion()

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
        self.session.sid = ""
        self.session.descend_via = ""
        self.session.star_join = ""
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
        self.state.sid_nav = None
        self.state.star_nav = None
        self.state.sid_key = ()
        self.state.star_key = ()
        self.state.procedure_suspended = ""
        self.state.comms_tuned = False
        self.state.checked_in_with = ""
        self.state.squawk_dialled = ""
        self.state.altimeter_dialled = 0.0
        self.state.last_radio_at = 0.0
        self.state.service_started_at = 0.0
        self.state.gsx_asked = False
        self.state.gsx_busy = False
        self.state.fuel_wrote = False
        self.state.service_warned = False
        self.state.alternate = None
        self.state.mtow_warned = False
        self.state.flap_gate_warned = False
        self.state.minimum_fuel_said = False
        self.state.depart_fuel_warned = False
        self._ga_flag_seen = False
        self._last_lever = 0.0
        self._ground_still_since = 0.0
        self._checklist_item_id = 0
        self._checklist_item_since = 0.0
        self._dialled_at = 0.0
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

    def _pushback_skippable(self) -> bool:
        """Whether this aeroplane can leave the stand under its own power
        when no tug ever answers: a light single or twin on a ramp taxis
        straight out, where an airliner nose-in at a gate cannot. The
        class comes from the aircraft table; anything heavier stays an
        error so a real failure is never silently taxied through a
        building."""
        from . import aircraft_class as ac
        return getattr(self, "_aircraft_class", "airliner") in (
            ac.PISTON_SINGLE, ac.PISTON_TWIN, ac.TURBOPROP_SINGLE,
            ac.HELICOPTER)

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
            # Speed hold only drives the levers where an autothrottle is
            # armed -- aircraft.cfg's autothrottle_arming_required makes
            # the arm a separate ask, sent once and only while the sim
            # says the AT is not already live.
            if not getattr(state, "autothrottle", False):
                d.autothrottle_arm()
            if target:
                d.altitude(target)
            d.level_change(True)
            self._manage_throttle(state, speed)
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
) -> AutonomousFlight:
    """The First Officer's machine, always built.

    It used to exist only when the setting was on at startup, which made
    the panel's engage button dead until a restart. The object is cheap
    and inert until engaged -- ``_enabled`` (the setting, default off)
    is what gates the tick, not the object's existence.
    """
    return AutonomousFlight(
        config, navdb, sim_source, session, crew, brain, get_state,
        on_event, use_route
    )


__all__ = ["AutonomousFlight", "AutonomousPhase", "AutonomousState", "create_autonomous_flight"]
