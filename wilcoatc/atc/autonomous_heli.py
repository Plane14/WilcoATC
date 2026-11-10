# -*- coding: utf-8 -*-
"""Autonomous flight for rotorcraft: the helicopter's own controller.

:class:`~wilcoatc.atc.autonomous.AutonomousFlight` delegates to this when the
aeroplane in the sim is a helicopter (``aircraft_class.HELICOPTER``). It is
a separate controller rather than more branches in the fixed-wing one because
a rotorcraft is flown differently in every phase: there is no takeoff roll,
no rotation, no flaps or gear, and the "autopilot" is the pilot's own hands
on the cyclic, the collective and the pedals.

**VFR only.** There are no helicopter instrument approach procedures in the
navdata this program carries, and the fixed-wing IFR phases (runway lineup,
ILS capture) do not apply. An IFR session is downgraded to VFR with an event,
never flown through the airplane logic.

**Best effort.** MSFS has no universal helicopter start or shutdown event --
every add-on wires its own -- so the engine sequence is: the generic
``ENGINE_AUTO_START``/``ENGINE_AUTO_SHUTDOWN`` events, the rotor governor and
clutch switches where the model answers them, and then watching
``engines_running`` and ``rotor_rpm_pct`` with timeouts rather than hoping.
The same goes for the flight controls: ``AXIS_COLLECTIVE_SET``,
``AXIS_CYCLIC_*_SET``, ``AXIS_STEERING_SET`` and the helicopter throttle
events are sent and their effect is judged from the state, not assumed.
``AUTO_HOVER_*`` is documented by the SDK as not implemented and is never
sent.

Every gain below is UNVERIFIED: no running 2024 has been flown with this.
The controllers are deliberately soft -- small proportional terms inside
tight clamps -- so a model that answers the axes differently from the
assumption degrades into a bad flight the pilot can see, not an unrecoverable
one. ``logs`` and the ``autonomous.*`` events say what was tried.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .aircraft_class import HELICOPTER
from .autonomous import AutonomousPhase
from .vspeeds import speeds_for

log = logging.getLogger(__name__)

# --- the axes -------------------------------------------------------------
# MSFS 2024 helicopter key events (SDK "Helicopter Specific Events"). The
# collective and the tail rotor take 0..16384; the cyclics and the steering
# axis take -16384..16384. ``_send`` masks to a DWORD, which is the
# two's-complement form SimConnect expects for the signed axes.
AXIS_MAX = 16384
COLLECTIVE_SET = b"AXIS_COLLECTIVE_SET"
CYCLIC_LONG_SET = b"AXIS_CYCLIC_LONGITUDINAL_SET"
CYCLIC_LAT_SET = b"AXIS_CYCLIC_LATERAL_SET"
STEERING_SET = b"AXIS_STEERING_SET"
HELI_THROTTLE_SET = b"AXIS_HELICOPTER_THROTTLE_SET"
# Named throttle events as fallbacks, per the same page: "set all
# throttles", and the per-engine forms a twin might answer.
THROTTLE_ALL_SET = b"HELICOPTER_THROTTLE_SET"
THROTTLE_ALL_FULL = b"HELICOPTER_THROTTLE_FULL"
THROTTLE_ALL_CUT = b"HELICOPTER_THROTTLE_CUT"
ROTOR_GOV_SET = b"ROTOR_GOV_SWITCH_SET"
ROTOR_CLUTCH_SET = b"ROTOR_CLUTCH_SWITCH_SET"
ROTOR_BRAKE = b"ROTOR_BRAKE"
ENGINE_START = b"ENGINE_AUTO_START"
ENGINE_STOP = b"ENGINE_AUTO_SHUTDOWN"

# --- the flight, conservatively -------------------------------------------
# UNVERIFIED everywhere: chosen inside the light singles' envelopes the
# FLIGHT Checklist profiles fly (H125, Bell 407, EC135, UH-1H).
HOVER_AGL_FT = 30.0            # the check hover, and the departure hover
TRANSITION_KT = 40.0           # out of ground effect, through the H-V curve
CLIMB_VS_FPM = 800.0           # best-rate-ish, well inside the profiles' 700-1000
MAX_VS_FPM = 1200.0            # the envelope: never command more
MAX_IAS_KT = 120.0             # a light single's VNE is near here; never above
PATTERN_AGL_FT = 800.0         # the altitude the circuit is flown at
FINAL_AGL_FT = 150.0           # where the approach becomes the landing
LANDING_VS_FPM = -300.0        # the descent onto the spot
FLARE_AGL_FT = 25.0            # collective comes up under this
CRUISE_CAP_FT = 6000.0         # a light single does not cruise at 39,000
START_TIMEOUT_S = 240.0        # spool-up on a cold turbine is slow
ROTOR_MIN_PCT = 80.0           # "flight RPM" for takeoff, lower than the 85 call
SHUTDOWN_COOLDOWN_S = 120.0    # best-effort idle cool-down before the brake
ROTOR_BRAKE_BELOW_PCT = 40.0   # apply the rotor brake under this, where fitted

# --- the controllers ------------------------------------------------------
# Collective: feet of altitude error per per-cent of collective, and the
# vertical-speed damping. UNVERIFIED gains; the clamps are the protection.
COLL_KP_PER_FT = 0.12          # per cent collective per foot of error
COLL_KD_PER_FPM = 0.04         # per cent collective per fpm of sink/climb
COLL_TRIM_PCT = 55.0           # the hover collective, guessed then adapted
COLL_ADAPT = 0.002             # how fast the trim learns, per second
# Cyclic longitudinal: axis units per knot of airspeed error. The SIGN is
# UNVERIFIED -- which way the axis pitches the model -- so the constant is
# written once, here, and flipped if a flight shows it backwards.
CYCLIC_KT_PER_AXIS = 30.0
CYCLIC_MAX = 6000              # about a third of full deflection
# Pedals/steering: axis units per degree of heading error.
PEDAL_DEG_PER_AXIS = 8.0
PEDAL_MAX = 5000
# The trim the cyclic starts from each regime, in axis units: a little
# forward to fly, centred to hover.
CYCLIC_TRIM_AIRBORNE = -1500
CYCLIC_TRIM_HOVER = 0
# How hard the engines are asked for in the air: the helicopter throttle
# event, 0..16384, full for takeoff and climb, governed in cruise.
THROTTLE_CLIMB = AXIS_MAX
THROTTLE_CRUISE = int(AXIS_MAX * 0.75)

# --- per-type envelopes ---------------------------------------------------
# Each rotorcraft is not the same aeroplane: an R44 and an S-76 do not
# transition at the same speed, climb at the same rate or see the same
# ceiling. The figures below are derived from the EUROCONTROL ATCPFDB
# entries for the type (indicative, not operational data) -- cruise TAS
# for the cruise speed, initial-climb ROC for the climb rate, approach
# IAS for the approach speed, ceiling and range as published. Where the
# database's helicopter rows are fixed-wing-looking (the H25x entries
# publish jet figures) or absent, the code maps to the nearest sane row
# or keeps the DEFAULT profile. UNVERIFIED like everything else here.


@dataclass(frozen=True)
class HeliProfile:
    """One rotorcraft's envelope: speeds, rates and the ceiling."""

    ias_cruise_kt: float
    ias_approach_kt: float
    climb_vs_fpm: float
    max_vs_fpm: float
    max_ias_kt: float
    ceiling_ft: float
    transition_kt: float
    landing_vs_fpm: float


def _heli_profile(cruise_tas: float, approach_ias: float, roc_fpm: float,
                  ceiling_ft: float) -> HeliProfile:
    """Derive a profile from the database's raw figures, clamped to what
    a light helicopter envelope looks like."""
    def clamp(v, lo, hi):
        return min(hi, max(lo, v))

    cruise = clamp(cruise_tas, 60.0, 140.0)
    approach = clamp(approach_ias, 45.0, 90.0)
    climb = clamp(roc_fpm * 0.75, 600.0, 1400.0)
    return HeliProfile(
        ias_cruise_kt=cruise,
        ias_approach_kt=approach,
        climb_vs_fpm=climb,
        max_vs_fpm=min(climb * 1.5, 2000.0),
        max_ias_kt=min(cruise + 30.0, 160.0),
        ceiling_ft=ceiling_ft,
        transition_kt=clamp(cruise * 0.4, 30.0, 60.0),
        landing_vs_fpm=-clamp(roc_fpm * 0.3, 200.0, 400.0))


DEFAULT_HELI_PROFILE = HeliProfile(
    ias_cruise_kt=80.0, ias_approach_kt=60.0, climb_vs_fpm=CLIMB_VS_FPM,
    max_vs_fpm=MAX_VS_FPM, max_ias_kt=MAX_IAS_KT, ceiling_ft=CRUISE_CAP_FT,
    transition_kt=TRANSITION_KT, landing_vs_fpm=LANDING_VS_FPM)

# Keyed by the type code the session carries; where WilcoATC's code is not
# the database's, the row is the nearest equivalent (B430 stands in for
# the AS350 family, EC35 for the EC130/135/145 family, B412 for the
# medium twins, AS32 for the Super Puma). UNVERIFIED mappings.
_HELI_PROFILES: dict[str, HeliProfile] = {
    # Robinson R44: piston, 110 kt, 1,000 fpm.
    "R44": _heli_profile(110, 80, 1000, 14000),
    # Airbus H125 / Leonardo H125 and the AS350 family via Bell 430.
    "H125": _heli_profile(130, 60, 1400, 14600),
    "B407": _heli_profile(130, 60, 1400, 14600),
    "R66": _heli_profile(130, 60, 1400, 14600),
    # Leonardo H145 / EC145 via the EC135 row.
    "H145": _heli_profile(135, 90, 1500, 20000),
    # Airbus H160 has no database row; EC35 is its closest published kin.
    "H160": _heli_profile(135, 90, 1500, 20000),
    # Eurocopter EC130/135/145 family.
    "EC30": _heli_profile(135, 90, 1500, 20000),
    "EC35": _heli_profile(135, 90, 1500, 20000),
    "EC45": _heli_profile(135, 90, 1500, 20000),
    # Airbus AS365 Dauphin via the medium-twin row.
    "AS50": _heli_profile(125, 60, 1300, 20000),
    # Bell 206 light single.
    "B06": _heli_profile(120, 80, 1350, 13500),
    # Bell 412 / Agusta A139 medium twin.
    "A139": _heli_profile(125, 60, 1300, 20000),
    "B412": _heli_profile(125, 60, 1300, 20000),
    # Sikorsky S-76 medium-lift twin.
    "S76": _heli_profile(150, 60, 1800, 13800),
    # Leonardo AW109 light-medium twin.
    "A109": _heli_profile(140, 100, 1900, 19600),
    # Airbus AS332 Super Puma heavy twin.
    "AS32": _heli_profile(140, 90, 1500, 17000),
    # Bell UH-1 utility single.
    "UH1": _heli_profile(110, 60, 1755, 19000),
}


def heli_profile_for(type_code: str) -> HeliProfile:
    """The envelope for this rotorcraft, or the conservative default."""
    return _HELI_PROFILES.get((type_code or "").strip().upper(),
                              DEFAULT_HELI_PROFILE)


# The internal machine. The owner-visible phase (AutonomousPhase) is a
# coarser view of the same flight.
STAGES = ("preflight", "start", "hover", "departure", "climb", "cruise",
          "descent", "approach", "landing", "shutdown", "done")

# stage -> the AutonomousPhase the rest of the program sees
_STAGE_PHASE = {
    "preflight": AutonomousPhase.PREFLIGHT,
    "start": AutonomousPhase.PREFLIGHT,
    "hover": AutonomousPhase.TAXI_OUT,
    "departure": AutonomousPhase.TAKEOFF,
    "climb": AutonomousPhase.CLIMB,
    "cruise": AutonomousPhase.CRUISE,
    "descent": AutonomousPhase.DESCENT,
    "approach": AutonomousPhase.APPROACH,
    "landing": AutonomousPhase.LANDING,
    "shutdown": AutonomousPhase.SHUTDOWN,
    # "done" is deliberately absent: entering it must not park the owner in
    # COMPLETE, or the chain step in ``_done`` would never run.
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _heading_error(want: float, have: float) -> float:
    """The shortest way round, -180..180."""
    return ((want - have + 540.0) % 360.0) - 180.0


@dataclass
class HeliState:
    """What the rotorcraft controller remembers between ticks."""

    stage: str = "preflight"
    stage_entered_at: float = 0.0
    # What the flight is for.
    dest_lat: float = 0.0
    dest_lon: float = 0.0
    dest_elev_ft: float = 0.0
    dest_ident: str = ""
    cruise_alt_ft: float = 0.0
    pattern_alt_ft: float = 0.0
    ias_cruise_kt: float = 80.0
    ias_approach_kt: float = 60.0
    # The learned hover collective, per cent depressed.
    coll_trim: float = COLL_TRIM_PCT
    last_agl_ft: float = 0.0
    last_ias_kt: float = 0.0
    # One-shots.
    systems_sent: bool = False
    start_sent: bool = False
    gov_sent: bool = False
    rotor_brake_sent: bool = False
    shutdown_sent: bool = False
    # IFR was seen and downgraded, once.
    ifr_downgraded: bool = False
    # The shutdown cool-down clock.
    shutdown_at: float = 0.0


class RotorcraftFlight:
    """Flies the helicopter the fixed-wing controller cannot.

    Same lifecycle as :class:`AutonomousFlight`: ``begin_flight`` once per
    leg, ``tick`` from the situation loop. The owner's ``state.phase`` is
    kept on the coarse :class:`AutonomousPhase` for anything that watches.
    """

    def __init__(self, owner):
        self.owner = owner
        self.config = owner.config
        self.session = owner.session
        self.sim = owner.sim
        self.get_state = owner.get_state
        self.crew = owner.crew
        self.on_event = owner.on_event
        self.state = HeliState()
        self._profile = DEFAULT_HELI_PROFILE
        self._transitioned_owner = ""
        self._last_tick = 0.0
        # The situation loop runs faster than a pilot's hand can; the
        # controls are closed-loop, so a few passes a second is plenty and
        # every pass is a SimConnect round trip.
        self.TICK_S = 0.25

    # -- plumbing ---------------------------------------------------------

    def _send(self, name: bytes, value: int) -> bool:
        return self.owner._send_event(name, value)

    def _emit(self, kind: str, text: str, key: str = "") -> None:
        self.owner._emit(kind, text, key=key)

    def _goto(self, stage: str) -> None:
        self.state.stage = stage
        self.state.stage_entered_at = time.monotonic()
        shown = _STAGE_PHASE.get(stage)
        if shown is not None and shown != self.owner.state.phase:
            self.owner._transition(shown)
        log.info("rotorcraft stage: %s", stage)

    def _checklist(self, name: str) -> None:
        done = self.owner.state.checklists_completed
        if name not in done:
            self.crew.checklists.start(name)
            done.add(name)

    def _elapsed(self) -> float:
        return time.monotonic() - self.state.stage_entered_at

    # -- entry ------------------------------------------------------------

    def begin(self, session) -> None:
        """A new leg (or a chained one): everything is re-planned."""
        self.session = session
        self.state = HeliState()
        # The type's own envelope where the table knows it -- an R44 and
        # an S-76 do not share a transition speed or a ceiling.
        self._profile = heli_profile_for(
            getattr(session.aircraft, "type_code", ""))
        dest = session.destination
        if dest is not None:
            self.state.dest_lat = dest.lat
            self.state.dest_lon = dest.lon
            self.state.dest_elev_ft = dest.elev_ft
            self.state.dest_ident = dest.ident
        cruise = session.cruise_altitude_ft or 3000.0
        self.state.cruise_alt_ft = min(cruise, self._profile.ceiling_ft)
        self.state.pattern_alt_ft = min(
            max(self.state.dest_elev_ft + PATTERN_AGL_FT, 1500.0),
            self.state.cruise_alt_ft)
        speeds = speeds_for(getattr(session.aircraft, "type_code", ""),
                            HELICOPTER)
        # The model's or the table's approach speed first, the type's own
        # envelope for the cruise it does not declare.
        self.state.ias_approach_kt = (
            float(speeds.vapp or 0.0) or self._profile.ias_approach_kt)
        self.state.ias_cruise_kt = self._profile.ias_cruise_kt
        self._goto("preflight")

    # -- the loop ---------------------------------------------------------

    def tick(self, state) -> None:
        """One pass of the flight, at whatever rate the situation loop runs."""
        if state is None or not getattr(state, "connected", False):
            return
        # The owner gave up (a timeout, an exception): hands off, whatever
        # stage this was in.
        if self.owner.state.phase == AutonomousPhase.ERROR:
            return
        now = time.monotonic()
        if now - self._last_tick < self.TICK_S:
            return
        self._last_tick = now
        # A chained leg: the owner's chain phase has loaded the next flight
        # plan and put the session back to PREFLIGHT; this controller
        # re-plans for it rather than starting a new object.
        if self.state.stage == "done":
            phase = self.owner.state.phase
            if phase == AutonomousPhase.PREFLIGHT:
                self.begin(self.session)
                return
            if phase == AutonomousPhase.CHAIN_NEXT:
                # Drive the owner's chain step here: the fixed-wing phase
                # loop is bypassed for a helicopter.
                self.owner._phase_chain_next(state, time.monotonic())
                phase = self.owner.state.phase
                if phase == AutonomousPhase.PREFLIGHT:
                    self.begin(self.session)
                elif phase in (AutonomousPhase.SHUTDOWN,
                               AutonomousPhase.COMPLETE):
                    # The chain found nothing more: the flight is over.
                    self._finish()
                return
            if phase in (AutonomousPhase.COMPLETE, AutonomousPhase.ERROR):
                return
            # SHUTDOWN (or anything else): the leg ended; chain or finish.
            self._done(state)
            return
        stage = self.state.stage
        try:
            if stage == "preflight":
                self._preflight(state)
            elif stage == "start":
                self._start(state)
            elif stage == "hover":
                self._hover(state)
            elif stage == "departure":
                self._departure(state)
            elif stage == "climb":
                self._climb(state)
            elif stage == "cruise":
                self._cruise(state)
            elif stage == "descent":
                self._descent(state)
            elif stage == "approach":
                self._approach(state)
            elif stage == "landing":
                self._landing(state)
            elif stage == "shutdown":
                self._shutdown(state)
            elif stage == "done":
                self._done(state)
        except Exception as exc:                       # pragma: no cover
            log.error("rotorcraft stage %s failed: %s", stage, exc,
                      exc_info=True)
            self.owner._set_error(f"rotorcraft {stage}: {exc}")

    # -- preflight and start ----------------------------------------------

    def _preflight(self, state) -> None:
        if self.session.ifr and not self.state.ifr_downgraded:
            # Explicit, not silent: the flight is flown VFR because that is
            # all a rotorcraft gets here, and the log says why.
            self.state.ifr_downgraded = True
            self.session.ifr = False
            self._emit("info",
                       "Autonomous rotorcraft is VFR only; the flight is "
                       "flown VFR", "autonomous.rotorcraft.vfr_only")
        if self.session.destination is None:
            self.owner._set_error("no destination for the rotorcraft leg")
            return
        self._checklist("before_start")
        self._goto("start")

    def _start(self, state) -> None:
        """Best-effort engine start: the generic events, then watch."""
        if not self.state.systems_sent:
            self.state.systems_sent = True
            if not getattr(state, "battery_master", False):
                self._send(b"MASTER_BATTERY_SET", 1)
            if not getattr(state, "avionics_master", False):
                self._send(b"AVIONICS_MASTER_SET", 1)
        if not self.state.start_sent:
            self.state.start_sent = True
            # The governor up for the start where the model has one; the
            # clutch is the model's own decision on most turbines, so it is
            # only touched if nothing is running after a while.
            self._send(ROTOR_GOV_SET, 1)
            self.state.gov_sent = True
            self._send(ENGINE_START, 1)
            self._emit("info", "Engine start requested (best effort)",
                       "autonomous.rotorcraft.start")
        running = bool(getattr(state, "engines_running", False))
        rpm = float(getattr(state, "rotor_rpm_pct", 0.0) or 0.0)
        # Rotor RPM is published by some models and not others: where it is
        # 0 all flight, engines_running and the timeout carry the phase.
        rotor_ok = rpm >= ROTOR_MIN_PCT or (rpm == 0.0 and running)
        if running and rotor_ok:
            self._checklist("after_start")
            self._goto("hover")
            return
        if self._elapsed() > START_TIMEOUT_S:
            self.owner._set_error(
                "rotorcraft engine start timed out; the model may need its "
                "own start event")

    # -- the flight --------------------------------------------------------

    def _hover(self, state) -> None:
        """The start checks: hold the spot until the rotor is flight RPM
        and the clock says the pilot has looked."""
        self._hold_hover(state)
        rpm = float(getattr(state, "rotor_rpm_pct", 0.0) or 0.0)
        rotor_ok = rpm >= ROTOR_MIN_PCT or (rpm == 0.0 and self._elapsed() > 30.0)
        if rotor_ok and self._elapsed() > 8.0:
            self._checklist("before_takeoff")
            self._goto("departure")
        elif self._elapsed() > 120.0:
            # A model that never reports rotor RPM still goes; the spool is
            # judged by the engine and the clock, not by a missing variable.
            self._checklist("before_takeoff")
            self._goto("departure")

    def _departure(self, state) -> None:
        """Transition through the H-V curve: light on the skids, ~40 kt,
        then climb out of ground effect."""
        agl = float(getattr(state, "altitude_agl_ft", 0.0) or 0.0)
        ias = float(getattr(state, "indicated_airspeed_kt", 0.0) or 0.0)
        prof = self._profile
        self._fly(state, agl_target=HOVER_AGL_FT + 20.0,
                  ias_target=prof.transition_kt,
                  vs_target=prof.climb_vs_fpm, throttle=THROTTLE_CLIMB)
        if agl > 100.0 and ias >= prof.transition_kt:
            self._checklist("after_takeoff")
            self._goto("climb")
        elif self._elapsed() > 90.0:
            # Never rotated: something the controller cannot see. Say so
            # rather than charge into the climb.
            self.owner._set_error("rotorcraft transition timed out")

    def _climb(self, state) -> None:
        agl = float(getattr(state, "altitude_agl_ft", 0.0) or 0.0)
        self._fly(state, agl_target=None,
                  alt_target=self.state.cruise_alt_ft,
                  ias_target=self.state.ias_cruise_kt,
                  vs_target=self._profile.climb_vs_fpm,
                  throttle=THROTTLE_CLIMB)
        if (state.altitude_ft or 0.0) >= self.state.cruise_alt_ft - 150.0 \
                and abs(getattr(state, "vertical_speed_fpm", 0.0) or 0.0) < 200:
            self._goto("cruise")
        elif self._elapsed() > 900.0:
            self.owner._set_error("rotorcraft climb timed out")

    def _cruise(self, state) -> None:
        self._checklist("cruise")
        dist = self._distance_nm(state)
        if dist is not None and dist < 5.0:
            self._checklist("descent")
            self._goto("descent")
            return
        self._fly(state, agl_target=None,
                  alt_target=self.session.assigned_altitude_ft
                  or self.state.cruise_alt_ft,
                  ias_target=self.session.assigned_speed_kt
                  or self.state.ias_cruise_kt,
                  vs_target=0.0, throttle=THROTTLE_CRUISE)

    def _descent(self, state) -> None:
        agl = float(getattr(state, "altitude_agl_ft", 0.0) or 0.0)
        self._fly(state, agl_target=None,
                  alt_target=self.state.pattern_alt_ft,
                  ias_target=self.state.ias_approach_kt,
                  vs_target=-self._profile.climb_vs_fpm,
                  throttle=THROTTLE_CRUISE)
        dist = self._distance_nm(state)
        if agl <= PATTERN_AGL_FT and (dist is None or dist < 3.0):
            self._checklist("approach")
            self._goto("approach")
        elif self._elapsed() > 900.0:
            self.owner._set_error("rotorcraft descent timed out")

    def _approach(self, state) -> None:
        agl = float(getattr(state, "altitude_agl_ft", 0.0) or 0.0)
        ias = float(getattr(state, "indicated_airspeed_kt", 0.0) or 0.0)
        # Down the profile's approach: slow to approach speed first, then
        # descend into the hover over the spot.
        if ias > self.state.ias_approach_kt + 5.0 or agl > FINAL_AGL_FT + 100:
            self._fly(state, agl_target=FINAL_AGL_FT + 100.0,
                      ias_target=self.state.ias_approach_kt,
                      vs_target=-400.0, throttle=THROTTLE_CRUISE)
        else:
            self._fly(state, agl_target=0.0,
                      ias_target=max(self.state.ias_approach_kt - 10.0, 40.0),
                      vs_target=self._profile.landing_vs_fpm,
                      throttle=THROTTLE_CRUISE)
        if agl <= FLARE_AGL_FT + 20.0:
            self._checklist("before_landing")
            self._goto("landing")

    def _landing(self, state) -> None:
        agl = float(getattr(state, "altitude_agl_ft", 0.0) or 0.0)
        if getattr(state, "on_ground", False):
            # Touchdown: collective down, cyclic and pedals centred, and the
            # aircraft's own brakes where it has them.
            self._collective(5.0)
            self._send(CYCLIC_LONG_SET, CYCLIC_TRIM_HOVER)
            self._send(CYCLIC_LAT_SET, 0)
            self._send(STEERING_SET, 0)
            self._send(b"PARKING_BRAKE_SET", 1)
            self._checklist("after_landing")
            self._goto("shutdown")
            return
        # The flare: the rate comes off under FLARE_AGL_FT so the skids
        # meet the ground rather than the airframe meeting it fast.
        vs_target = (self._profile.landing_vs_fpm if agl > FLARE_AGL_FT
                     else -120.0)
        self._fly(state, agl_target=0.0,
                  ias_target=max(self.state.ias_approach_kt - 15.0, 35.0),
                  vs_target=vs_target, throttle=THROTTLE_CRUISE)
        if self._elapsed() > 300.0:
            self.owner._set_error("rotorcraft landing timed out")

    # -- shutdown -----------------------------------------------------------

    def _shutdown(self, state) -> None:
        """Cool-down, rotor brake where fitted, then the engine events."""
        rpm = float(getattr(state, "rotor_rpm_pct", 0.0) or 0.0)
        running = bool(getattr(state, "engines_running", False))
        if not self.state.shutdown_at:
            self.state.shutdown_at = time.monotonic()
        elapsed = time.monotonic() - self.state.shutdown_at
        if running and not self.state.rotor_brake_sent \
                and getattr(state, "rotor_brake_available", False) \
                and (rpm == 0.0 or rpm < ROTOR_BRAKE_BELOW_PCT):
            self._send(ROTOR_BRAKE, 1)
            self.state.rotor_brake_sent = True
        cooled = elapsed >= SHUTDOWN_COOLDOWN_S or (rpm and rpm < 30.0)
        if running and cooled and not self.state.shutdown_sent:
            self.state.shutdown_sent = True
            self._send(ENGINE_STOP, 1)
            self._send(b"AVIONICS_MASTER_SET", 0)
            self._send(b"MASTER_BATTERY_SET", 0)
            self._emit("info", "Engine shutdown requested (best effort)",
                       "autonomous.rotorcraft.shutdown")
        if (not running and self.state.shutdown_sent) \
                or (self.state.shutdown_sent and elapsed > 60.0) \
                or elapsed > 300.0:
            self._goto("done")

    def _done(self, state) -> None:
        """The leg is over: chain the next or finish, as the owner does."""
        if self.owner._chained:
            if self.owner.state.phase != AutonomousPhase.CHAIN_NEXT:
                self.owner._transition(AutonomousPhase.CHAIN_NEXT)
            return
        self._finish()

    def _finish(self) -> None:
        """Emit the completion once and park the owner in COMPLETE."""
        if self.owner.state.phase != AutonomousPhase.COMPLETE:
            self._emit("info", "Autonomous rotorcraft flight completed",
                       "autonomous.completed")
            self.owner._transition(AutonomousPhase.COMPLETE)

    # -- the controls -------------------------------------------------------

    def _fly(self, state, *, agl_target=None, alt_target=None,
             ias_target=None, vs_target=0.0, throttle=None) -> None:
        """One pass of the three axes, in the order a pilot thinks of them:
        collective for height, cyclic for speed, pedals for heading."""
        agl = float(getattr(state, "altitude_agl_ft", 0.0) or 0.0)
        alt = float(getattr(state, "altitude_ft", 0.0) or 0.0)
        vs = float(getattr(state, "vertical_speed_fpm", 0.0) or 0.0)
        ias = float(getattr(state, "indicated_airspeed_kt", 0.0) or 0.0)
        hdg = float(getattr(state, "heading_true", 0.0) or 0.0)
        airborne = not bool(getattr(state, "on_ground", True))

        # -- collective: height, or rate of climb --------------------------
        if agl_target is not None:
            # Hover/landing: feet above the ground, with sink damping.
            err = agl_target - agl
            coll = self.state.coll_trim + COLL_KP_PER_FT * err \
                - COLL_KD_PER_FPM * vs
            # Learn the trim only in a steady hover, so a climb does not
            # teach the collective the wrong number.
            if airborne and abs(vs) < 150.0 and abs(err) < 40.0:
                self.state.coll_trim += COLL_ADAPT * (
                    float(getattr(state, "collective_position", 0.0) or 0.0)
                    - self.state.coll_trim) * 0.1
        elif alt_target is not None:
            # Cruise/climb/descent: feet to the level, rate of climb damped.
            err = alt_target - alt
            want_vs = _clamp(err * 2.0, -self._profile.max_vs_fpm,
                             self._profile.max_vs_fpm)
            coll = self.state.coll_trim + COLL_KP_PER_FT * err / 10.0 \
                + COLL_KD_PER_FPM * (want_vs - vs) / 10.0
        else:
            coll = self.state.coll_trim
        self._collective(coll)

        # -- cyclic longitudinal: airspeed ----------------------------------
        if ias_target is not None and airborne:
            err = ias_target - ias
            axis = CYCLIC_TRIM_AIRBORNE + CYCLIC_KT_PER_AXIS * err
            # A helicopter cannot climb vertically forever: bleed speed
            # protection -- if the collective is asking hard and the speed
            # is low, lower the nose rather than stall in the H-V curve.
            if ias < 45.0 and agl > 200.0:
                axis -= 1500
            self._send(CYCLIC_LONG_SET, int(_clamp(axis, -CYCLIC_MAX,
                                                 CYCLIC_MAX)))
        elif not airborne:
            self._send(CYCLIC_LONG_SET, CYCLIC_TRIM_HOVER)
        self._send(CYCLIC_LAT_SET, self._lateral_trim_axis(state))

        # -- pedals: heading -------------------------------------------------
        want_hdg = self._desired_track(state)
        if want_hdg:
            err = _heading_error(want_hdg, hdg)
            self._send(STEERING_SET,
                       int(_clamp(-PEDAL_DEG_PER_AXIS * err, -PEDAL_MAX,
                                  PEDAL_MAX)))

        # -- throttle ---------------------------------------------------------
        if throttle is not None:
            self._throttle(throttle)

        # -- envelope -----------------------------------------------------------
        self._protect(state, vs, ias, agl)
        self.state.last_agl_ft = agl
        self.state.last_ias_kt = ias

    def _collective(self, pct: float) -> None:
        pct = _clamp(pct, 0.0, 100.0)
        self._send(COLLECTIVE_SET, int(pct / 100.0 * AXIS_MAX))

    def _throttle(self, value: int) -> None:
        """The helicopter throttle events, best effort: the axis where the
        model has it, the named events where it does not."""
        if not self._send(HELI_THROTTLE_SET, value):
            self._send(THROTTLE_ALL_SET, value)
        if value >= AXIS_MAX * 0.9:
            self._send(THROTTLE_ALL_FULL, 0)

    def _protect(self, state, vs: float, ias: float, agl: float) -> None:
        """The envelope: never more than the type's max VS either way,
        never above its max IAS, never a big collective at skid height --
        and never past the torque the engine is rated for, where the
        model reports it."""
        max_vs = self._profile.max_vs_fpm
        if vs > max_vs:
            self._collective(max(0.0, self.state.coll_trim - 10.0))
        elif vs < -max_vs:
            self._collective(min(100.0, self.state.coll_trim + 15.0))
        if ias > self._profile.max_ias_kt:
            self._send(CYCLIC_LONG_SET, CYCLIC_MAX)   # nose down, UNVERIFIED sign
        torque = float(getattr(state, "eng_torque_pct", 0.0) or 0.0)
        # The variable is documented as a percent but some models publish
        # torque in foot-pounds on the same scale; a figure outside the
        # percent range is not a percent, and limiting on it would hold
        # the collective down all flight.
        if 95.0 < torque <= 115.0:
            # At red line the collective comes off, whatever else wants.
            self._collective(max(0.0, self.state.coll_trim - 8.0))
        if agl < 15.0 and not getattr(state, "on_ground", True):
            # Effectively landed but not flagged: kill the climb input.
            self._collective(min(self.state.coll_trim, 30.0))

    def _lateral_trim_axis(self, state=None) -> int:
        """The model's own lateral cyclic trim as an axis value.

        ROTOR LATERAL TRIM PCT is the trim the model carries; 50 is
        centred, and the departure from it is the bias a wings-level
        hover needs. UNVERIFIED: the sign and the scale are the
        simulator's, and a wrong one shows as a slow roll, not a departure.
        """
        if state is None:
            state = self.get_state()
        trim = float(getattr(state, "rotor_lateral_trim_pct", 0.0)
                     or 0.0) if state is not None else 0.0
        # Only a value that could be a percent is trusted; anything else
        # is a unit the conversion did not expect.
        if not 0.0 < trim < 100.0:
            return 0
        return int(_clamp((trim - 50.0) * 163.84, -3000, 3000))

    # -- navigation ---------------------------------------------------------

    def _distance_nm(self, state) -> float | None:
        if not getattr(state, "has_position", False) \
                or not self.state.dest_lat:
            return None
        return state.distance_to(self.state.dest_lat, self.state.dest_lon)

    def _desired_track(self, state) -> float:
        """Where to point: the controller's heading, else the destination."""
        assigned = getattr(self.session, "assigned_heading", None)
        if assigned:
            return float(assigned)
        if not getattr(state, "has_position", False) \
                or not self.state.dest_lat:
            return float(getattr(state, "heading_true", 0.0) or 0.0)
        return state.bearing_to(self.state.dest_lat, self.state.dest_lon)

    def _hold_hover(self, state) -> None:
        """Feet on the ground, rotor turning: centred controls."""
        self._collective(self.state.coll_trim)
        self._send(CYCLIC_LONG_SET, CYCLIC_TRIM_HOVER)
        self._send(CYCLIC_LAT_SET, self._lateral_trim_axis(state))
        self._send(STEERING_SET, 0)


__all__ = ["RotorcraftFlight", "HELI_THROTTLE_SET", "STAGES"]
