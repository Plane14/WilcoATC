# -*- coding: utf-8 -*-
"""Autonomous flight control for the First Officer.

When enabled, the First Officer manages the entire flight from startup to
shutdown using the simulator's autopilot and aircraft systems. This includes
all phases: preflight, pushback, taxi, takeoff, climb, cruise, descent,
approach, landing, taxi-in, and shutdown.

Two autopilot management modes are supported:
- "managed": Uses the aircraft's FMS/autopilot modes (HDG, ALT, NAV, APR)
  via SimConnect events. More realistic for modern airliners.
- "direct": Sets autopilot references directly via SimConnect events.
  Works on more aircraft types.

When chained flights are enabled, the First Officer will automatically
select random destinations within the aircraft's range at airports that
support the aircraft type and have ground services (fuel, parking, etc.).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from ..config import Config
from ..navdata.db import NavDB, Airport
from ..sim.source import SimConnectSource
from .session import FlightSession, Phase
from .checklist import ChecklistRunner
from .callouts import Callouts, Situation
from .crew import CrewDirector
from .profile import FlightProfile
from .phraseology import Aircraft
from .copilot import confirmation, readback
from .controller import ControllerBrain

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
    # Autopilot state
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
    # Chain flights
    chain_flight_index: int = 0
    chain_flights: list[dict] = field(default_factory=list)
    # Aircraft performance for chain planning
    aircraft_range_nm: float = 0.0
    aircraft_type: str = ""
    # Error tracking
    error_message: str = ""
    error_count: int = 0


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
    ):
        self.config = config
        self.navdb = navdb
        self.sim = sim_source
        self.session = session
        self.crew = crew
        self.brain = brain
        self.get_state = get_state
        self.on_event = on_event

        self.state = AutonomousState()
        self._enabled = config.immersion.autonomous_flight
        self._chained = config.immersion.autonomous_chained_flights
        self._ap_mode = config.immersion.autonomous_ap_mode
        self._chain_delay = config.immersion.autonomous_chain_delay_s
        self._request_ifr = config.immersion.autonomous_request_ifr
        self._use_sim_ai = config.immersion.autonomous_use_sim_ai

        # Compute aircraft range for chain planning
        self._compute_aircraft_performance(session)

        # Timing constants
        self.ACTION_INTERVAL = 1.0  # seconds between autonomous actions
        self.CLEARANCE_TIMEOUT = 60.0
        self.PUSHBACK_TIMEOUT = 120.0
        self.TAXI_TIMEOUT = 300.0
        self.TAKEOFF_TIMEOUT = 180.0
        self.LEVEL_OFF_TIMEOUT = 30.0

    def _compute_aircraft_performance(self, session: FlightSession) -> None:
        """Compute aircraft range and performance for chain flight planning."""
        type_code = session.aircraft.type_code.upper()
        self.state.aircraft_type = type_code

        # Estimate range based on aircraft type (nautical miles)
        # These are rough estimates for typical configurations
        ranges = {
            # Light aircraft
            "C152": 350, "C172": 550, "C182": 750, "C208": 800,
            "PA28": 500, "PA34": 700, "DA40": 600, "DA42": 800,
            "SR22": 700, "BE36": 800, "TB20": 700,
            # Turboprops
            "TBM9": 1500, "PC12": 1300, "AT76": 800, "DH8D": 900,
            "C25A": 1200, "C25C": 1400, "C700": 1800,
            # Business jets
            "C525": 1500, "C560": 1800, "C680": 2200, "C750": 2500,
            "E55P": 1500, "E50P": 1800, "F2TH": 2000, "F900": 2500,
            "GLEX": 3500, "GLF5": 3500, "GLF6": 4000, "G280": 2000,
            "H25B": 1800, "LJ45": 1800, "LJ60": 2000,
            # Airliners - narrowbody
            "A319": 3200, "A320": 3000, "A321": 3000, "A20N": 3500, "A21N": 3500,
            "B737": 2800, "B738": 3000, "B739": 3000, "B38M": 3500, "B39M": 3500,
            "B752": 3500, "B753": 3300,
            # Airliners - widebody
            "A332": 5500, "A333": 5500, "A339": 6000, "A343": 6500, "A346": 7000,
            "A359": 7000, "A35K": 8000, "A388": 8000,
            "B763": 5500, "B764": 5500, "B772": 5500, "B77W": 7500, "B77L": 8500,
            "B77X": 8000, "B788": 7500, "B789": 7500, "B78X": 6500,
            "B744": 7000, "B748": 7500,
            # Regional jets
            "CRJ7": 1500, "CRJ9": 1500, "CRJX": 1500,
            "E170": 1800, "E175": 1800, "E190": 2200, "E195": 2200,
        }

        # Try exact match first, then prefix match
        range_nm = ranges.get(type_code, 0)
        if not range_nm:
            for prefix, r in ranges.items():
                if type_code.startswith(prefix):
                    range_nm = r
                    break

        # Default fallback based on category
        if not range_nm:
            if session.profile.airliner:
                range_nm = 2500
            elif session.profile.business_jet:
                range_nm = 2000
            elif session.profile.turboprop:
                range_nm = 1000
            else:
                range_nm = 500

        self.state.aircraft_range_nm = range_nm
        log.info("Aircraft %s: estimated range %d NM", type_code, range_nm)

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
        # Compute aircraft performance for chain planning
        self._compute_aircraft_performance(session)
        self._emit("info", "Autonomous flight started", "autonomous.started")

    def set_chain_flights(self, flights: list[dict]) -> None:
        """Set up chained flights from a list of flight plans."""
        self.state.chain_flights = flights
        self.state.chain_flight_index = 0

    def tick(self) -> None:
        """Main autonomous flight loop. Called from the situation loop."""
        if not self._enabled:
            return

        state = self.get_state()
        if state is None or not state.connected:
            return

        now = time.monotonic()

        # Rate limit actions
        if now - self.state.last_action_at < self.ACTION_INTERVAL:
            return

        # Execute phase logic
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

    # Phase implementations

    def _phase_preflight(self, state, now: float) -> None:
        """Preflight: configure aircraft, load flight plan, run before-start checklist."""
        # Ensure avionics and battery are on
        self._ensure_systems_on(state)

        # Run before-start checklist if available
        self._run_checklist_for_phase("before_start")

        # If we have a flight plan and IFR, request clearance
        if self.session.ifr and self._request_ifr and self.session.destination:
            self._transition(AutonomousPhase.REQUEST_CLEARANCE)
        else:
            # VFR or no clearance needed - go to pushback
            self._transition(AutonomousPhase.PUSHBACK)

    def _phase_request_clearance(self, state, now: float) -> None:
        """Request and receive IFR clearance."""
        if not self.state.has_ifr_clearance:
            # Request clearance via first officer
            self._request_clearance_via_fo()
            self.state.has_ifr_clearance = True  # Will be confirmed by ATC
        elif self.session.clearance_read_back:
            # Clearance received and read back
            self._transition(AutonomousPhase.PUSHBACK)
        elif now - self.state.phase_entered_at > self.CLEARANCE_TIMEOUT:
            self._set_error("Clearance timeout")

    def _phase_pushback(self, state, now: float) -> None:
        """Request pushback and start engines."""
        if not state.engines_running:
            self._start_engines(state)

        if state.parking_brake:
            self._release_parking_brake()

        # Request pushback via GSX or sim
        if not self.state.pushing:
            self._request_pushback()

        # Check if pushback complete (stationary after moving)
        if self.state.pushing and abs(state.ground_speed_kt) < 1.0:
            if now - self.state.still_since > 10.0:
                self._transition(AutonomousPhase.TAXI_OUT)

        if now - self.state.phase_entered_at > self.PUSHBACK_TIMEOUT:
            self._set_error("Pushback timeout")

    def _phase_taxi_out(self, state, now: float) -> None:
        """Taxi to departure runway following ATC instructions."""
        # Run taxi checklist
        self._run_checklist_for_phase("taxi")

        # Taxi is managed by ATC - we follow clearances
        # The session phase will be TAXI_OUT when cleared
        if self.session.phase == Phase.HOLDING_SHORT:
            self._transition(AutonomousPhase.HOLDING_SHORT)

        if now - self.state.phase_entered_at > self.TAXI_TIMEOUT:
            self._set_error("Taxi timeout")

    def _phase_holding_short(self, state, now: float) -> None:
        """Holding short of runway, run before-takeoff checklist."""
        self._run_checklist_for_phase("before_takeoff")

        # Wait for takeoff clearance
        if self.session.cleared_takeoff:
            self._transition(AutonomousPhase.LINE_UP)

    def _phase_line_up(self, state, now: float) -> None:
        """Line up on runway and prepare for takeoff."""
        # Ensure lined up on correct runway
        if not state.on_ground:
            self._transition(AutonomousPhase.TAKEOFF)
            return

        # Configure for takeoff: flaps, trim, lights
        self._configure_for_takeoff(state)

        # Wait for lineup confirmation or just proceed
        if self.session.lined_up:
            self._transition(AutonomousPhase.TAKEOFF)

    def _phase_takeoff(self, state, now: float) -> None:
        """Execute takeoff roll and initial climb."""
        if not state.on_ground:
            # Airborne - engage climb autopilot
            self._engage_climb_autopilot(state)
            self._run_checklist_for_phase("after_takeoff")
            self._transition(AutonomousPhase.CLIMB)
            return

        # Apply takeoff power
        self._set_takeoff_power()

        # Steer with nose wheel until rotation
        if state.indicated_airspeed_kt > 80:
            self._rotate(state)

        if now - self.state.phase_entered_at > self.TAKEOFF_TIMEOUT:
            self._set_error("Takeoff timeout")

    def _phase_climb(self, state, now: float) -> None:
        """Climb to cruise altitude following ATC clearances."""
        # Follow ATC altitude assignments
        target_alt = self.session.assigned_altitude_ft or self.session.cruise_altitude_ft
        if target_alt:
            self._set_autopilot_altitude(target_alt)

        # Follow SID or heading assignments
        if self.session.assigned_heading:
            self._set_autopilot_heading(self.session.assigned_heading)

        # Speed management
        if self.session.assigned_speed_kt:
            self._set_autopilot_speed(self.session.assigned_speed_kt)

        # Check if level at cruise
        if (abs(state.altitude_ft - target_alt) < 200 and
                abs(state.vertical_speed_fpm) < 200):
            self._level_off_at_cruise(state)

        # Check for top of descent
        if self.session.phase == Phase.DESCENT:
            self._transition(AutonomousPhase.DESCENT)

    def _phase_cruise(self, state, now: float) -> None:
        """Cruise: follow route, manage fuel, monitor systems."""
        # Run cruise checklist
        self._run_checklist_for_phase("cruise")

        # Follow flight plan (NAV mode)
        self._engage_nav_mode()

        # Check for descent point
        if self.session.phase == Phase.DESCENT:
            self._transition(AutonomousPhase.DESCENT)

    def _phase_descent(self, state, now: float) -> None:
        """Descend via STAR or ATC vectors."""
        # Run descent checklist
        self._run_checklist_for_phase("descent")

        # Follow ATC descent clearances
        target_alt = self.session.assigned_altitude_ft
        if target_alt:
            self._set_autopilot_altitude(target_alt)

        # Speed management
        if self.session.assigned_speed_kt:
            self._set_autopilot_speed(self.session.assigned_speed_kt)

        # Check for approach phase
        if self.session.phase == Phase.APPROACH:
            self._transition(AutonomousPhase.APPROACH)

    def _phase_approach(self, state, now: float) -> None:
        """Fly approach: intercept localizer, capture glideslope."""
        # Run approach checklist
        self._run_checklist_for_phase("approach")

        # Configure for approach: flaps, gear, speed
        self._configure_for_approach(state)

        # Engage approach mode (APR) for ILS
        self._engage_approach_mode()

        # Check for landing clearance
        if self.session.cleared_landing:
            self._transition(AutonomousPhase.LANDING)

    def _phase_landing(self, state, now: float) -> None:
        """Land the aircraft."""
        # Run landing checklist
        self._run_checklist_for_phase("before_landing")

        # Autoland if capable, otherwise manual flare
        if state.on_ground:
            self._transition(AutonomousPhase.TAXI_IN)

    def _phase_taxi_in(self, state, now: float) -> None:
        """Taxi to parking after landing."""
        # Run after-landing/taxi-in checklist
        self._run_checklist_for_phase("after_landing")

        # Follow ATC taxi instructions to parking
        if self.session.phase == Phase.PARKED:
            self._transition(AutonomousPhase.PARKED)

    def _phase_parked(self, state, now: float) -> None:
        """At parking position, shutdown engines."""
        self._set_parking_brake()
        self._shutdown_engines()
        self._run_checklist_for_phase("shutdown")

        if self._chained and self.state.chain_flights:
            self._transition(AutonomousPhase.CHAIN_NEXT)
        else:
            self._transition(AutonomousPhase.SHUTDOWN)

    def _phase_shutdown(self, state, now: float) -> None:
        """Flight complete, systems off."""
        self._emit("info", "Autonomous flight completed", "autonomous.completed")
        self._transition(AutonomousPhase.COMPLETE)

    def _phase_chain_next(self, state, now: float) -> None:
        """Load and start next flight in chain with random destination."""
        # Select next destination if not pre-planned
        if not self.state.chain_flights:
            self._select_next_destination()
            if not self.state.chain_flights:
                self._emit("info", "No suitable destination found for next flight",
                           "autonomous.chain.no_destination")
                self._transition(AutonomousPhase.SHUTDOWN)
                return

        # Wait for chain delay
        if now - self.state.phase_entered_at < self._chain_delay:
            return

        next_flight = self.state.chain_flights[self.state.chain_flight_index]
        self.state.chain_flight_index += 1

        # Load next flight plan
        self._load_next_flight(next_flight)
        self._transition(AutonomousPhase.PREFLIGHT)

    def _select_next_destination(self) -> None:
        """Select a random destination within aircraft range with ground services."""
        current = self.session.departure or self._get_current_airport()
        if not current:
            log.warning("Cannot select next destination: no current airport")
            return

        # Find suitable airports within range
        candidates = self._find_suitable_destinations(current)
        if not candidates:
            log.warning("No suitable destinations within range of %s", current.ident)
            return

        # Pick one at random, weighted by suitability
        destination = self._pick_destination(candidates)
        if not destination:
            return

        # Create flight plan for the leg
        flight_plan = self._create_flight_plan(current, destination)
        self.state.chain_flights.append(flight_plan)
        log.info("Chained flight: %s -> %s (%d NM)", current.ident, destination.ident,
                 flight_plan.get("distance_nm", 0))

    def _get_current_airport(self) -> Optional[Airport]:
        """Get the airport the aircraft is currently at."""
        state = self.get_state()
        if state and state.has_position:
            return self.navdb.home_airport(state.latitude, state.longitude)
        return self.session.departure

    def _find_suitable_destinations(self, origin: Airport) -> list[tuple[Airport, float]]:
        """Find airports within range that can host the aircraft with ground services."""
        candidates = []
        max_range = self.state.aircraft_range_nm * 0.8  # 80% of max range for reserves

        # Get all airports within range
        for airport in self.navdb.airports.values():
            if airport.ident == origin.ident:
                continue

            distance = origin.distance_to(airport)
            if distance > max_range:
                continue

            # Check if airport can handle this aircraft type
            if not self._airport_suitable(airport):
                continue

            # Check for ground services (fuel, parking, etc.)
            if not self._has_ground_services(airport):
                continue

            candidates.append((airport, distance))

        # Sort by distance (prefer medium-range flights for variety)
        candidates.sort(key=lambda x: abs(x[1] - max_range * 0.5))
        return candidates[:20]  # Limit to top 20

    def _airport_suitable(self, airport: Airport) -> bool:
        """Check if airport can handle the aircraft type."""
        type_code = self.state.aircraft_type

        # Check runway length
        min_length = self._min_runway_length_for_type(type_code)
        has_long_runway = any(
            rw.length_ft >= min_length for rw in self.navdb.runways(airport.ident)
        )
        if not has_long_runway:
            return False

        # Check if airport has appropriate facilities
        # Large aircraft need major airports
        if type_code in ("A388", "B744", "B748", "B77W", "B77L", "B77X", "A35K"):
            # Major airports only - check for multiple runways or long runways
            runways = self.navdb.runways(airport.ident)
            if len(runways) < 2 and not any(rw.length_ft >= 10000 for rw in runways):
                return False

        return True

    def _has_ground_services(self, airport: Airport) -> bool:
        """Check if airport has ground services (fuel, parking, handling)."""
        # In the sim, most airports with runways have basic services
        # Check for FBO/fuel availability via airport type
        if airport.type in ("large_airport", "medium_airport"):
            return True
        # Small airports might work for light aircraft
        if self.state.aircraft_type in ("C152", "C172", "C182", "PA28", "DA40", "SR22"):
            return airport.type in ("small_airport", "medium_airport", "large_airport")
        return False

    def _min_runway_length_for_type(self, type_code: str) -> int:
        """Minimum runway length in feet for aircraft type."""
        # Light aircraft
        if type_code in ("C152", "C172", "C182", "PA28", "DA40", "SR22", "BE36", "TB20"):
            return 2500
        # Turboprops / small jets
        if type_code in ("TBM9", "PC12", "C25A", "C25C", "C525", "E55P", "E50P",
                         "AT76", "DH8D", "C208"):
            return 4000
        # Regional jets / business jets
        if type_code in ("CRJ7", "CRJ9", "E170", "E175", "E190", "E195",
                         "F2TH", "H25B", "LJ45", "LJ60", "C680", "C750", "G280"):
            return 5500
        # Narrowbody airliners
        if type_code.startswith(("A3", "B73", "B38", "B39", "B75")):
            return 7000
        # Widebody
        if type_code in ("A332", "A333", "A339", "A343", "A346", "A359", "A35K", "A388",
                         "B763", "B764", "B772", "B77W", "B77L", "B77X", "B788", "B789", "B78X",
                         "B744", "B748"):
            return 9000
        # Default
        return 6000

    def _pick_destination(self, candidates: list[tuple[Airport, float]]) -> Optional[Airport]:
        """Pick a destination from candidates with some randomness."""
        if not candidates:
            return None

        # Weight by inverse distance from ideal range (prefer medium-haul)
        max_range = self.state.aircraft_range_nm * 0.8
        weights = []
        for _, dist in candidates:
            # Prefer flights around 40-60% of max range
            ideal = max_range * 0.5
            diff = abs(dist - ideal) / max_range
            weight = 1.0 / (1.0 + diff * 2.0)
            weights.append(weight)

        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]

        # Pick randomly with weights
        return random.choices([c[0] for c in candidates], weights=weights, k=1)[0]

    def _create_flight_plan(self, origin: Airport, destination: Airport) -> dict:
        """Create a flight plan dictionary for the leg."""
        distance = origin.distance_to(destination)

        # Determine cruise altitude based on distance and aircraft type
        if distance < 200:
            cruise_ft = 15000
        elif distance < 500:
            cruise_ft = 25000
        elif distance < 1000:
            cruise_ft = 35000
        else:
            cruise_ft = 39000

        # Adjust for aircraft type
        if self.state.aircraft_type in ("C152", "C172", "C182", "PA28", "DA40", "SR22"):
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
            "ifr": True,  # Default to IFR for autonomous
            "aircraft_type": self.state.aircraft_type,
        }

    def _load_next_flight(self, flight_plan: dict) -> None:
        """Load the next flight in the chain."""
        # Update config with new flight plan
        self.config.flight.departure = flight_plan["origin"]
        self.config.flight.destination = flight_plan["destination"]
        self.config.flight.cruise_altitude_ft = flight_plan["cruise_altitude_ft"]
        self.config.flight.ifr = flight_plan.get("ifr", True)

        # Update session
        self.session.destination = self.navdb.airport(flight_plan["destination"])
        self.session.cruise_altitude_ft = flight_plan["cruise_altitude_ft"]
        self.session.ifr = flight_plan.get("ifr", True)
        self.session.phase = Phase.PREFLIGHT

        # Reset clearance state
        self.session.clearance_issued = False
        self.session.clearance_read_back = False
        self.state.has_ifr_clearance = False
        self.state.clearance_read_back = False
        self.state.checklists_completed.clear()

        # Rebuild route
        from .route import from_ofp
        # Create a minimal OFP-like object
        class SimplePlan:
            def __init__(self, fp):
                self.origin = fp["origin"]
                self.destination = fp["destination"]
                self.cruise_altitude_ft = fp["cruise_altitude_ft"]
                self.route = f"DCT {fp['destination']}"
                self.sid = ""
                self.star = ""
                self.arrival_runway = ""
                self.planned_runway = ""
                self.ifr = fp.get("ifr", True)

        plan = SimplePlan(flight_plan)
        self.use_route(from_ofp(plan))

        self._emit("info", f"Next flight: {flight_plan['origin']} -> {flight_plan['destination']}",
                   "autonomous.chain.next")
        """Error state - disable autonomous flight."""
        self._emit("error", f"Autonomous flight error: {self.state.error_message}",
                   "autonomous.error")
        self.disable()

    # Helper methods

    def _transition(self, new_phase: AutonomousPhase) -> None:
        """Transition to a new autonomous phase."""
        log.info("Autonomous phase: %s -> %s", self.state.phase.value, new_phase.value)
        self.state.previous_phase = self.state.phase
        self.state.phase = new_phase
        self.state.phase_entered_at = time.monotonic()
        self._emit("info", f"Autonomous: {new_phase.value}", f"autonomous.phase.{new_phase.value}")

    def _set_error(self, message: str) -> None:
        """Set error state."""
        self.state.error_message = message
        self.state.error_count += 1
        self._transition(AutonomousPhase.ERROR)

    def _emit(self, kind: str, text: str, key: str = "") -> None:
        """Emit an event."""
        self.on_event(kind, text, "Autonomous", key=key)

    def _ensure_systems_on(self, state) -> None:
        """Ensure battery, avionics, and essential systems are on."""
        if not getattr(state, "battery_master", False):
            self._send_event(b"ELECTRICAL_MASTER_BATTERY", 1)
        if not getattr(state, "avionics_master", False):
            self._send_event(b"AVIONICS_MASTER_SWITCH", 1)

    def _start_engines(self, state) -> None:
        """Start engines using starter events."""
        # This is aircraft-specific; use generic starter event
        if not state.engines_running:
            self._send_event(b"ENGINE_AUTO_START", 1)

    def _release_parking_brake(self) -> None:
        self._send_event(b"PARKING_BRAKES", 0)

    def _set_parking_brake(self) -> None:
        self._send_event(b"PARKING_BRAKES", 1)

    def _request_pushback(self) -> None:
        """Request pushback via GSX or sim pushback."""
        # Try GSX first
        if hasattr(self, '_gsx_pushback'):
            self._gsx_pushback()
        else:
            self._send_event(b"TOGGLE_PUSHBACK", 0)

    def _request_clearance_via_fo(self) -> None:
        """Request IFR clearance through the first officer."""
        if self.crew.radio_call:
            text, why = self.crew.radio_call("clearance")
            if text:
                self.crew.transmit(text)

    def _run_checklist_for_phase(self, checklist_name: str) -> None:
        """Trigger a checklist for the current phase."""
        if checklist_name not in self.state.checklists_completed:
            self.crew.checklists.start(checklist_name)
            self.state.checklists_completed.add(checklist_name)

    def _configure_for_takeoff(self, state) -> None:
        """Set flaps, trim, lights for takeoff."""
        # Set takeoff flaps (typically 1 or 5)
        self._send_event(b"FLAPS_SET", 1)  # Index 1 = takeoff flaps
        # Lights
        self._send_event(b"LIGHT_LANDING_TOGGLE", 1)
        self._send_event(b"LIGHT_STROBE_TOGGLE", 1)

    def _set_takeoff_power(self) -> None:
        """Set takeoff thrust."""
        self._send_event(b"THROTTLE_FULL", 0)

    def _rotate(self, state) -> None:
        """Rotate at Vr."""
        # Pull back on yoke slightly
        self._send_event(b"ELEVATOR_UP", 1)

    def _engage_climb_autopilot(self, state) -> None:
        """Engage autopilot for climb."""
        self._engage_autopilot_master()
        # Set climb speed
        climb_speed = 250  # Default, will be overridden by ATC
        self._set_autopilot_speed(climb_speed)
        # Engage vertical speed or flight level change
        self._send_event(b"AP_VS_VAR_SET_ENGLISH", 1500)  # 1500 fpm climb

    def _level_off_at_cruise(self, state) -> None:
        """Level off at cruise altitude."""
        self._engage_altitude_hold()
        self._engage_nav_mode()
        self._set_autopilot_speed(self.session.cruise_altitude_ft)

    def _engage_autopilot_master(self) -> None:
        """Turn on autopilot master."""
        if not self.state.ap_master:
            self._send_event(b"AP_MASTER", 0)
            self.state.ap_master = True

    def _engage_altitude_hold(self) -> None:
        if not self.state.ap_alt_mode:
            self._send_event(b"AP_ALT_HOLD_ON", 0)
            self.state.ap_alt_mode = True

    def _engage_heading_hold(self) -> None:
        if not self.state.ap_hdg_mode:
            self._send_event(b"AP_HDG_HOLD_ON", 0)
            self.state.ap_hdg_mode = True

    def _engage_nav_mode(self) -> None:
        if not self.state.ap_nav_mode:
            self._send_event(b"AP_NAV1_HOLD_ON", 0)
            self.state.ap_nav_mode = True

    def _engage_approach_mode(self) -> None:
        if not self.state.ap_apr_mode:
            self._send_event(b"AP_APR_HOLD_ON", 0)
            self.state.ap_apr_mode = True

    def _set_autopilot_altitude(self, altitude_ft: float) -> None:
        """Set autopilot altitude target."""
        # Use AP_ALT_VAR_SET_ENGLISH for feet
        self._send_event(b"AP_ALT_VAR_SET_ENGLISH", int(altitude_ft))

    def _set_autopilot_heading(self, heading: float) -> None:
        """Set autopilot heading target."""
        self._engage_heading_hold()
        self._send_event(b"AP_HDG_VAR_SET", int(heading))

    def _set_autopilot_speed(self, speed_kt: float) -> None:
        """Set autopilot speed target."""
        self._send_event(b"AP_AIRSPEED_SET", 1)  # Turn on
        self._send_event(b"AP_SPD_VAR_SET", int(speed_kt))

    def _configure_for_approach(self, state) -> None:
        """Configure aircraft for approach."""
        # Flaps approach
        self._send_event(b"FLAPS_SET", 2)  # Approach flaps
        # Gear down
        if not state.gear_down:
            self._send_event(b"GEAR_DOWN", 0)
        # Approach speed
        self._set_autopilot_speed(140)

    def _shutdown_engines(self) -> None:
        """Shutdown engines."""
        self._send_event(b"ENGINE_AUTO_SHUTDOWN", 1)

    def _load_next_flight(self, flight_plan: dict) -> None:
        """Load the next flight in the chain."""
        # This would load a new flight plan via SimBrief or .pln file
        log.info("Loading next flight in chain: %s", flight_plan)

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
) -> Optional[AutonomousFlight]:
    """Factory function to create AutonomousFlight if enabled."""
    if config.immersion.autonomous_flight:
        return AutonomousFlight(
            config, navdb, sim_source, session, crew, brain, get_state, on_event
        )
    return None


__all__ = ["AutonomousFlight", "AutonomousPhase", "AutonomousState", "create_autonomous_flight"]