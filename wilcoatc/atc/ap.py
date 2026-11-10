# -*- coding: utf-8 -*-
"""The First Officer's hands: every command the aeroplane is sent.

One place, so the phase machine in :mod:`wilcoatc.atc.autonomous` never has
to know which of the three ways a command can reach the sim:

* **Events.** The SDK's key events (``AP_HDG_HOLD_ON``, ``GEAR_DOWN``,
  ``THROTTLE_IDLE`` ...), mapped once each and sent as a DWORD. What the
  autopilot's mode knobs and the aircraft's switches are.

* **Variables.** ``SimConnect_SetDataOnSimObject`` through
  :meth:`SimConnectSource.set_variable`, for what no event expresses: the
  throttle lever's per-cent, a control surface's position where an event
  would only step it.

* **Axes.** The ``*_SET`` control events (``ELEVATOR_SET``,
  ``AILERON_SET``, ``RUDDER_SET``), which take a position in the
  simulator's own -16383..16384 scale -- what the FO's hand-flying needs
  for the moments the autopilot cannot fly: rotation, the flare, the
  go-around's initial pitch.

Two autopilot styles, as ``immersion.autonomous_ap_mode`` sets them:

* ``managed`` -- what an airline flies: flight-level change for the climb
  and the descent (``FLIGHT_LEVEL_CHANGE_ON``), NAV for the route, APR for
  the approach, and the autothrottle where the model has one
  (``AUTO_THROTTLE_TO_GA`` for the go-around's power).

* ``direct`` -- what a light aeroplane's steam gauge AP does: heading
  hold, altitude hold, vertical speed, indicated airspeed, each set
  explicitly.

Every event name goes through :meth:`SimConnectSource._send`, which maps
it once and answers False for a name the simulator does not have, so an
aircraft or sim that lacks a mode simply keeps the previous one rather
than throwing mid-flight. Nothing here reads the state; the caller does,
and asks again next tick.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# The simulator's own control scale: -16383 full negative, +16384 full
# positive, 0 centred. ELEVATOR_SET's -16K is full down, so a positive
# value here is a pull.
AXIS_MIN = -16383
AXIS_MAX = 16384


def _axis(position: float) -> int:
    """A -1..1 control position in the simulator's axis units."""
    return int(round(max(-1.0, min(1.0, position)) * AXIS_MAX))


class FlightDirector:
    """Sends the aeroplane where it is told, by the best means it has."""

    def __init__(self, sim, mode: str = "managed"):
        self.sim = sim
        self.mode = "managed" if mode == "managed" else "direct"
        # The logical modes already asked for, so a 1 Hz loop does not
        # resend what the sim already has. Cleared by ``reset`` between
        # legs; individual modes are also dropped when disengaged.
        self._sent: set[str] = set()

    # -- plumbing ---------------------------------------------------------

    def reset(self) -> None:
        """A new leg: nothing is assumed engaged until it is asked for."""
        self._sent.clear()

    def send(self, name: bytes, value: int = 0) -> bool:
        """One event, through the source's map-once-and-probe path."""
        if self.sim is None or not getattr(self.sim, "connected", False):
            return False
        try:
            return bool(self.sim._send(name, value))
        except Exception as exc:
            log.debug("sim event %s failed: %s", name, exc)
            return False

    def once(self, key: str, name: bytes, value: int = 0) -> bool:
        """An event sent at most once until ``reset`` or ``forget``."""
        if key in self._sent:
            return True
        if self.send(name, value):
            self._sent.add(key)
            return True
        return False

    def forget(self, *keys: str) -> None:
        for key in keys:
            self._sent.discard(key)

    def write(self, variable: str, value: float) -> bool:
        """A sim variable, where the source can write one."""
        if self.sim is None or not hasattr(self.sim, "set_variable"):
            return False
        try:
            return bool(self.sim.set_variable(variable, value))
        except Exception as exc:
            log.debug("simvar %s failed: %s", variable, exc)
            return False

    # -- the master --------------------------------------------------------

    def autopilot(self, on: bool = True) -> bool:
        return self.send(b"AP_MASTER", 1 if on else 0)

    def flight_director(self, on: bool = True) -> bool:
        return self.send(b"TOGGLE_FLIGHT_DIRECTOR", 0)

    def yaw_damper(self, on: bool = True) -> bool:
        return self.send(b"YAW_DAMPER_ON" if on else b"YAW_DAMPER_OFF", 0)

    # -- lateral -----------------------------------------------------------

    def heading(self, degrees: float) -> bool:
        """Command a heading through the heading bug and HDG hold.

        ``AP_HEADING_BUG_SET_EX1`` is the MSFS 2024 setter: the parameter
        is the heading scaled by 16384/360 with the slot index in [1],
        which this single-DWORD send leaves at its default. The older
        ``AP_HDG_VAR_SET`` name is tried after it for sims that still map
        it, and ``HEADING_BUG_SET`` last of all.
        """
        bug = int(round((degrees % 360.0) * AXIS_MAX / 360.0))
        ok = self.send(b"AP_HEADING_BUG_SET_EX1", bug)
        if not ok:
            ok = self.send(b"AP_HDG_VAR_SET", int(round(degrees)) % 360)
        if not ok:
            ok = self.send(b"HEADING_BUG_SET", int(round(degrees)) % 360)
        self.send(b"AP_HDG_HOLD_ON", 0)
        return ok

    def nav(self, on: bool = True) -> bool:
        """NAV1 lateral guidance: the flight plan, legs and all."""
        return self.send(b"AP_NAV1_HOLD_ON" if on else b"AP_NAV1_HOLD_OFF", 0)

    def nav_source(self, index: int = 1) -> bool:
        """Which NAV receiver the holds follow (1 or 2)."""
        return self.send(b"AP_NAV_SELECT_SET", int(index))

    def bank_hold(self, on: bool = True) -> bool:
        return self.send(b"AP_BANK_HOLD_ON" if on else b"AP_BANK_HOLD_OFF", 0)

    # -- vertical ----------------------------------------------------------

    def altitude(self, feet: float) -> bool:
        """The altitude bug, in feet (``AP_ALT_VAR_SET_ENGLISH``)."""
        self.send(b"AP_ALT_VAR_SET_ENGLISH", int(round(feet)))
        return True

    def alt_hold(self, on: bool = True) -> bool:
        return self.send(b"AP_ALT_HOLD_ON" if on else b"AP_ALT_HOLD_OFF", 0)

    def vertical_speed(self, fpm: float) -> bool:
        """A vertical-speed command, feet a minute."""
        self.send(b"AP_VS_VAR_SET_ENGLISH", int(round(fpm)))
        return True

    def vs_hold(self, on: bool = True) -> bool:
        return self.send(b"AP_VS_ON" if on else b"AP_VS_OFF", 0)

    def level_change(self, on: bool = True) -> bool:
        """The managed climb/descent: power for the profile, airspeed held."""
        name = (b"FLIGHT_LEVEL_CHANGE_ON" if on
                else b"FLIGHT_LEVEL_CHANGE_OFF")
        return self.send(name, 0)

    def att_hold(self, on: bool = True) -> bool:
        """Wing leveler with pitch hold: wings level, attitude kept."""
        return self.send(b"AP_ATT_HOLD_ON" if on else b"AP_ATT_HOLD_OFF", 0)

    def wing_leveler(self, on: bool = True) -> bool:
        return self.send(b"AP_WING_LEVELER_ON" if on else b"AP_WING_LEVELER_OFF", 0)

    def pitch_leveler(self, on: bool = True) -> bool:
        return self.send(b"AP_PITCH_LEVELER_ON" if on else b"AP_PITCH_LEVELER_OFF", 0)

    def radio_altitude(self, on: bool = True) -> bool:
        return self.send(b"AP_ALT_RADIO_MODE_ON" if on else b"AP_ALT_RADIO_MODE_OFF", 0)

    # -- speed -------------------------------------------------------------

    def speed(self, knots: float) -> bool:
        """Indicated airspeed bug. ``AP_SPD_VAR_SET``'s parameter is the
        speed in hundredths of a knot, so 140 kt is 14000 -- sending the
        bare number commands one-point-four knots, which is what the old
        code did."""
        self.send(b"AP_SPD_VAR_SET", int(round(knots * 100.0)))
        return True

    def airspeed_hold(self, on: bool = True) -> bool:
        return self.send(b"AP_AIRSPEED_ON" if on else b"AP_AIRSPEED_OFF", 0)

    def mach_hold(self, on: bool = True) -> bool:
        return self.send(b"AP_MACH_HOLD_ON" if on else b"AP_MACH_HOLD_OFF", 0)

    # -- approach ----------------------------------------------------------

    def localizer(self, on: bool = True) -> bool:
        return self.send(b"AP_LOC_HOLD_ON" if on else b"AP_LOC_HOLD_OFF", 0)

    def approach(self, on: bool = True) -> bool:
        """LOC and glideslope together, armed or engaged."""
        return self.send(b"AP_APR_HOLD_ON" if on else b"AP_APR_HOLD_OFF", 0)

    # -- takeoff and go-around power ---------------------------------------

    def toga(self, on: bool = True) -> bool:
        """Autothrottle takeoff/go-around mode where the model has one."""
        return self.send(b"AUTO_THROTTLE_TO_GA", 0)

    def go_around(self) -> bool:
        """The TO/GA press itself, where the model has the switch: the
        autopilot's go-around mode and the autothrottle's, together."""
        ok = self.send(b"AUTOPILOT_TO_GA", 0)
        self.send(b"AUTO_THROTTLE_TO_GA", 0)
        return ok

    def throttle(self, position: float) -> bool:
        """The thrust levers, 0..1 of travel: 0 idle, 1 takeoff. The
        throttle lever variable first, the axis event after it, so a
        model that takes the variable does not also get the event's
        step."""
        position = max(0.0, min(1.0, position))
        if self.write("GENERAL ENG THROTTLE LEVER POSITION:1",
                      position * 100.0):
            return True
        return self.send(b"AXIS_THROTTLE_SET", int(round(position * AXIS_MAX)))

    def throttle_idle(self) -> bool:
        return self.send(b"THROTTLE_IDLE", 0)

    def throttle_cut(self) -> bool:
        return self.send(b"THROTTLE_CUT", 0)

    def takeoff_power(self) -> bool:
        return self.send(b"THROTTLE_FULL", 0)

    # -- flight controls, for the moments the AP cannot fly -----------------

    def aileron(self, position: float) -> bool:
        """-1 full left .. +1 full right."""
        return self.send(b"AILERON_SET", _axis(position))

    def elevator(self, position: float) -> bool:
        """-1 full down .. +1 full up (a pull is positive)."""
        return self.send(b"ELEVATOR_SET", _axis(position))

    def rudder(self, position: float) -> bool:
        """-1 full left .. +1 full right; ignored where gyro controls
        are enabled, which is the pilot's own business then."""
        return self.send(b"RUDDER_SET", _axis(position))

    def center_controls(self) -> bool:
        ok = self.send(b"CENTER_AILER_RUDDER", 0)
        self.send(b"ELEVATOR_SET", 0)
        return ok

    # -- configuration ------------------------------------------------------

    def flaps(self, position: float) -> bool:
        """Flap handle, 0 retracted .. 1 full, nearest detent."""
        return self.send(b"AXIS_FLAPS_SET", int(round(position * AXIS_MAX)))

    def gear(self, down: bool) -> bool:
        return self.send(b"GEAR_DOWN" if down else b"GEAR_UP", 0)

    def gear_set(self, position: float) -> bool:
        return self.send(b"GEAR_SET", int(round(position * AXIS_MAX)))

    def spoilers(self, position: float) -> bool:
        """Speedbrake handle, 0 up .. 1 down."""
        return self.send(b"SPOILERS_SET", int(round(position * AXIS_MAX)))

    def spoilers_arm(self, on: bool = True) -> bool:
        return self.send(b"SPOILERS_ARM_ON" if on else b"SPOILERS_ARM_OFF", 0)

    # -- ground --------------------------------------------------------------

    def parking_brake(self, on: bool) -> bool:
        name = b"PARKING_BRAKES_ON" if on else b"PARKING_BRAKES_OFF"
        return self.send(name, 0)

    def pushback(self) -> bool:
        return self.send(b"TOGGLE_PUSHBACK", 0)

    # -- electrics and switches ------------------------------------------------

    def battery(self, on: bool = True) -> bool:
        return self.send(b"BATTERY1_SET", 1 if on else 0)

    def avionics(self, on: bool = True) -> bool:
        return self.send(b"AVIONICS_MASTER_SET", 1 if on else 0)

    def pitot_heat(self, on: bool = True) -> bool:
        return self.send(b"PITOT_HEAT_ON" if on else b"PITOT_HEAT_OFF", 0)

    def seatbelts(self, on: bool = True) -> bool:
        return self.send(b"CABIN_SEATBELTS_ALERT_SWITCH_TOGGLE", 0)

    def beacon(self, on: bool = True) -> bool:
        return self.send(b"BEACON_LIGHTS_ON" if on else b"BEACON_LIGHTS_OFF", 0)

    def nav_lights(self, on: bool = True) -> bool:
        return self.send(b"NAV_LIGHTS_ON" if on else b"NAV_LIGHTS_OFF", 0)

    def strobes(self, on: bool = True) -> bool:
        return self.send(b"STROBES_ON" if on else b"STROBES_OFF", 0)

    def taxi_light(self, on: bool = True) -> bool:
        return self.send(b"TAXI_LIGHTS_ON" if on else b"TAXI_LIGHTS_OFF", 0)

    def landing_light(self, on: bool = True) -> bool:
        return self.send(b"LANDING_LIGHTS_ON" if on else b"LANDING_LIGHTS_OFF", 0)

    def transponder(self, code: str) -> bool:
        """Squawk through the source's own BCD packing."""
        if hasattr(self.sim, "set_squawk"):
            try:
                return bool(self.sim.set_squawk(code))
            except Exception:
                return False
        return False

    def kohlsman(self, hpa: float) -> bool:
        """The altimeter setting. ``KOHLSMAN_SET``'s parameter is
        hectopascals where the sim defines it that way; where it does
        not the write simply fails and the pilot's own setting stands."""
        return self.send(b"KOHLSMAN_SET", int(round(hpa)))

    def starter(self, on: bool = True) -> bool:
        return self.send(b"STARTER1_SET", 1 if on else 0)

    def all_starters(self, on: bool = True) -> bool:
        return self.send(b"TOGGLE_ALL_STARTERS", 0)


__all__ = ["FlightDirector", "AXIS_MAX", "AXIS_MIN"]
