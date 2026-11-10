# -*- coding: utf-8 -*-
"""Per-type performance for the *invented* traffic, from the EUROCONTROL
Aircraft Performance Database (ATCPFDB).

The traffic on the frequency used to be three aeroplanes: an airliner, a
turboprop and a light single, each a class average (:mod:`.flying`). An A320
and an A380 climbed at the same rate, a CRJ9 descended like a 737 and every
airliner rotated within a few knots of every other. The database states the
figures per type -- takeoff distance and V2, initial-climb rate and speed,
cruise speed, ceiling, still-air range, descent rate, approach and landing
speeds -- for 398 types, and nearly every code WilcoATC's fleets carry is in
it.

Two rules keep it honest:

*It is indicative.* The database itself says "all data presented is only
indicative and should not be used operationally", and its copyright belongs
to EUROCONTROL. What is stored here is the small set of figures the traffic
model actually uses, for the types the injector can spawn -- not the
database. Each row names the code it was read from, because several WilcoATC
codes are aliases of one entry (``AT76`` is read from ``AT72``, ``E75L`` from
``E170``, ``TBM9`` from ``TBM8``).

*It is for invented traffic only.* The first officer flying the pilot's own
aeroplane does not read this table: the model's own declared speeds and the
simulator's variables are better evidence there (``vspeeds``,
``sim.source``). This file exists so the aeroplane somebody else is flying
behaves like the type it says it is.

What is not in the database stays the class's: turn rate, pattern altitude,
taxi and braking figures, and the takeoff-roll shape. ``performance_for``
starts from :func:`~wilcoatc.atc.flying.performance_for` and replaces only
the fields a row actually states.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import flying as fly
from .flying import Performance


@dataclass(frozen=True)
class TypePerf:
    """One type's figures, as ATCPFDB states them.

    ``db`` is the database's own ICAO code -- different from the WilcoATC
    code where the two disagree. Speeds are knots, rates feet a minute,
    distances metres and nautical miles, ceiling feet. Zeros are "no data".
    """

    db: str
    v2_kt: float = 0.0
    takeoff_m: float = 0.0
    climb_ias_kt: float = 0.0       # initial climb to 5,000 ft
    climb_roc_fpm: float = 0.0
    climb150_ias_kt: float = 0.0    # the enroute climb speed at FL150
    cruise_tas_kt: float = 0.0
    ceiling_ft: float = 0.0
    range_nm: float = 0.0           # still-air range at long-range cruise
    descent_rod_fpm: float = 0.0    # descent to FL100
    approach_ias_kt: float = 0.0
    vat_kt: float = 0.0             # landing threshold speed
    wtc: str = ""                   # wake turbulence category
    recat: str = ""                 # RECAT-EU


# The rotorcraft codes: their "landing Vat" is a hover-ish figure, not a
# threshold speed, so the approach speed is read from the approach IAS.
HELI_TYPES: frozenset[str] = frozenset({
    "R44", "R66", "H125", "AS50", "EC30", "B06", "B407", "EC35", "EC45",
    "S76", "A109", "A139", "B412", "UH1",
})

# The figures, keyed by the type code the traffic flies. ``db`` names the
# ATCPFDB entry each row was read from; where it differs from the key the
# row is an alias and the difference is the database's, not ours.
TABLE: dict[str, TypePerf] = {
    "A320": TypePerf(db="A320", v2_kt=145, takeoff_m=2190, climb_ias_kt=175, climb_roc_fpm=2500, climb150_ias_kt=290, cruise_tas_kt=450, ceiling_ft=39000, range_nm=2700, descent_rod_fpm=3500, approach_ias_kt=250, vat_kt=137, wtc="M", recat="Upper Medium"),
    "A321": TypePerf(db="A321", v2_kt=145, takeoff_m=2210, climb_ias_kt=175, climb_roc_fpm=2500, climb150_ias_kt=290, cruise_tas_kt=450, ceiling_ft=41000, range_nm=2350, descent_rod_fpm=2500, approach_ias_kt=225, vat_kt=141, wtc="M", recat="Upper Medium"),
    "A319": TypePerf(db="A319", v2_kt=135, takeoff_m=1750, climb_ias_kt=165, climb_roc_fpm=2500, climb150_ias_kt=290, cruise_tas_kt=450, ceiling_ft=39000, range_nm=1800, descent_rod_fpm=3500, approach_ias_kt=230, vat_kt=130, wtc="M", recat="Upper Medium"),
    "B738": TypePerf(db="B738", v2_kt=145, takeoff_m=2300, climb_ias_kt=165, climb_roc_fpm=3000, climb150_ias_kt=290, cruise_tas_kt=460, ceiling_ft=41000, range_nm=2000, descent_rod_fpm=3500, approach_ias_kt=250, vat_kt=147, wtc="M", recat="Upper Medium"),
    "B38M": TypePerf(db="B38M", v2_kt=145, takeoff_m=2500, climb_ias_kt=165, climb_roc_fpm=2500, climb150_ias_kt=290, cruise_tas_kt=453, ceiling_ft=41000, range_nm=3550, descent_rod_fpm=3500, approach_ias_kt=250, vat_kt=145, wtc="M", recat="Upper Medium"),
    "B37M": TypePerf(db="B37M", v2_kt=145, takeoff_m=2100, climb_ias_kt=165, climb_roc_fpm=2800, climb150_ias_kt=290, cruise_tas_kt=453, ceiling_ft=41000, range_nm=3850, descent_rod_fpm=3500, approach_ias_kt=250, vat_kt=145, wtc="M", recat="Upper Medium"),
    "A20N": TypePerf(db="A20N", v2_kt=145, takeoff_m=1951, climb_ias_kt=175, climb_roc_fpm=2200, climb150_ias_kt=290, cruise_tas_kt=450, ceiling_ft=39000, range_nm=3500, descent_rod_fpm=3000, approach_ias_kt=250, vat_kt=135, wtc="M", recat="Upper Medium"),
    "A21N": TypePerf(db="A21N", v2_kt=145, takeoff_m=2150, climb_ias_kt=175, climb_roc_fpm=2000, climb150_ias_kt=290, cruise_tas_kt=450, ceiling_ft=39000, range_nm=4000, descent_rod_fpm=2500, approach_ias_kt=250, vat_kt=140, wtc="M", recat="Upper Medium"),
    "E190": TypePerf(db="E190", v2_kt=138, takeoff_m=2056, climb_ias_kt=190, climb_roc_fpm=3400, climb150_ias_kt=300, cruise_tas_kt=460, ceiling_ft=41000, range_nm=2400, descent_rod_fpm=2200, approach_ias_kt=230, vat_kt=131, wtc="M", recat="Lower Medium"),
    "E195": TypePerf(db="E195", v2_kt=140, takeoff_m=1432, climb_ias_kt=190, climb_roc_fpm=3000, climb150_ias_kt=300, cruise_tas_kt=447, ceiling_ft=41000, range_nm=2300, descent_rod_fpm=2500, approach_ias_kt=230, vat_kt=135, wtc="M", recat="Lower Medium"),
    "E75L": TypePerf(db="E170", v2_kt=138, takeoff_m=1644, climb_ias_kt=190, climb_roc_fpm=3400, climb150_ias_kt=300, cruise_tas_kt=460, ceiling_ft=41000, range_nm=2100, descent_rod_fpm=2600, approach_ias_kt=250, vat_kt=130, wtc="M", recat="Lower Medium"),
    "CRJ9": TypePerf(db="CRJ9", v2_kt=140, takeoff_m=1770, climb_ias_kt=165, climb_roc_fpm=2500, climb150_ias_kt=290, cruise_tas_kt=450, ceiling_ft=41000, range_nm=1550, descent_rod_fpm=3500, approach_ias_kt=250, vat_kt=135, wtc="M", recat="Lower Medium"),
    "AT76": TypePerf(db="AT72", v2_kt=110, takeoff_m=1500, climb_ias_kt=140, climb_roc_fpm=1500, climb150_ias_kt=210, cruise_tas_kt=275, ceiling_ft=25000, range_nm=1500, descent_rod_fpm=1500, approach_ias_kt=200, vat_kt=120, wtc="M", recat="Lower Medium"),
    "DH8D": TypePerf(db="DH8D", v2_kt=115, takeoff_m=1300, climb_ias_kt=150, climb_roc_fpm=3500, climb150_ias_kt=210, cruise_tas_kt=360, ceiling_ft=25000, range_nm=1300, descent_rod_fpm=1500, approach_ias_kt=245, vat_kt=121, wtc="M", recat="Lower Medium"),
    "B77W": TypePerf(db="B77W", v2_kt=168, takeoff_m=3000, climb_ias_kt=200, climb_roc_fpm=3000, climb150_ias_kt=300, cruise_tas_kt=490, ceiling_ft=43000, range_nm=7930, descent_rod_fpm=3000, approach_ias_kt=240, vat_kt=149, wtc="H", recat="Upper Heavy"),
    "A359": TypePerf(db="A359", v2_kt=150, takeoff_m=888, climb_ias_kt=220, climb_roc_fpm=3000, climb150_ias_kt=300, cruise_tas_kt=490, ceiling_ft=43000, range_nm=8100, descent_rod_fpm=3000, approach_ias_kt=240, vat_kt=140, wtc="H", recat="Upper Heavy"),
    "B789": TypePerf(db="B789", v2_kt=165, takeoff_m=2800, climb_ias_kt=190, climb_roc_fpm=2700, climb150_ias_kt=290, cruise_tas_kt=490, ceiling_ft=43000, range_nm=7565, descent_rod_fpm=2500, approach_ias_kt=240, vat_kt=150, wtc="H", recat="Upper Heavy"),
    "A333": TypePerf(db="A333", v2_kt=145, takeoff_m=2300, climb_ias_kt=175, climb_roc_fpm=2000, climb150_ias_kt=290, cruise_tas_kt=475, ceiling_ft=41000, range_nm=5650, descent_rod_fpm=3000, approach_ias_kt=220, vat_kt=140, wtc="H", recat="Upper Heavy"),
    "B78X": TypePerf(db="B78X", v2_kt=180, takeoff_m=2800, climb_ias_kt=190, climb_roc_fpm=2700, climb150_ias_kt=290, cruise_tas_kt=488, ceiling_ft=41000, range_nm=6430, descent_rod_fpm=2500, approach_ias_kt=240, vat_kt=150, wtc="H", recat="Upper Heavy"),
    "A388": TypePerf(db="A388", v2_kt=150, takeoff_m=2970, climb_ias_kt=190, climb_roc_fpm=1500, climb150_ias_kt=320, cruise_tas_kt=520, ceiling_ft=43000, range_nm=8000, descent_rod_fpm=2000, approach_ias_kt=250, vat_kt=140, wtc="J", recat="Super Heavy"),
    "B763": TypePerf(db="B763", v2_kt=160, takeoff_m=2900, climb_ias_kt=190, climb_roc_fpm=3000, climb150_ias_kt=290, cruise_tas_kt=460, ceiling_ft=45000, range_nm=6105, descent_rod_fpm=3000, approach_ias_kt=230, vat_kt=140, wtc="H", recat="Lower Heavy"),
    "B77L": TypePerf(db="B77L", v2_kt=123, takeoff_m=2970, climb_ias_kt=175, climb_roc_fpm=2500, climb150_ias_kt=280, cruise_tas_kt=490, ceiling_ft=43100, range_nm=9380, descent_rod_fpm=3000, approach_ias_kt=210, vat_kt=140, wtc="H", recat="Upper Heavy"),
    "A306": TypePerf(db="A306", v2_kt=160, takeoff_m=2240, climb_ias_kt=190, climb_roc_fpm=3000, climb150_ias_kt=290, cruise_tas_kt=470, ceiling_ft=37000, range_nm=4150, descent_rod_fpm=2000, approach_ias_kt=240, vat_kt=139, wtc="H", recat="Lower Heavy"),
    "B752": TypePerf(db="B752", v2_kt=145, takeoff_m=1900, climb_ias_kt=175, climb_roc_fpm=3500, climb150_ias_kt=300, cruise_tas_kt=470, ceiling_ft=42000, range_nm=3900, descent_rod_fpm=3500, approach_ias_kt=220, vat_kt=130, wtc="M", recat="Lower Heavy"),
    "C172": TypePerf(db="C172", v2_kt=60, takeoff_m=300, climb_ias_kt=90, climb_roc_fpm=400, climb150_ias_kt=0, cruise_tas_kt=115, ceiling_ft=13000, range_nm=430, descent_rod_fpm=500, approach_ias_kt=110, vat_kt=65, wtc="L", recat="Light"),
    "C152": TypePerf(db="C152", v2_kt=60, takeoff_m=220, climb_ias_kt=85, climb_roc_fpm=500, climb150_ias_kt=0, cruise_tas_kt=100, ceiling_ft=14700, range_nm=315, descent_rod_fpm=500, approach_ias_kt=80, vat_kt=55, wtc="L", recat="Light"),
    "C182": TypePerf(db="C182", v2_kt=65, takeoff_m=200, climb_ias_kt=80, climb_roc_fpm=500, climb150_ias_kt=85, cruise_tas_kt=140, ceiling_ft=18000, range_nm=550, descent_rod_fpm=500, approach_ias_kt=110, vat_kt=65, wtc="L", recat="Light"),
    "P28A": TypePerf(db="P28A", v2_kt=65, takeoff_m=300, climb_ias_kt=80, climb_roc_fpm=700, climb150_ias_kt=0, cruise_tas_kt=110, ceiling_ft=12000, range_nm=500, descent_rod_fpm=500, approach_ias_kt=110, vat_kt=65, wtc="L", recat="Light"),
    "PA28": TypePerf(db="P28A", v2_kt=65, takeoff_m=300, climb_ias_kt=80, climb_roc_fpm=700, climb150_ias_kt=0, cruise_tas_kt=110, ceiling_ft=12000, range_nm=500, descent_rod_fpm=500, approach_ias_kt=110, vat_kt=65, wtc="L", recat="Light"),
    "DA40": TypePerf(db="DA40", v2_kt=73, takeoff_m=305, climb_ias_kt=130, climb_roc_fpm=700, climb150_ias_kt=130, cruise_tas_kt=130, ceiling_ft=16000, range_nm=830, descent_rod_fpm=800, approach_ias_kt=120, vat_kt=80, wtc="L", recat="Light"),
    "SR22": TypePerf(db="SR22", v2_kt=0, takeoff_m=462, climb_ias_kt=75, climb_roc_fpm=1260, climb150_ias_kt=110, cruise_tas_kt=180, ceiling_ft=17500, range_nm=1049, descent_rod_fpm=1000, approach_ias_kt=120, vat_kt=70, wtc="L", recat="Light"),
    "BE36": TypePerf(db="BE36", v2_kt=75, takeoff_m=350, climb_ias_kt=95, climb_roc_fpm=500, climb150_ias_kt=95, cruise_tas_kt=160, ceiling_ft=18000, range_nm=850, descent_rod_fpm=500, approach_ias_kt=120, vat_kt=70, wtc="L", recat="Light"),
    "TB20": TypePerf(db="TB30", v2_kt=80, takeoff_m=400, climb_ias_kt=110, climb_roc_fpm=1800, climb150_ias_kt=150, cruise_tas_kt=190, ceiling_ft=23000, range_nm=700, descent_rod_fpm=800, approach_ias_kt=190, vat_kt=80, wtc="L", recat="Light"),
    "DR40": TypePerf(db="DR40", v2_kt=0, takeoff_m=350, climb_ias_kt=100, climb_roc_fpm=800, climb150_ias_kt=110, cruise_tas_kt=140, ceiling_ft=15000, range_nm=450, descent_rod_fpm=500, approach_ias_kt=140, vat_kt=65, wtc="L", recat="Light"),
    "DA42": TypePerf(db="DA42", v2_kt=85, takeoff_m=733, climb_ias_kt=140, climb_roc_fpm=1700, climb150_ias_kt=160, cruise_tas_kt=180, ceiling_ft=18000, range_nm=1061, descent_rod_fpm=1500, approach_ias_kt=120, vat_kt=75, wtc="L", recat="Light"),
    "PA34": TypePerf(db="PA34", v2_kt=80, takeoff_m=300, climb_ias_kt=110, climb_roc_fpm=1300, climb150_ias_kt=140, cruise_tas_kt=160, ceiling_ft=19000, range_nm=980, descent_rod_fpm=800, approach_ias_kt=140, vat_kt=80, wtc="L", recat="Light"),
    "BE58": TypePerf(db="BE58", v2_kt=100, takeoff_m=700, climb_ias_kt=130, climb_roc_fpm=1500, climb150_ias_kt=130, cruise_tas_kt=175, ceiling_ft=20000, range_nm=1000, descent_rod_fpm=800, approach_ias_kt=140, vat_kt=95, wtc="L", recat="Light"),
    "C25A": TypePerf(db="C25A", v2_kt=120, takeoff_m=1024, climb_ias_kt=145, climb_roc_fpm=3500, climb150_ias_kt=230, cruise_tas_kt=418, ceiling_ft=41000, range_nm=1781, descent_rod_fpm=2000, approach_ias_kt=220, vat_kt=110, wtc="L", recat="Light"),
    "C56X": TypePerf(db="C56X", v2_kt=115, takeoff_m=1055, climb_ias_kt=145, climb_roc_fpm=2500, climb150_ias_kt=270, cruise_tas_kt=430, ceiling_ft=45000, range_nm=2000, descent_rod_fpm=3000, approach_ias_kt=210, vat_kt=117, wtc="M", recat="Light"),
    "PC12": TypePerf(db="PC12", v2_kt=110, takeoff_m=600, climb_ias_kt=130, climb_roc_fpm=1700, climb150_ias_kt=130, cruise_tas_kt=270, ceiling_ft=30000, range_nm=2000, descent_rod_fpm=1500, approach_ias_kt=180, vat_kt=85, wtc="L", recat="Light"),
    "TBM9": TypePerf(db="TBM8", v2_kt=110, takeoff_m=870, climb_ias_kt=160, climb_roc_fpm=2400, climb150_ias_kt=180, cruise_tas_kt=300, ceiling_ft=31000, range_nm=1520, descent_rod_fpm=1500, approach_ias_kt=160, vat_kt=95, wtc="L", recat="Light"),
    "R44": TypePerf(db="R44", v2_kt=0, takeoff_m=0, climb_ias_kt=60, climb_roc_fpm=1000, climb150_ias_kt=0, cruise_tas_kt=110, ceiling_ft=14000, range_nm=300, descent_rod_fpm=1000, approach_ias_kt=80, vat_kt=15, wtc="L", recat="Light"),
    "R66": TypePerf(db="H25A", v2_kt=124, takeoff_m=1247, climb_ias_kt=140, climb_roc_fpm=3700, climb150_ias_kt=290, cruise_tas_kt=400, ceiling_ft=41000, range_nm=1680, descent_rod_fpm=2200, approach_ias_kt=210, vat_kt=125, wtc="M", recat="Light"),
    "H125": TypePerf(db="H25A", v2_kt=124, takeoff_m=1247, climb_ias_kt=140, climb_roc_fpm=3700, climb150_ias_kt=290, cruise_tas_kt=400, ceiling_ft=41000, range_nm=1680, descent_rod_fpm=2200, approach_ias_kt=210, vat_kt=125, wtc="M", recat="Light"),
    "AS50": TypePerf(db="H25C", v2_kt=125, takeoff_m=1900, climb_ias_kt=155, climb_roc_fpm=3500, climb150_ias_kt=290, cruise_tas_kt=450, ceiling_ft=43000, range_nm=3100, descent_rod_fpm=4000, approach_ias_kt=250, vat_kt=132, wtc="M", recat="Light"),
    "EC30": TypePerf(db="EC35", v2_kt=0, takeoff_m=0, climb_ias_kt=110, climb_roc_fpm=1500, climb150_ias_kt=110, cruise_tas_kt=135, ceiling_ft=20000, range_nm=340, descent_rod_fpm=1500, approach_ias_kt=90, vat_kt=5, wtc="L", recat="Light"),
    "B06": TypePerf(db="B06", v2_kt=0, takeoff_m=0, climb_ias_kt=60, climb_roc_fpm=1350, climb150_ias_kt=0, cruise_tas_kt=120, ceiling_ft=13500, range_nm=374, descent_rod_fpm=1100, approach_ias_kt=80, vat_kt=40, wtc="L", recat="Light"),
    "B407": TypePerf(db="B430", v2_kt=0, takeoff_m=0, climb_ias_kt=60, climb_roc_fpm=1400, climb150_ias_kt=0, cruise_tas_kt=130, ceiling_ft=14600, range_nm=324, descent_rod_fpm=1000, approach_ias_kt=60, vat_kt=40, wtc="L", recat="Light"),
    "EC35": TypePerf(db="EC35", v2_kt=0, takeoff_m=0, climb_ias_kt=110, climb_roc_fpm=1500, climb150_ias_kt=110, cruise_tas_kt=135, ceiling_ft=20000, range_nm=340, descent_rod_fpm=1500, approach_ias_kt=90, vat_kt=5, wtc="L", recat="Light"),
    "EC45": TypePerf(db="EC35", v2_kt=0, takeoff_m=0, climb_ias_kt=110, climb_roc_fpm=1500, climb150_ias_kt=110, cruise_tas_kt=135, ceiling_ft=20000, range_nm=340, descent_rod_fpm=1500, approach_ias_kt=90, vat_kt=5, wtc="L", recat="Light"),
    "S76": TypePerf(db="S76", v2_kt=0, takeoff_m=0, climb_ias_kt=70, climb_roc_fpm=1800, climb150_ias_kt=0, cruise_tas_kt=150, ceiling_ft=13800, range_nm=410, descent_rod_fpm=1500, approach_ias_kt=60, vat_kt=15, wtc="L", recat="Light"),
    "A109": TypePerf(db="A109", v2_kt=0, takeoff_m=0, climb_ias_kt=90, climb_roc_fpm=1900, climb150_ias_kt=100, cruise_tas_kt=140, ceiling_ft=19600, range_nm=521, descent_rod_fpm=1000, approach_ias_kt=100, vat_kt=40, wtc="L", recat="Light"),
    "A139": TypePerf(db="B412", v2_kt=0, takeoff_m=0, climb_ias_kt=80, climb_roc_fpm=1300, climb150_ias_kt=100, cruise_tas_kt=125, ceiling_ft=20000, range_nm=402, descent_rod_fpm=1000, approach_ias_kt=60, vat_kt=40, wtc="L", recat="Light"),
    "B412": TypePerf(db="B412", v2_kt=0, takeoff_m=0, climb_ias_kt=80, climb_roc_fpm=1300, climb150_ias_kt=100, cruise_tas_kt=125, ceiling_ft=20000, range_nm=402, descent_rod_fpm=1000, approach_ias_kt=60, vat_kt=40, wtc="L", recat="Light"),
    "UH1": TypePerf(db="UH1", v2_kt=0, takeoff_m=0, climb_ias_kt=60, climb_roc_fpm=1755, climb150_ias_kt=80, cruise_tas_kt=110, ceiling_ft=19000, range_nm=275, descent_rod_fpm=1200, approach_ias_kt=60, vat_kt=40, wtc="L", recat="Light"),
}

# The takeoff roll's acceleration is worked out from V2 and the takeoff
# distance rather than taken from the class: v^2 / 2d is the average the roll
# actually makes, and it is why an A320 needs half again the roll of the
# class figure. Outside this band the estimate is a database quirk, not a
# roll, and the class figure stands.
_TAKEOFF_ACCEL_MIN, _TAKEOFF_ACCEL_MAX = 0.5, 8.0


def _takeoff_accel_kt_s(perf: TypePerf) -> float:
    """Average acceleration to V2 over the published takeoff distance."""
    if perf.v2_kt <= 0 or perf.takeoff_m <= 0:
        return 0.0
    v_ms = perf.v2_kt * 0.514444
    a_ms2 = v_ms * v_ms / (2.0 * perf.takeoff_m)
    return a_ms2 / 0.514444


def traffic_performance(type_code: str, light: bool = False) -> Performance:
    """How this type flies, for the invented traffic.

    The class profile from :func:`flying.performance_for` first -- it owns
    the turn rate, the pattern and everything the database does not state --
    then the row's figures where it has them. A type the table does not
    know is exactly what it was before.
    """
    base = fly.performance_for(type_code, light=light)
    perf = TABLE.get((type_code or "").strip().upper())
    if perf is None:
        return base

    more: dict[str, float] = {}
    if perf.climb_roc_fpm > 0:
        more["climb_fpm"] = perf.climb_roc_fpm
    if perf.climb_ias_kt > 0:
        more["climb_kt"] = perf.climb_ias_kt
    if perf.climb150_ias_kt > 0:
        # The enroute climb speed is the closest the database gets to the
        # "cruise" a departure is flown at in the terminal model; the true
        # cruise TAS would have it racing at flight level, which is not what
        # this number is for.
        more["cruise_kt"] = perf.climb150_ias_kt
    if perf.descent_rod_fpm > 0:
        more["descend_fpm"] = perf.descent_rod_fpm
    # Final-approach speed: the landing Vat for a fixed-wing, the approach
    # IAS for a rotorcraft whose Vat is a near-zero figure.
    approach = perf.approach_ias_kt if perf.db and \
        (type_code or "").upper() in HELI_TYPES else perf.vat_kt
    if approach <= 0:
        approach = perf.approach_ias_kt
    if approach > 0:
        more["approach_kt"] = approach
    accel = _takeoff_accel_kt_s(perf)
    if _TAKEOFF_ACCEL_MIN <= accel <= _TAKEOFF_ACCEL_MAX:
        more["takeoff_kt_s"] = accel
    if perf.range_nm > 0:
        more["range_nm"] = perf.range_nm
    if perf.ceiling_ft > 0:
        more["ceiling_ft"] = perf.ceiling_ft
    return replace(base, **more) if more else base


def range_for(type_code: str, light: bool = False) -> float:
    """Still-air range for routing, table first, class figure after.

    Zero only where even the class cannot say -- which for this model is
    nowhere, since the class fallback covers it.
    """
    perf = type_perf(type_code)
    if perf is not None and perf.range_nm > 0:
        return perf.range_nm
    code = (type_code or "").strip().upper()
    if code in fly._LIGHT_TYPES or (light and code not in fly._TURBOPROP_TYPES):
        return 450.0
    if code in fly._TURBOPROP_TYPES:
        return 1200.0
    return 2500.0


def type_perf(type_code: str) -> TypePerf | None:
    """The row for a type, for anything else that wants the raw figures."""
    return TABLE.get((type_code or "").strip().upper())


__all__ = ["HELI_TYPES", "TABLE", "TypePerf", "range_for",
           "traffic_performance", "type_perf"]
