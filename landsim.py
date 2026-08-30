"""
LandSim Light - 1D (vertical axis) propulsive landing simulation.

Scenario
--------
A rocket falls from a given altitude (default 150 m) in free fall (with drag).
At some altitude the solid motor is ignited. The motor cannot be shut down or
regulated internally - the vehicle only has flaps ("thrust spoilers") that can
block up to 90 % of the thrust. The effective thrust is therefore

        F_eff(t) = throttle(t) * F_motor(t),   throttle in <0.1 , 1.0>

The blocked thrust is simply wasted: the propellant is consumed according to the
nominal (un-throttled) thrust curve regardless of the flap position, so the mass
history does not depend on the throttle profile and the burn time is fixed.

The throttle profile is discretised into 100 ms phases; every phase gets its own
throttle value, which the optimiser is free to choose.

The script answers: in what range of ignition altitudes can the rocket land with
a touchdown speed below 3 m/s, and what throttle profile achieves it?

Usage
-----
    python3 landsim.py                       # default case
    python3 landsim.py --drop-alt 200 --max-touchdown 3
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
#  Motor
# --------------------------------------------------------------------------- #
# Table given by the user: for every fraction of the peak thrust the time at
# which the curve rises through that level and the time at which it decays back
# through it.
#   fraction [%], level [N], rise-time [s], decay-time [s]
MOTOR_TABLE_LONG = [
    (5,   6.02,   1.4500, 4.0633),
    (10,  12.04,  1.5000, 3.9964),
    (25,  30.10,  1.5643, 3.7841),
    (50,  60.20,  1.5914, 3.5436),
    (75,  90.31,  1.7555, 3.2416),
    (90,  108.37, 1.8732, 3.0369),
    (95,  114.39, 1.9048, 2.9239),
    (100, 120.41, 2.7047, 2.7047),
]

# Alternative motor: same propellant mass, shorter and much punchier burn.
MOTOR_TABLE_SHORT = [
    (5,   13.47,  1.7980, 3.3520),
    (10,  26.94,  1.8247, 3.2804),
    (25,  67.36,  1.8720, 3.1343),
    (50,  134.72, 1.9345, 2.9153),
    (75,  202.08, 2.0091, 2.7106),
    (90,  242.50, 2.1147, 2.5615),
    (95,  255.97, 2.1752, 2.4846),
    (100, 269.44, 2.3040, 2.3150),
]

MOTOR_TABLES = {
    "long": MOTOR_TABLE_LONG,     # 120 N peak, 2.61 s burn
    "short": MOTOR_TABLE_SHORT,   # 269 N peak, 1.55 s burn
}
MOTOR_TABLE = MOTOR_TABLE_LONG    # backwards-compatible default

PROPELLANT_MASS = 0.277        # kg, consumed propellant
GROSS_MASS = 3.2               # kg, lift-off / total mass incl. propellant
DIAMETER = 0.105               # m
CD = 0.35                      # [-]
RHO = 1.225                    # kg/m3, sea-level air density
G = 9.80665                    # m/s2

AREA = math.pi * (DIAMETER / 2.0) ** 2


class Motor:
    """Thrust curve reconstructed from the fraction-of-peak table."""

    def __init__(self, table=MOTOR_TABLE, propellant_mass=PROPELLANT_MASS,
                 thrust_multiplier=1.0, name=None):
        if thrust_multiplier <= 0.0:
            raise ValueError("thrust multiplier must be > 0")
        if isinstance(table, str):
            if table not in MOTOR_TABLES:
                raise ValueError(f"unknown motor '{table}', "
                                 f"choose from {sorted(MOTOR_TABLES)}")
            name, table = table, MOTOR_TABLES[table]
        self.name = name or "custom"
        rise = [(t, f) for _, f, t, _ in table]
        decay = [(t, f) for _, f, _, t in table]
        # ascending branch, then descending branch (reversed -> increasing time)
        pts = sorted(rise) + sorted(decay, reverse=False)[::-1]
        # remove the duplicated peak point
        pts = sorted(rise) + [p for p in sorted(decay) if p[0] > sorted(rise)[-1][0]]

        t0 = pts[0][0]
        # zero thrust just before the first tabulated point and just after the last
        times = [0.0] + [t - t0 for t, _ in pts] + [pts[-1][0] - t0 + 1e-6]
        thrust = [0.0] + [f for _, f in pts] + [0.0]

        self.t = np.asarray(times, dtype=float)
        # the whole lookup table is scaled by the multiplier before anything else
        self.thrust_multiplier = float(thrust_multiplier)
        self.f = np.asarray(thrust, dtype=float) * self.thrust_multiplier
        self.burn_time = float(self.t[-1])
        self.peak_thrust = float(self.f.max())

        # total impulse and the cumulative impulse (used for the mass history)
        dt_seg = np.diff(self.t)
        seg_imp = 0.5 * (self.f[:-1] + self.f[1:]) * dt_seg
        self.cum_impulse = np.concatenate(([0.0], np.cumsum(seg_imp)))
        self.total_impulse = float(self.cum_impulse[-1])
        self.propellant_mass = propellant_mass

    def thrust(self, t):
        """Nominal (un-throttled) thrust [N] at time t after ignition."""
        return np.interp(t, self.t, self.f, left=0.0, right=0.0)

    def burned_mass(self, t):
        """Propellant burned [kg] at time t after ignition (throttle independent)."""
        imp = np.interp(t, self.t, self.cum_impulse, left=0.0,
                        right=self.total_impulse)
        return self.propellant_mass * imp / self.total_impulse


# --------------------------------------------------------------------------- #
#  Vehicle / simulation configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    drop_altitude: float = 150.0     # m, altitude where the free fall starts
    gross_mass: float = GROSS_MASS   # kg
    diameter: float = DIAMETER       # m
    cd: float = CD                   # [-]
    rho: float = RHO                 # kg/m3
    initial_velocity: float = 0.0    # m/s, positive = upwards
    max_touchdown_speed: float = 3.0 # m/s
    throttle_min: float = 0.1        # flaps block at most 90 % of the thrust
    throttle_max: float = 1.0
    phase_length: float = 0.1        # s, one throttle step
    dt: float = 0.002                # s, integration step
    max_time: float = 30.0           # s, simulation cut-off after ignition
    motor: Motor = field(default_factory=Motor)

    @staticmethod
    def with_propellant(propellant_mass: float, **kwargs) -> "Config":
        """Build a Config whose motor burns a different propellant mass."""
        return Config(motor=Motor(propellant_mass=propellant_mass), **kwargs)

    @property
    def area(self) -> float:
        return math.pi * (self.diameter / 2.0) ** 2

    @property
    def n_phases(self) -> int:
        return int(math.ceil(self.motor.burn_time / self.phase_length))


# --------------------------------------------------------------------------- #
#  Free fall down to the ignition altitude
# --------------------------------------------------------------------------- #
def free_fall_to(cfg: Config, ignition_altitude: float):
    """Integrate the drag free fall from drop_altitude to ignition_altitude.

    Returns (velocity [m/s, negative = down], elapsed time [s]).
    """
    y = cfg.drop_altitude
    v = cfg.initial_velocity
    t = 0.0
    m = cfg.gross_mass
    k = 0.5 * cfg.rho * cfg.cd * cfg.area
    dt = cfg.dt
    while y > ignition_altitude and t < 120.0:
        a = -G - k * v * abs(v) / m
        v += a * dt
        y += v * dt
        t += dt
    return v, t


# --------------------------------------------------------------------------- #
#  Powered phase - vectorised over a whole population of throttle profiles
# --------------------------------------------------------------------------- #
def simulate_population(cfg: Config, y0: float, v0: float, profiles: np.ndarray):
    """Simulate the powered descent for many throttle profiles at once.

    profiles : (N, n_phases) array of throttle values in <throttle_min, throttle_max>

    Returns a dict with, per profile:
        touchdown_speed : |v| at ground contact  (np.inf if it never lands)
        landed          : bool
        min_altitude    : lowest altitude reached
        max_altitude    : highest altitude reached after ignition
    """
    profiles = np.atleast_2d(np.asarray(profiles, dtype=float))
    n = profiles.shape[0]

    y = np.full(n, float(y0))
    v = np.full(n, float(v0))
    alive = np.ones(n, dtype=bool)
    touchdown = np.full(n, np.inf)
    min_alt = np.full(n, float(y0))
    max_alt = np.full(n, float(y0))

    k = 0.5 * cfg.rho * cfg.cd * cfg.area
    dt = cfg.dt
    dry_mass = cfg.gross_mass - cfg.motor.propellant_mass
    t = 0.0
    steps = int(cfg.max_time / dt)

    for _ in range(steps):
        if not alive.any():
            break
        thrust_nom = float(cfg.motor.thrust(t))
        mass = dry_mass + (cfg.motor.propellant_mass - float(cfg.motor.burned_mass(t)))

        if thrust_nom > 0.0:
            idx = min(int(t / cfg.phase_length), profiles.shape[1] - 1)
            thrust = thrust_nom * profiles[:, idx]
        else:
            thrust = np.zeros(n)

        drag = -k * v * np.abs(v)
        a = (thrust + drag) / mass - G

        v_new = np.where(alive, v + a * dt, v)
        y_new = np.where(alive, y + 0.5 * (v + v_new) * dt, y)

        # ground contact -> linear interpolation of the impact speed
        hit = alive & (y_new <= 0.0)
        if hit.any():
            frac = np.where(y - y_new != 0.0, y / np.maximum(y - y_new, 1e-12), 0.0)
            frac = np.clip(frac, 0.0, 1.0)
            v_hit = v + (v_new - v) * frac
            touchdown[hit] = np.abs(v_hit[hit])
            alive[hit] = False

        v, y = v_new, y_new
        min_alt = np.minimum(min_alt, np.where(alive | hit, y, min_alt))
        max_alt = np.maximum(max_alt, np.where(alive, y, max_alt))
        t += dt

    return {
        "touchdown_speed": touchdown,
        "landed": np.isfinite(touchdown),
        "min_altitude": min_alt,
        "max_altitude": max_alt,
    }


def simulate(cfg: Config, ignition_altitude: float, profile):
    """Convenience wrapper: full flight for a single throttle profile."""
    v0, t_fall = free_fall_to(cfg, ignition_altitude)
    res = simulate_population(cfg, ignition_altitude, v0, np.asarray(profile)[None, :])
    return {
        "ignition_altitude": ignition_altitude,
        "entry_speed": abs(v0),
        "fall_time": t_fall,
        "touchdown_speed": float(res["touchdown_speed"][0]),
        "landed": bool(res["landed"][0]),
        "max_altitude": float(res["max_altitude"][0]),
    }


def trajectory(cfg: Config, ignition_altitude: float, profile):
    """Single-profile run returning the full time history (for plotting/inspection)."""
    v, t_fall = free_fall_to(cfg, ignition_altitude)
    y, t = ignition_altitude, 0.0
    k = 0.5 * cfg.rho * cfg.cd * cfg.area
    dry = cfg.gross_mass - cfg.motor.propellant_mass
    hist = []
    while y > 0.0 and t < cfg.max_time:
        thrust_nom = float(cfg.motor.thrust(t))
        mass = dry + (cfg.motor.propellant_mass - float(cfg.motor.burned_mass(t)))
        idx = min(int(t / cfg.phase_length), len(profile) - 1)
        thrust = thrust_nom * (profile[idx] if thrust_nom > 0.0 else 0.0)
        a = (thrust - k * v * abs(v)) / mass - G
        hist.append((t + t_fall, y, v, thrust, mass))
        v_new = v + a * cfg.dt
        y += 0.5 * (v + v_new) * cfg.dt
        v = v_new
        t += cfg.dt
    return hist


# --------------------------------------------------------------------------- #
#  Side simulation: how high would the rocket fly if launched from the ground?
# --------------------------------------------------------------------------- #
def ascent(cfg: Config, throttle=1.0, launch_altitude=0.0):
    """Vertical ascent from rest at the ground - a side calculation, it has
    nothing to do with the landing search.

    throttle : constant throttle (1.0 = full thrust) or a sequence of per-phase
               throttle values, same convention as in the landing simulation.

    Returns a dict with the apogee, the burnout state and the peak velocity.
    """
    if np.isscalar(throttle):
        prof = np.full(cfg.n_phases, float(throttle))
    else:
        prof = np.asarray(throttle, dtype=float)

    k = 0.5 * cfg.rho * cfg.cd * cfg.area
    dry = cfg.gross_mass - cfg.motor.propellant_mass
    y, v, t = float(launch_altitude), 0.0, 0.0
    burnout = None
    v_max = 0.0
    a_max = 0.0
    liftoff_time = None

    while t < cfg.max_time * 10:
        thrust_nom = float(cfg.motor.thrust(t))
        mass = dry + (cfg.motor.propellant_mass - float(cfg.motor.burned_mass(t)))
        if thrust_nom > 0.0:
            idx = min(int(t / cfg.phase_length), len(prof) - 1)
            thrust = thrust_nom * prof[idx]
        else:
            thrust = 0.0
            if burnout is None:
                burnout = (t, y, v)
        a = (thrust - k * v * abs(v)) / mass - G
        if y <= launch_altitude and a <= 0.0:
            # still sitting on the pad, thrust below the weight
            y = float(launch_altitude)
            v = 0.0
            if burnout is not None:
                break          # motor finished and it never left the pad
            t += cfg.dt
            continue
        if liftoff_time is None:
            liftoff_time = t
        a_max = max(a_max, a)
        v_new = v + a * cfg.dt
        y += 0.5 * (v + v_new) * cfg.dt
        v = v_new
        v_max = max(v_max, v)
        t += cfg.dt
        if v <= 0.0 and burnout is not None:
            break

    if burnout is None:
        burnout = (t, y, v)
    return {
        "apogee": y,
        "time_to_apogee": t,
        "burnout_time": burnout[0],
        "burnout_altitude": burnout[1],
        "burnout_speed": burnout[2],
        "max_speed": v_max,
        "max_acceleration": a_max,
        "liftoff_time": liftoff_time,
        "coast_height": y - burnout[1],
    }


# --------------------------------------------------------------------------- #
#  Throttle-profile optimiser (differential evolution, vectorised)
# --------------------------------------------------------------------------- #
def _cost(cfg: Config, res):
    """Lower is better. Landing softly is the only goal."""
    c = np.where(res["landed"], res["touchdown_speed"],
                 1000.0 + res["min_altitude"])   # never reached the ground
    return c


def optimise_profile(cfg: Config, ignition_altitude: float, pop_size=64,
                     generations=120, seed=0, x0=None):
    """Find the throttle profile that minimises the touchdown speed."""
    rng = np.random.default_rng(seed)
    d = cfg.n_phases
    lo, hi = cfg.throttle_min, cfg.throttle_max
    v0, t_fall = free_fall_to(cfg, ignition_altitude)

    pop = rng.uniform(lo, hi, size=(pop_size, d))
    # a few structured seeds: constant throttles and simple ramps
    seeds = [np.full(d, x) for x in (0.1, 0.25, 0.5, 0.75, 1.0)]
    seeds += [np.linspace(lo, hi, d), np.linspace(hi, lo, d)]
    if x0 is not None:
        seeds.append(np.clip(np.asarray(x0, dtype=float), lo, hi))
    for i, s in enumerate(seeds[:pop_size]):
        pop[i] = s

    cost = _cost(cfg, simulate_population(cfg, ignition_altitude, v0, pop))

    for _ in range(generations):
        # DE/rand/1/bin with a randomised F and CR
        a, b, c = (pop[rng.permutation(pop_size)] for _ in range(3))
        f = rng.uniform(0.4, 0.9)
        mutant = np.clip(a + f * (b - c), lo, hi)
        cr = rng.uniform(0.7, 0.95)
        mask = rng.random((pop_size, d)) < cr
        mask[np.arange(pop_size), rng.integers(0, d, pop_size)] = True
        trial = np.where(mask, mutant, pop)

        t_cost = _cost(cfg, simulate_population(cfg, ignition_altitude, v0, trial))
        better = t_cost < cost
        pop[better] = trial[better]
        cost[better] = t_cost[better]

    best = int(np.argmin(cost))
    res = simulate_population(cfg, ignition_altitude, v0, pop[best][None, :])
    return {
        "profile": pop[best].copy(),
        "touchdown_speed": float(res["touchdown_speed"][0]),
        "landed": bool(res["landed"][0]),
        "entry_speed": abs(v0),
        "fall_time": t_fall,
        "cost": float(cost[best]),
    }


# --------------------------------------------------------------------------- #
#  Search for the feasible ignition-altitude window
# --------------------------------------------------------------------------- #
class Cancelled(Exception):
    """Raised when a caller-supplied stop callback aborts the search."""


def find_window(cfg: Config, coarse_step=5.0, tol=0.25, verbose=True,
                pop_size=64, generations=120, on_progress=None,
                should_stop=None, search_min=1.0, search_max=None):
    """Return the lowest and the highest ignition altitude with a soft landing.

    on_progress : optional callable(str) receiving human-readable progress lines
    should_stop : optional callable() -> bool; when it returns True the search
                  is aborted with a Cancelled exception
    """
    def report(msg):
        if on_progress is not None:
            on_progress(msg)
        elif verbose:
            print(msg)

    def check_stop():
        if should_stop is not None and should_stop():
            raise Cancelled()

    limit = cfg.max_touchdown_speed
    lo_bound = max(0.0, float(search_min))
    hi_bound = cfg.drop_altitude if search_max is None else float(search_max)
    hi_bound = min(hi_bound, cfg.drop_altitude)
    if hi_bound <= lo_bound:
        raise ValueError("the search range is empty "
                         f"(<{lo_bound:.2f}, {hi_bound:.2f}> m)")

    # --- coarse scan -------------------------------------------------------
    grid = np.arange(lo_bound, hi_bound + 1e-9, coarse_step)
    results, warm = {}, None
    seen_ok = False
    for h in grid:
        check_stop()
        r = optimise_profile(cfg, float(h), pop_size=pop_size,
                             generations=generations, seed=int(h * 7) & 0xFFFF,
                             x0=warm)
        results[float(h)] = r
        if r["touchdown_speed"] < limit:
            warm = r["profile"]
        flag = "OK " if r["touchdown_speed"] <= limit else "   "
        report(f"  {flag}h = {h:6.1f} m   entry {r['entry_speed']:5.1f} m/s"
               f"   best touchdown {r['touchdown_speed']:7.2f} m/s")
        check_stop()
        # the feasible set is a single contiguous window: the first failure
        # after a series of successes closes it, so the scan can stop there
        if r["touchdown_speed"] <= limit:
            seen_ok = True
        elif seen_ok:
            report("  first failure after the OK series - window closed, "
                   "stopping the scan")
            break

    # --- if the coarse grid missed the (possibly narrow) window, zoom in around
    #     the most promising altitude a few times before giving up ------------
    step = coarse_step
    for _ in range(4):
        feasible = [h for h, r in results.items() if r["touchdown_speed"] <= limit]
        if feasible:
            break
        best_h = min(results, key=lambda h: results[h]["touchdown_speed"])
        step = step / 4.0
        if step < tol:
            break
        report(f"  no soft landing on this grid, refining around "
               f"h = {best_h:.1f} m with step {step:.2f} m")
        fine = np.arange(max(lo_bound, best_h - 4 * step),
                         min(hi_bound, best_h + 4 * step) + 1e-9, step)
        for h in fine:
            check_stop()
            h = float(h)
            if h in results:
                continue
            r = optimise_profile(cfg, h, pop_size=pop_size, generations=generations,
                                 seed=int(h * 17) & 0xFFFF,
                                 x0=results[best_h]["profile"])
            results[h] = r
            flag = "OK " if r["touchdown_speed"] <= limit else "   "
            report(f"  {flag}h = {h:6.2f} m   entry {r['entry_speed']:5.1f} m/s"
                   f"   best touchdown {r['touchdown_speed']:7.2f} m/s")

    feasible = [h for h, r in results.items() if r["touchdown_speed"] <= limit]
    if not feasible:
        return None

    h_lo_ok, h_hi_ok = min(feasible), max(feasible)

    def ok(h, warm_profile):
        check_stop()
        r = optimise_profile(cfg, h, pop_size=pop_size, generations=generations,
                             seed=int(h * 131) & 0xFFFF, x0=warm_profile)
        return r["touchdown_speed"] <= limit, r

    # --- refine the lower edge --------------------------------------------
    lo_fail = max([h for h in results if h < h_lo_ok], default=lo_bound - coarse_step)
    lo_fail = max(lo_fail, 0.0)
    best_lo = results[h_lo_ok]
    a, b = lo_fail, h_lo_ok
    while b - a > tol:
        mid = 0.5 * (a + b)
        good, r = ok(mid, best_lo["profile"])
        if good:
            b, best_lo = mid, r
        else:
            a = mid
        report(f"  lower edge  in <{a:6.2f}, {b:6.2f}> m")
    low_alt, low_res = b, best_lo

    # --- refine the upper edge --------------------------------------------
    hi_fail = min([h for h in results if h > h_hi_ok], default=hi_bound + coarse_step)
    hi_fail = min(hi_fail, cfg.drop_altitude)
    best_hi = results[h_hi_ok]
    a, b = h_hi_ok, hi_fail
    while b - a > tol:
        mid = 0.5 * (a + b)
        good, r = ok(mid, best_hi["profile"])
        if good:
            a, best_hi = mid, r
        else:
            b = mid
        report(f"  upper edge  in <{a:6.2f}, {b:6.2f}> m")
    high_alt, high_res = a, best_hi

    return {"low": (low_alt, low_res), "high": (high_alt, high_res)}


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def print_profile(cfg: Config, res, label):
    prof = res["profile"]
    print(f"\n{label}")
    print(f"  ignition altitude   : {res['altitude']:.2f} m")
    print(f"  time from release   : {res['fall_time']:.3f} s")
    print(f"  speed at ignition   : {res['entry_speed']:.2f} m/s (down)")
    print(f"  touchdown speed     : {res['touchdown_speed']:.2f} m/s")
    print("  throttle profile (100 ms phases, 1.00 = full thrust, "
          "0.10 = flaps block 90 %):")
    for i in range(0, len(prof), 10):
        chunk = prof[i:i + 10]
        t_start = i * cfg.phase_length
        print(f"    t={t_start:5.2f}s : " + " ".join(f"{x:.2f}" for x in chunk))
    print("  resulting thrust [N] per phase:")
    for i in range(0, len(prof), 10):
        chunk = prof[i:i + 10]
        t_start = i * cfg.phase_length
        vals = [cfg.motor.thrust((i + j + 0.5) * cfg.phase_length) * chunk[j]
                for j in range(len(chunk))]
        print(f"    t={t_start:5.2f}s : " + " ".join(f"{x:6.1f}" for x in vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop-alt", type=float, default=150.0,
                    help="altitude where the free fall starts [m]")
    ap.add_argument("--mass", type=float, default=GROSS_MASS, help="gross mass [kg]")
    ap.add_argument("--propellant", type=float, default=PROPELLANT_MASS,
                    help="propellant mass [kg]")
    ap.add_argument("--diameter", type=float, default=DIAMETER, help="diameter [m]")
    ap.add_argument("--cd", type=float, default=CD, help="drag coefficient")
    ap.add_argument("--rho", type=float, default=RHO, help="air density [kg/m3]")
    ap.add_argument("--max-touchdown", type=float, default=3.0,
                    help="soft-landing limit [m/s]")
    ap.add_argument("--throttle-min", type=float, default=0.1,
                    help="minimum thrust fraction (flaps at 90 %% blocking)")
    ap.add_argument("--phase", type=float, default=0.1, help="throttle phase length [s]")
    ap.add_argument("--dt", type=float, default=0.002, help="integration step [s]")
    ap.add_argument("--motor", choices=sorted(MOTOR_TABLES), default="long",
                    help="which motor lookup table to use "
                         "(long = 120 N / 2.6 s, short = 269 N / 1.55 s)")
    ap.add_argument("--thrust-mult", type=float, default=1.0,
                    help="multiplier applied to the whole motor lookup table")
    ap.add_argument("--search-min", type=float, default=1.0,
                    help="lowest ignition altitude to search [m]")
    ap.add_argument("--search-max", type=float, default=None,
                    help="highest ignition altitude to search [m], "
                         "default = drop altitude")
    ap.add_argument("--coarse-step", type=float, default=5.0,
                    help="ignition-altitude scan step [m]")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="bisection tolerance of the window edges [m]")
    ap.add_argument("--pop", type=int, default=64, help="DE population size")
    ap.add_argument("--gen", type=int, default=120, help="DE generations")
    ap.add_argument("--ascent", action="store_true",
                    help="only run the side calculation: how high the rocket "
                         "would fly if launched vertically from the ground")
    ap.add_argument("--ascent-throttle", type=float, default=1.0,
                    help="constant throttle used for --ascent")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = Config(motor=Motor(args.motor, propellant_mass=args.propellant,
                             thrust_multiplier=args.thrust_mult),
                 drop_altitude=args.drop_alt, gross_mass=args.mass,
                 diameter=args.diameter, cd=args.cd, rho=args.rho,
                 max_touchdown_speed=args.max_touchdown,
                 throttle_min=args.throttle_min, phase_length=args.phase,
                 dt=args.dt)
    m = cfg.motor

    print("=" * 72)
    print("LandSim Light - 1D propulsive landing")
    print("=" * 72)
    print(f"  gross mass        : {cfg.gross_mass:.3f} kg "
          f"(propellant {m.propellant_mass * 1000:.0f} g)")
    print(f"  diameter / area   : {cfg.diameter * 1000:.0f} mm / {cfg.area * 1e4:.1f} cm2")
    print(f"  Cd                : {cfg.cd}")
    print(f"  drop altitude     : {cfg.drop_altitude:.1f} m")
    print(f"  motor             : {m.name}")
    print(f"  thrust multiplier : {m.thrust_multiplier:g} x table")
    print(f"  motor burn time   : {m.burn_time:.3f} s, peak {m.peak_thrust:.1f} N")
    print(f"  total impulse     : {m.total_impulse:.1f} Ns  "
          f"(avg {m.total_impulse / m.burn_time:.1f} N)")
    print(f"  weight            : {cfg.gross_mass * G:.1f} N  ->  "
          f"peak T/W = {m.peak_thrust / (cfg.gross_mass * G):.2f}")
    print(f"  throttle          : {cfg.throttle_min:.2f} - {cfg.throttle_max:.2f} "
          f"in {cfg.n_phases} phases of {cfg.phase_length * 1000:.0f} ms")
    print(f"  terminal velocity : "
          f"{math.sqrt(cfg.gross_mass * G / (0.5 * cfg.rho * cfg.cd * cfg.area)):.1f} m/s")
    s_max = cfg.drop_altitude if args.search_max is None else args.search_max
    print(f"  soft-landing limit: {cfg.max_touchdown_speed:.1f} m/s")
    print(f"  searched altitudes: {args.search_min:.1f} - {s_max:.1f} m\n")

    asc = ascent(cfg, throttle=args.ascent_throttle)
    print(f"Side calculation - vertical launch from the ground "
          f"(throttle {args.ascent_throttle:g}):")
    if asc["liftoff_time"] is None:
        print("  the rocket never lifts off (thrust stays below the weight)\n")
    else:
        print(f"  burnout           : {asc['burnout_time']:.2f} s at "
              f"{asc['burnout_altitude']:.1f} m, {asc['burnout_speed']:.1f} m/s")
        print(f"  max speed / accel : {asc['max_speed']:.1f} m/s / "
              f"{asc['max_acceleration'] / G:.1f} g")
        print(f"  APOGEE            : {asc['apogee']:.1f} m "
              f"at t = {asc['time_to_apogee']:.2f} s "
              f"(coast {asc['coast_height']:.1f} m)\n")
    if args.ascent:
        return

    print("Scanning ignition altitudes ...")
    win = find_window(cfg, coarse_step=args.coarse_step, tol=args.tol,
                      verbose=not args.quiet, pop_size=args.pop,
                      generations=args.gen, search_min=args.search_min,
                      search_max=args.search_max)

    print("\n" + "=" * 72)
    if win is None:
        print("No ignition altitude produced a soft landing "
              f"(<= {cfg.max_touchdown_speed} m/s).")
        return

    lo_alt, lo_res = win["low"]
    hi_alt, hi_res = win["high"]
    lo_res["altitude"], hi_res["altitude"] = lo_alt, hi_alt
    dt_ign = lo_res["fall_time"] - hi_res["fall_time"]
    print(f"RESULT: soft landing possible for ignition altitudes "
          f"{lo_alt:.2f} m ... {hi_alt:.2f} m "
          f"(window {hi_alt - lo_alt:.2f} m)")
    print(f"        time from the release at {cfg.drop_altitude:.1f} m: "
          f"{hi_res['fall_time']:.3f} s (highest) ... "
          f"{lo_res['fall_time']:.3f} s (lowest)")
    print(f"        -> time window for the ignition command: {dt_ign:.3f} s")
    print("=" * 72)
    print_profile(cfg, lo_res, "LOWEST ignition altitude that still lands softly")
    print_profile(cfg, hi_res, "HIGHEST ignition altitude that still lands softly")


if __name__ == "__main__":
    main()
