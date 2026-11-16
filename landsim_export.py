#!/usr/bin/env python3
"""
LandSim Light -> flight software export (Daidalos).

    python landsim_export.py --vehicle vehicle.json --out band.json [--gains]
                             [--motor-table]

Turns a vehicle description into the numbers the flight computer needs on board,
computed by the SAME model the campaigns fly (`tvc_sim.project_rh` /
`tvc_sim.TvcConfig`) - nothing in this file knows a vehicle number.

band.json (schema 1)
--------------------
    "schema"        1. Bumped only for a change that breaks a reader; new optional
                    keys are added without a bump.
    "landsim_git"   commit the numbers were computed from ("unknown" outside git);
                    "landsim_dirty" is true when tracked files differed from it.
    "vz0", "dvz", "n"
                    the grid of descent speeds |vz| [m/s]: vz0 + i * dvz, i < n.
                    Default 4, 2, 32  ->  4, 6, ... 66 m/s.
    "hmin", "hmax"  for every |vz| on the grid, the band of altitudes [m above the
                    ground] at which the main motor may START PRODUCING THRUST and
                    still land: project_rh(h, -|vz|) >= FEASIBLE_VZ ("feasible_vz",
                    -2.3 m/s, against the planner's aim "plan_target_vz"). The edges
                    are found by bisection to "edge_tol_m" (0.1 m) and reported on the
                    FEASIBLE side, so the band never claims more than the model
                    proved. hmax <= 0 means NO feasible band at that speed (hmin is
                    then 0 as well).
                    The band is for thrust onset: the command has to go out earlier
                    by the igniter delay - see delay_pad_s.
    "delay_pad_s"   how much igniter delay the guidance pads for [s]. The ignition
                    command is issued the fall of `delay_pad_s` ahead of the altitude
                    it wants the motor to light at.
    "fin_cd_zero"   drag coefficient of the whole vehicle (body + fins) on the body
                    cross-section, fins at zero deflection (the burn's Cd)
    "fin_cd_brake"  the same with the fins splayed as airbrakes - the free-fall Cd
                    the vehicle actually flies. Equals fin_cd_zero when the airbrake
                    or the fins are switched off, so the free fall can always use it.
    "feasible_vz", "plan_target_vz", "edge_tol_m", "vehicle"
                    how to read the band, and the resolved vehicle it belongs to.
    "gains"         only with --gains: controller gains re-fitted for THIS mass and
                    inertia - wn_tvc, zeta_tvc, wn_fin, zeta_fin, roll_gain,
                    sched_tvc, sched_fin, att_ki - and "gains_fit", the fit's report.
    "motor_table"   only with --motor-table: [[t [s], F [N]], ...] after
                    thrust_scale, exactly the curve the model flies (two rows with
                    the same t are a step in thrust).

vehicle.json
------------
Every key is optional; an empty object is the default rocket.

    mass_gross_kg, mass_prop_kg, cd_body, diameter_m, thrust_scale,
    fin_enable (0/1), fin_brake (0/1, or "auto" | "always" | "off"),
    motor_curve [[t, N], ...]   explicit thrust curve instead of the built-in motor
    fin_area_m2, fin_arm_m, fin_cl_alpha, fin_stall_deg
                                one fin's area, its arm to the CG, the lift slope the
                                controller inverts with [1/rad], the stall angle
                                (fin_cl_alpha without fin_aero selects the analytic
                                lift model, the CFD table belongs to the stock fin)
    inertia {"transverse": ..., "roll": ...}   kg m^2 at the GROSS mass; omitted =
                                slender rod / solid cylinder from mass and diameter
    delay_pad_s                 omitted = the optimum for ign_delay_max_s
    ign_delay_max_s, n_boosters, ...

Any other TvcConfig field can be given by its own name (e.g. "att_ki", "k_min",
"l_gimbal", "cn_alpha"); a key that is neither an alias nor a field is an error,
so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import replace

import numpy as np

import tvc_sim
from tvc_sim import FEASIBLE_VZ, TvcConfig, find_ignition, project_rh  # noqa: F401

SCHEMA = 1

# The default speed grid of band.json: |vz| = VZ0 + i * DVZ, i < N.
DEFAULT_VZ0 = 4.0
DEFAULT_DVZ = 2.0
DEFAULT_N = 32

# Numerical settings of the edge search (not vehicle numbers).
EDGE_TOL_M = 0.1          # bisection stops when the bracket is this narrow
H_FLOOR_M = 1.0           # lowest ignition altitude considered
H_CAP_M = 300.0           # highest ignition altitude considered
SCAN_STEP_M = 2.0         # coarse scan that brackets the band before bisecting
PEAK_ITER = 16            # ternary iterations when the scan sees no feasible point

# vehicle.json key -> TvcConfig field, for the names that differ.
ALIASES = {
    "mass_gross_kg": "gross_mass",
    "mass_prop_kg": "propellant",
    "cd_body": "cd",
    "diameter_m": "diameter",
    "thrust_scale": "thrust_mult",
    "fin_area_m2": "fin_area",
    "fin_arm_m": "fin_arm",
    "fin_cl_alpha": "fin_cl_alpha",
    "fin_stall_deg": "fin_stall",
    "delay_pad_s": "delay_pad",
    "ign_delay_max_s": "ign_delay_max",
}
SPECIAL = {"fin_enable", "fin_brake", "motor_curve", "inertia"}

# controller gain names as band.json spells them -> TvcConfig field
GAIN_EXPORT = {
    "wn_tvc": "wn", "zeta_tvc": "zeta", "wn_fin": "wn_fin", "zeta_fin": "zeta_fin",
    "roll_gain": "roll_gain", "sched_tvc": "sched_tvc", "sched_fin": "sched_fin",
    "att_ki": "att_ki",
}
ATT_KI_FACTORS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)   # of the configured att_ki


# ---------------------------------------------------------------------------
# vehicle.json -> TvcConfig
# ---------------------------------------------------------------------------
def _fields():
    return {f.name: f for f in dataclasses.fields(TvcConfig)}


def _coerce(name, value, default):
    """Convert a JSON value to the type of the field's default."""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                raise ValueError("expected 0/1 or true/false")
            return bool(value)
        if isinstance(default, int):
            if isinstance(value, bool) or float(value) != int(value):
                raise ValueError("expected an integer")
            return int(value)
        if isinstance(default, float):
            if isinstance(value, (bool, str)):
                raise ValueError("expected a number")
            v = float(value)
            if not math.isfinite(v):
                raise ValueError("expected a finite number")
            return v
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vehicle key '{name}': {exc} (got {value!r})") from None
    return value


def _brake_mode(value):
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("auto", "always", "off"):
            return v
        raise ValueError("fin_brake must be 0/1 or 'auto' | 'always' | 'off'")
    if value in (0, 1, False, True):
        return "auto" if value else "off"
    raise ValueError("fin_brake must be 0 or 1")


def config_from_vehicle(vehicle: dict) -> TvcConfig:
    """Build the TvcConfig a vehicle.json describes. Strict: unknown keys raise."""
    if not isinstance(vehicle, dict):
        raise ValueError("vehicle.json must hold a JSON object")
    fields = _fields()
    base = TvcConfig()
    kw = {}
    for key, value in vehicle.items():
        if key.startswith("_") or key == "comment":
            continue                       # free-form annotations
        if key in SPECIAL:
            continue
        name = ALIASES.get(key, key)
        if name not in fields or name in SPECIAL:
            raise ValueError(f"unknown vehicle key '{key}'")
        if name in kw:
            raise ValueError(f"vehicle key '{key}' given twice (alias and field)")
        kw[name] = _coerce(key, value, getattr(base, name))

    if "fin_enable" in vehicle:
        if vehicle["fin_enable"] not in (0, 1, False, True):
            raise ValueError("fin_enable must be 0 or 1")
        kw["fins"] = bool(vehicle["fin_enable"])
    if "fin_brake" in vehicle:
        kw["fin_brake"] = _brake_mode(vehicle["fin_brake"])
    if vehicle.get("motor_curve") is not None:
        curve = vehicle["motor_curve"]
        try:
            kw["motor_curve"] = tuple((float(t), float(f)) for t, f in curve)
        except (TypeError, ValueError):
            raise ValueError("motor_curve must be a list of [t, N] pairs") from None
        kw.setdefault("motor", "curve")
    if kw.get("fin_cl_alpha", 0.0) > 0.0:
        kw.setdefault("fin_aero", "analytic")

    mass = kw.get("gross_mass", base.gross_mass)
    diam = kw.get("diameter", base.diameter)
    length = kw.get("body_length", base.body_length)
    inertia = vehicle.get("inertia") or {}
    if not isinstance(inertia, dict) or set(inertia) - {"transverse", "roll"}:
        raise ValueError("inertia must be {\"transverse\": kg m^2, \"roll\": kg m^2}")
    if "mmoi_transverse" not in kw:
        kw["mmoi_transverse"] = float(inertia.get(
            "transverse", mass * length * length / 12.0))
    elif "transverse" in inertia:
        raise ValueError("inertia.transverse given twice")
    if "mmoi_roll" not in kw:
        kw["mmoi_roll"] = float(inertia.get("roll", mass * 0.5 * (diam / 2.0) ** 2))
    elif "roll" in inertia:
        raise ValueError("inertia.roll given twice")
    if "delay_pad" not in kw:
        kw["delay_pad"] = tvc_sim.default_delay_pad(
            kw.get("ign_delay_max", base.ign_delay_max))

    cfg = replace(base, **kw)
    _validate(cfg)
    return cfg


def _validate(cfg: TvcConfig):
    positive = ("gross_mass", "propellant", "diameter", "cd", "thrust_mult",
                "mmoi_transverse", "mmoi_roll", "k_max", "fin_stall", "fin_span",
                "fin_root", "fin_tip", "fin_arm", "l_gimbal")
    for name in positive:
        if not getattr(cfg, name) > 0.0:
            raise ValueError(f"{name} must be > 0 (got {getattr(cfg, name)})")
    for name in ("fin_area", "fin_cl_alpha", "k_min", "delay_pad"):
        if getattr(cfg, name) < 0.0:
            raise ValueError(f"{name} must be >= 0 (got {getattr(cfg, name)})")
    if cfg.propellant >= cfg.gross_mass:
        raise ValueError("mass_prop_kg must be less than mass_gross_kg")
    if cfg.k_min >= cfg.k_max:
        raise ValueError("k_min must be below k_max")
    cfg.tables()                    # validates the motor curve / motor name


def resolved_vehicle(cfg: TvcConfig) -> dict:
    """The vehicle as the model resolved it, in vehicle.json's own words."""
    fins = cfg.fin_set()
    return {
        "mass_gross_kg": cfg.gross_mass, "mass_prop_kg": cfg.propellant,
        "cd_body": cfg.cd, "diameter_m": cfg.diameter,
        "thrust_scale": cfg.thrust_mult,
        "fin_enable": int(fins.enabled),
        "fin_brake": int(cfg.fin_brake != "off" and fins.enabled),
        "fin_area_m2": fins.area, "fin_arm_m": fins.arm,
        "fin_cl_alpha": fins.cl_alpha, "fin_stall_deg": fins.stall,
        "fin_aero": cfg.fin_aero,
        "inertia": {"transverse": cfg.mmoi_transverse, "roll": cfg.mmoi_roll},
        "n_boosters": cfg.n_boosters,
        "motor": ("curve" if cfg.motor_curve is not None else cfg.motor),
    }


# ---------------------------------------------------------------------------
# the band
# ---------------------------------------------------------------------------
def _vz_touchdown(p, h, vz_abs):
    return float(project_rh(float(h), -float(vz_abs), 0.0, p.mass_extra,
                            p.mt, p.mf, p.mc, p.prop_mass, p.i_total, p.cd_burn,
                            p.area, p.k_min, p.k_max, p.bt, p.bf, p.bb, p.b_prop,
                            p.b_ign, p.plan_target))


def band_at(cfg: TvcConfig, vz_abs: float, tol=EDGE_TOL_M, h_floor=H_FLOOR_M,
            h_cap=H_CAP_M, step=SCAN_STEP_M, planner=None):
    """(hmin, hmax, note) for one descent speed. hmax <= 0: no feasible band.

    The altitude axis is scanned coarsely from the floor up, because feasibility is
    an INTERVAL (igniting too high burns the grain out before the ground), then
    each edge is bisected inside the bracket the scan left. If the scan sees no
    feasible point the peak is searched for - a band narrower than the scan step is
    still a band.
    """
    p = planner or tvc_sim.planner_inputs(cfg)

    def vz_td(h):
        return _vz_touchdown(p, h, vz_abs)

    first = last = None
    best_h, best_v = h_floor, -1e9
    i = 0
    while h_floor + i * step <= h_cap:
        h = h_floor + i * step
        v = vz_td(h)
        if v > best_v:
            best_h, best_v = h, v
        if v >= FEASIBLE_VZ:
            if first is None:
                first = h
            last = h
        elif first is not None:
            break
        i += 1
    ran_off_cap = first is not None and last + step > h_cap
    note = ""

    if first is None:                      # nothing on the grid: look for a sliver
        a, c = max(h_floor, best_h - step), min(h_cap, best_h + step)
        for _ in range(PEAK_ITER):
            m1, m2 = a + (c - a) / 3.0, c - (c - a) / 3.0
            v1, v2 = vz_td(m1), vz_td(m2)
            for hh, vv in ((m1, v1), (m2, v2)):
                if vv > best_v:
                    best_h, best_v = hh, vv
            if v1 < v2:
                a = m1
            else:
                c = m2
        if best_v < FEASIBLE_VZ:
            return 0.0, 0.0, "no feasible altitude"
        first = last = best_h
        lo_start = max(h_floor, best_h - step)
        hi_end = min(h_cap, best_h + step)
        note = "band narrower than the scan step"
    else:
        lo_start = max(h_floor, first - step)
        hi_end = last + step

    # lower edge: lo infeasible, hi feasible
    if first <= h_floor:
        hmin, note = h_floor, (note or "feasible down to the altitude floor")
    else:
        lo, hi = lo_start, first
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if vz_td(mid) >= FEASIBLE_VZ:
                hi = mid
            else:
                lo = mid
        hmin = hi
    # upper edge: lo feasible, hi infeasible
    if ran_off_cap:
        hmax, note = last, (note or "feasible up to the altitude cap")
    else:
        lo, hi = last, hi_end
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if vz_td(mid) >= FEASIBLE_VZ:
                lo = mid
            else:
                hi = mid
        hmax = lo
    return float(hmin), float(hmax), note


def _band_job(job):
    cfg, vz, kw = job
    return band_at(cfg, vz, **kw)


def compute_band(cfg: TvcConfig, vz0=DEFAULT_VZ0, dvz=DEFAULT_DVZ, n=DEFAULT_N,
                 tol=EDGE_TOL_M, workers=None, log=None, **kw):
    """hmin, hmax (lists) for |vz| = vz0 + i * dvz. Speeds are independent, so they
    spread over worker processes; the result does not depend on how."""
    if n < 1 or dvz <= 0.0 or vz0 < 0.0:
        raise ValueError("need n >= 1, dvz > 0 and vz0 >= 0")
    speeds = [vz0 + i * dvz for i in range(int(n))]
    jobs = [(cfg, v, dict(tol=tol, **kw)) for v in speeds]
    if workers is None:
        workers = min(os.cpu_count() or 1, 8)
    workers = max(1, min(workers, len(jobs)))
    results = None
    if workers > 1:
        try:
            with cf.ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_band_job, jobs))
        except (OSError, ValueError, ImportError, RuntimeError, cf.BrokenExecutor
                ) as exc:   # noqa: BLE001 - restricted environments cannot fork
            if log:
                log(f"  worker processes unavailable ({exc}); computing in this one")
            results = None
    if results is None:
        results = [_band_job(j) for j in jobs]
    if log:
        for v, (lo, hi, note) in zip(speeds, results):
            if note:
                log(f"  |vz| {v:5.1f} m/s: {note}")
    hmin = [round(r[0], 3) for r in results]
    hmax = [round(r[1], 3) for r in results]
    return hmin, hmax


# ---------------------------------------------------------------------------
# gains
# ---------------------------------------------------------------------------
def fit_att_ki(cfg: TvcConfig, runs=12, cells=3, workers=None, log=None):
    """Pick the attitude-integral gain on the tuner's own campaign (common random
    numbers, the ignition altitudes solved once) with everything else held at `cfg`.
    Returns (att_ki, {candidate: cost})."""
    small = tvc_sim.tuning_campaign_config(cfg, runs * 3, cells)
    h_cmd = tvc_sim.ignition_grid(small)
    ref = cfg.att_ki if cfg.att_ki > 0.0 else tvc_sim.ATT_KI
    costs = {}
    for ki in sorted({round(ref * f, 6) for f in ATT_KI_FACTORS} | {cfg.att_ki}):
        camp = tvc_sim.run_campaign(replace(small, att_ki=ki), h_cmd_grid=h_cmd,
                                    workers=workers)
        costs[ki] = tvc_sim.gain_cost(camp)[0]
        if log:
            log(f"      att_ki {ki:6.2f}  cost {costs[ki]:.4f}")
    best = min(costs, key=lambda k: (costs[k], k != cfg.att_ki))   # ties keep cfg's
    return best, costs


def fit_gains(cfg: TvcConfig, budget=60, runs=12, cells=3, workers=None, log=None):
    """Re-fit the loop gains for this mass / inertia / actuators. Returns
    (gains dict in band.json's names, report dict)."""
    g, rep = tvc_sim.tune_gains(cfg, budget=budget, runs=runs, cells=cells,
                                workers=workers, on_progress=log)
    tuned = cfg.with_gains(g)
    ki, ki_costs = fit_att_ki(tuned, runs=runs, cells=cells, workers=workers, log=log)
    tuned = replace(tuned, att_ki=ki)
    gains = {out: float(getattr(tuned, field)) for out, field in GAIN_EXPORT.items()}
    rep = dict(rep)
    rep["att_ki_costs"] = {f"{k:g}": round(v, 5) for k, v in ki_costs.items()}
    return gains, rep


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------
def git_state(root=None):
    """(commit hash, dirty) of the LandSim checkout, ("unknown", False) without git."""
    root = root or os.path.dirname(os.path.abspath(__file__))

    def run(*args):
        return subprocess.run(["git", "-C", root, *args], capture_output=True,
                              text=True, timeout=20)
    try:
        h = run("rev-parse", "HEAD")
        if h.returncode != 0:
            return "unknown", False
        st = run("status", "--porcelain", "--untracked-files=no")
        return h.stdout.strip(), bool(st.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown", False


def motor_table(cfg: TvcConfig):
    m, _ = cfg.tables()
    return [[round(float(t), 9), round(float(f), 6)] for t, f in zip(m.t, m.f)]


def build_document(cfg: TvcConfig, vz0=DEFAULT_VZ0, dvz=DEFAULT_DVZ, n=DEFAULT_N,
                   tol=EDGE_TOL_M, with_gains=False, with_motor_table=False,
                   workers=None, gains_budget=60, gains_runs=12, gains_cells=3,
                   log=None):
    git, dirty = git_state()
    hmin, hmax = compute_band(cfg, vz0, dvz, n, tol=tol, workers=workers, log=log)
    _, cd_brake = cfg.fin_args()
    doc = {
        "schema": SCHEMA,
        "landsim_git": git,
        "landsim_dirty": dirty,
        "vz0": float(vz0), "dvz": float(dvz), "n": int(n),
        "hmin": hmin, "hmax": hmax,
        "delay_pad_s": float(cfg.delay_pad),
        "fin_cd_zero": round(float(cfg.cd_burn), 6),
        "fin_cd_brake": round(float(cd_brake), 6),
        "feasible_vz": FEASIBLE_VZ,
        "plan_target_vz": float(cfg.plan_target),
        "edge_tol_m": float(tol),
        "vehicle": resolved_vehicle(cfg),
    }
    if with_gains:
        gains, rep = fit_gains(cfg, budget=gains_budget, runs=gains_runs,
                               cells=gains_cells, workers=workers, log=log)
        doc["gains"] = gains
        doc["gains_fit"] = rep
    if with_motor_table:
        doc["motor_table"] = motor_table(cfg)
    return doc


def write_json(doc, path):
    """Write atomically - a flight computer must never read half a file."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".band-", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_vehicle(path):
    if path is None:
        return {}
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Export the ignition band (and optionally re-fitted gains and "
                    "the motor table) of a vehicle for the Daidalos flight software.")
    ap.add_argument("--vehicle", help="vehicle.json (omitted = the default rocket)")
    ap.add_argument("--out", default="band.json", help="output file (default band.json)")
    ap.add_argument("--gains", action="store_true",
                    help="re-fit the loop gains for this mass / inertia and export them")
    ap.add_argument("--motor-table", action="store_true",
                    help="export the thrust table (t, N) after thrust_scale")
    ap.add_argument("--vz0", type=float, default=DEFAULT_VZ0,
                    help="first |vz| of the grid [m/s] (default %(default)g)")
    ap.add_argument("--dvz", type=float, default=DEFAULT_DVZ,
                    help="grid spacing [m/s] (default %(default)g)")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="grid points (default %(default)d)")
    ap.add_argument("--tol", type=float, default=EDGE_TOL_M,
                    help="edge bisection tolerance [m] (default %(default)g)")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes (0 = one per core, up to 8)")
    ap.add_argument("--gains-budget", type=int, default=60,
                    help="candidate gain sets flown by --gains (default %(default)d)")
    ap.add_argument("--gains-runs", type=int, default=12,
                    help="flights per entry state while fitting (default %(default)d)")
    ap.add_argument("--gains-cells", type=int, default=3,
                    help="entry grid of the fit is cells x cells (default %(default)d)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr, flush=True)

    try:
        cfg = config_from_vehicle(load_vehicle(args.vehicle))
    except (OSError, ValueError) as exc:
        ap.error(f"vehicle: {exc}")
    doc = build_document(cfg, args.vz0, args.dvz, args.n, tol=args.tol,
                         with_gains=args.gains, with_motor_table=args.motor_table,
                         workers=args.workers or None, gains_budget=args.gains_budget,
                         gains_runs=args.gains_runs, gains_cells=args.gains_cells,
                         log=log)
    write_json(doc, args.out)
    ok = sum(1 for h in doc["hmax"] if h > 0.0)
    log(f"wrote {args.out}: {ok}/{doc['n']} speeds have a feasible band "
        f"(|vz| {doc['vz0']:g} .. {doc['vz0'] + (doc['n'] - 1) * doc['dvz']:g} m/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
