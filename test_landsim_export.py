"""Tests of the flight-software export (landsim_export.py).

    python -m pytest test_landsim_export.py        (or: python -m unittest -v)

The contract that matters is the first one: the band that is written to band.json
must be the band the simulation's own ignition planner (`find_ignition`) works out
for the same vehicle. The rest guard the interface Daidalos reads - the names of the
public functions, the shape of the file, and that every vehicle number really goes
into the model instead of a default.

The default-rocket band is computed once (about ten seconds on four cores).
Set LANDSIM_SLOW_TESTS=1 to also run the real gain fit end to end (about a minute).
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

import landsim
import landsim_export as ex
import tvc_sim
from tvc_sim import FEASIBLE_VZ, find_ignition, project_rh

# Entry states (h0 [m], vx0 [m/s]) the planner is compared on: the corners and the
# middle of the default campaign grid (140-180 m, +/-7 m/s).
ENTRY_STATES = [(150.0, 0.0), (170.0, 3.0), (140.0, -5.0), (180.0, 7.0), (160.0, 0.0)]


def planner_band(cfg, h0, vx0):
    """(h_min, h_max, no_band) the simulation's planner finds for one entry state."""
    p = tvc_sim.planner_inputs(cfg)
    _, _, h_min, h_max, no_band = tvc_sim.find_ignition_full(
        h0, vx0, float(cfg.vz0), p.cd_burn, p.cd_free, p.area, p.k_min, p.k_max,
        p.mass0, p.delay_pad, p.ready, p.mt, p.mf, p.mc, p.prop_mass, p.i_total,
        p.bt, p.bf, p.bb, p.b_prop, p.plan_target)
    return h_min, h_max, no_band


def speed_at(cfg, h0, vx0, h):
    """|vz| the free fall from (h0, vx0) has reached at altitude h."""
    p = tvc_sim.planner_inputs(cfg)
    _, vz, _ = tvc_sim.freefall_to(h0, h, vx0, float(cfg.vz0), p.cd_free, p.area,
                                   p.mass0, 0.005)
    return -vz


def small_band(vehicle, n=2, **kw):
    cfg = ex.config_from_vehicle(vehicle)
    hmin, hmax = ex.compute_band(cfg, n=n, workers=1, **kw)
    return cfg, hmin, hmax


class DefaultRocket(unittest.TestCase):
    """band.json for the default rocket, written through the command line."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls.tmp.name, "band.json")
        cls.veh = os.path.join(cls.tmp.name, "vehicle.json")
        with open(cls.veh, "w") as fh:
            json.dump({}, fh)
        rc = ex.main(["--vehicle", cls.veh, "--out", cls.out, "--quiet"])
        assert rc == 0
        with open(cls.out) as fh:
            cls.band = json.load(fh)
        cls.cfg = ex.config_from_vehicle({})
        cls.v = cls.band["vz0"] + cls.band["dvz"] * np.arange(cls.band["n"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # --- the contract -----------------------------------------------------
    def test_band_matches_find_ignition_within_1_m(self):
        b = self.band
        for h0, vx0 in ENTRY_STATES:
            h_min, h_max, no_band = planner_band(self.cfg, h0, vx0)
            self.assertEqual(no_band, 0.0, "default rocket must have a band")
            for edge, ref in (("hmin", h_min), ("hmax", h_max)):
                v = speed_at(self.cfg, h0, vx0, ref)
                self.assertTrue(self.v[0] <= v <= self.v[-1],
                                f"|vz| {v:.1f} is off the exported grid")
                got = float(np.interp(v, self.v, b[edge]))
                self.assertLess(abs(got - ref), 1.0,
                                f"{edge} at {h0:g} m / {vx0:+g} m/s: band {got:.2f} m "
                                f"against find_ignition {ref:.2f} m")

    def test_edges_sit_on_the_threshold(self):
        """Inside the band is feasible, a bisection tolerance outside it is not."""
        p = tvc_sim.planner_inputs(self.cfg)

        def vz_td(h, v):
            return project_rh(h, -v, 0.0, p.mass_extra, p.mt, p.mf, p.mc, p.prop_mass,
                              p.i_total, p.cd_burn, p.area, p.k_min, p.k_max, p.bt,
                              p.bf, p.bb, p.b_prop, p.b_ign, p.plan_target)
        for i in (0, 5, 13, 21, 31):
            v, lo, hi = self.v[i], self.band["hmin"][i], self.band["hmax"][i]
            self.assertGreaterEqual(vz_td(lo, v), FEASIBLE_VZ)
            self.assertGreaterEqual(vz_td(hi, v), FEASIBLE_VZ)
            self.assertLess(vz_td(lo - 0.25, v), FEASIBLE_VZ)
            self.assertLess(vz_td(hi + 0.25, v), FEASIBLE_VZ)

    # --- the file ----------------------------------------------------------
    def test_schema_and_grid(self):
        b = self.band
        self.assertEqual(b["schema"], 1)
        self.assertEqual((b["vz0"], b["dvz"], b["n"]), (4.0, 2.0, 32))
        self.assertEqual(len(b["hmin"]), 32)
        self.assertEqual(len(b["hmax"]), 32)
        for key in ("delay_pad_s", "fin_cd_zero", "fin_cd_brake"):
            self.assertIsInstance(b[key], float)
        self.assertAlmostEqual(b["delay_pad_s"], tvc_sim.IGN_DELAY_PLAN)
        self.assertAlmostEqual(b["fin_cd_zero"], self.cfg.cd_burn, places=5)
        self.assertAlmostEqual(b["fin_cd_brake"], self.cfg.fin_args()[1], places=5)
        self.assertGreater(b["fin_cd_brake"], b["fin_cd_zero"])
        self.assertEqual(b["feasible_vz"], FEASIBLE_VZ)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}|unknown", b["landsim_git"]))
        self.assertNotIn("gains", b)
        self.assertNotIn("motor_table", b)

    def test_band_is_a_physical_band(self):
        hmin, hmax = np.array(self.band["hmin"]), np.array(self.band["hmax"])
        self.assertTrue(np.all(hmax > 0.0) and np.all(hmax > hmin))
        self.assertTrue(np.all(np.diff(hmin) > 0.0), "faster fall needs a higher ignition")
        self.assertTrue(np.all(np.diff(hmax) > 0.0))

    def test_motor_table_round_trips_as_a_motor_curve(self):
        with_table = ex.build_document(self.cfg, n=3, with_motor_table=True, workers=1)
        again = ex.config_from_vehicle({"motor_curve": with_table["motor_table"]})
        self.assertEqual(ex.compute_band(again, n=3, workers=1),
                         (with_table["hmin"], with_table["hmax"]))


class PublicInterface(unittest.TestCase):
    """The names and shapes the flight-software side relies on."""

    def test_stable_names_and_signatures(self):
        self.assertEqual(FEASIBLE_VZ, -2.3)
        self.assertEqual(list(inspect.signature(project_rh).parameters)[:7],
                         ["h0", "vz0", "t0", "mass_extra", "mt", "mf", "mc"])
        self.assertEqual(
            list(inspect.signature(find_ignition).parameters),
            ["h_start", "vx0", "vz0", "cd", "cd_free", "area", "k_min", "k_max",
             "mass0", "delay_pad", "boost_ready", "mt", "mf", "mc", "prop_mass",
             "i_total", "bt", "bf", "bb", "b_prop", "plan_target"])

    def test_find_ignition_is_the_prefix_of_find_ignition_full(self):
        cfg = ex.config_from_vehicle({})
        p = tvc_sim.planner_inputs(cfg)
        args = (150.0, 0.0, 0.0, p.cd_burn, p.cd_free, p.area, p.k_min, p.k_max,
                p.mass0, p.delay_pad, p.ready, p.mt, p.mf, p.mc, p.prop_mass,
                p.i_total, p.bt, p.bf, p.bb, p.b_prop, p.plan_target)
        short, full = find_ignition(*args), tvc_sim.find_ignition_full(*args)
        self.assertEqual(len(short), 3)
        self.assertEqual(tuple(short), tuple(full[:3]))
        self.assertAlmostEqual(tvc_sim.plan_ignition(cfg, 150.0, 0.0), short[0])


class VehicleReachesTheModel(unittest.TestCase):
    """Nothing is hardcoded: every vehicle.json number lands in TvcConfig or the
    motor, and moves the exported numbers where physics says it must."""

    FULL = {
        "mass_gross_kg": 3.4, "mass_prop_kg": 0.30, "cd_body": 0.42,
        "diameter_m": 0.11, "thrust_scale": 0.92, "fin_brake": 1, "fin_enable": 1,
        "fin_area_m2": 0.0075, "fin_arm_m": 0.55, "fin_cl_alpha": 2.0,
        "fin_stall_deg": 14.0, "inertia": {"transverse": 0.62, "roll": 0.0055},
    }

    def test_every_key_lands_in_the_config(self):
        cfg = ex.config_from_vehicle(self.FULL)
        self.assertEqual(cfg.gross_mass, 3.4)
        self.assertEqual(cfg.propellant, 0.30)
        self.assertEqual(cfg.cd, 0.42)
        self.assertEqual(cfg.diameter, 0.11)
        self.assertEqual(cfg.thrust_mult, 0.92)
        self.assertTrue(cfg.fins)
        self.assertEqual(cfg.fin_brake, "auto")
        veh = cfg.vehicle()
        self.assertAlmostEqual(veh[tvc_sim.V_AREA], math.pi * 0.055 ** 2)
        self.assertAlmostEqual(veh[tvc_sim.V_IT], 0.62 / 3.4)
        self.assertAlmostEqual(veh[tvc_sim.V_IA], 0.0055 / 3.4)
        fa, cd_free = cfg.fin_args()
        n, area, arm, _, cl_alpha = fa[:5]
        self.assertEqual((n, area, arm, cl_alpha), (4.0, 0.0075, 0.55, 2.0))
        self.assertEqual(cfg.fin_set().stall, 14.0)
        self.assertEqual(cfg.fin_aero, "analytic")        # a CL_alpha was supplied
        m, _ = cfg.tables()
        default_peak = landsim.Motor("long").peak_thrust
        self.assertAlmostEqual(m.peak_thrust, 0.92 * default_peak)
        self.assertEqual(m.propellant_mass, 0.30)
        self.assertGreater(cd_free, cfg.cd_burn)

    def test_inertia_defaults_follow_the_mass(self):
        a = ex.config_from_vehicle({"mass_gross_kg": 2.85})
        b = ex.config_from_vehicle({"mass_gross_kg": 5.7})
        self.assertAlmostEqual(b.mmoi_transverse, 2.0 * a.mmoi_transverse)
        self.assertAlmostEqual(b.mmoi_roll, 2.0 * a.mmoi_roll)

    def test_fin_switches(self):
        on = ex.config_from_vehicle({"fin_enable": 1, "fin_brake": 1})
        no_brake = ex.config_from_vehicle({"fin_enable": 1, "fin_brake": 0})
        off = ex.config_from_vehicle({"fin_enable": 0, "fin_brake": 1})
        cd = lambda c: (c.cd_burn, c.fin_args()[1])               # noqa: E731
        self.assertGreater(cd(on)[1], cd(on)[0])
        self.assertEqual(cd(no_brake)[1], cd(no_brake)[0])
        self.assertGreater(cd(no_brake)[0], on.cd)                 # fins still drag
        self.assertEqual(cd(off), (off.cd, off.cd))                # no fins, no drag
        for cfg, brake in ((on, 1), (no_brake, 0), (off, 0)):
            doc = ex.build_document(cfg, n=1, workers=1)
            self.assertEqual(doc["vehicle"]["fin_brake"], brake)
        self.assertEqual(ex.build_document(off, n=1, workers=1)["vehicle"]["fin_enable"], 0)

    def test_fin_stall_limits_the_brake(self):
        free = ex.config_from_vehicle({})
        stalled = ex.config_from_vehicle({"fin_stall_deg": 12.0})
        self.assertLess(stalled.fin_args()[1], free.fin_args()[1])

    def test_motor_curve_replaces_the_table(self):
        curve = [[0, 0], [0.1, 60], [0.5, 120], [2.0, 110], [2.6, 40], [2.8, 0]]
        cfg = ex.config_from_vehicle({"motor_curve": curve, "thrust_scale": 1.1})
        m, _ = cfg.tables()
        self.assertTrue(np.allclose(m.t, [t for t, _ in curve]))
        self.assertTrue(np.allclose(m.f, [1.1 * f for _, f in curve]))
        self.assertAlmostEqual(
            m.total_impulse, 1.1 * np.trapezoid([f for _, f in curve],
                                                [t for t, _ in curve]))

    def test_the_band_moves_with_the_vehicle(self):
        base_cfg, base_lo, base_hi = small_band({})
        for change in ({"mass_gross_kg": 3.4}, {"mass_prop_kg": 0.40},
                       {"cd_body": 0.50}, {"diameter_m": 0.13},
                       {"thrust_scale": 0.85}, {"fin_enable": 0},
                       {"fin_area_m2": 0.012}):
            cfg, lo, hi = small_band(change)
            self.assertNotEqual((lo, hi), (base_lo, base_hi),
                                f"{change} did not move the band")
            self.assertGreater(len(cfg.tables()[0].t), 1)
        # direction, not just difference: a heavier vehicle needs more room ...
        _, lo, _ = small_band({"mass_gross_kg": 3.4})
        self.assertTrue(all(a > b for a, b in zip(lo, base_lo)))
        # ... and a weaker motor too
        _, lo, _ = small_band({"thrust_scale": 0.85})
        self.assertTrue(all(a > b for a, b in zip(lo, base_lo)))

    def test_no_feasible_band_is_reported_as_hmax_zero(self):
        # peak thrust under the vehicle's weight: nothing can be done at any height
        _, lo, hi = small_band({"thrust_scale": 0.15})
        self.assertEqual(hi, [0.0, 0.0])
        self.assertEqual(lo, [0.0, 0.0])

    def test_bad_input_is_refused(self):
        bad = [
            {"mas_gross_kg": 3.0},                       # typo
            {"mass_gross_kg": 0.2},                      # lighter than its propellant
            {"thrust_scale": 0},
            {"thrust_scale": "big"},
            {"fin_enable": 2},
            {"fin_brake": "sometimes"},
            {"cd_body": float("nan")},
            {"gross_mass": 3.0, "mass_gross_kg": 3.0},   # alias and field together
            {"inertia": {"pitch": 1.0}},
            {"inertia": {"roll": 0.1}, "mmoi_roll": 0.1},
            {"motor_curve": [[0, 0]]},
            {"motor_curve": [[0, 10], [1, 5], [0.5, 0]]},
            {"motor_curve": [[0, 0], [1, 0]]},
            [],
        ]
        for vehicle in bad:
            with self.assertRaises(ValueError, msg=str(vehicle)):
                ex.config_from_vehicle(vehicle)

    def test_any_tvcconfig_field_is_reachable_by_name(self):
        cfg = ex.config_from_vehicle({"att_ki": 3.5, "k_min": 0.2, "n_boosters": 0,
                                      "use_booster_rule": 0, "comment": "ignored"})
        self.assertEqual((cfg.att_ki, cfg.k_min, cfg.n_boosters), (3.5, 0.2, 0))
        self.assertFalse(cfg.use_booster_rule)


class Outputs(unittest.TestCase):
    def test_motor_table_is_after_thrust_scale(self):
        base = ex.motor_table(ex.config_from_vehicle({}))
        scaled = ex.motor_table(ex.config_from_vehicle({"thrust_scale": 0.9}))
        self.assertEqual([r[0] for r in base], [r[0] for r in scaled])
        for (_, f0), (_, f1) in zip(base, scaled):
            self.assertAlmostEqual(f1, 0.9 * f0, places=4)
        self.assertEqual(base[-1][1], 0.0)

    def test_grid_options_and_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "b.json")
            ex.main(["--out", out, "--vz0", "10", "--dvz", "5", "--n", "3",
                     "--workers", "1", "--motor-table", "--quiet"])
            with open(out) as fh:
                doc = json.load(fh)
            self.assertEqual((doc["vz0"], doc["dvz"], doc["n"]), (10.0, 5.0, 3))
            self.assertEqual(len(doc["hmin"]), 3)
            self.assertIn("motor_table", doc)
            self.assertEqual(os.listdir(d), ["b.json"], "no temp file left behind")

    def test_bad_vehicle_file_exits_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            veh = os.path.join(d, "v.json")
            with open(veh, "w") as fh:
                json.dump({"mas_gross_kg": 3}, fh)
            with self.assertRaises(SystemExit) as cm:
                ex.main(["--vehicle", veh, "--out", os.path.join(d, "b.json"),
                         "--quiet"])
            self.assertEqual(cm.exception.code, 2)
            self.assertFalse(os.path.exists(os.path.join(d, "b.json")))


class Gains(unittest.TestCase):
    def test_gains_are_fitted_for_the_given_vehicle(self):
        """--gains hands the tuner THIS vehicle's mass and inertia and exports what
        it returns under band.json's own names (the tuner itself is replaced here)."""
        seen = {}

        def fake_tune(cfg, **kw):
            seen["cfg"] = cfg
            return (7.0, 1.1, 5.0, 0.9, 2.5, 0.2, -0.3), {"cost": 0.01, "success": 0.99}

        def fake_ki(cfg, **kw):
            seen["ki_cfg"] = cfg
            return 3.0, {2.0: 0.2, 3.0: 0.1}

        vehicle = {"mass_gross_kg": 3.4, "inertia": {"transverse": 0.62, "roll": 0.0055}}
        cfg = ex.config_from_vehicle(vehicle)
        with mock.patch.object(tvc_sim, "tune_gains", fake_tune), \
                mock.patch.object(ex, "fit_att_ki", fake_ki):
            doc = ex.build_document(cfg, n=2, with_gains=True, workers=1)
        self.assertEqual(seen["cfg"].gross_mass, 3.4)
        self.assertEqual(seen["cfg"].mmoi_transverse, 0.62)
        self.assertEqual(seen["cfg"].mmoi_roll, 0.0055)
        self.assertEqual(seen["ki_cfg"].wn, 7.0)          # att_ki is fitted on the tuned set
        self.assertEqual(doc["gains"], {
            "wn_tvc": 7.0, "zeta_tvc": 1.1, "wn_fin": 5.0, "zeta_fin": 0.9,
            "roll_gain": 2.5, "sched_tvc": 0.2, "sched_fin": -0.3, "att_ki": 3.0})
        self.assertEqual(doc["gains_fit"]["cost"], 0.01)

    @unittest.skipUnless(os.environ.get("LANDSIM_SLOW_TESTS"),
                         "set LANDSIM_SLOW_TESTS=1 to fly the real (small) fit")
    def test_real_fit_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "b.json")
            ex.main(["--out", out, "--n", "2", "--gains", "--gains-budget", "6",
                     "--gains-runs", "3", "--quiet"])
            with open(out) as fh:
                gains = json.load(fh)["gains"]
        self.assertEqual(set(gains), set(ex.GAIN_EXPORT))
        lo, hi = tvc_sim.GAIN_BOUNDS[:, 0], tvc_sim.GAIN_BOUNDS[:, 1]
        for (name, bound_lo, bound_hi) in zip(tvc_sim.GAIN_NAMES, lo, hi):
            export = [k for k, v in ex.GAIN_EXPORT.items() if v == name][0]
            self.assertTrue(bound_lo <= gains[export] <= bound_hi, export)
        self.assertGreaterEqual(gains["att_ki"], 0.0)


if __name__ == "__main__":
    unittest.main()
