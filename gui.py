"""
Very simple Tkinter GUI for LandSim Light.

Lets the user edit the vehicle / scenario parameters and shows the resulting
ignition-altitude window for a soft landing: the lowest and the highest ignition
altitude, the speed at ignition and the touchdown speed for both, plus the
throttle profile that achieves it.

Run with:   python3 gui.py
"""

from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import landsim
from landsim import (Config, Motor, Booster, MOTOR_TABLES, NEVER, Cancelled,
                      find_window, ascent, G)

# label, attribute key, default, unit, tooltip-ish hint
FIELDS = [
    ("Drop altitude",      "drop_alt",     "150",   "m",     "where the free fall starts"),
    ("Initial velocity",   "v0",           "0",     "m/s",   "+ up, 0 = released from rest"),
    ("Gross mass",         "mass",         "3.2",   "kg",    "incl. propellant"),
    ("Propellant mass",    "propellant",   "0.277", "kg",    "consumed during the burn"),
    ("Diameter",           "diameter",     "105",   "mm",    "reference diameter"),
    ("Reference area",     "area",         "",      "cm2",   "leave empty = from diameter"),
    ("Cd",                 "cd",           "0.35",  "-",     "drag coefficient"),
    ("Air density",        "rho",          "1.225", "kg/m3", ""),
    ("Thrust multiplier",  "thrust_mult",  "1.0",   "x",     "scales the whole motor table"),
    ("Soft-landing limit", "limit",        "3.0",   "m/s",   "max touchdown speed"),
    ("Search from",        "search_min",   "1",     "m",     "lowest ignition altitude tried"),
    ("Search to",          "search_max",   "",      "m",     "empty = drop altitude"),
    ("Min. throttle",      "throttle_min", "0.1",   "-",     "0.1 = flaps block 90 %"),
    ("Throttle phase",     "phase",        "0.1",   "s",     "length of one throttle step"),
    ("Time step",          "dt",           "0.002", "s",     "integration step"),
    ("Scan step",          "coarse_step",  "5",     "m",     "ignition-altitude scan"),
    ("Edge tolerance",     "tol",          "0.25",  "m",     "bisection precision"),
    ("Population",         "pop",          "64",    "-",     "optimiser population"),
    ("Generations",        "gen",          "120",   "-",     "optimiser generations"),
]


class CircularConfig(Config):
    """Config that allows the reference area to be overridden directly."""

    area_override: float | None = None

    @property
    def area(self) -> float:
        if getattr(self, "area_override", None):
            return self.area_override
        return math.pi * (self.diameter / 2.0) ** 2


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LandSim Light - propulsive landing")
        root.minsize(760, 560)

        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()

        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)

        # ---------------- inputs ----------------
        inp = ttk.LabelFrame(main, text="Inputs", padding=8)
        inp.pack(fill="x")

        # motor selector
        motor_row = ttk.Frame(inp)
        motor_row.grid(row=99, column=0, columnspan=8, sticky="w", pady=(6, 2))
        ttk.Label(motor_row, text="Motor:").pack(side="left", padx=(6, 6))
        self.motor_var = tk.StringVar(value="long")
        for key, text in (("long", "long  (120 N peak, 2.61 s burn)"),
                          ("short", "short  (269 N peak, 1.55 s burn)")):
            ttk.Radiobutton(motor_row, text=text, value=key,
                            variable=self.motor_var,
                            command=self.show_motor).pack(side="left", padx=(0, 14))
        self.motor_info = tk.StringVar(value="")
        ttk.Label(motor_row, textvariable=self.motor_info,
                  foreground="#666").pack(side="left")

        # D9 boosters - non-throttleable, single-shot, the optimiser decides
        boost_row = ttk.Frame(inp)
        boost_row.grid(row=100, column=0, columnspan=8, sticky="w", pady=(0, 4))
        ttk.Label(boost_row, text="Klima D9 boosters:").pack(side="left", padx=(6, 6))
        self.boosters = tk.StringVar(value="0")
        ttk.Spinbox(boost_row, from_=0, to=3, width=3,
                    textvariable=self.boosters).pack(side="left")
        _b = Booster()
        ttk.Label(boost_row,
                  text=f"  ({_b.peak_thrust:.0f} N peak, {_b.burn_time:.2f} s, "
                       f"{_b.total_impulse:.1f} Ns, {_b.total_mass * 1000:.1f} g each) "
                       f"- cannot be throttled, stopped or relit; the optimiser "
                       f"decides if and when to light each one",
                  foreground="#666").pack(side="left")
        ttk.Label(boost_row, text="ignition window:").pack(side="left", padx=(10, 2))
        self.booster_window = tk.StringVar(value="")
        ttk.Entry(boost_row, textvariable=self.booster_window, width=6).pack(side="left")
        ttk.Label(boost_row, text="s (empty = burn + 4 s)",
                  foreground="#666").pack(side="left", padx=(3, 0))

        self.vars: dict[str, tk.StringVar] = {}
        for i, (label, key, default, unit, hint) in enumerate(FIELDS):
            col, row = (i % 2) * 4, i // 2
            ttk.Label(inp, text=label + ":").grid(row=row, column=col, sticky="e",
                                                  padx=(6, 4), pady=2)
            var = tk.StringVar(value=default)
            self.vars[key] = var
            entry = ttk.Entry(inp, textvariable=var, width=9)
            entry.grid(row=row, column=col + 1, sticky="w", pady=2)
            if key == "thrust_mult":
                var.trace_add("write", lambda *_: self.show_motor())
            ttk.Label(inp, text=unit, foreground="#666", width=6).grid(
                row=row, column=col + 2, sticky="w")
            ttk.Label(inp, text=hint, foreground="#999").grid(
                row=row, column=col + 3, sticky="w", padx=(0, 10))

        # ---------------- buttons ----------------
        bar = ttk.Frame(main)
        bar.pack(fill="x", pady=(8, 4))
        self.run_btn = ttk.Button(bar, text="Compute", command=self.start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Defaults", command=self.reset).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar, text="Apogee from ground",
                   command=self.run_ascent).pack(side="left")
        ttk.Label(bar, text="throttle:").pack(side="left", padx=(8, 2))
        self.ascent_throttle = tk.StringVar(value="1.0")
        ttk.Entry(bar, textvariable=self.ascent_throttle, width=5).pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        # ---------------- results ----------------
        res = ttk.LabelFrame(main, text="Result", padding=8)
        res.pack(fill="x")
        self.summary = tk.StringVar(value="-")
        ttk.Label(res, textvariable=self.summary, font=("TkDefaultFont", 11, "bold"),
                  foreground="#0a5").pack(anchor="w", pady=(0, 2))
        self.timing = tk.StringVar(value="")
        ttk.Label(res, textvariable=self.timing,
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 6))

        grid = ttk.Frame(res)
        grid.pack(fill="x")
        heads = ["", "Ignition altitude", "Time from release", "Speed at ignition",
                 "Touchdown speed"]
        for c, h in enumerate(heads):
            ttk.Label(grid, text=h, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=c, sticky="w", padx=8)
        self.cells = {}
        for r, name in enumerate(("LOWEST ignition", "HIGHEST ignition"), start=1):
            ttk.Label(grid, text=name).grid(row=r, column=0, sticky="w", padx=8)
            for c, key in enumerate(("alt", "t", "entry", "td"), start=1):
                v = tk.StringVar(value="-")
                self.cells[(r, key)] = v
                ttk.Label(grid, textvariable=v).grid(row=r, column=c, sticky="w", padx=8)

        # ---------------- log ----------------
        logf = ttk.LabelFrame(main, text="Progress / throttle profiles", padding=6)
        logf.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(logf, height=14, wrap="none",
                           font=("TkFixedFont", 9))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

        self.show_motor()
        self.root.after(100, self.poll)

    def show_motor(self):
        try:
            m = Motor(self.motor_var.get(),
                      thrust_multiplier=self.num("thrust_mult", 1.0))
        except (ValueError, KeyError):
            self.motor_info.set("")
            return
        self.motor_info.set(f"burn {m.burn_time:.3f} s, peak {m.peak_thrust:.1f} N, "
                            f"impulse {m.total_impulse:.1f} Ns")

    # ------------------------------------------------------------------ #
    def reset(self):
        for _, key, default, _, _ in FIELDS:
            self.vars[key].set(default)
        self.motor_var.set("long")
        self.boosters.set("0")
        self.booster_window.set("")
        self.show_motor()

    def num(self, key, default=None):
        txt = self.vars[key].get().strip().replace(",", ".")
        if txt == "":
            if default is not None:
                return default
            raise ValueError(f"'{key}' is empty")
        return float(txt)

    def build_config(self) -> CircularConfig:
        cfg = CircularConfig(
            drop_altitude=self.num("drop_alt"),
            gross_mass=self.num("mass"),
            diameter=self.num("diameter") / 1000.0,
            cd=self.num("cd"),
            rho=self.num("rho"),
            initial_velocity=self.num("v0", 0.0),
            max_touchdown_speed=self.num("limit"),
            throttle_min=self.num("throttle_min"),
            phase_length=self.num("phase"),
            dt=self.num("dt"),
            n_boosters=max(0, int(float(self.boosters.get() or 0))),
            booster_window=(float(self.booster_window.get().replace(",", "."))
                            if self.booster_window.get().strip() else None),
            motor=Motor(self.motor_var.get(),
                        propellant_mass=self.num("propellant"),
                        thrust_multiplier=self.num("thrust_mult", 1.0)),
        )
        area_cm2 = self.vars["area"].get().strip()
        cfg.area_override = float(area_cm2.replace(",", ".")) / 1e4 if area_cm2 else None
        if cfg.gross_mass <= cfg.motor.propellant_mass:
            raise ValueError("gross mass must be larger than the propellant mass")
        return cfg

    # ------------------------------------------------------------------ #
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self.build_config()
            s_max = self.vars["search_max"].get().strip()
            params = dict(coarse_step=self.num("coarse_step"), tol=self.num("tol"),
                          pop_size=int(self.num("pop")), generations=int(self.num("gen")),
                          search_min=self.num("search_min", 1.0),
                          search_max=float(s_max.replace(",", ".")) if s_max else None)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.log.delete("1.0", "end")
        for key in self.cells:
            self.cells[key].set("-")
        self.timing.set("")
        self.summary.set("computing ...")
        m = cfg.motor
        self.put("log", f"motor '{m.name}', thrust multiplier {m.thrust_multiplier:g}x, "
                        f"{cfg.n_boosters} D9 booster(s) available\n"
                        f"burn {m.burn_time:.3f} s, peak {m.peak_thrust:.1f} N, "
                        f"total impulse {m.total_impulse:.1f} Ns, "
                        f"peak T/W {m.peak_thrust / (cfg.gross_mass * G):.2f}, "
                        f"area {cfg.area * 1e4:.1f} cm2\n"
                        f"{cfg.n_phases} throttle phases of "
                        f"{cfg.phase_length * 1000:.0f} ms\n")

        self.stop_flag.clear()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(12)
        self.worker = threading.Thread(target=self.work, args=(cfg, params), daemon=True)
        self.worker.start()

    def run_ascent(self):
        """Side calculation: vertical launch from the ground, no optimisation."""
        try:
            cfg = self.build_config()
            thr = float(self.ascent_throttle.get().strip().replace(",", ".") or 1.0)
            if not 0.0 < thr <= 1.0:
                raise ValueError("ascent throttle must be in (0, 1]")
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        a = ascent(cfg, throttle=thr)
        self.log.insert("end", f"\n--- vertical launch from the ground, motor "
                               f"'{cfg.motor.name}', throttle {thr:g}, "
                               f"{cfg.n_boosters} D9 lit at lift-off ---\n")
        if a["liftoff_time"] is None:
            self.summary.set("The rocket never lifts off "
                             "(thrust stays below the weight).")
            self.log.insert("end", "  no lift-off\n")
        else:
            self.summary.set(f"Apogee from the ground: {a['apogee']:.1f} m "
                             f"(burnout {a['burnout_altitude']:.1f} m, "
                             f"max speed {a['max_speed']:.1f} m/s)")
            self.log.insert("end",
                            f"  burnout   : {a['burnout_time']:.2f} s at "
                            f"{a['burnout_altitude']:.1f} m, "
                            f"{a['burnout_speed']:.1f} m/s\n"
                            f"  max speed : {a['max_speed']:.1f} m/s, "
                            f"max accel {a['max_acceleration'] / G:.1f} g\n"
                            f"  APOGEE    : {a['apogee']:.1f} m at "
                            f"t = {a['time_to_apogee']:.2f} s "
                            f"(coast {a['coast_height']:.1f} m)\n")
        self.timing.set("")
        self.log.see("end")

    def stop(self):
        self.stop_flag.set()
        self.put("log", "stopping ...")

    def put(self, kind, payload):
        self.queue.put((kind, payload))

    # ------------------------------------------------------------------ #
    def work(self, cfg, params):
        try:
            win = find_window(cfg, verbose=False,
                              on_progress=lambda s: self.put("log", s),
                              should_stop=self.stop_flag.is_set, **params)
            self.put("done", (cfg, win))
        except Cancelled:
            self.put("cancelled", None)
        except Exception as exc:                      # noqa: BLE001
            self.put("error", f"{type(exc).__name__}: {exc}")

    def poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "done":
                    self.finish(*payload)
                elif kind == "cancelled":
                    self.summary.set("cancelled")
                    self.idle()
                elif kind == "error":
                    self.summary.set("error")
                    messagebox.showerror("Simulation failed", payload)
                    self.idle()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def idle(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def finish(self, cfg, win):
        self.idle()
        if win is None:
            self.summary.set(f"No ignition altitude in the searched range lands "
                             f"below {cfg.max_touchdown_speed:g} m/s.")
            self.timing.set("")
            return
        (lo_alt, lo), (hi_alt, hi) = win["low"], win["high"]
        self.summary.set(f"Soft landing possible for ignition altitudes "
                         f"{lo_alt:.2f} m ... {hi_alt:.2f} m "
                         f"(window {hi_alt - lo_alt:.2f} m)")
        for r, alt, res in ((1, lo_alt, lo), (2, hi_alt, hi)):
            self.cells[(r, "alt")].set(f"{alt:.2f} m")
            self.cells[(r, "t")].set(f"{res['fall_time']:.3f} s")
            self.cells[(r, "entry")].set(f"{res['entry_speed']:.2f} m/s")
            self.cells[(r, "td")].set(f"{res['touchdown_speed']:.2f} m/s")
        dt_ign = lo["fall_time"] - hi["fall_time"]
        self.timing.set(f"Ignition must happen {hi['fall_time']:.3f} s ... "
                        f"{lo['fall_time']:.3f} s after the release from "
                        f"{cfg.drop_altitude:g} m  ->  time window {dt_ign:.3f} s")

        for name, alt, res in (("LOWEST", lo_alt, lo), ("HIGHEST", hi_alt, hi)):
            self.log.insert("end", f"\n{name} ignition altitude {alt:.2f} m -> "
                                   f"touchdown {res['touchdown_speed']:.2f} m/s\n")
            ign = res.get("booster_ignitions")
            if ign is not None and len(ign):
                used = [f"{x:.2f} s" for x in ign if x < NEVER]
                self.log.insert("end", "  D9 booster ignitions: "
                                       + (", ".join(used) if used else "none used")
                                       + f"  (of {len(ign)} carried)\n")
            self.log.insert("end",
                            f"  throttle per {cfg.phase_length * 1000:.0f} ms phase:\n")
            prof = res["profile"]
            for i in range(0, len(prof), 10):
                chunk = prof[i:i + 10]
                self.log.insert("end", f"    t={i * cfg.phase_length:5.2f}s : "
                                       + " ".join(f"{x:.2f}" for x in chunk) + "\n")
            self.log.insert("end", "  total thrust [N] (main + boosters):\n")
            for i in range(0, len(prof), 10):
                chunk = prof[i:i + 10]
                vals = [landsim.phase_thrust(cfg, (i + j + 0.5) * cfg.phase_length,
                                             chunk[j], res.get("booster_ignitions"))
                        for j in range(len(chunk))]
                self.log.insert("end", f"    t={i * cfg.phase_length:5.2f}s : "
                                       + " ".join(f"{x:6.1f}" for x in vals) + "\n")
        self.log.see("end")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
