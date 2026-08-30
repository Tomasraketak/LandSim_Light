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
from landsim import Config, Motor, Cancelled, find_window, G

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
    ("Soft-landing limit", "limit",        "3.0",   "m/s",   "max touchdown speed"),
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

        self.vars: dict[str, tk.StringVar] = {}
        for i, (label, key, default, unit, hint) in enumerate(FIELDS):
            col, row = (i % 2) * 4, i // 2
            ttk.Label(inp, text=label + ":").grid(row=row, column=col, sticky="e",
                                                  padx=(6, 4), pady=2)
            var = tk.StringVar(value=default)
            self.vars[key] = var
            ttk.Entry(inp, textvariable=var, width=9).grid(row=row, column=col + 1,
                                                           sticky="w", pady=2)
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
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        # ---------------- results ----------------
        res = ttk.LabelFrame(main, text="Result", padding=8)
        res.pack(fill="x")
        self.summary = tk.StringVar(value="-")
        ttk.Label(res, textvariable=self.summary, font=("TkDefaultFont", 11, "bold"),
                  foreground="#0a5").pack(anchor="w", pady=(0, 6))

        grid = ttk.Frame(res)
        grid.pack(fill="x")
        heads = ["", "Ignition altitude", "Speed at ignition", "Touchdown speed"]
        for c, h in enumerate(heads):
            ttk.Label(grid, text=h, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=c, sticky="w", padx=8)
        self.cells = {}
        for r, name in enumerate(("LOWEST ignition", "HIGHEST ignition"), start=1):
            ttk.Label(grid, text=name).grid(row=r, column=0, sticky="w", padx=8)
            for c, key in enumerate(("alt", "entry", "td"), start=1):
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

        self.root.after(100, self.poll)

    # ------------------------------------------------------------------ #
    def reset(self):
        for _, key, default, _, _ in FIELDS:
            self.vars[key].set(default)

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
            motor=Motor(propellant_mass=self.num("propellant")),
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
            params = dict(coarse_step=self.num("coarse_step"), tol=self.num("tol"),
                          pop_size=int(self.num("pop")), generations=int(self.num("gen")))
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.log.delete("1.0", "end")
        for key in self.cells:
            self.cells[key].set("-")
        self.summary.set("computing ...")
        m = cfg.motor
        self.put("log", f"burn {m.burn_time:.3f} s, peak {m.peak_thrust:.1f} N, "
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
            self.summary.set(f"No ignition altitude lands below "
                             f"{cfg.max_touchdown_speed:g} m/s.")
            return
        (lo_alt, lo), (hi_alt, hi) = win["low"], win["high"]
        self.summary.set(f"Soft landing possible for ignition altitudes "
                         f"{lo_alt:.2f} m ... {hi_alt:.2f} m "
                         f"(window {hi_alt - lo_alt:.2f} m)")
        for r, alt, res in ((1, lo_alt, lo), (2, hi_alt, hi)):
            self.cells[(r, "alt")].set(f"{alt:.2f} m")
            self.cells[(r, "entry")].set(f"{res['entry_speed']:.2f} m/s")
            self.cells[(r, "td")].set(f"{res['touchdown_speed']:.2f} m/s")

        for name, alt, res in (("LOWEST", lo_alt, lo), ("HIGHEST", hi_alt, hi)):
            self.log.insert("end", f"\n{name} ignition altitude {alt:.2f} m -> "
                                   f"touchdown {res['touchdown_speed']:.2f} m/s\n"
                                   f"  throttle per {cfg.phase_length * 1000:.0f} ms phase:\n")
            prof = res["profile"]
            for i in range(0, len(prof), 10):
                chunk = prof[i:i + 10]
                self.log.insert("end", f"    t={i * cfg.phase_length:5.2f}s : "
                                       + " ".join(f"{x:.2f}" for x in chunk) + "\n")
            self.log.insert("end", "  effective thrust [N]:\n")
            for i in range(0, len(prof), 10):
                chunk = prof[i:i + 10]
                vals = [cfg.motor.thrust((i + j + 0.5) * cfg.phase_length) * chunk[j]
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
