"""
Tkinter front end for LandSim Light.

Three pages:

  * **Vehicle**  - everything about the rocket itself: airframe, inertia, arms, the
    two actuators, the motor and the fins. Both simulations read this page, so the
    vehicle is described once and the two scenarios below only describe the flight.
  * **1-D ignition window** - the throttle-profile search from `landsim.py`.
  * **3-D / TVC Monte Carlo** - the campaign from `tvc_sim.py`, with its own large
    log: a campaign has a lot to say and it is worth reading.

Run with:   python3 gui.py
"""

from __future__ import annotations

import io
import math
import os
import numpy as np
import contextlib
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import landsim
import tvc_sim
import orkimport

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D             # noqa: F401
    HAVE_MPL, MPL_ERR = True, ""
except Exception as _exc:                                # noqa: BLE001
    HAVE_MPL, MPL_ERR = False, f"{type(_exc).__name__}: {_exc}"
from landsim import (Config, Motor, Booster, MOTOR_TABLES, NEVER, Cancelled,
                     find_window, ascent, G)

# --------------------------------------------------------------------------- #
#  Field tables.  (label, key, default, unit, hint)
#  Every numeric input in the whole GUI lives in one dict of StringVars keyed by
#  the second column, so any page can read any field.
# --------------------------------------------------------------------------- #
AIRFRAME_FIELDS = [
    ("Gross mass",        "mass",         "2.85",  "kg",    "incl. propellant"),
    ("Propellant mass",   "propellant",   "0.277", "kg",    "consumed during the burn"),
    ("Diameter",          "diameter",     "105",   "mm",    "reference diameter"),
    ("Reference area",    "area",         "",      "cm2",   "empty = from the diameter (1-D only)"),
    ("Cd (axial)",        "cd",           "0.35",  "-",     "drag coefficient"),
    ("Air density",       "rho",          "1.225", "kg/m3", "sea level = 1.225"),
]

INERTIA_FIELDS = [
    ("MMOI transverse",   "mmoi",         "0.466", "kg m2", "pitch/yaw, at the gross mass"),
    ("MMOI roll",         "mmoi_r",       "0.0039", "kg m2", "about the long axis"),
    ("Gimbal arm",        "lgimb",        "0.50",  "m",     "CG -> nozzle pivot"),
    ("CP arm (aft)",      "lcp",          "0.35",  "m",     "CG -> centre of pressure"),
    ("Body CN_alpha",     "cna",          "2.0",   "1/rad", "normal-force slope"),
]

ACTUATOR_FIELDS = [
    ("Servo travel",      "srvmax",       "10",    "deg +/-", "TVC servo limit"),
    ("Servo ratio",       "srvratio",     "2.0",   "srv/nozzle", "linkage"),
    ("Servo speed",       "srvspd",       "500",   "deg/s", ""),
    ("Servo accel",       "srvacc",       "2000",  "deg/s2", ""),
    ("Servo step",        "srvq",         "0.15",  "deg",   "command quantisation"),
    ("Clamp min",         "throttle_min", "0.10",  "-",     "0.10 = flaps spoil 90 %"),
    ("Clamp max",         "kmax",         "1.00",  "-",     ""),
    ("Clamp speed",       "thrspd",       "12.84", "1/s",   "clamp travel per second"),
    ("Clamp accel",       "thracc",       "257",   "1/s2",  ""),
]

MOTOR_FIELDS = [
    ("Thrust multiplier", "thrust_mult",  "1.0",   "x",     "scales the whole table"),
    ("D9 boosters",       "boosters",     "1",     "-",     "carried; the rule decides"),
    ("D9 cant",           "b_cant",       "15",    "deg",   "aimed through the CG"),
    ("D9 mount azimuth",  "b_azim",       "0",     "deg",   "body-fixed"),
    ("D9 window (1-D)",   "booster_window", "",    "s",     "empty = burn + 4 s"),
]

FIN_FIELDS = [
    ("Fin count",         "fin_count",    "4",     "-",     ""),
    ("Max deflection",    "fin_deflect",  "15",    "deg +/-", ""),
    ("Travel time",       "fin_travel",   "0.05",  "s",     "end stop to end stop "
                                                    "(BMS-117WV+ derated)"),
    ("Arm from CG",       "fin_arm",      "0.50",  "m",     "aft"),
    ("Root chord",        "fin_root",     "120",   "mm",    ""),
    ("Tip chord",         "fin_tip",      "63",    "mm",    ""),
    ("Span (height)",     "fin_span",     "70",    "mm",    ""),
]

SCENARIO_1D_FIELDS = [
    ("Drop altitude",     "drop_alt",     "150",   "m",     "where the free fall starts"),
    ("Initial velocity",  "v0",           "0",     "m/s",   "+ up, 0 = from rest"),
    ("Soft-landing limit", "limit",       "3.0",   "m/s",   "max touchdown speed"),
    ("Search from",       "search_min",   "1",     "m",     "lowest ignition altitude"),
    ("Search to",         "search_max",   "",      "m",     "empty = drop altitude"),
    ("Throttle phase",    "phase",        "0.1",   "s",     "one throttle step"),
    ("Time step",         "dt",           "0.002", "s",     "integration step"),
    ("Scan step",         "coarse_step",  "5",     "m",     "altitude scan"),
    ("Edge tolerance",    "tol",          "0.25",  "m",     "bisection precision"),
    ("Population",        "pop",          "64",    "-",     "optimiser population"),
    ("Generations",       "gen",          "120",   "-",     "optimiser generations"),
]

CAMPAIGN_FIELDS = [
    ("Flights per cell",  "runs",         "40",    "-",     ""),
    ("Release from",      "h_lo",         "140",   "m",     ""),
    ("Release to",        "h_hi",         "180",   "m",     ""),
    ("Release step",      "h_step",       "5",     "m",     ""),
    ("|vx| max",          "vx_max",       "7",     "m/s",   ""),
    ("vx step",           "vx_step",      "1",     "m/s",   ""),
    ("Igniter delay",     "ign_delay",    "0.3",   "s",     "U(0, x) from command"),
    ("Delay pad",         "delay_pad",    "",      "s",     "what guidance plans "
                                                   "for; empty = delay + 0.1 s"),
    ("Thrust scatter",    "scatter",      "0.15",  "+/- frac", "instantaneous"),
    ("Scatter window",    "tau",          "0.7",   "s",     "correlation time"),
    ("Roll rate max",     "roll",         "90",    "deg/s", "U(0, x), either sign"),
    ("Figure directory",  "figdir",       "figures", "",    "each run gets a subfolder"),
]

# What the landing gear will survive. These decide pass/fail and nothing else - the
# controller never sees them.
GATE_FIELDS = [
    ("Vertical limit",    "gate_vz",      "4.0",   "m/s",   "touchdown |vz|"),
    ("Horizontal limit",  "gate_vh",      "0.5",   "m/s",   "touchdown |vh| - a scrape tips it over"),
    ("Tilt limit",        "gate_tilt",    "4.0",   "deg",   "off vertical at touchdown"),
    ("Rate limit",        "gate_rate",    "30",    "deg/s", "transverse rate (roll is not gated)"),
]

GAIN_FIELDS = [
    ("TVC bandwidth",     "wn",           "7.886", "rad/s", "motor lit"),
    ("TVC damping",       "zeta",         "0.600", "-",     ""),
    ("TVC schedule",      "sched_tvc",    "0.600", "exp",   "on (T / 100 N)"),
    ("Fin bandwidth",     "wn_fin",       "9.210", "rad/s", "motor unlit"),
    ("Fin damping",       "zeta_fin",     "1.483", "-",     ""),
    ("Fin schedule",      "sched_fin",    "0.470", "exp",   "on (q / 700 Pa)"),
    ("Roll damper",       "roll_gain",    "0.300", "rad/s", ""),
    ("Tilt cone at pad",  "tilt_min",     "1.5",   "deg",   "keep it WELL under the tilt gate"),
    ("Tilt cone slope",   "tilt_slope",   "1.5",   "deg/m", "how fast the cone opens with altitude"),
    ("Tilt cone cap",     "tilt_cap",     "20",    "deg",   "the widest it ever gets"),
    ("Tune candidates",   "tune_budget",  "60",    "sets",  ""),
    ("Tune runs/cell",    "tune_runs",    "12",    "flights", ""),
    ("Tune workers",      "tune_workers", "0",     "-",     "0 = one per core"),
]

ALL_FIELDS = (AIRFRAME_FIELDS + INERTIA_FIELDS + ACTUATOR_FIELDS + MOTOR_FIELDS
              + FIN_FIELDS + SCENARIO_1D_FIELDS + CAMPAIGN_FIELDS + GATE_FIELDS
              + GAIN_FIELDS)


MPL_HINT = ("  A 'No module named matplotlib.backends.registry' error means a HALF-"
            "UPGRADED matplotlib:\n  new .py files over an old install. Clean it out "
            "and reinstall:\n"
            "      pip uninstall -y matplotlib\n"
            "      pip uninstall -y matplotlib      (run it twice - old copies hide "
            "behind each other)\n"
            "      pip install --no-cache-dir matplotlib\n"
            "  The campaign results are unaffected; re-draw them later from a saved "
            "run with\n      python tvc_sim.py --load results.npz\n")


# One palette for the whole window, so the pages look like one program.
INK, INK2, ACCENT = "#12232e", "#5a6b73", "#2a78d6"
GOOD, BAD, SURFACE = "#1baf7a", "#e34948", "#f4f5f6"


def style_app(root):
    """A calmer, denser look than the Tk defaults."""
    st = ttk.Style(root)
    for theme in ("clam", "vista", "default"):
        if theme in st.theme_names():
            st.theme_use(theme)
            break
    root.configure(background=SURFACE)
    st.configure(".", background=SURFACE, foreground=INK, fieldbackground="#ffffff")
    st.configure("TLabelframe", background=SURFACE, borderwidth=1, relief="solid")
    st.configure("TLabelframe.Label", background=SURFACE, foreground=ACCENT,
                 font=("TkDefaultFont", 9, "bold"))
    st.configure("TNotebook", background=SURFACE, borderwidth=0)
    st.configure("TNotebook.Tab", padding=(14, 7))
    st.configure("TButton", padding=(10, 4))
    st.configure("Run.TButton", padding=(14, 6), font=("TkDefaultFont", 9, "bold"))
    st.configure("Hint.TLabel", foreground=INK2, background=SURFACE)
    st.configure("Head.TLabel", font=("TkDefaultFont", 11, "bold"), foreground=INK,
                 background=SURFACE)
    st.configure("Good.TLabel", font=("TkDefaultFont", 11, "bold"), foreground=GOOD,
                 background=SURFACE)
    st.configure("TProgressbar", background=ACCENT)
    return st


class Tooltip:
    """Minimal hover tooltip.

    The hints used to sit in a fourth grid column, which is what made this page
    unreadable: a grid column is as wide as its widest cell in ANY row, so one long
    hint pushed every field in that column sideways and the groups ran into each
    other. Hovering is where an explanation belongs anyway.
    """

    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, padx=6, pady=3).pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class CircularConfig(Config):
    """1-D config that allows the reference area to be overridden directly."""

    area_override: float | None = None

    @property
    def area(self) -> float:
        if getattr(self, "area_override", None):
            return self.area_override
        return math.pi * (self.diameter / 2.0) ** 2


class App:
    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LandSim Light")
        root.minsize(1160, 780)
        style_app(root)

        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.t_start = 0.0

        self.vars: dict[str, tk.StringVar] = {
            key: tk.StringVar(value=default) for _, key, default, _, _ in ALL_FIELDS}
        self.motor_var = tk.StringVar(value="long")
        self.motor_info = tk.StringVar(value="")
        self.fin_on = tk.BooleanVar(value=True)
        self.fin_brake = tk.StringVar(value="auto")
        self.fin_drift = tk.BooleanVar(value=False)
        self.fin_info = tk.StringVar(value="")

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)
        veh_tab = ttk.Frame(nb, padding=10)
        nb.add(veh_tab, text="Vehicle")
        oned_tab = ttk.Frame(nb, padding=10)
        nb.add(oned_tab, text="1-D ignition window")
        tvc_tab = ttk.Frame(nb, padding=10)
        nb.add(tvc_tab, text="3-D / TVC Monte Carlo")
        view_tab = ttk.Frame(nb, padding=10)
        nb.add(view_tab, text="Flight viewer")
        chart_tab = ttk.Frame(nb, padding=10)
        nb.add(chart_tab, text="Campaign charts")

        self.build_vehicle_tab(veh_tab)
        self.build_1d_tab(oned_tab)
        self.build_tvc_tab(tvc_tab)
        self.build_viewer_tab(view_tab)
        self.build_charts_tab(chart_tab)
        self.show_motor()
        self.show_fins()
        self.root.after(100, self.poll)

    # ------------------------------------------------------------------ #
    def _fields(self, parent, fields, columns=3, trace=None):
        """Lay a field table out as `columns` groups of label / entry / unit.

        Each group gets its own equally weighted column pair, the label column is
        right-aligned and the hint is a tooltip rather than a fourth cell - a grid
        column is as wide as its widest member, so a long hint used to shove
        everything to its right into the next group.
        """
        grid = ttk.Frame(parent)
        grid.pack(fill="x")
        # every column keeps its natural width and one trailing spacer takes the
        # slack, so the groups stay together on the left instead of being stretched
        # across the whole window
        grid.columnconfigure(columns * 3, weight=1)
        for i, (label, key, _default, unit, hint) in enumerate(fields):
            col, row = (i % columns) * 3, i // columns
            lab = ttk.Label(grid, text=label + ":", anchor="e", width=17)
            lab.grid(row=row, column=col, sticky="e", padx=(8, 4), pady=3)
            var = self.vars[key]
            if trace is not None:
                var.trace_add("write", lambda *_a, f=trace: f())
            ent = ttk.Entry(grid, textvariable=var, width=9)
            ent.grid(row=row, column=col + 1, sticky="w", pady=3)
            unit_lab = ttk.Label(grid, text=unit, foreground="#777", width=10,
                                 anchor="w")
            unit_lab.grid(row=row, column=col + 2, sticky="w", padx=(4, 16))
            if hint:
                for w in (lab, ent, unit_lab):
                    Tooltip(w, hint)
        return grid

    def _note(self, parent, text):
        """A full-width explanatory line under a group."""
        ttk.Label(parent, text=text, foreground="#777", wraplength=1050,
                  justify="left").pack(anchor="w", padx=8, pady=(2, 0))

    # ------------------------------------------------------------------ #
    #  Page 1: the vehicle
    # ------------------------------------------------------------------ #
    def _scrollable(self, tab):
        """A vertically scrollable page, so a small screen scrolls instead of
        clipping the bottom of the vehicle."""
        canvas = tk.Canvas(tab, highlightthickness=0)
        bar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner.bind("<Configure>",
                   lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win, width=e.width))
        # The wheel has to be grabbed globally to reach the child frames, so it is
        # grabbed only while the pointer is actually over this page - otherwise it
        # would scroll the vehicle while the user is reading the log on another tab.
        def on_wheel(event, fixed=None):
            step = fixed if fixed is not None else int(-event.delta / 120)
            canvas.yview_scroll(step, "units")

        def grab(_e=None):
            canvas.bind_all("<MouseWheel>", on_wheel)
            canvas.bind_all("<Button-4>", lambda e: on_wheel(e, -1))
            canvas.bind_all("<Button-5>", lambda e: on_wheel(e, 1))

        def release(_e=None):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(seq)

        canvas.bind("<Enter>", grab)
        canvas.bind("<Leave>", release)
        inner.bind("<Enter>", grab)
        inner.bind("<Leave>", release)
        return inner

    def build_vehicle_tab(self, tab):
        tab = self._scrollable(tab)
        ttk.Label(tab, foreground="#555", wraplength=1050, justify="left",
                  text="The rocket itself. Both simulations read this page - describe "
                       "the vehicle once here, and the other two pages only describe "
                       "the flight. Hover any field for what it means.").pack(
            anchor="w", pady=(0, 4))

        mot = ttk.LabelFrame(tab, text="Landing motor", padding=8)
        mot.pack(fill="x", pady=(4, 0))
        row = ttk.Frame(mot)
        row.pack(fill="x", pady=(0, 4))
        for key, text in (("long", "long  (120 N peak, 2.61 s, 222 Ns)"),
                          ("short", "short  (269 N peak, 1.55 s, 259 Ns)")):
            ttk.Radiobutton(row, text=text, value=key, variable=self.motor_var,
                            command=self.show_motor).pack(side="left", padx=(8, 18))
        ttk.Label(mot, textvariable=self.motor_info, foreground="#777").pack(
            anchor="w", padx=8, pady=(0, 4))
        self._fields(mot, MOTOR_FIELDS, columns=3, trace=self.show_motor)

        air = ttk.LabelFrame(tab, text="Airframe", padding=8)
        air.pack(fill="x", pady=(8, 0))
        self._fields(air, AIRFRAME_FIELDS, columns=3)

        inr = ttk.LabelFrame(tab, text="Inertia and arms   (3-D only)", padding=8)
        inr.pack(fill="x", pady=(8, 0))
        self._fields(inr, INERTIA_FIELDS, columns=3)
        rowb = ttk.Frame(inr)
        rowb.pack(fill="x", pady=(6, 0))
        ttk.Button(rowb, text="MMOI from a 1.40 m rod",
                   command=self.mmoi_from_rod).pack(side="left", padx=(8, 10))
        ttk.Label(rowb, foreground="#777",
                  text="slender rod m*L^2/12 and solid cylinder m*R^2/2 - a starting "
                       "point, not a substitute for a CAD number").pack(side="left")

        act = ttk.LabelFrame(tab, text="Actuators   (3-D only, except the clamp range)",
                             padding=8)
        act.pack(fill="x", pady=(8, 0))
        self._fields(act, ACTUATOR_FIELDS, columns=3)

        fin = ttk.LabelFrame(tab, text="Fin control   (NACA 0012, all-moving)",
                             padding=8)
        fin.pack(fill="x", pady=(8, 0))
        head = ttk.Frame(fin)
        head.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(head, text="fins active", variable=self.fin_on,
                        command=self.show_fins).pack(side="left", padx=(8, 16))
        ttk.Label(head, text="airbrake:").pack(side="left")
        ttk.Combobox(head, textvariable=self.fin_brake, width=8, state="readonly",
                     values=("auto", "always", "off")).pack(side="left", padx=(4, 16))
        ttk.Checkbutton(head, text="aerodynamic drift nulling",
                        variable=self.fin_drift).pack(side="left")
        ttk.Label(fin, textvariable=self.fin_info, foreground="#777",
                  wraplength=1050, justify="left").pack(anchor="w", padx=8,
                                                        pady=(0, 4))
        self._fields(fin, FIN_FIELDS, columns=3, trace=self.show_fins)

        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(10, 0))
        ttk.Button(bar, text="Import an OpenRocket .ork ...",
                   command=self.import_ork).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Reset the vehicle to defaults",
                   command=self.reset_vehicle).pack(side="left")
        self.ork_info = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self.ork_info, style="Hint.TLabel",
                  wraplength=1050, justify="left").pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------------ #
    #  Page 2: the 1-D ignition-window search
    # ------------------------------------------------------------------ #
    def build_1d_tab(self, tab):
        ttk.Label(tab, foreground="#555", wraplength=1050, justify="left",
                  text="Vertical axis only. Searches for the range of ignition "
                       "altitudes from which a throttle profile exists that lands "
                       "softly, and prints the profile for both edges.").pack(anchor="w")

        inp = ttk.LabelFrame(tab, text="Scenario and search", padding=8)
        inp.pack(fill="x", pady=(6, 0))
        self._fields(inp, SCENARIO_1D_FIELDS, columns=3)

        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(8, 4))
        self.run_btn = ttk.Button(bar, text="Compute", command=self.start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar, text="Apogee from ground",
                   command=self.run_ascent).pack(side="left")
        ttk.Label(bar, text="throttle:").pack(side="left", padx=(8, 2))
        self.ascent_throttle = tk.StringVar(value="1.0")
        ttk.Entry(bar, textvariable=self.ascent_throttle, width=5).pack(side="left")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.progress.pack(side="right")

        res = ttk.LabelFrame(tab, text="Result", padding=8)
        res.pack(fill="x")
        self.summary = tk.StringVar(value="-")
        ttk.Label(res, textvariable=self.summary,
                  font=("TkDefaultFont", 11, "bold"),
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
                ttk.Label(grid, textvariable=v).grid(row=r, column=c, sticky="w",
                                                     padx=8)

        logf = ttk.LabelFrame(tab, text="Progress / throttle profiles", padding=6)
        logf.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(logf, height=18, wrap="none", font=("TkFixedFont", 9))
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ #
    #  Page 3: the 3-D campaign
    # ------------------------------------------------------------------ #
    def build_tvc_tab(self, tab):
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, foreground="#555", wraplength=1050, justify="left",
                  text="Two translational axes, a rolling airframe, two TVC channels "
                       "and four fins. Every cell of the entry grid is flown N times "
                       "with a random igniter delay, thrust scatter and roll rate; the "
                       "on-board rule decides in flight whether to light the D9. The "
                       "vehicle comes from the Vehicle page.").pack(anchor="w")
        try:
            import matplotlib                          # noqa: F401
            self.mpl_ok, self.mpl_err = True, ""
        except Exception as exc:                       # noqa: BLE001
            self.mpl_ok, self.mpl_err = False, f"{type(exc).__name__}: {exc}"
        if not self.mpl_ok:
            ttk.Label(top, foreground="#c0392b", wraplength=1050, justify="left",
                      text=f"matplotlib is BROKEN or missing, so no figures can be "
                           f"drawn ({self.mpl_err}). The campaign itself runs fine.  "
                           f"{MPL_HINT.strip()}").pack(anchor="w")
        if tvc_sim.HAVE_NUMBA:
            txt, col = ("numba active - the flight kernel is compiled (~14 ms per "
                        "flight; the first call spends a few seconds compiling)", "#0a5")
        else:
            txt, col = ("numba NOT installed - the flight kernel runs as plain Python, "
                        "roughly 40x slower.  pip install numba", "#c0392b")
        ttk.Label(top, text=txt, foreground=col, wraplength=1050,
                  justify="left").pack(anchor="w")

        cols = ttk.Frame(tab)
        cols.pack(fill="x", pady=(6, 0))
        camp = ttk.LabelFrame(cols, text="Campaign and dispersions", padding=8)
        camp.pack(side="left", fill="both", expand=True)
        self._fields(camp, CAMPAIGN_FIELDS, columns=2)
        gates = ttk.LabelFrame(cols, text="Touchdown gates (the landing gear)",
                               padding=8)
        gates.pack(side="left", fill="both", padx=(8, 0))
        self._fields(gates, GATE_FIELDS, columns=1)
        gains = ttk.LabelFrame(cols, text="Controller gains", padding=8)
        gains.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._fields(gains, GAIN_FIELDS, columns=2)

        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(8, 4))
        self.tvc_btn = ttk.Button(bar, text="Run campaign", command=self.start_tvc)
        self.tvc_btn.pack(side="left")
        self.tvc_stop = ttk.Button(bar, text="Stop", command=self.stop,
                                   state="disabled")
        self.tvc_stop.pack(side="left", padx=6)
        ttk.Button(bar, text="Single flight + figure",
                   command=self.tvc_single).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.tune_btn = ttk.Button(bar, text="Tune gains", command=self.start_tune)
        self.tune_btn.pack(side="left")
        ttk.Button(bar, text="Clear log",
                   command=lambda: self.tvc_log.delete("1.0", "end")).pack(side="left",
                                                                           padx=(10, 0))
        self.tvc_prog = ttk.Progressbar(bar, mode="indeterminate", length=180)
        self.tvc_prog.pack(side="right")

        res = ttk.LabelFrame(tab, text="Result", padding=6)
        res.pack(fill="x")
        self.tvc_summary = tk.StringVar(value="-")
        ttk.Label(res, textvariable=self.tvc_summary,
                  font=("TkDefaultFont", 11, "bold"),
                  foreground="#0a5").pack(anchor="w")
        self.tvc_gates = tk.StringVar(value="")
        ttk.Label(res, textvariable=self.tvc_gates,
                  font=("TkFixedFont", 9)).pack(anchor="w", pady=(4, 0))

        logf = ttk.LabelFrame(tab, text="Simulation log", padding=6)
        logf.pack(fill="both", expand=True, pady=(6, 0))
        # The campaign has a lot to say - progress, timing, the full report, the
        # tuner's search - so this is the one widget that gets all the spare room.
        self.tvc_log = tk.Text(logf, height=34, wrap="none", font=("TkFixedFont", 9),
                               background="#111111", foreground="#e6e6e6",
                               insertbackground="#e6e6e6")
        sb = ttk.Scrollbar(logf, command=self.tvc_log.yview)
        sbx = ttk.Scrollbar(logf, orient="horizontal", command=self.tvc_log.xview)
        self.tvc_log.configure(yscrollcommand=sb.set, xscrollcommand=sbx.set)
        sb.pack(side="right", fill="y")
        sbx.pack(side="bottom", fill="x")
        self.tvc_log.pack(fill="both", expand=True)

    # ------------------------------------------------------------------ #
    def reset_vehicle(self):
        for group in (AIRFRAME_FIELDS, INERTIA_FIELDS, ACTUATOR_FIELDS,
                      MOTOR_FIELDS, FIN_FIELDS):
            for _label, key, default, _u, _h in group:
                self.vars[key].set(default)
        self.motor_var.set("long")
        self.fin_on.set(True)
        self.fin_brake.set("auto")
        self.fin_drift.set(False)
        self.show_motor()
        self.show_fins()

    def import_ork(self):
        """Read an OpenRocket file into the vehicle fields.

        The landing vehicle is NOT the rocket that left the pad: the motor it
        launched on has burned, so its propellant is gone and its casing is not.
        The dialog asks for that, because an .ork does not carry motor masses.
        """
        from tkinter import filedialog, simpledialog
        path = filedialog.askopenfilename(
            title="OpenRocket file",
            filetypes=[("OpenRocket", "*.ork"), ("XML", "*.xml"), ("All", "*.*")])
        if not path:
            return
        try:
            rock = orkimport.read_ork(path)
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("Import failed", f"{type(exc).__name__}: {exc}")
            return

        spent = simpledialog.askfloat(
            "Spent motor",
            "Propellant [g] of the motor that has ALREADY burned\n"
            "(the one it launched on - its casing stays on board):",
            initialvalue=0.0, minvalue=0.0, parent=self.root)
        extra = simpledialog.askfloat(
            "Motor hardware",
            "Total mass [g] of the motors themselves\n"
            "(.ork files do not store motor masses - casings, landing motor, D9):",
            initialvalue=0.0, minvalue=0.0, parent=self.root)
        spent = (spent or 0.0) / 1000.0
        extra = (extra or 0.0) / 1000.0

        mass = rock.mass + extra - spent
        if mass <= self.num("propellant", 0.277):
            messagebox.showerror("Import failed",
                                 f"the imported mass comes out at {mass:.3f} kg, "
                                 f"which is not more than the landing propellant")
            return
        self.vars["mass"].set(f"{mass:.3f}")
        if rock.diameter > 0:
            self.vars["diameter"].set(f"{rock.diameter * 1000:.1f}")
        if rock.length > 0:
            self.vars["mmoi"].set(f"{mass * rock.length ** 2 / 12.0:.4f}")
            self.vars["mmoi_r"].set(
                f"{mass * 0.5 * (rock.diameter / 2.0) ** 2:.5f}")
        if rock.fin_count:
            self.vars["fin_count"].set(str(rock.fin_count))
            self.vars["fin_root"].set(f"{rock.fin_root * 1000:.0f}")
            self.vars["fin_tip"].set(f"{rock.fin_tip * 1000:.0f}")
            self.vars["fin_span"].set(f"{rock.fin_span * 1000:.0f}")
            if rock.fin_position and rock.length:
                # fins sit at the tail; the arm is from the CG, taken at mid-body
                arm = max(0.05, rock.fin_position + 0.5 * rock.fin_root
                          - 0.5 * rock.length)
                self.vars["fin_arm"].set(f"{arm:.2f}")
        self.ork_info.set(rock.summary() + f"\n  -> gross mass set to {mass:.3f} kg "
                          f"({rock.mass:.3f} imported + {extra:.3f} motors "
                          f"- {spent:.3f} spent propellant); MMOI from a "
                          f"{rock.length:.2f} m rod. Check every field before flying.")
        self.show_motor()
        self.show_fins()

    def mmoi_from_rod(self):
        """Fill the two MMOI boxes from the simple geometric bodies."""
        try:
            m = self.num("mass")
            d = self.num("diameter") / 1000.0
        except ValueError:
            return
        length = 1.40
        self.vars["mmoi"].set(f"{m * length * length / 12.0:.4f}")
        self.vars["mmoi_r"].set(f"{m * 0.5 * (d / 2.0) ** 2:.5f}")

    def num(self, key, default=None):
        txt = self.vars[key].get().strip().replace(",", ".")
        if txt == "":
            if default is not None:
                return default
            raise ValueError(f"'{key}' is empty")
        return float(txt)

    def show_motor(self):
        try:
            m = Motor(self.motor_var.get(),
                      propellant_mass=self.num("propellant", 0.277),
                      thrust_multiplier=self.num("thrust_mult", 1.0))
        except (ValueError, KeyError):
            self.motor_info.set("")
            return
        self.motor_info.set(f"burn {m.burn_time:.3f} s, peak {m.peak_thrust:.1f} N, "
                            f"impulse {m.total_impulse:.1f} Ns")

    def show_fins(self):
        """Live read-out of what the entered fin geometry is actually worth."""
        try:
            cfg = self.tvc_config()
        except (ValueError, KeyError):
            self.fin_info.set("")
            return
        f = cfg.fin_set()
        if not f.enabled:
            self.fin_info.set("no fins - roll uncontrolled, no airbrake")
            return
        self.fin_info.set(f.describe())

    # ------------------------------------------------------------------ #
    #  Configs
    # ------------------------------------------------------------------ #
    def build_config(self) -> CircularConfig:
        """The 1-D config: vehicle from page 1, scenario from page 2."""
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
            n_boosters=max(0, int(self.num("boosters", 0))),
            booster_window=(self.num("booster_window")
                            if self.vars["booster_window"].get().strip() else None),
            motor=Motor(self.motor_var.get(),
                        propellant_mass=self.num("propellant"),
                        thrust_multiplier=self.num("thrust_mult", 1.0)),
        )
        area_cm2 = self.vars["area"].get().strip()
        cfg.area_override = (float(area_cm2.replace(",", ".")) / 1e4
                             if area_cm2 else None)
        if cfg.gross_mass <= cfg.motor.propellant_mass:
            raise ValueError("gross mass must be larger than the propellant mass")
        return cfg

    def tvc_config(self) -> "tvc_sim.TvcConfig":
        n = self.num
        return tvc_sim.TvcConfig(
            motor=self.motor_var.get(),
            gross_mass=n("mass"), propellant=n("propellant"),
            diameter=n("diameter") / 1000.0, cd=n("cd"),
            thrust_mult=n("thrust_mult", 1.0),
            mmoi_transverse=n("mmoi"), mmoi_roll=n("mmoi_r"),
            l_gimbal=n("lgimb"), l_cp=n("lcp"), cn_alpha=n("cna"),
            servo_max=n("srvmax"), servo_ratio=n("srvratio"),
            servo_speed=n("srvspd"), servo_accel=n("srvacc"),
            servo_quant=n("srvq"), throttle_speed=n("thrspd"),
            throttle_accel=n("thracc"),
            k_min=n("throttle_min"), k_max=n("kmax"),
            n_boosters=int(n("boosters", 0)),
            booster_cant=n("b_cant"), booster_azimuth=n("b_azim"),
            h_lo=n("h_lo"), h_hi=n("h_hi"), h_step=n("h_step"),
            vx_max=n("vx_max"), vx_step=n("vx_step"), runs=int(n("runs")),
            ign_delay_max=n("ign_delay"),
            delay_pad=(n("delay_pad") if self.vars["delay_pad"].get().strip()
                       else tvc_sim.default_delay_pad(n("ign_delay"))),
            thrust_scatter=n("scatter"), thrust_tau=n("tau"), roll_max=n("roll"),
            gate_vz=n("gate_vz"), gate_vh=n("gate_vh"), gate_tilt=n("gate_tilt"),
            gate_omega=n("gate_rate"),
            tilt_min=n("tilt_min"), tilt_slope=n("tilt_slope"),
            tilt_cap=n("tilt_cap"),
            wn=n("wn"), zeta=n("zeta"), wn_fin=n("wn_fin"), zeta_fin=n("zeta_fin"),
            roll_gain=n("roll_gain"), sched_tvc=n("sched_tvc"),
            sched_fin=n("sched_fin"),
            fins=bool(self.fin_on.get()), fin_count=int(n("fin_count")),
            fin_root=n("fin_root") / 1000.0, fin_tip=n("fin_tip") / 1000.0,
            fin_span=n("fin_span") / 1000.0, fin_arm=n("fin_arm"),
            fin_max_deflect=n("fin_deflect"), fin_travel_time=n("fin_travel"),
            fin_brake=self.fin_brake.get(),
            fin_drift_null=bool(self.fin_drift.get()))

    # ------------------------------------------------------------------ #
    #  Page 5: the campaign's charts, live
    # ------------------------------------------------------------------ #
    def build_charts_tab(self, tab):
        ttk.Label(tab, style="Head.TLabel",
                  text="The last campaign, drawn").pack(anchor="w")
        ttk.Label(tab, style="Hint.TLabel", wraplength=1050, justify="left",
                  text="The same five figures the campaign writes to disk, drawn here "
                       "from the run in memory - run a campaign on the Monte Carlo "
                       "page, or load a saved .npz.").pack(anchor="w")
        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(6, 4))
        ttk.Label(bar, text="figure:").pack(side="left", padx=(8, 4))
        self.chart_kind = tk.StringVar(value="envelope")
        cb = ttk.Combobox(bar, textvariable=self.chart_kind, width=16, state="readonly",
                          values=list(tvc_sim.FIGURES))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _e: self.draw_chart())
        ttk.Button(bar, text="Redraw", command=self.draw_chart).pack(side="left",
                                                                     padx=8)
        ttk.Button(bar, text="Load a saved run (.npz) ...",
                   command=self.load_campaign_file).pack(side="left")
        self.chart_info = tk.StringVar(value="no campaign in memory yet")
        ttk.Label(bar, textvariable=self.chart_info,
                  style="Hint.TLabel").pack(side="left", padx=12)

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)
        self.camp = None
        if not HAVE_MPL:
            ttk.Label(body, foreground=BAD, wraplength=900, justify="left",
                      text=f"matplotlib is not usable ({MPL_ERR}).\n{MPL_HINT}").pack(
                anchor="w")
            self.chart_fig = None
            return
        self.chart_fig = Figure(figsize=(12.5, 6.6), dpi=96)
        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=body)
        NavigationToolbar2Tk(self.chart_canvas, body).update()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw_chart(self):
        if not HAVE_MPL or self.camp is None:
            return
        self.chart_fig.clear()
        try:
            tvc_sim.draw_figure(self.chart_kind.get(), self.chart_fig, self.camp)
            self.chart_fig.tight_layout()
        except Exception as exc:                        # noqa: BLE001
            self.chart_info.set(f"drawing failed: {type(exc).__name__}: {exc}")
        self.chart_canvas.draw_idle()

    def load_campaign_file(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Saved campaign",
                                          filetypes=[("numpy", "*.npz"),
                                                     ("All", "*.*")])
        if not path:
            return
        try:
            self.camp = tvc_sim.load_campaign(path)
        except Exception as exc:                        # noqa: BLE001
            messagebox.showerror("Load failed", f"{type(exc).__name__}: {exc}")
            return
        n = self.camp["out"][:, :, :, 0].size
        self.chart_info.set(f"{os.path.basename(path)}: {n} flights")
        self.draw_chart()

    # ------------------------------------------------------------------ #
    #  Page 4: the flight viewer
    # ------------------------------------------------------------------ #
    def build_viewer_tab(self, tab):
        ttk.Label(tab, style="Head.TLabel",
                  text="Fly one trajectory and watch it").pack(anchor="w")
        ttk.Label(tab, style="Hint.TLabel", wraplength=1050, justify="left",
                  text="One flight at a time, with the vehicle drawn where it "
                       "actually is and pointing where it actually points. The panels "
                       "on the right are the same flight, so a wobble in the 3-D view "
                       "has a cause you can read off the traces. Playback runs on the "
                       "wall clock: 1.0x is real time, and if the machine cannot draw "
                       "every 20 ms telemetry frame it drops frames instead of "
                       "running slow.").pack(anchor="w")

        bar = ttk.Frame(tab)
        bar.pack(fill="x", pady=(6, 4))
        for label, key, default, width in (("release [m]", "v_h0", "160", 6),
                                           ("vx [m/s]", "v_vx", "5", 5),
                                           ("seed", "v_seed", "3", 6)):
            ttk.Label(bar, text=label + ":").pack(side="left", padx=(8, 3))
            self.vars.setdefault(key, tk.StringVar(value=default))
            ttk.Entry(bar, textvariable=self.vars[key], width=width).pack(side="left")
        ttk.Button(bar, text="Fly it", style="Run.TButton",
                   command=self.viewer_fly).pack(side="left", padx=(12, 6))
        self.play_btn = ttk.Button(bar, text="Play", command=self.viewer_play,
                                   state="disabled")
        self.play_btn.pack(side="left")
        ttk.Label(bar, text="speed:").pack(side="left", padx=(10, 3))
        self.play_speed = tk.StringVar(value="1.0")
        ttk.Combobox(bar, textvariable=self.play_speed, width=5, state="readonly",
                     values=("0.1", "0.25", "0.5", "1.0", "2.0", "4.0")
                     ).pack(side="left")
        self.equal_z = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="1:1 axes", variable=self.equal_z,
                        command=self.viewer_rescale).pack(side="left", padx=(10, 0))
        Tooltip(bar.winfo_children()[-1],
                "One metre is the same length on x, y and altitude. The two "
                "horizontal axes are always equal to each other; unticking this only "
                "stretches the altitude axis so a 160 m fall over 20 m of drift is "
                "readable.")
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(bar, from_=0, to=1, orient="horizontal",
                                     variable=self.frame_var,
                                     command=self.viewer_scrub)
        self.frame_scale.pack(side="left", fill="x", expand=True, padx=(12, 8))
        self.viewer_status = tk.StringVar(value="no flight yet")
        ttk.Label(tab, textvariable=self.viewer_status,
                  font=("TkFixedFont", 9)).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)
        if not HAVE_MPL:
            ttk.Label(body, foreground=BAD, wraplength=900, justify="left",
                      text=f"matplotlib is not usable, so there is nothing to draw "
                           f"here ({MPL_ERR}).\n{MPL_HINT}").pack(anchor="w")
            self.view_fig = None
            return
        self.view_fig = Figure(figsize=(12.5, 6.4), dpi=96)
        self.view_fig.patch.set_facecolor("#fcfcfb")
        self.ax3d = self.view_fig.add_subplot(1, 2, 1, projection="3d")
        self.ax_a = self.view_fig.add_subplot(3, 2, 2)
        self.ax_b = self.view_fig.add_subplot(3, 2, 4)
        self.ax_c = self.view_fig.add_subplot(3, 2, 6)
        self.view_canvas = FigureCanvasTkAgg(self.view_fig, master=body)
        self.view_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.tel = None
        self.playing = False
        self.art = None            # the artists that get moved, see viewer_build()
        self._bg = None            # cached background for blitting
        self._bg_size = None
        self._scrub_lock = False   # set while playback drives the slider itself

    # ---------------- the flight ----------------
    def viewer_fly(self):
        try:
            cfg = self.tvc_config()
            h0 = self.num("v_h0")
            vx = self.num("v_vx")
            seed = int(self.num("v_seed"))
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        out, tel = tvc_sim.fly_one(cfg, seed, h0, vx, n_tel=6000)
        if len(tel) < 2:
            messagebox.showerror("Flight viewer", "the flight produced no telemetry")
            return
        self.playing = False
        self.play_btn.configure(text="Play", state="normal")
        self.out, self.tel, self.cfg_view = out, tel, cfg
        self.tel_dt = max(1e-3, float(tel[1, 0] - tel[0, 0]))
        self.frame_scale.configure(to=len(tel) - 1)
        self.viewer_status.set(
            f"{'LANDED' if out[0] > 0.5 else 'FAILED'}  -  touchdown {out[1]:.2f} m/s "
            f"down, {out[2]:.2f} m/s across, tilt {out[3]:.1f} deg, rate "
            f"{out[4]:.1f} deg/s, roll {out[15]:.1f} deg/s   |   ignition commanded "
            f"{out[5]:.1f} m, lit {out[6]:.1f} m after {out[7] * 1000:.0f} ms, D9 "
            f"{'lit' if out[8] > 0.5 else 'unused'}, clamp waste {out[11]:.1f} m/s, "
            f"steering {out[16]:.2f} m/s")
        if HAVE_MPL:
            self.viewer_build()
            self.set_frame(len(tel) - 1)

    # ---------------- drawing ----------------
    #
    # The viewer used to rebuild all four panels from scratch on every frame:
    # ax.clear(), replot ~6000 points per trace, rebuild three legends and run
    # tight_layout - 150-250 ms of work for one 20 ms step of flight, which is why
    # "1.0x" crawled at a tenth of real time. Now everything static is drawn ONCE
    # by viewer_build() and every frame only moves a handful of artists.
    def viewer_build(self):
        """Draw everything that does not change during playback, once."""
        tel = self.tel
        t, h, x, y = tel[:, 0], tel[:, 1], tel[:, 3], tel[:, 15]
        # The flown path is redrawn every frame, so cap it at ~400 points. At 20 ms
        # sampling that is still 8 s of trajectory per point-pair at worst, which no
        # screen can tell apart from the full record.
        self.path_stride = max(1, len(tel) // 400)
        ax = self.ax3d
        ax.clear()
        ax.set_facecolor("#fcfcfb")
        ax.plot(x, y, h, color="#cfd6db", lw=1, ls="--")           # the whole flight
        ax.scatter([0], [0], [0], color=GOOD, s=40, marker="o")    # the pad
        flown, = ax.plot([], [], [], color=ACCENT, lw=2)
        shadow, = ax.plot([], [], [], color="#dfe4e8", lw=1)
        stick, = ax.plot([], [], [], color=INK, lw=4,
                         solid_capstyle="round")
        flame, = ax.plot([], [], [], color="#eb6834", lw=3, alpha=0.85)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("altitude [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        # The readout lives INSIDE the axes, not in the title: only the axes
        # rectangles are blitted per frame, and a title sits outside them.
        title3d = ax.text2D(0.0, 1.0, "", transform=ax.transAxes, fontsize=9,
                            color=INK, ha="left", va="top",
                            family="monospace")

        vlines = []
        for a, series, title, xlabel in (
                (self.ax_a, ((h, ACCENT, "altitude [m]"),
                             (-tel[:, 2], "#eb6834", "descent rate [m/s]")),
                 "Altitude and descent rate", ""),
                (self.ax_b, ((tel[:, 6], ACCENT, "total thrust [N]"),
                             (tel[:, 7] * 100.0, "#eb6834", "clamp [%]")),
                 "Thrust and clamp", ""),
                (self.ax_c, ((tel[:, 5], ACCENT, "tilt [deg]"),
                             (tel[:, 12], "#1baf7a", "fin 1 [deg]"),
                             (tel[:, 8], "#eb6834", "servo 1 [deg]")),
                 "Attitude and actuators", "time [s]")):
            a.clear()
            a.set_facecolor("#fcfcfb")
            for data, colour, label in series:
                a.plot(t, data, color=colour, lw=1.4, label=label)
            vlines.append(a.axvline(t[0], color=INK2, lw=1.0))
            a.grid(True, color="#e7e6e1", lw=0.7)
            a.set_axisbelow(True)
            a.tick_params(labelsize=7, colors=INK2)
            a.set_title(title, fontsize=9, loc="left", color=INK)
            if xlabel:
                a.set_xlabel(xlabel, fontsize=8, color=INK2)
            a.legend(fontsize=7, frameon=False, loc="upper right")
            for side in ("top", "right"):
                a.spines[side].set_visible(False)

        self.art = dict(flown=flown, shadow=shadow, stick=stick, flame=flame,
                        title=title3d, vlines=vlines)
        self.thrust_max = max(float(np.max(tel[:, 6])), 1.0)
        self.stick_len = max(4.0, 0.04 * float(np.max(h)))
        self.viewer_rescale(redraw=False)
        self.view_fig.tight_layout()          # once per flight, not once per frame
        self._bg = None

    def viewer_rescale(self, redraw=True):
        """Axis limits and the box aspect. Both horizontal axes always share one
        scale; the altitude axis joins them when '1:1 axes' is ticked."""
        if not HAVE_MPL or self.tel is None or self.art is None:
            return
        tel = self.tel
        h, x, y = tel[:, 1], tel[:, 3], tel[:, 15]
        # One symmetric span for x and y, so a metre across is a metre across
        # whichever way the vehicle drifted.
        span = max(12.0, float(np.max(np.abs(x))) * 1.3,
                   float(np.max(np.abs(y))) * 1.3)
        zmax = float(np.max(h)) * 1.05
        ax = self.ax3d
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(0, zmax)
        if self.equal_z.get():
            # True 1:1:1. The box comes out tall and narrow - that IS the flight:
            # 160 m of fall against 20 m of drift.
            ax.set_box_aspect((2.0 * span, 2.0 * span, zmax))
        else:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        self._bg = None                       # the cached background is now stale
        if redraw:
            self.viewer_draw()

    def viewer_scrub(self, _value=None):
        """The slider moved. Ignore it while playback is the one moving it."""
        if self._scrub_lock:
            return
        self.playing = False
        self.play_btn.configure(text="Play")
        self.viewer_draw()

    def set_frame(self, i):
        self._scrub_lock = True
        self.frame_var.set(int(i))
        self._scrub_lock = False
        self.viewer_draw()

    def viewer_draw(self):
        """Move the artists to the current frame. No clearing, no re-layout."""
        if not HAVE_MPL or self.tel is None or self.art is None:
            return
        tel = self.tel
        i = max(0, min(int(self.frame_var.get()), len(tel) - 1))
        t, h, vz, x = tel[:, 0], tel[:, 1], tel[:, 2], tel[:, 3]
        y = tel[:, 15]
        a = self.art
        st = self.path_stride
        sl = slice(0, i + 1, st)
        a["flown"].set_data_3d(x[sl], y[sl], h[sl])
        a["shadow"].set_data_3d(x[sl], y[sl], np.zeros(len(h[sl])))
        bx, by, bz = tel[i, 16], tel[i, 17], tel[i, 18]
        L = self.stick_len
        a["stick"].set_data_3d([x[i] - 0.5 * L * bx, x[i] + 0.5 * L * bx],
                               [y[i] - 0.5 * L * by, y[i] + 0.5 * L * by],
                               [h[i] - 0.5 * L * bz, h[i] + 0.5 * L * bz])
        if tel[i, 6] > 1.0:
            f = 1.2 * L * tel[i, 6] / self.thrust_max
            a["flame"].set_data_3d(
                [x[i] - 0.5 * L * bx, x[i] - 0.5 * L * bx - f * bx],
                [y[i] - 0.5 * L * by, y[i] - 0.5 * L * by - f * by],
                [h[i] - 0.5 * L * bz, h[i] - 0.5 * L * bz - f * bz])
        else:
            a["flame"].set_data_3d([], [], [])
        a["title"].set_text(f"t = {t[i]:5.2f} s    h = {h[i]:6.1f} m    "
                            f"vz = {vz[i]:6.1f} m/s    thrust = {tel[i, 6]:5.1f} N")
        for v in a["vlines"]:
            v.set_xdata([t[i], t[i]])
        try:
            self.viewer_blit()
        except Exception:                      # noqa: BLE001
            # Any matplotlib version that will not blit still gets a correct, if
            # slower, picture.
            self._bg = None
            self.view_canvas.draw_idle()

    def viewer_moving(self):
        a = self.art
        return [a["flown"], a["shadow"], a["stick"], a["flame"], a["title"],
                *a["vlines"]]

    def viewer_blit(self):
        """Repaint only the artists that moved.

        A full redraw of this figure costs ~190 ms on a laptop - the four sets of
        ticks, the 3-D panes and the font lookups dominate, not the data. All of that
        is identical from frame to frame, so it is rendered once into a bitmap and
        restored; only six artists are actually drawn per frame. That is the
        difference between 5 fps and real time.
        """
        cv = self.view_canvas
        size = (int(self.view_fig.bbox.width), int(self.view_fig.bbox.height))
        if self._bg is None or self._bg_size != size:
            for art in self.viewer_moving():
                art.set_visible(False)
            cv.draw()
            self._bg = cv.copy_from_bbox(self.view_fig.bbox)
            self._bg_size = size
            for art in self.viewer_moving():
                art.set_visible(True)
        cv.restore_region(self._bg)
        for art in self.viewer_moving():
            art.axes.draw_artist(art)
        # Push only the four panels, not the whole canvas: the tk blit is charged by
        # the pixel and the margins never change.
        for a in (self.ax3d, self.ax_a, self.ax_b, self.ax_c):
            cv.blit(a.bbox)

    # ---------------- playback ----------------
    def viewer_play(self):
        if not HAVE_MPL or self.tel is None:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            i = self.frame_var.get()
            if i >= len(self.tel) - 1:
                i = 0
            self._play_i0 = i
            self._play_t0 = time.perf_counter()
            self.viewer_step()

    def viewer_step(self):
        """Wall-clock playback.

        The frame index comes from how much real time has passed, NOT from counting
        redraws, so a slow machine drops frames and still finishes the flight in the
        right number of seconds. The old version advanced exactly one 20 ms frame per
        redraw, so it ran at whatever rate matplotlib could manage - about a tenth of
        real time.
        """
        if not self.playing or self.tel is None:
            return
        try:
            speed = float(self.play_speed.get())
        except ValueError:
            speed = 1.0
        elapsed = (time.perf_counter() - self._play_t0) * speed
        i = self._play_i0 + int(elapsed / self.tel_dt)
        last = len(self.tel) - 1
        if i >= last:
            self.set_frame(last)
            self.playing = False
            self.play_btn.configure(text="Play")
            return
        t0 = time.perf_counter()
        self.set_frame(i)
        draw_ms = (time.perf_counter() - t0) * 1000.0
        # Aim for 30 fps of DISPLAY. Whatever a redraw actually costs is subtracted
        # from the wait, and a redraw slower than the budget simply becomes the frame
        # rate - the flight still plays at the right speed, just less smoothly.
        self.root.after(int(max(1.0, 33.0 - draw_ms)), self.viewer_step)

    # ------------------------------------------------------------------ #
    #  1-D run
    # ------------------------------------------------------------------ #
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self.build_config()
            s_max = self.vars["search_max"].get().strip()
            params = dict(coarse_step=self.num("coarse_step"), tol=self.num("tol"),
                          pop_size=int(self.num("pop")),
                          generations=int(self.num("gen")),
                          search_min=self.num("search_min", 1.0),
                          search_max=(float(s_max.replace(",", "."))
                                      if s_max else None))
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.log.delete("1.0", "end")
        for key in self.cells:
            self.cells[key].set("-")
        self.timing.set("")
        self.summary.set("computing ...")
        m = cfg.motor
        self.put("log", f"motor '{m.name}', thrust multiplier {m.thrust_multiplier:g}x,"
                        f" {cfg.n_boosters} D9 booster(s) available\n"
                        f"burn {m.burn_time:.3f} s, peak {m.peak_thrust:.1f} N, "
                        f"total impulse {m.total_impulse:.1f} Ns, "
                        f"peak T/W {m.peak_thrust / (cfg.start_mass * G):.2f}, "
                        f"area {cfg.area * 1e4:.1f} cm2\n"
                        f"{cfg.n_phases} throttle phases of "
                        f"{cfg.phase_length * 1000:.0f} ms\n")
        self.stop_flag.clear()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.start(12)
        self.worker = threading.Thread(target=self.work, args=(cfg, params),
                                       daemon=True)
        self.worker.start()

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

    # ------------------------------------------------------------------ #
    #  3-D campaign
    # ------------------------------------------------------------------ #
    def tvc_header(self, cfg, what):
        """Everything about the run, written into the log before it starts."""
        m, b = cfg.tables()
        f = cfg.fin_set()
        hg, vg = cfg.entry_grid()
        n = len(hg) * len(vg) * cfg.runs
        w = cfg.gross_mass * G
        return (
            f"{'=' * 78}\n{what}   {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 78}\n"
            f"  motor        : {m.name}, {m.peak_thrust:.1f} N peak, "
            f"{m.burn_time:.2f} s, {m.total_impulse:.1f} Ns, "
            f"x{m.thrust_multiplier:g}\n"
            f"  vehicle      : {cfg.gross_mass:.3f} kg gross "
            f"({cfg.propellant * 1000:.0f} g propellant), {cfg.diameter * 1000:.0f} mm, "
            f"Cd {cfg.cd}, weight {w:.1f} N, peak T/W {m.peak_thrust / w:.2f}\n"
            f"  inertia      : {cfg.mmoi_transverse:.4f} kg m2 transverse, "
            f"{cfg.mmoi_roll:.5f} roll;  arms gimbal {cfg.l_gimbal:.2f} m, "
            f"CP {cfg.l_cp:.2f} m\n"
            f"  TVC          : +/-{cfg.servo_max:.0f} deg servo at "
            f"{cfg.servo_ratio:.1f}:1 = +/-{cfg.servo_max / cfg.servo_ratio:.1f} deg "
            f"nozzle, {cfg.servo_speed:.0f} deg/s, step {cfg.servo_quant:.2f} deg\n"
            f"  clamp        : {cfg.k_min:.2f}-{cfg.k_max:.2f} at "
            f"{cfg.throttle_speed:.1f} /s\n"
            f"  fins         : {f.describe() if f.enabled else 'none'}\n"
            f"  airbrake     : {cfg.fin_brake}, drift nulling "
            f"{'on' if cfg.fin_drift_null else 'off'}\n"
            f"  boosters     : {cfg.n_boosters} x D9, canted {cfg.booster_cant:.0f} deg "
            f"at azimuth {cfg.booster_azimuth:.0f} deg\n"
            f"  tilt cone    : {cfg.tilt_min:g} deg at the pad + "
            f"{cfg.tilt_slope:g} deg/m, capped {cfg.tilt_cap:g} deg\n"
            f"  gates        : |vz|<{cfg.gate_vz:g} m/s, |vh|<{cfg.gate_vh:g} m/s, "
            f"tilt<{cfg.gate_tilt:g} deg, rate<{cfg.gate_omega:g} deg/s\n"
            f"  gains        : TVC {cfg.wn:.2f}/{cfg.zeta:.2f} sched "
            f"{cfg.sched_tvc:+.2f}   fins {cfg.wn_fin:.2f}/{cfg.zeta_fin:.2f} sched "
            f"{cfg.sched_fin:+.2f}   roll {cfg.roll_gain:.2f}\n"
            f"  entry grid   : {cfg.h_lo:.0f}-{cfg.h_hi:.0f} m step {cfg.h_step:.0f}, "
            f"vx 0 +/-{cfg.vx_max:.0f} m/s step {cfg.vx_step:.0f}\n"
            f"  dispersions  : igniter U(0, {cfg.ign_delay_max * 1000:.0f}) ms "
            f"(pad {cfg.delay_pad * 1000:.0f} ms), thrust "
            f"+/-{cfg.thrust_scatter * 100:.0f} % over "
            f"{cfg.thrust_tau * 1000:.0f} ms, roll U(0, {cfg.roll_max:.0f}) deg/s\n"
            f"  flights      : {len(hg)} x {len(vg)} cells x {cfg.runs} = {n}\n"
            f"{'-' * 78}")

    def start_tvc(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self.tvc_config()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self.tvc_summary.set("running ...")
        self.tvc_gates.set("")
        self.put("tlog", self.tvc_header(cfg, "MONTE CARLO CAMPAIGN"))
        self.t_start = time.time()
        self.total_cells = len(cfg.entry_grid()[0]) * len(cfg.entry_grid()[1])
        self.stop_flag.clear()
        self.tvc_btn.configure(state="disabled")
        self.tune_btn.configure(state="disabled")
        self.tvc_stop.configure(state="normal")
        self.tvc_prog.start(12)
        self.worker = threading.Thread(target=self.tvc_work, args=(cfg,), daemon=True)
        self.worker.start()

    def campaign_progress(self, line):
        """Add wall-clock timing to the simulation's own progress line."""
        el = time.time() - self.t_start
        try:
            done = int(line.split("/")[0].strip())
            frac = done / max(self.total_cells, 1)
            eta = el / frac - el if frac > 0 else 0.0
            self.put("tlog", f"{line}   [{el:5.0f} s elapsed, {eta:5.0f} s left, "
                             f"{frac * 100:4.0f} %]")
        except (ValueError, IndexError):
            self.put("tlog", line)

    def tvc_work(self, cfg):
        try:
            camp = tvc_sim.run_campaign(cfg, on_progress=self.campaign_progress,
                                        should_stop=self.stop_flag.is_set)
            self.put("tlog", f"{'-' * 78}\nflying done in "
                             f"{time.time() - self.t_start:.0f} s - writing figures ...")
            # A campaign is minutes of flying. A plotting failure must not throw it
            # away, so the figures are drawn in their own try and the numbers are
            # delivered either way.
            paths, fig_err = [], None
            try:
                paths = tvc_sim.make_figures(camp, self.vars["figdir"].get().strip()
                                             or "figures")
            except Exception as exc:                   # noqa: BLE001
                import traceback
                fig_err = f"{type(exc).__name__}: {exc}"
                self.put("tlog", "FIGURES FAILED (the campaign itself is fine):\n"
                                 + traceback.format_exc())
            self.put("tdone", (camp, paths, fig_err))
        except Cancelled:
            self.put("tcancelled", None)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            self.put("tlog", traceback.format_exc())
            self.put("terror", f"{type(exc).__name__}: {exc}")

    def tvc_single(self):
        """One flight with telemetry, straight to a figure - no campaign."""
        try:
            cfg = self.tvc_config()
            h0 = 0.5 * (cfg.h_lo + cfg.h_hi)
            out, tel = tvc_sim.fly_one(cfg, cfg.seed0, h0, cfg.vx_max, n_tel=4000)
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("Single flight failed", str(exc))
            return
        self.tvc_log.insert("end",
            f"\n--- single flight, release {h0:.0f} m, vx {cfg.vx_max:+.0f} m/s ---\n"
            f"  {'LANDED' if out[0] > 0.5 else 'FAILED'}: touchdown {out[1]:.2f} m/s "
            f"down, {out[2]:.2f} m/s across, tilt {out[3]:.1f} deg, transverse rate "
            f"{out[4]:.1f} deg/s, roll {out[15]:.1f} deg/s\n"
            f"  ignition commanded {out[5]:.1f} m, lit {out[6]:.1f} m after "
            f"{out[7] * 1000:.0f} ms; D9 "
            f"{'lit at %.2f s' % out[9] if out[8] > 0.5 else 'unused'}; "
            f"burnout before touchdown: {'yes' if out[10] > 0.5 else 'no'}\n"
            f"  mean clamp {out[12]:.2f}, clamp waste {out[11]:.1f} m/s, "
            f"steering dV {out[16]:.2f} m/s, max tilt in the burn {out[13]:.1f} deg, "
            f"flight time {out[14]:.2f} s\n")
        self.tvc_log.see("end")
        self.tvc_summary.set(f"single flight: "
                             f"{'LANDED' if out[0] > 0.5 else 'FAILED'} at "
                             f"{out[1]:.2f} m/s down / {out[2]:.2f} m/s across")

    def tvc_finish(self, camp, paths, fig_err=None):
        s = tvc_sim.summarise(camp)
        cfg = camp["cfg"]
        self.camp = camp
        if HAVE_MPL:
            n = camp["out"][:, :, :, 0].size
            self.chart_info.set(f"latest campaign: {n} flights, "
                                f"{s['success']:.1f} % success")
            self.draw_chart()
        self.tvc_summary.set(f"Success {s['success']:.1f} %   "
                             f"({camp['out'][:, :, :, 0].size} flights in "
                             f"{time.time() - self.t_start:.0f} s)")
        self.tvc_gates.set(
            f"|vz|<{cfg.gate_vz:g} {s['gate_vz']:5.1f} %   "
            f"|vh|<{cfg.gate_vh:g} {s['gate_vh']:5.1f} %   "
            f"tilt<{cfg.gate_tilt:g} {s['gate_tilt']:5.1f} %   "
            f"rate<{cfg.gate_omega:g} {s['gate_om']:5.1f} %   "
            f"D9 used {s['boost_rate']:.0f} %\n"
            f"p95: vz {s['p95_vz']:.2f} m/s   vh {s['p95_vh']:.2f} m/s   "
            f"tilt {s['p95_tilt']:.1f} deg   rate {s['p95_om']:.1f} deg/s\n"
            f"over the {s['n_surv']} flights that survived the vertical gate:  "
            f"|vh| {s['gate_vh_c']:.1f} %   tilt {s['gate_tilt_c']:.1f} %   "
            f"rate {s['gate_om_c']:.1f} %   (p95 |vh| {s['p95_vh_c']:.2f} m/s)")
        # the full CLI report, verbatim, into the log
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tvc_sim.print_report(camp)
        self.tvc_log.insert("end", "\n" + buf.getvalue())
        if paths:
            self.tvc_log.insert("end", "\nfigures:\n")
            for p in paths:
                self.tvc_log.insert("end", f"  {p}\n")
        else:
            # Repeat the reason HERE. The traceback is printed the moment it happens,
            # which by now is a full report further up the log - "see the error above"
            # is useless advice when "above" is two screens away.
            self.tvc_log.insert("end", f"\nNO FIGURES WERE WRITTEN - the numbers "
                                       f"above are unaffected.\n  reason: "
                                       f"{fig_err or 'unknown'}\n")
            if fig_err and "matplotlib" in fig_err:
                self.tvc_log.insert("end", MPL_HINT)
        self.tvc_log.see("end")

    # ------------------------------------------------------------------ #
    #  Gain tuning
    # ------------------------------------------------------------------ #
    def start_tune(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            cfg = self.tvc_config()
            budget = int(self.num("tune_budget"))
            runs = int(self.num("tune_runs"))
            workers = int(self.num("tune_workers", 0)) or None
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        self.tvc_summary.set("tuning gains ...")
        self.put("tlog", self.tvc_header(cfg, "GAIN TUNING"))
        self.t_start = time.time()
        self.stop_flag.clear()
        self.tvc_btn.configure(state="disabled")
        self.tune_btn.configure(state="disabled")
        self.tvc_stop.configure(state="normal")
        self.tvc_prog.start(12)
        self.worker = threading.Thread(target=self.tune_work,
                                       args=(cfg, budget, runs, workers), daemon=True)
        self.worker.start()

    def tune_work(self, cfg, budget, runs, workers=None):
        try:
            g, rep = tvc_sim.tune_gains(cfg, budget=budget, runs=runs,
                                        workers=workers,
                                        on_progress=lambda s: self.put("tlog", s),
                                        should_stop=self.stop_flag.is_set)
            tvc_sim.save_gains(g, rep)
            self.put("ttuned", (g, rep))
        except Cancelled:
            self.put("tcancelled", None)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            self.put("tlog", traceback.format_exc())
            self.put("terror", f"{type(exc).__name__}: {exc}")

    def apply_gains(self, g, rep):
        for key, val in zip(("wn", "zeta", "wn_fin", "zeta_fin", "roll_gain",
                             "sched_tvc", "sched_fin"), g):
            self.vars[key].set(f"{val:.3f}")
        self.tvc_summary.set(f"gains tuned in {time.time() - self.t_start:.0f} s - "
                             f"{rep['success'] * 100:.1f} % on the tuning grid "
                             f"(cost {rep['cost']:.4f}); written to tvc_gains.json")
        self.tvc_log.insert("end", "  the gain boxes now hold the tuned values; "
                                   "run the campaign to fly them\n")
        self.tvc_log.see("end")

    # ------------------------------------------------------------------ #
    #  Plumbing
    # ------------------------------------------------------------------ #
    def stop(self):
        self.stop_flag.set()
        self.put("log", "stopping ...")
        self.put("tlog", "stopping ...")

    def put(self, kind, payload):
        self.queue.put((kind, payload))

    def poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "tlog":
                    self.tvc_log.insert("end", payload + "\n")
                    self.tvc_log.see("end")
                elif kind == "done":
                    self.finish(*payload)
                elif kind == "cancelled":
                    self.summary.set("cancelled")
                    self.idle()
                elif kind == "error":
                    self.summary.set("error")
                    messagebox.showerror("Simulation failed", payload)
                    self.idle()
                elif kind == "tdone":
                    self.tvc_finish(*payload)
                    self.tvc_idle()
                elif kind == "ttuned":
                    self.apply_gains(*payload)
                    self.tvc_idle()
                elif kind == "tcancelled":
                    self.tvc_summary.set("cancelled")
                    self.tvc_idle()
                elif kind == "terror":
                    self.tvc_summary.set("error")
                    messagebox.showerror("Campaign failed", payload)
                    self.tvc_idle()
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def idle(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def tvc_idle(self):
        self.tvc_prog.stop()
        self.tvc_btn.configure(state="normal")
        self.tune_btn.configure(state="normal")
        self.tvc_stop.configure(state="disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
