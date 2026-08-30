# LandSim Light

1D (vertical axis only) simulation of a propulsive landing of a small rocket with
a solid motor that can only be throttled by thrust-spoiler flaps.

## Model

* Rocket: 3.2 kg gross (277 g propellant), 105 mm diameter, Cd = 0.35, ISA sea-level air.
* Motor: thrust curve reconstructed from the fraction-of-peak / rise-time / decay-time
  table by linear interpolation between the tabulated points
  (burn ≈ 2.61 s, peak 120.4 N, total impulse ≈ 222 Ns).
* Throttling: flaps block up to 90 % of the thrust, so the effective thrust is
  `throttle * F_motor(t)` with `throttle ∈ <0.1, 1.0>`. The blocked thrust is
  **wasted** – the propellant burns according to the nominal curve, so the burn
  time and the mass history do not depend on the throttle setting.
* The throttle profile is split into 100 ms phases; every phase has its own value.

## What it computes

For a free fall from a configurable altitude (default 150 m) it searches, with a
vectorised differential-evolution optimiser over the throttle profile, the range of
ignition altitudes for which a touchdown below 3 m/s is achievable, and prints the
lowest and the highest such altitude together with the throttle / thrust profile.

## Extra knobs

* `--motor long|short` (GUI: radio buttons) switches between the two motor lookup
  tables — both burn the same 277 g of propellant:
  * **long** – 120.4 N peak, 2.613 s burn, 221.8 Ns
  * **short** – 269.4 N peak, 1.554 s burn, 258.8 Ns
* `--thrust-mult` (GUI: *Thrust multiplier*) scales the whole motor lookup table by a
  float > 0 before the simulation starts; everything else (burn time, mass history)
  stays as it is.
* `--search-min` / `--search-max` (GUI: *Search from* / *Search to*) limit the range of
  ignition altitudes that is searched.
* The feasible set is treated as a single contiguous window, so the coarse scan stops
  at the first failure after a series of successes.
* The report also gives the time from the release down to each of the two ignition
  altitudes and the resulting time window for the ignition command.

## Side calculation: apogee from the ground

`ascent(cfg, throttle)` simulates a vertical launch from rest at ground level with
the same vehicle, motor and drag model and returns the apogee, the burnout state,
the peak speed and the peak acceleration. It is completely separate from the
landing search.

* CLI: printed before every run; `--ascent` runs only this, `--ascent-throttle`
  sets a constant throttle.
* GUI: the **Apogee from ground** button with its own throttle box.

With the default vehicle: **long** motor -> apogee ≈ 171 m (burnout 70.7 m, 45.8 m/s),
**short** motor -> apogee ≈ 264 m (burnout 61.7 m, 67.3 m/s, 7.8 g).

## Usage

```
python3 landsim.py                                  # default 150 m case
python3 landsim.py --drop-alt 200 --max-touchdown 3
python3 landsim.py --coarse-step 5 --gen 200 --pop 80 --tol 0.1   # higher quality
```

Requires only `numpy`.

## GUI

```
python3 gui.py
```

A minimal Tkinter window where you can edit the drop altitude, initial velocity,
gross and propellant mass, diameter (or the reference area directly), Cd, air
density, the soft-landing limit, the minimum throttle, the phase length, the
integration step and the optimiser settings. It runs the search in a background
thread (with a Stop button and a live progress log) and shows the two resulting
ignition altitudes together with the speed at ignition and the touchdown speed,
plus the throttle / thrust profile for each.

Tkinter ships with the standard Python installers on Windows and macOS; on Debian/
Ubuntu install it with `sudo apt install python3-tk`.
