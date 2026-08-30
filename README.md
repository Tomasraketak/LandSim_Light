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

## Usage

```
python3 landsim.py                                  # default 150 m case
python3 landsim.py --drop-alt 200 --max-touchdown 3
python3 landsim.py --coarse-step 5 --gen 200 --pop 80 --tol 0.1   # higher quality
```

Requires only `numpy`.
