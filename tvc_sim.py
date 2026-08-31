"""
LandSim Light - 3-DOF-attitude / 2-axis-translation TVC landing simulation.

This is the *third* simulation in the project. The 1-D modes in `landsim.py` ask
"in what window of ignition altitudes is a soft landing possible at all?"; this one
asks whether a real controller, on a real vehicle, with a real igniter, actually
flies it.

What is modelled
----------------
* **Two translational axes** - vertical z and horizontal x - and the full attitude
  problem in three dimensions: the airframe **rolls** about its own long axis at a
  rate it inherited from separation, with no actuator able to stop it.
* **Two TVC channels.** The nozzle deflects about two perpendicular body axes. The
  servos are bolted to the airframe, so the gimbal axes roll with it; attitude is
  therefore carried as *two* body-fixed unit vectors (thrust axis `b` and a
  transverse reference `g`) and never as Euler angles.
* **The throttle clamp** of the 1-D model: 10-100 % of axial thrust, the diverted
  part wasted, burn duration fixed.
* **One optional Klima D9 booster**, non-throttleable and single-shot, lit by an
  on-board rule when the main motor alone cannot close the landing.
* **Dispersions**: ignition delay U(0, 300 ms), instantaneous thrust scatter of up
  to +/-15 % correlated over a 700 ms window, roll rate U(0, 90) deg/s either way,
  and a grid of entry states. Avionics sensor noise is deliberately NOT modelled -
  the controller sees the true state.

Guidance (structure and gains follow Landing-Rocket-Sim)
-------------------------------------------------------
* ZEV-style commanded thrust *direction* with drag credit and an altitude-dependent
  tilt cone; no Euler angles anywhere in the loop.
* Cascade attitude control: outer attitude-P on the error rotation vector
  `e = asin|b x u| * unit(b x u)`, inner rate-P on the gyro rate, gyroscopic
  decoupling of the roll-induced cross-axis torque, and **dynamic inversion**
  `s = (b x tau)/(L*T)` so one gain set is valid across a 100:1 thrust range.
* A 4-segment throttle plan re-solved at 10 Hz by an Illinois root find on one
  scalar, plus a terminal descent-rate law for the last few metres.

Run `python3 tvc_sim.py --help` for the command line, or use the *3-D / TVC* panel
in `gui.py`.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import numpy as np

import landsim
from landsim import Motor, Booster

try:                                    # optional - ~40x faster when present
    from numba import njit
    HAVE_NUMBA = True
except Exception:                       # noqa: BLE001
    HAVE_NUMBA = False

    def njit(*args, **kwargs):          # type: ignore[misc]
        def deco(f):
            return f
        return deco(args[0]) if args and callable(args[0]) else deco


# ======================================================================
# --- VEHICLE / ENVIRONMENT --------------------------------------------
# Airframe geometry follows the reference vehicle: the landing motor sits under the
# nose, which is the LEADING end during the descent, so the gimbal is below the CG
# and behaves like a tail-mounted motor. The fins trail above -> the CP is aft of
# the CG, the airframe is statically stable nose-down and weathercocks into the
# relative wind. That weathercock torque is what the TVC fights.
# ======================================================================
G = landsim.G
RHO = 1.225

L_BODY = 1.40             # airframe length [m]
L_GIMBAL = 0.70           # CG -> nozzle pivot [m]
L_CP = 0.35               # CG -> centre of pressure (aft) [m]
DIAMETER = landsim.DIAMETER
AREA = math.pi * (DIAMETER / 2.0) ** 2
I_AXIAL_COEF = 0.5 * (DIAMETER / 2.0) ** 2       # I_roll = mass * this

CD_AXIAL = landsim.CD     # axial drag coefficient (same 0.35 as the 1-D modes)
C_N_ALPHA = 2.0           # slender-body normal-force slope

# Actuators
TVC_SERVO_MAX = 10.0      # servo command limit [deg]
TVC_RATIO = 2.0           # servo deg per nozzle deg -> +/-5 deg of nozzle
TVC_MAX_SPEED = 500.0     # deg/s at the servo shaft
TVC_MAX_ACCEL = 2000.0    # deg/s^2
SERVO_QUANT = 0.15        # servo command quantisation [deg]
THROTTLE_SPEED = 12.84    # clamp slew [1/s]
THROTTLE_ACCEL = 257.0    # clamp acceleration [1/s^2]
K_MIN, K_MAX = 0.10, 1.00

# ---------------------------------------------------------------------------
# FIN CONTROL - four cruciform fins, NACA 0012, all-moving
#
# Geometry is the airframe's own: 120 mm root, 63 mm tip, 70 mm span, 35 mm sweep
# (26.6 deg), four of them, no cant. They sit 0.50 m aft of the CG - further back
# than the 0.35 m aero CP arm, which is the body-plus-fins average.
#
# What they buy, in order of importance to THIS vehicle:
#   * they work with the motor OFF. During the free fall the gimbal has nothing to
#     push against, so before ignition the fins are the only actuator there is.
#   * splayed alternately (+,-,+,-) they cancel their own lift and roll torque and
#     leave pure drag: AIRBRAKES. On a vehicle whose entire problem is arriving with
#     more energy than the grain can absorb, that is worth more than the steering.
#   * they can produce a ROLL torque, which the gimbal provably cannot - the gimbal
#     moment (-L*b) x F is perpendicular to b by construction.
# The D9 booster is not mounted axially: it is canted 15 deg and aimed THROUGH the
# CG, so it makes no moment, but its thrust has a lateral component that pushes on
# the horizontal channel - and, being bolted to the airframe, that component rotates
# with the roll. The cant also costs 1 - cos(15 deg) = 3.4 % of its impulse upwards.
D9_CANT = 15.0            # deg from the body axis
D9_AZIMUTH = 0.0          # deg, body-fixed azimuth of the mounting, from the u1 axis

FIN_COUNT = 4
FIN_ROOT = 0.120          # m, root chord
FIN_TIP = 0.063           # m, tip chord
FIN_SPAN = 0.070          # m, exposed semi-span (the "height" on the drawing)
FIN_ARM = 0.50            # m, CG -> fin quarter-chord, aft
FIN_MAX_DEFLECT = 15.0    # deg
FIN_TRAVEL_TIME = 0.09    # s to go from one end stop to the other (2 x max)
FIN_CD0 = 0.012           # NACA 0012 profile drag at this Reynolds number
FIN_ALPHA_STALL = 14.0    # deg
FIN_CL_MAX = 0.95
FIN_ROLL_GAIN = 1.5       # rad/s, roll-rate damping bandwidth. Deliberately slow:
                          # the roll inertia is m*R^2/2 = 0.004 kg m^2, so a single
                          # degree of fin deflection is worth ~800 deg/s^2. A loop
                          # sized by "authority" rather than by inertia demands
                          # deflections the 333 deg/s actuator cannot track and the
                          # roll axis limit-cycles at hundreds of deg/s.
FIN_ROLL_MAX = 2.0        # deg of the travel the roll channel may spend
FIN_MIN_AIRSPEED = 8.0    # m/s below which the fins are not used for control
DRIFT_TILT_GAIN = 1.5     # deg of commanded tilt per m/s of drift
DRIFT_TILT_MAX = 8.0      # deg, cap on the aerodynamic drift-nulling tilt. NOT a
                          # taste setting: an all-moving fin must spend the body's
                          # angle of attack on cancelling its own crossflow before it
                          # can steer at all, so past ~10 deg of trim the set
                          # saturates - asymmetrically, which puts the roll straight
                          # back. Measured, 18 deg re-introduced 290 deg/s of spin.
FIN_CTRL_LEAD = 40.0      # m above the commanded ignition altitude at which the fin
                          # attitude loop wakes up - far enough for the transient to
                          # settle before the motor lights, late enough that the long
                          # free fall is flown at trim

# Rates
DT_PHYS = 0.001           # plant step [s]
CTRL_DIV = 5              # -> 200 Hz control
PLAN_DIV = 100            # -> 10 Hz planner

# Guidance constants
H_TERM = 3.00             # terminal law engages below this altitude [m]
A_TERM = 8.00             # deceleration setting the terminal descent-rate profile
A_TERM_MAX = 70.0         # cap on commanded specific force [m/s^2]
KP_VZ = 3.00              # descent-rate tracking gain
TILT_SLOPE = 1.5          # tilt cone opens with altitude [deg/m]
TILT_MIN = 4.0            # ... from this floor at the pad [deg]
TILT_CAP = 20.0           # ... up to this cap [deg]. Tighter than the reference's 45
                          # deg cone: this vehicle's vertical margin is thin enough
                          # that a wide cone spends more dV on steering than the
                          # horizontal channel is worth.
KI_VX = 0.0               # horizontal trim integral gain (see the plant)
WN_DEFAULT = 9.0          # attitude loop bandwidth while the motor is lit [rad/s]
ZETA_DEFAULT = 1.0        # damping ratio
WN_FIN_DEFAULT = 7.0      # attitude loop bandwidth on the fins alone [rad/s]
ZETA_FIN_DEFAULT = 1.0

# Gain SCHEDULING references. Both actuators are already dynamically inverted - the
# nozzle command is divided by T*L and the fin command by q*S*CL_alpha*arm - so the
# loop gain is nominally independent of throttle and airspeed. What inversion cannot
# undo is the AUTHORITY LIMIT: at 12 N of clamped thrust the nozzle can make a tenth
# of the torque it makes at 120 N, and a loop asking for the same bandwidth simply
# saturates. The schedules below trim the demanded bandwidth by how much authority is
# actually there, with the exponent left for the tuner to find (0 = no schedule).
T_SCHED_REF = 100.0       # N of real, post-clamp thrust the TVC bandwidth refers to
Q_SCHED_REF = 700.0       # Pa of dynamic pressure the fin bandwidth refers to
SCHED_LO, SCHED_HI = 0.35, 2.0   # clamp on either schedule factor

# Touchdown gates (the reference's five, read on magnitudes)
GATE_VZ = 3.00            # m/s
GATE_VH = 0.75            # m/s
GATE_TILT = 10.0          # deg
GATE_OMEGA = 60.0         # deg/s
GATE_H = 0.25             # m

# The planner aims for -0.5 m/s at the pad and calls an altitude usable while the
# projection lands inside the gate with margin (-2.3 m/s against a -3.0 m/s gate).
PLAN_TARGET_VZ = -0.5
FEASIBLE_VZ = -2.3

# Dispersions asked for in the test plan
IGN_DELAY_MAX = 0.300     # s, U(0, IGN_DELAY_MAX) from command to thrust onset
IGN_DELAY_PLAN = 0.300    # s, what the guidance pads for (worst case, see below)
THRUST_SCATTER = 0.15     # +/- of the tabulated thrust, at any instant
THRUST_TAU = 0.70         # s, correlation window of that scatter
ROLL_RATE_MAX = 90.0      # deg/s, U(0, max) with a random sign
SIG_ALIGN = 0.25          # deg, initial attitude alignment error (physical)

N_OUT = 17                # length of the per-flight result vector


# ======================================================================
# --- MOTOR TABLES -----------------------------------------------------
# Both motors are described to the compiled kernels by plain arrays, so the same
# code runs with or without numba.
# ======================================================================
class Fins:
    """Four all-moving cruciform fins.

    The lift slope is the low-aspect-ratio (Helmbold) value, not 2*pi: these fins
    have an effective aspect ratio near 1.5 once mirrored in the body, and using the
    thin-aerofoil slope would over-state their authority by a factor of four.
    """

    def __init__(self, count=FIN_COUNT, root=FIN_ROOT, tip=FIN_TIP, span=FIN_SPAN,
                 arm=FIN_ARM, max_deflect=FIN_MAX_DEFLECT,
                 travel_time=FIN_TRAVEL_TIME, body_diameter=DIAMETER,
                 enabled=True):
        self.count = int(count)
        self.enabled = bool(enabled) and self.count > 0
        self.area = 0.5 * (root + tip) * span              # m2, one fin
        self.span = span
        self.arm = arm
        self.max_deflect = max_deflect
        self.travel_time = travel_time
        # rate limit: the quoted time is END STOP TO END STOP, i.e. 2 x max_deflect
        self.rate = 2.0 * max_deflect / travel_time if travel_time > 0 else 1e6
        ar = 2.0 * span * span / self.area                 # mirrored in the body
        self.aspect = ar
        self.cl_alpha = 2.0 * math.pi * ar / (2.0 + math.sqrt(ar * ar + 4.0))
        # spanwise centre of pressure -> the roll arm
        self.roll_arm = 0.5 * body_diameter + 0.4 * span

    def cd_extra(self, deflect_deg=0.0):
        """Fin drag expressed as an addition to the BODY drag coefficient, so the
        1-D planner can use one number. Induced drag included."""
        a = math.radians(min(abs(deflect_deg), FIN_ALPHA_STALL))
        cl = min(self.cl_alpha * a, FIN_CL_MAX)  # same cap as fin_cl
        cd = FIN_CD0 + cl * cl / (math.pi * self.aspect * 0.85)
        return self.count * self.area * cd / AREA

    def describe(self):
        return (f"{self.count} fins, {self.area * 1e4:.1f} cm2 each, AR {self.aspect:.2f}, "
                f"CL_alpha {self.cl_alpha:.2f} /rad, +/-{self.max_deflect:.0f} deg in "
                f"{self.travel_time * 1000:.0f} ms ({self.rate:.0f} deg/s), arm "
                f"{self.arm:.2f} m, roll arm {self.roll_arm * 1000:.0f} mm; "
                f"airbrake adds dCd {self.cd_extra(self.max_deflect):.3f}")


@njit(cache=True, inline='always')
def fin_cl(alpha, cl_alpha):
    """NACA 0012 lift curve, linear then stalled - and it does stall: 15 deg of
    deflection on top of a few degrees of crossflow is right at the edge."""
    a_st = math.radians(FIN_ALPHA_STALL)
    cl_st = cl_alpha * a_st
    if cl_st > FIN_CL_MAX:
        cl_st = FIN_CL_MAX
    if alpha > a_st:
        return cl_st * math.exp(-(alpha - a_st) * 1.5)
    if alpha < -a_st:
        return -cl_st * math.exp(-(-alpha - a_st) * 1.5)
    cl = cl_alpha * alpha
    if cl > cl_st:
        cl = cl_st
    elif cl < -cl_st:
        cl = -cl_st
    return cl


@njit(cache=True)
def fin_forces(bx, by, bz, u1x, u1y, u1z, u2x, u2y, u2z,
               vx, vy, vz, wx, wy, wz, defl, n_fin, fin_area, fin_arm, roll_arm,
               cl_alpha, aspect):
    """Total force and moment of the fin set, in world axes.

    Each fin is a flat surface whose hinge is the spanwise direction r_i; positive
    deflection pushes along n_i = b x r_i. The fin sees the vehicle's airspeed plus
    the local velocity from the body rate (w x r), so the fins damp rotation on
    their own - that damping is a real, and here useful, part of the model.

    The force is applied at fin_arm along +b (the fins trail ABOVE the CG during a
    nose-down descent) and roll_arm out along r_i, which is what gives the set a
    roll moment the gimbal cannot produce.
    """
    fx = fy = fz = 0.0
    tx = ty = tz = 0.0
    for i in range(n_fin):
        ang = 2.0 * math.pi * i / n_fin
        ca, sa = math.cos(ang), math.sin(ang)
        rx = ca * u1x + sa * u2x
        ry = ca * u1y + sa * u2y
        rz = ca * u1z + sa * u2z
        nx = by * rz - bz * ry
        ny = bz * rx - bx * rz
        nz = bx * ry - by * rx
        # position of this fin's centre of pressure
        px = fin_arm * bx + roll_arm * rx
        py = fin_arm * by + roll_arm * ry
        pz = fin_arm * bz + roll_arm * rz
        # local airspeed at the fin
        lx = vx + (wy * pz - wz * py)
        ly = vy + (wz * px - wx * pz)
        lz = vz + (wx * py - wy * px)
        v2 = lx * lx + ly * ly + lz * lz
        if v2 < 1.0:
            continue
        v = math.sqrt(v2)
        va = -(lx * bx + ly * by + lz * bz)      # axial airspeed, nose-to-tail
        if va < 2.0:
            va = 2.0
        cross = lx * nx + ly * ny + lz * nz      # crossflow along +n
        alpha = math.radians(defl[i]) - cross / va
        q = 0.5 * RHO * v2
        cl = fin_cl(alpha, cl_alpha)
        cd = FIN_CD0 + cl * cl / (math.pi * aspect * 0.85)
        lift = q * fin_area * cl
        drag = q * fin_area * cd
        # lift along +n, drag OPPOSING the vehicle's motion through the air
        ffx = lift * nx - drag * lx / v
        ffy = lift * ny - drag * ly / v
        ffz = lift * nz - drag * lz / v
        fx += ffx
        fy += ffy
        fz += ffz
        tx += py * ffz - pz * ffy
        ty += pz * ffx - px * ffz
        tz += px * ffy - py * ffx
    return fx, fy, fz, tx, ty, tz


def motor_arrays(motor: Motor):
    return motor.t.copy(), motor.f.copy(), motor.cum_impulse.copy()


def booster_arrays(booster: Booster):
    return booster.t.copy(), booster.f.copy(), booster.burned.copy()


@njit(cache=True, inline='always')
def _interp(x, xp, fp):
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    return np.interp(x, xp, fp)


@njit(cache=True, inline='always')
def main_thrust(t, mt, mf):
    """Tabulated (un-throttled, un-dispersed) main-motor thrust [N]."""
    if t < 0.0 or t > mt[-1]:
        return 0.0
    return np.interp(t, mt, mf)


@njit(cache=True, inline='always')
def main_burned(t, mt, mc, prop_mass, i_total):
    """Propellant burned [kg]. The clamp wastes thrust, it does not slow the grain."""
    if t <= 0.0:
        return 0.0
    if t >= mt[-1]:
        return prop_mass
    return prop_mass * np.interp(t, mt, mc) / i_total


@njit(cache=True, inline='always')
def boost_thrust(tau, bt, bf):
    if tau < 0.0 or tau > bt[-1]:
        return 0.0
    return np.interp(tau, bt, bf)


@njit(cache=True, inline='always')
def boost_burned(tau, bt, bb, prop_mass):
    if tau <= 0.0:
        return 0.0
    if tau >= bt[-1]:
        return prop_mass
    return np.interp(tau, bt, bb)


# ======================================================================
# --- GUIDANCE ---------------------------------------------------------
@njit(cache=True, inline='always')
def clampf(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@njit(cache=True, inline='always')
def guidance_dir(h, vx, vy, vz, mass, cd, ax_bias, ay_bias):
    """Commanded thrust direction as a unit vector, plus t_go.

    A direction rather than a pair of angles: the two horizontal channels enter
    symmetrically, there is nothing to wrap and no gimbal lock. `a = -v/t_go` is
    the minimum-effort law for a terminal VELOCITY constraint with free final
    position; the drag term credits the deceleration the air is already providing,
    so the law stops over-tilting and over-throttling.
    """
    svz = abs(vz) if abs(vz) > 0.5 else 0.5
    hh = h if h > 0.1 else 0.1
    t_go = 2.0 * hh / svz
    if t_go < 0.25:
        t_go = 0.25

    req_ax = -vx / t_go + ax_bias
    req_ay = -vy / t_go + ay_bias
    req_az = abs(vz) / t_go + G

    v = math.sqrt(vx * vx + vy * vy + vz * vz)
    if v > 0.01:
        df = 0.5 * RHO * AREA * cd * v * v
        req_ax += df * vx / v / mass
        req_ay += df * vy / v / mass
        req_az += df * vz / v / mass

    if req_az < 0.1:
        req_az = 0.1
    # tilt cone: wide up high, tight near the pad, identical in every azimuth
    max_tilt = math.radians(clampf(h * TILT_SLOPE + TILT_MIN, 0.0, TILT_CAP))
    tmax = math.tan(max_tilt)
    rh = math.sqrt(req_ax * req_ax + req_ay * req_ay)
    if rh > tmax * req_az and rh > 1e-12:
        s = tmax * req_az / rh
        req_ax *= s
        req_ay *= s
    n = math.sqrt(req_ax * req_ax + req_ay * req_ay + req_az * req_az)
    return req_ax / n, req_ay / n, req_az / n, t_go


@njit(cache=True, inline='always')
def gimbal_basis(bx, by, bz, gx, gy, gz):
    """Orthonormal basis (u1, u2) of the plane perpendicular to b.

    u1 is the body-fixed reference g made perpendicular to b, u2 = b x u1. Because
    g turns with the airframe, so do the two gimbal axes - which is what a
    bolted-on servo actually does.
    """
    d = gx * bx + gy * by + gz * bz
    ax, ay, az = gx - d * bx, gy - d * by, gz - d * bz
    n = math.sqrt(ax * ax + ay * ay + az * az)
    if n < 1e-6:
        ax, ay, az = 1.0 - bx * bx, -bx * by, -bx * bz
        n = math.sqrt(ax * ax + ay * ay + az * az)
    u1x, u1y, u1z = ax / n, ay / n, az / n
    return (u1x, u1y, u1z,
            by * u1z - bz * u1y, bz * u1x - bx * u1z, bx * u1y - by * u1x)


@njit(cache=True, inline='always')
def terminal_throttle(h, vz, mass, thrust_avail):
    """Track vz_ref = -sqrt(2*a*h) for the last few metres.

    The caller only ever takes max(plan, terminal): a terminal law that REPLACES
    the plan cuts thrust while the vehicle is still fast and flies it into the
    ground with propellant left in the grain.
    """
    hh = h if h > 0.05 else 0.05
    vz_ref = -clampf(math.sqrt(2.0 * A_TERM * hh), 0.8, 2.5)
    a_cmd = G + KP_VZ * (vz_ref - vz)
    if a_cmd > A_TERM_MAX:
        a_cmd = A_TERM_MAX
    if thrust_avail < 1.0:
        return K_MIN
    return clampf(mass * a_cmd / thrust_avail, K_MIN, K_MAX)


@njit(cache=True, inline='always')
def terminal_active(h, vz, thrust_avail, mass):
    """Gate for the terminal law - and it is a NARROW gate on this vehicle.

    The reference lander can hold a slow 2 m/s approach for a second or two,
    because its grain still has thrust to spare. This one cannot: the burn is over
    2.6 s after ignition whatever the clamp does, and a vehicle that eases to
    -1 m/s at 3 m simply burns out up there and falls the rest of the way, arriving
    at 3+ m/s. So the profile is only allowed when the motor can actually still
    support it - thrust above weight - and the vehicle is already slow and low.
    Everywhere else the receding-horizon level flies it onto the pad with the
    motor still lit, which is what the fixed burn duration demands.
    """
    return h < H_TERM and vz > -3.0 and thrust_avail > 1.6 * mass * G


PLAN_HOLD = 0.30          # s the planned level is held before the projection
                          # assumes full clamp again - see k_from_plan


@njit(cache=True, inline='always')
def k_from_plan(k_level, t_since_plan, h, vz, mass, thrust_avail):
    """The commanded clamp level.

    THE PLAN IS ONE NUMBER. The reference vehicle carries a four-segment throttle
    ladder solved once per planner tick, which is the right family for a classic
    coast-then-slam suicide burn. It is the wrong family here: the clamp cannot
    save the grain, so the burn duration is fixed the moment the motor lights and
    the shape that matters is the TAIL, not the head. What this vehicle needs is a
    receding-horizon level - "the constant clamp that, applied from now on, puts me
    on the pad at half a metre per second" - re-solved ten times a second. The
    replan is what makes a constant level adaptive: it comes down through the burn
    exactly as much as the flight so far turned out to need.

    The level is held for PLAN_HOLD seconds and the projection then assumes FULL
    clamp for the rest of the burn. That detail is what makes one knob enough: the
    honest question at any instant is "how much do I waste right now, given that I
    can still use everything later", and a family that assumed the level was held
    to burnout answers a different question - it cannot express "coast now, slam
    later", so from a high ignition it burns too much too early and stops the
    vehicle in mid-air.

    The terminal law then takes over for the last few metres, and it REPLACES the
    level rather than adding to it: with 120 N against a 31 N weight the endgame
    problem is too much thrust, not too little.
    """
    if terminal_active(h, vz, thrust_avail, mass):
        return terminal_throttle(h, vz, mass, thrust_avail)
    if t_since_plan > PLAN_HOLD:
        return K_MAX
    return k_level


# ======================================================================
# --- PLANNER PROJECTION ----------------------------------------------
@njit(cache=True)
def project_vz(h0, vz0, t0, mass_extra, k_level,
               mt, mf, mc, prop_mass, i_total, cd,
               bt, bf, bb, b_prop, b_t_ign, dt):
    """Forward-integrate the vertical channel under a candidate plan.

    Returns the vertical speed at ground contact (negative = descending).

    It integrates all the way to the ground, INCLUDING the ballistic fall after
    burnout. Stopping the projection the moment the vehicle starts going back up
    would report a hoverslam that over-braked, coasted to a stop 8 m up and then
    fell back at 12 m/s as a success - which is exactly the failure mode that has
    to be planned against, because the grain cannot be saved for later.
    """
    h, vz, t = h0, vz0, t0
    dry = mass_extra
    kd = 0.5 * RHO * AREA * cd
    n = 0
    n_max = int(30.0 / dt)
    while h > 0.0 and n < n_max:
        tn = main_thrust(t, mt, mf)
        mass = dry + (prop_mass - main_burned(t, mt, mc, prop_mass, i_total))
        if b_t_ign < 1.0e5:
            mass -= boost_burned(t - b_t_ign, bt, bb, b_prop)
        k = k_from_plan(k_level, t - t0, h, vz, mass, tn)
        thrust = tn * k
        if b_t_ign < 1.0e5:
            thrust += boost_thrust(t - b_t_ign, bt, bf)
        a = (thrust - kd * vz * abs(vz)) / mass - G
        vz_new = vz + a * dt
        h += 0.5 * (vz + vz_new) * dt
        vz = vz_new
        t += dt
        n += 1
    return vz


@njit(cache=True)
def solve_plan(h, vz, t, mass_extra, mt, mf, mc, prop_mass, i_total, cd,
               bt, bf, bb, b_prop, b_t_ign, dt, target):
    """Pick the clamp level whose projected touchdown speed reaches `target`.

    The outcome is NOT monotone in the level: too little thrust crashes, enough
    lands, and too much stops the vehicle in mid-air with a spent grain and drops
    it from there. A plain bracketed root find on [0.1, 1.0] is therefore invalid -
    it converges happily onto the over-braking branch. The family is scanned
    coarsely instead, the FIRST crossing from below is bracketed, and only that
    bracket is bisected. The lowest level that meets the target is also the one
    that wastes the least and leaves the most room for the next replan.

    Returns (level, residual). When nothing reaches the target, the best level
    found comes back with its negative residual - which is what the booster rule
    reads to decide that the main motor alone cannot close this landing.
    """
    n_scan = 10
    best_k = K_MIN
    best_f = -1.0e9
    prev_k = K_MIN
    prev_f = 0.0
    lo = -1.0
    hi = -1.0
    for i in range(n_scan):
        k = K_MIN + (K_MAX - K_MIN) * i / (n_scan - 1)
        f = project_vz(h, vz, t, mass_extra, k,
                       mt, mf, mc, prop_mass, i_total, cd,
                       bt, bf, bb, b_prop, b_t_ign, dt) - target
        if f > best_f:
            best_f = f
            best_k = k
        if f >= 0.0:
            if i == 0:
                return K_MIN, f
            lo, hi = prev_k, k
            break
        prev_k, prev_f = k, f
    if lo < 0.0:
        # No crossing on the scan grid. The feasible band in k can be narrower than
        # the grid step, so refine the PEAK by ternary search instead of giving up:
        # f(k) rises to a maximum and falls again, and the maximum is both the best
        # achievable landing and the honest input to the booster rule.
        a = best_k - (K_MAX - K_MIN) / (n_scan - 1)
        c = best_k + (K_MAX - K_MIN) / (n_scan - 1)
        if a < K_MIN:
            a = K_MIN
        if c > K_MAX:
            c = K_MAX
        for _ in range(12):
            m1 = a + (c - a) / 3.0
            m2 = c - (c - a) / 3.0
            f1 = project_vz(h, vz, t, mass_extra, m1,
                            mt, mf, mc, prop_mass, i_total, cd,
                            bt, bf, bb, b_prop, b_t_ign, dt) - target
            f2 = project_vz(h, vz, t, mass_extra, m2,
                            mt, mf, mc, prop_mass, i_total, cd,
                            bt, bf, bb, b_prop, b_t_ign, dt) - target
            if f1 > best_f:
                best_f, best_k = f1, m1
            if f2 > best_f:
                best_f, best_k = f2, m2
            if f1 < f2:
                a = m1
            else:
                c = m2
        return best_k, best_f
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        f = project_vz(h, vz, t, mass_extra, mid,
                       mt, mf, mc, prop_mass, i_total, cd,
                       bt, bf, bb, b_prop, b_t_ign, dt) - target
        if f >= 0.0:
            hi = mid
        else:
            lo = mid
    return hi, 0.0


@njit(cache=True)
def project_rh(h0, vz0, t0, mass_extra, mt, mf, mc, prop_mass, i_total, cd,
               bt, bf, bb, b_prop, b_t_ign, target):
    """Projection that RE-SOLVES the clamp level as it goes - the pre-flight model
    of what the flight computer will actually do.

    A single constant level, judged once at ignition, badly under-states what the
    vehicle can do from a high ignition: the receding-horizon controller starts
    near the clamp minimum, wastes the early part of the grain and comes up to full
    later, and no constant level can imitate that. Using the honest closed loop
    here is what makes the pre-flight ignition window match the flight.

    Only used pre-flight (once per entry state), never inside the 10 Hz loop.
    """
    dt = 0.01
    h, vz, t = h0, vz0, t0
    kd = 0.5 * RHO * AREA * cd
    k_level = K_MAX
    n = 0
    n_max = 4000
    since = 1.0e9
    while h > 0.0 and n < n_max:
        if since >= 0.1:
            k_level, _ = solve_plan(h, vz, t, mass_extra,
                                    mt, mf, mc, prop_mass, i_total, cd,
                                    bt, bf, bb, b_prop, b_t_ign, 0.03, target)
            since = 0.0
        tn = main_thrust(t, mt, mf)
        mass = mass_extra + (prop_mass - main_burned(t, mt, mc, prop_mass, i_total))
        if b_t_ign < 1.0e5:
            mass -= boost_burned(t - b_t_ign, bt, bb, b_prop)
        k = k_from_plan(k_level, since, h, vz, mass, tn)
        thrust = tn * k
        if b_t_ign < 1.0e5:
            thrust += boost_thrust(t - b_t_ign, bt, bf)
        a = (thrust - kd * vz * abs(vz)) / mass - G
        vz_new = vz + a * dt
        h += 0.5 * (vz + vz_new) * dt
        vz = vz_new
        t += dt
        since += dt
        n += 1
    return vz


@njit(cache=True)
def freefall_to(h_start, h_target, vx0, vz0, cd, mass, dt):
    """Coupled 2-D drag free fall - the horizontal speed enters the drag magnitude,
    so a decoupled 1-D propagation over-predicts vx at ignition."""
    h, vx, vz, t = h_start, vx0, vz0, 0.0
    kd = 0.5 * RHO * AREA * cd / mass
    while h > h_target and t < 30.0:
        v = math.sqrt(vx * vx + vz * vz)
        ax = -kd * v * vx
        az = -kd * v * vz - G
        vxm, vzm = vx + 0.5 * ax * dt, vz + 0.5 * az * dt
        vm = math.sqrt(vxm * vxm + vzm * vzm)
        h += vzm * dt
        vx += -kd * vm * vxm * dt
        vz += (-kd * vm * vzm - G) * dt
        t += dt
    return vx, vz, t


@njit(cache=True)
def find_ignition(h_start, vx0, vz0, cd, cd_free, mass0, delay_pad, boost_ready,
                  mt, mf, mc, prop_mass, i_total, bt, bf, bb, b_prop):
    """Commanded ignition altitude and the matching command time.

    Three numbers decide it:

    * `h_min` - the lowest altitude from which SOME clamp level still lands softly.
      Below it nothing can be done, so it is the floor.
    * the igniter pad - the fall between the command and thrust onset, sized on the
      WORST-CASE delay and evaluated at the velocity actually reached down there. A
      hoverslam is asymmetric: igniting high is recoverable by throttling down,
      igniting low is not recoverable at all.
    * `h_max` - the highest altitude from which the landing still closes. The grain
      burns for its fixed 2.6 s whatever the clamp does, so igniting too high means
      burning out with altitude left and falling the rest of the way. The pad may
      never push the command above it.

    The altitude axis is SCANNED rather than bisected, because feasibility is an
    interval and not a half-line: bisection on an interval-valued predicate walks
    off one of its two edges.

    If a booster is carried and the on-board rule may use it, both edges are
    evaluated WITH it - the vehicle plans on the lower, safer ignition and the rule
    then decides in flight whether the D9 is actually needed.
    """
    b_ign = 0.0 if boost_ready > 0.5 else 1.0e6
    n = 24
    h_lo_ok = -1.0
    h_hi_ok = -1.0
    h_best = 0.5 * h_start
    vz_best = -1.0e9
    step = (h_start - 5.0) / (n - 1)
    for i in range(n):
        hh = 5.0 + i * step
        _, vzi, _ = freefall_to(h_start, hh, vx0, vz0, cd_free, mass0, 0.005)
        vz_td = project_rh(hh, vzi, 0.0, mass0 - prop_mass,
                           mt, mf, mc, prop_mass, i_total, cd,
                           bt, bf, bb, b_prop, b_ign, PLAN_TARGET_VZ)
        if vz_td > vz_best:
            vz_best = vz_td
            h_best = hh
        if vz_td >= FEASIBLE_VZ:
            if h_lo_ok < 0.0:
                h_lo_ok = hh
            h_hi_ok = hh
        elif h_lo_ok > 0.0:
            break                      # the feasible band is one interval
    if h_lo_ok < 0.0:
        # No altitude closes the landing at all. Take the least-bad one rather than
        # an arbitrary default: the flight is going to be scored on how hard it
        # arrives, and this is where it arrives softest.
        h_lo_ok = h_hi_ok = h_best
    # refine both edges
    lo, hi = max(5.0, h_lo_ok - step), h_lo_ok
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        _, vzi, _ = freefall_to(h_start, mid, vx0, vz0, cd_free, mass0, 0.005)
        vz_td = project_rh(mid, vzi, 0.0, mass0 - prop_mass,
                           mt, mf, mc, prop_mass, i_total, cd,
                           bt, bf, bb, b_prop, b_ign, PLAN_TARGET_VZ)
        if vz_td >= FEASIBLE_VZ:
            hi = mid
        else:
            lo = mid
    h_min = hi
    lo, hi = h_hi_ok, min(h_start - 1.0, h_hi_ok + step)
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        _, vzi, _ = freefall_to(h_start, mid, vx0, vz0, cd_free, mass0, 0.005)
        vz_td = project_rh(mid, vzi, 0.0, mass0 - prop_mass,
                           mt, mf, mc, prop_mass, i_total, cd,
                           bt, bf, bb, b_prop, b_ign, PLAN_TARGET_VZ)
        if vz_td >= FEASIBLE_VZ:
            lo = mid
        else:
            hi = mid
    h_max = lo

    _, vz_at, _ = freefall_to(h_start, h_min, vx0, vz0, cd_free, mass0, 0.005)
    pad = abs(vz_at) * delay_pad + 0.5 * G * delay_pad * delay_pad
    h_cmd = h_min + pad
    if h_cmd > h_max:
        # The igniter spread is wider than the whole feasible band, so no command
        # altitude can cover every delay. Sitting in the middle of the BAND is the
        # instinctive choice and it is wrong: the ignition point is spread over
        # ~14 m whatever we do, so what maximises the overlap is putting the
        # command at the band's TOP and letting the spread hang down through it.
        # Anything lower throws away the short-delay flights at the top as well as
        # the long-delay ones at the bottom.
        h_cmd = h_max
    if h_cmd > h_start - 1.0:
        h_cmd = h_start - 1.0
    _, _, t_cmd = freefall_to(h_start, h_cmd, vx0, vz0, cd_free, mass0, 0.005)
    return h_cmd, t_cmd, h_min


# ======================================================================
# --- THE PLANT --------------------------------------------------------
# One flight, from the entry altitude to touchdown. Everything the vehicle does -
# free fall, the igniter delay, the burn, the clamp, both gimbal channels, the
# uncontrolled roll - happens inside this loop at DT_PHYS.
# ======================================================================
@njit(cache=True)
def fly(seed, h_start, vx0, vz0, m_gross, cd, wn, zeta, wn_fin, zeta_fin,
        roll_gain, sched_tvc, sched_fin,
        n_boost, use_boost_rule, roll_max, ign_delay_max, delay_pad,
        thr_scatter, thr_tau, gyro_ff,
        n_fin, fin_area, fin_arm, fin_roll_arm, fin_cl_alpha, fin_aspect,
        fin_max, fin_rate, fin_brake_mode, fin_drift, cd_free, b_cant, b_azim,
        mt, mf, mc, prop_mass, i_total, bt, bf, bb, b_prop, b_total,
        tel, n_tel, h_cmd_in):
    np.random.seed(seed)
    out = np.zeros(N_OUT, dtype=np.float64)

    # ---- per-flight draws ------------------------------------------------
    ign_delay = np.random.uniform(0.0, ign_delay_max)
    roll_rate = math.radians(np.random.uniform(0.0, roll_max))
    if np.random.random() < 0.5:
        roll_rate = -roll_rate
    align = math.radians(np.random.normal() * SIG_ALIGN)
    align_az = np.random.uniform(0.0, 2.0 * math.pi)
    s_thr = 0.0                                  # instantaneous thrust deviation

    mass0 = m_gross + n_boost * b_total          # carried boosters are dead weight
    dry = mass0 - prop_mass

    # ---- pre-flight plan -------------------------------------------------
    # The pre-flight plan depends only on the ENTRY STATE, never on the dispersion
    # draws (the vehicle cannot know its igniter delay in advance), so a campaign
    # solves it once per grid cell and passes it in - it costs more than the flight
    # itself.
    # The planner's 1-D projection only cares about the vertical, so it is handed
    # the booster's AXIAL component - a 15 deg cant costs 3.4 % of its impulse
    # upwards, and the rest goes sideways where the horizontal channel gets it.
    bf_ax = bf * math.cos(b_cant)
    boost_ready = 1.0 if (n_boost > 0.5 and use_boost_rule > 0.5) else 0.0
    if h_cmd_in > 0.0:
        h_cmd = h_cmd_in
    else:
        h_cmd, _, _ = find_ignition(h_start, vx0, vz0, cd, cd_free, mass0,
                                    delay_pad, boost_ready, mt, mf, mc,
                                    prop_mass, i_total, bt, bf_ax, bb, b_prop)

    # ---- state -----------------------------------------------------------
    x, y, h = 0.0, 0.0, h_start
    vx, vy, vz = vx0, 0.0, vz0
    # The vehicle is released nose-down and upright, so the thrust axis starts along
    # +z. It is NOT initialised along -v: the airframe weathercocks into the relative
    # wind by itself during the free fall, and letting the plant do that is what makes
    # the entry attitude consistent with the entry state instead of assumed.
    bx, by, bz = 0.0, 0.0, 1.0
    # small physical alignment error
    px, py, pz = 1.0, 0.0, 0.0
    pn = math.sqrt(px * px + py * py + pz * pz)
    if pn < 1e-6:
        px, py, pz, pn = 1.0, 0.0, 0.0, 1.0
    px, py, pz = px / pn, py / pn, pz / pn
    qx = by * pz - bz * py
    qy = bz * px - bx * pz
    qz = bx * py - by * px
    ex = math.cos(align_az) * px + math.sin(align_az) * qx
    ey = math.cos(align_az) * py + math.sin(align_az) * qy
    ez = math.cos(align_az) * pz + math.sin(align_az) * qz
    bx += align * ex
    by += align * ey
    bz += align * ez
    bn = math.sqrt(bx * bx + by * by + bz * bz)
    bx, by, bz = bx / bn, by / bn, bz / bn
    # body-fixed transverse reference axis (marks the roll orientation)
    gx, gy, gz = px, py, pz
    d = gx * bx + gy * by + gz * bz
    gx, gy, gz = gx - d * bx, gy - d * by, gz - d * bz
    gn = math.sqrt(gx * gx + gy * gy + gz * gz)
    gx, gy, gz = gx / gn, gy / gn, gz / gn
    # angular rate: the inherited spin about the body axis
    wx, wy, wz = roll_rate * bx, roll_rate * by, roll_rate * bz

    defl = np.zeros(4)           # fin deflections [deg]
    defl_cmd = np.zeros(4)
    ctrl_arr = np.zeros(4)
    xflow = np.zeros(4)
    fin_brake_now = 0.0
    srv1 = srv2 = 0.0            # servo angles [deg]
    sr1 = sr2 = 0.0              # servo rates [deg/s]
    cmd1 = cmd2 = 0.0
    k_act, k_rate_act = K_MIN, 0.0
    k_cmd = K_MIN
    k_level = K_MAX

    t = 0.0
    t_ign_cmd = -1.0
    t_burn0 = -1.0               # absolute time of thrust onset
    engine_on = False
    b_t_ign = 1.0e6              # booster ignition, in BURN time
    boost_used = 0.0
    ax_bias = ay_bias = 0.0
    max_tilt = 0.0
    dv_clamp = 0.0
    dv_tilt = 0.0
    k_sum = 0.0
    k_n = 0.0
    h_ign_real = 0.0
    step = 0
    tel_i = 0

    while t < 40.0:
        # ---------------- ignition logic ----------------
        if t_ign_cmd < 0.0 and h <= h_cmd:
            t_ign_cmd = t
        if t_burn0 < 0.0 and t_ign_cmd >= 0.0 and t - t_ign_cmd >= ign_delay:
            t_burn0 = t
            h_ign_real = h
        tb = t - t_burn0 if t_burn0 >= 0.0 else -1.0
        engine_on = tb >= 0.0 and tb <= mt[-1]

        # ---------------- thrust ----------------
        # instantaneous scatter: an OU process bounded at +/-15 %, correlated over
        # a 0.7 s window, so the deviation is a slow wander rather than white noise
        s_thr += (-s_thr / thr_tau) * DT_PHYS + thr_scatter * 0.5 * math.sqrt(
            2.0 * DT_PHYS / thr_tau) * np.random.normal()
        s_thr = clampf(s_thr, -thr_scatter, thr_scatter)

        t_nom = main_thrust(tb, mt, mf) * (1.0 + s_thr) if tb >= 0.0 else 0.0
        mass = dry + prop_mass
        if tb > 0.0:
            mass = dry + (prop_mass - main_burned(tb, mt, mc, prop_mass, i_total))
        thr_boost = 0.0
        if b_t_ign < 1.0e5 and tb >= 0.0:
            thr_boost = boost_thrust(tb - b_t_ign, bt, bf) * (1.0 + s_thr)
            mass -= boost_burned(tb - b_t_ign, bt, bb, b_prop)

        # ---------------- planner, 10 Hz ----------------
        if step % PLAN_DIV == 0 and t_ign_cmd >= 0.0 and h > 0.05:
            tb_p = tb if tb >= 0.0 else 0.0
            k_level, resid = solve_plan(h, vz, tb_p, dry,
                                        mt, mf, mc, prop_mass, i_total, cd,
                                        bt, bf_ax, bb, b_prop, b_t_ign, 0.02,
                                        PLAN_TARGET_VZ)
            # ---- the booster rule ----
            # A D9 is a one-way door: it cannot be throttled, stopped or relit. So it
            # is lit only once the plan says the main motor CANNOT close the landing
            # even at full clamp - and only while there is still burn left for it to
            # matter. If that never happens, it is never used.
            if (use_boost_rule > 0.5 and n_boost > 0.5 and b_t_ign > 1.0e5
                    and resid < -1.0):
                b_t_ign = tb_p
                boost_used = 1.0
        if engine_on or (t_burn0 < 0.0 and t_ign_cmd >= 0.0):
            k_cmd = k_level

        # ---------------- controller, 200 Hz ----------------
        if step % CTRL_DIV == 0:
            dt_c = DT_PHYS * CTRL_DIV
            # body-fixed gimbal / fin axes, needed by everything below
            c1x, c1y, c1z, c2x, c2y, c2z = gimbal_basis(bx, by, bz, gx, gy, gz)
            # Feed the canted booster's side force forward. The flight computer
            # knows it lit the D9 and knows which way the airframe is pointing, so
            # the 15 deg cant is a KNOWN horizontal input, not a disturbance to be
            # discovered through the velocity error. Subtracting it here lets the
            # main motor cancel it while the vehicle stays vertical, instead of the
            # guidance chasing the drift it causes.
            bxf = byf = 0.0
            if thr_boost > 0.0:
                mfx = math.cos(b_azim) * c1x + math.sin(b_azim) * c2x
                mfy = math.cos(b_azim) * c1y + math.sin(b_azim) * c2y
                sb = math.sin(b_cant)
                bxf = -thr_boost * (-sb * mfx) / mass
                byf = -thr_boost * (-sb * mfy) / mass
            ux, uy, uz, t_go = guidance_dir(h, vx, vy, vz, mass, cd,
                                            ax_bias + bxf, ay_bias + byf)
            # ---- aerodynamic drift nulling, before the motor is lit ----
            # The body's own normal force is a free horizontal actuator while the
            # vehicle is still falling: tilting the thrust axis TOWARDS the drift
            # increases the component of the airflow perpendicular to the body, and
            # that normal force pushes back against the drift. Killing the sideways
            # velocity here costs no propellant at all, and it is worth more than
            # that - the burn can then be flown nearly vertical instead of spending
            # its thrust on steering, which is the one thing this vehicle cannot
            # afford. It is handed over to the ordinary guidance law once the
            # attitude loop wakes up for the ignition.
            if (fin_drift > 0.5 and t_ign_cmd < 0.0 and h > h_cmd + FIN_CTRL_LEAD
                    and n_fin > 0.5):
                vh_m = math.sqrt(vx * vx + vy * vy)
                if vh_m > 0.2:
                    tl = math.tan(math.radians(clampf(vh_m * DRIFT_TILT_GAIN, 0.0,
                                                      DRIFT_TILT_MAX)))
                    ux = tl * vx / vh_m
                    uy = tl * vy / vh_m
                    uz = 1.0
                    nrm = math.sqrt(ux * ux + uy * uy + 1.0)
                    ux, uy, uz = ux / nrm, uy / nrm, uz / nrm
                else:
                    ux, uy, uz = 0.0, 0.0, 1.0
            # attitude error as a rotation vector: e = asin|b x u| * unit(b x u).
            # No atan2, no wrap, no gimbal lock - and perpendicular to b by
            # construction, so the loop can never ask for a roll it cannot make.
            cx = by * uz - bz * uy
            cy = bz * ux - bx * uz
            cz = bx * uy - by * ux
            cn = math.sqrt(cx * cx + cy * cy + cz * cz)
            if cn > 1e-9:
                ang = math.asin(clampf(cn, -1.0, 1.0))
                if bx * ux + by * uy + bz * uz < 0.0:
                    ang = math.pi - ang
                sc = ang / cn
                eax, eay, eaz = cx * sc, cy * sc, cz * sc
            else:
                eax = eay = eaz = 0.0
            tilt = math.degrees(math.acos(clampf(bz, -1.0, 1.0)))
            if tilt > max_tilt and engine_on:
                max_tilt = tilt

            # split the rate into roll and transverse: the gimbal makes no torque
            # about b, so feeding the roll rate into the rate loop asks for the
            # impossible and the inversion answers with a deflection that does
            # something else instead
            w_roll = wx * bx + wy * by + wz * bz
            gtx, gty, gtz = wx - w_roll * bx, wy - w_roll * by, wz - w_roll * bz

            I_t = mass * (L_BODY * L_BODY / 12.0)
            I_a = mass * I_AXIAL_COEF

            # aero feedforward: tau = L_CP * (b x F_normal)
            rvx, rvy, rvz = vx, vy, vz
            v2 = rvx * rvx + rvy * rvy + rvz * rvz
            q = 0.5 * RHO * v2
            tafx = tafy = tafz = 0.0
            if v2 > 0.0025:
                vr = math.sqrt(v2)
                ahx, ahy, ahz = -rvx / vr, -rvy / vr, -rvz / vr
                ca = ahx * bx + ahy * by + ahz * bz
                ffk = q * AREA * C_N_ALPHA * L_CP
                tafx = ffk * (by * ahz - bz * ahy)
                tafy = ffk * (bz * ahx - bx * ahz)
                tafz = ffk * (bx * ahy - by * ahx)

            # real, post-clamp thrust - the nozzle's actual authority
            t_est = t_nom * k_act
            if t_est < 8.0:
                t_est = 8.0

            # ---- which loop is flying, and how hard ----
            # One attitude loop, but its gains belong to whichever actuator is
            # authoritative: the nozzle once the motor is lit, the fins before that.
            # Each is then scheduled on the authority it actually has - real
            # post-clamp thrust for the nozzle, dynamic pressure for the fins.
            if engine_on:
                wn_a, zeta_a = wn, zeta
                fac = (t_est / T_SCHED_REF) ** sched_tvc if sched_tvc != 0.0 else 1.0
            else:
                wn_a, zeta_a = wn_fin, zeta_fin
                fac = (q / Q_SCHED_REF) ** sched_fin if sched_fin != 0.0 else 1.0
            wn_a = wn_a * clampf(fac, SCHED_LO, SCHED_HI)
            k_rate = 2.0 * zeta_a * wn_a
            k_th = wn_a / (2.0 * zeta_a)
            ocx, ocy, ocz = k_th * eax, k_th * eay, k_th * eaz
            arx = k_rate * (ocx - gtx)
            ary = k_rate * (ocy - gty)
            arz = k_rate * (ocz - gtz)
            trx = I_t * arx - tafx
            try_ = I_t * ary - tafy
            trz = I_t * arz - tafz
            # gyroscopic decoupling: for an axisymmetric body
            #   I_t dw_t/dt = tau_t - (I_t - I_a) w_roll (b x w_t)
            gyk = gyro_ff * (I_t - I_a) * w_roll
            trx += gyk * (by * gtz - bz * gty)
            try_ += gyk * (bz * gtx - bx * gtz)
            trz += gyk * (bx * gty - by * gtx)

            # dynamic inversion. tau = -L*T*(b x s) inverts EXACTLY to
            # s = (b x tau)/(L*T) because s is perpendicular to b. Without it the
            # loop gain would sweep by 100x as the thrust runs down the curve.
            inv = 1.0 / (t_est * L_GIMBAL)
            sqx = (by * trz - bz * try_) * inv
            sqy = (bz * trx - bx * trz) * inv
            sqz = (bx * try_ - by * trx) * inv
            sd1 = clampf(sqx * c1x + sqy * c1y + sqz * c1z, -0.999, 0.999)
            sd2 = clampf(sqx * c2x + sqy * c2y + sqz * c2z, -0.999, 0.999)
            cmd1 = math.degrees(math.asin(sd1)) * TVC_RATIO
            cmd2 = math.degrees(math.asin(sd2)) * TVC_RATIO
            cmd1 = clampf(cmd1, -TVC_SERVO_MAX, TVC_SERVO_MAX)
            cmd2 = clampf(cmd2, -TVC_SERVO_MAX, TVC_SERVO_MAX)
            cmd1 = math.floor(cmd1 / SERVO_QUANT + 0.5) * SERVO_QUANT
            cmd2 = math.floor(cmd2 / SERVO_QUANT + 0.5) * SERVO_QUANT
            if not engine_on:
                # A gimbal with no thrust behind it makes no torque, so there is
                # nothing to steer with during the free fall - the airframe simply
                # weathercocks. Parking the nozzle keeps the servo from chattering
                # against a demand it cannot meet and starts the burn centred.
                cmd1 = 0.0
                cmd2 = 0.0

            # ---------------- fin allocation ----------------
            # The fins pick up what the gimbal cannot: the whole demand while the
            # motor is unlit, whatever is left when the nozzle is at its stop, and
            # the roll axis, which the gimbal cannot touch at all.
            if n_fin > 0.5:
                # torque the gimbal will actually deliver with the command above
                s1c = math.sin(math.radians(cmd1 / TVC_RATIO))
                s2c = math.sin(math.radians(cmd2 / TVC_RATIO))
                gsx = s1c * c1x + s2c * c2x
                gsy = s1c * c1y + s2c * c2y
                gsz = s1c * c1z + s2c * c2z
                tgx = -L_GIMBAL * t_est * (by * gsz - bz * gsy)
                tgy = -L_GIMBAL * t_est * (bz * gsx - bx * gsz)
                tgz = -L_GIMBAL * t_est * (bx * gsy - by * gsx)
                if not engine_on:
                    tgx = tgy = tgz = 0.0
                rfx = trx - tgx
                rfy = try_ - tgy
                rfz = trz - tgz
                # roll: pure rate damping towards zero. There is no roll ATTITUDE
                # to hold - nothing in the mission cares which way round the vehicle
                # is - but arriving spinning is a gate, and a spin costs the two
                # transverse channels their gyroscopic coupling.
                tau_roll = I_a * roll_gain * (0.0 - w_roll)

                v_f = math.sqrt(v2)
                qf = 0.5 * RHO * v2
                k_t = 2.0 * fin_arm * qf * fin_area * fin_cl_alpha
                k_r = n_fin * fin_roll_arm * qf * fin_area * fin_cl_alpha
                # Fins do nothing at low airspeed, and the inversion knows it: 1/q
                # runs away as the vehicle leaves the release point at walking pace,
                # so every channel saturates on a demand it cannot meet, the four
                # deflections saturate ASYMMETRICALLY, and the set spins the airframe
                # up to hundreds of deg/s before it has any authority to stop it.
                # Below this speed the fins simply hold their airbrake position.
                if v_f < FIN_MIN_AIRSPEED:
                    k_t = 0.0
                    k_r = 0.0
                # During the free fall the fins do NOT fight the weathercock. The
                # airframe is statically stable and trims itself within a few degrees
                # of the airflow, which is where it wants to be at ignition anyway;
                # holding it dead vertical instead costs large deflections, and four
                # fins deflected hard and asymmetrically are a roll torque - measured,
                # it spun the airframe to 340 deg/s before the motor was even lit.
                # Attitude control starts with the ignition command; the airbrake and
                # the roll damper run the whole way down.
                if (t_ign_cmd < 0.0 and h > h_cmd + FIN_CTRL_LEAD
                        and fin_drift < 0.5):
                    k_t = 0.0
                if k_t < 1e-6:
                    a_cmd_f = b_cmd_f = 0.0
                else:
                    # the fin set's transverse torque runs OPPOSITE to the gimbal's
                    # for the same force, because the fins are aft of the CG
                    a_cmd_f = -(rfx * c1x + rfy * c1y + rfz * c1z) / k_t
                    b_cmd_f = -(rfx * c2x + rfy * c2y + rfz * c2z) / k_t
                d_roll = 0.0 if k_r < 1e-6 else tau_roll / k_r
                d_roll = clampf(d_roll, -math.radians(FIN_ROLL_MAX),
                                math.radians(FIN_ROLL_MAX))
                a_cmd_f = clampf(math.degrees(a_cmd_f), -fin_max, fin_max)
                b_cmd_f = clampf(math.degrees(b_cmd_f), -fin_max, fin_max)
                d_roll = math.degrees(d_roll)

                # airbrake: alternating +,-,+,- cancels both the lift and the roll
                # torque of the set and leaves pure drag. Deployed while the motor
                # is unlit, which is where the energy has to come out - and stowed
                # at ignition so the travel belongs to the controller.
                if fin_brake_mode > 1.5:
                    fin_brake_now = fin_max
                elif fin_brake_mode > 0.5 and not engine_on:
                    fin_brake_now = fin_max
                else:
                    fin_brake_now = 0.0

                # Control first, brake with what is left - and the brake magnitude
                # is the SAME on all four fins, taken from the tightest one.
                #
                # This is why the splayed brake was rolling the vehicle. The set is
                # roll-neutral only while every fin sits at the same |angle of
                # attack|; with sideslip that needs different DEFLECTIONS on the two
                # fins of a pair (the crossflow adds to one and subtracts from the
                # other), and the correction is in the command. But if the fins are
                # squeezed into whatever travel each one has left, the four end up at
                # different angles - measured, 0.02 N m at 2 deg of sideslip and
                # 0.65 N m at 30 deg. Against a roll inertia of 0.004 kg m^2 that is
                # already hundreds of deg/s. Equalised, the residual is 0.005 N m.
                # axial airspeed, for the crossflow correction
                va_f = -(vx * bx + vy * by + vz * bz)
                if va_f < 3.0:
                    va_f = 3.0
                room = fin_max
                for _i in range(int(n_fin)):
                    ang_i = 2.0 * math.pi * _i / n_fin
                    rix = math.cos(ang_i) * c1x + math.sin(ang_i) * c2x
                    riy = math.cos(ang_i) * c1y + math.sin(ang_i) * c2y
                    riz = math.cos(ang_i) * c1z + math.sin(ang_i) * c2z
                    nix = by * riz - bz * riy
                    niy = bz * rix - bx * riz
                    niz = bx * riy - by * rix
                    xflow[_i] = math.degrees((vx * nix + vy * niy + vz * niz) / va_f)
                    ctrl_i = (d_roll + a_cmd_f * math.cos(ang_i)
                              + b_cmd_f * math.sin(ang_i) + xflow[_i])
                    ctrl_i = clampf(ctrl_i, -fin_max, fin_max)
                    ctrl_arr[_i] = ctrl_i
                    r_i = fin_max - abs(ctrl_i)
                    if r_i < room:
                        room = r_i
                brake = fin_brake_now
                if brake > room:
                    brake = room
                for _i in range(int(n_fin)):
                    alt = 1.0 if _i % 2 == 0 else -1.0
                    defl_cmd[_i] = clampf(ctrl_arr[_i] + brake * alt,
                                          -fin_max, fin_max)

        # ---------------- servo actuators (one per gimbal axis) ----------------
        # Rate-limited with a DECELERATION-AWARE target rate: the servo may only run
        # as fast as it can still stop in the travel that is left,
        # v = sqrt(2*a*|e|). A plain "slew at max speed until you arrive" model
        # overshoots by tens of degrees at 500 deg/s and turns the actuator itself
        # into the dominant instability.
        e_s = cmd1 - srv1
        d_s = 1.0 if e_s > 0.0 else (-1.0 if e_s < 0.0 else 0.0)
        v_t = d_s * min(TVC_MAX_SPEED, math.sqrt(2.0 * TVC_MAX_ACCEL * abs(e_s)))
        dv_max = TVC_MAX_ACCEL * DT_PHYS
        sr1 += clampf(v_t - sr1, -dv_max, dv_max)
        srv1 += sr1 * DT_PHYS
        e_s = cmd2 - srv2
        d_s = 1.0 if e_s > 0.0 else (-1.0 if e_s < 0.0 else 0.0)
        v_t = d_s * min(TVC_MAX_SPEED, math.sqrt(2.0 * TVC_MAX_ACCEL * abs(e_s)))
        sr2 += clampf(v_t - sr2, -dv_max, dv_max)
        srv2 += sr2 * DT_PHYS
        srv1 = clampf(srv1, -TVC_SERVO_MAX, TVC_SERVO_MAX)
        srv2 = clampf(srv2, -TVC_SERVO_MAX, TVC_SERVO_MAX)

        # ---------------- fin actuators ----------------
        # Same deceleration-aware rate limit as the gimbal servos. The quoted 90 ms
        # is end stop to end stop, so the rate is 2*max/90 ms.
        if n_fin > 0.5:
            for _i in range(int(n_fin)):
                e_f = defl_cmd[_i] - defl[_i]
                if e_f > fin_rate * DT_PHYS:
                    defl[_i] += fin_rate * DT_PHYS
                elif e_f < -fin_rate * DT_PHYS:
                    defl[_i] -= fin_rate * DT_PHYS
                else:
                    defl[_i] = defl_cmd[_i]

        # ---------------- throttle actuator ----------------
        kd_des = clampf((k_cmd - k_act) / DT_PHYS, -THROTTLE_SPEED, THROTTLE_SPEED)
        dk = clampf(kd_des - k_rate_act, -THROTTLE_ACCEL * DT_PHYS,
                    THROTTLE_ACCEL * DT_PHYS)
        k_rate_act += dk
        k_act = clampf(k_act + k_rate_act * DT_PHYS, K_MIN, K_MAX)

        # ---------------- forces ----------------
        n1 = math.radians(srv1 / TVC_RATIO)
        n2 = math.radians(srv2 / TVC_RATIO)
        c1x, c1y, c1z, c2x, c2y, c2z = gimbal_basis(bx, by, bz, gx, gy, gz)
        s1, s2 = math.sin(n1), math.sin(n2)
        s_perp = math.sqrt(s1 * s1 + s2 * s2)
        if s_perp > 0.999:
            s1 *= 0.999 / s_perp
            s2 *= 0.999 / s_perp
            s_perp = 0.999
        sx = s1 * c1x + s2 * c2x
        sy = s1 * c1y + s2 * c2y
        sz = s1 * c1z + s2 * c2z
        ca_n = math.sqrt(1.0 - s_perp * s_perp)
        # thrust direction: the body axis tilted by the nozzle deflection
        tdx = ca_n * bx + sx
        tdy = ca_n * by + sy
        tdz = ca_n * bz + sz
        tn_ = math.sqrt(tdx * tdx + tdy * tdy + tdz * tdz)
        tdx, tdy, tdz = tdx / tn_, tdy / tn_, tdz / tn_

        thrust_main = t_nom * k_act
        thrust = thrust_main + thr_boost
        if engine_on:
            dv_clamp += (t_nom - t_nom * k_act) / mass * DT_PHYS
            # the vertical component the tilt gives away - "thrust spent on steering"
            dv_tilt += thrust * (1.0 - tdz) / mass * DT_PHYS
            k_sum += k_act
            k_n += 1.0

        rvx, rvy, rvz = vx, vy, vz
        v2 = rvx * rvx + rvy * rvy + rvz * rvz
        v_rel = math.sqrt(v2)
        q = 0.5 * RHO * v2
        f_ax = f_ay = f_az = 0.0
        tax = tay = taz = 0.0
        if v_rel > 0.05:
            ahx, ahy, ahz = -rvx / v_rel, -rvy / v_rel, -rvz / v_rel
            ca = ahx * bx + ahy * by + ahz * bz
            f_axial = q * AREA * cd * ca
            fnk = q * AREA * C_N_ALPHA
            f_ax = f_axial * bx + fnk * (ahx - ca * bx)
            f_ay = f_axial * by + fnk * (ahy - ca * by)
            f_az = f_axial * bz + fnk * (ahz - ca * bz)
            # aero damping acts on the TRANSVERSE rate only: an uncanted airframe
            # has almost no roll damping, and applying this coefficient to the spin
            # would quietly brake it and flatter the controller
            c_damp = 0.5 * RHO * v_rel * AREA * C_N_ALPHA * L_CP * L_CP
            wr_t = wx * bx + wy * by + wz * bz
            tax = L_CP * fnk * (by * ahz - bz * ahy) - c_damp * (wx - wr_t * bx)
            tay = L_CP * fnk * (bz * ahx - bx * ahz) - c_damp * (wy - wr_t * by)
            taz = L_CP * fnk * (bx * ahy - by * ahx) - c_damp * (wz - wr_t * bz)

        if n_fin > 0.5:
            ffx, ffy, ffz, ftx, fty, ftz = fin_forces(
                bx, by, bz, c1x, c1y, c1z, c2x, c2y, c2z,
                vx, vy, vz, wx, wy, wz, defl, int(n_fin), fin_area, fin_arm,
                fin_roll_arm, fin_cl_alpha, fin_aspect)
            f_ax += ffx
            f_ay += ffy
            f_az += ffz
            tax += ftx
            tay += fty
            taz += ftz

        # The booster's own thrust direction: canted b_cant from the body axis
        # towards a body-fixed azimuth, and aimed through the CG so it makes no
        # moment. Its lateral component is a real horizontal input - and it rolls
        # with the airframe, so a spinning vehicle averages it away and a
        # roll-stabilised one does not.
        f_tx = thrust_main * tdx
        f_ty = thrust_main * tdy
        f_tz = thrust_main * tdz
        if thr_boost > 0.0:
            cc_b, ss_b = math.cos(b_cant), math.sin(b_cant)
            mx = math.cos(b_azim) * c1x + math.sin(b_azim) * c2x
            my = math.cos(b_azim) * c1y + math.sin(b_azim) * c2y
            mz = math.cos(b_azim) * c1z + math.sin(b_azim) * c2z
            f_tx += thr_boost * (cc_b * bx - ss_b * mx)
            f_ty += thr_boost * (cc_b * by - ss_b * my)
            f_tz += thr_boost * (cc_b * bz - ss_b * mz)
        # gimbal moment = (-L*b) x F: perpendicular to b, so no roll torque exists
        # the booster is aimed through the CG, so only the main motor's nozzle
        # produces a moment
        ttx = -L_GIMBAL * thrust_main * (by * sz - bz * sy)
        tty = -L_GIMBAL * thrust_main * (bz * sx - bx * sz)
        ttz = -L_GIMBAL * thrust_main * (bx * sy - by * sx)

        I_t = mass * (L_BODY * L_BODY / 12.0)
        I_a = mass * I_AXIAL_COEF
        ax = (f_tx + f_ax) / mass
        ay = (f_ty + f_ay) / mass
        az = (f_tz + f_az) / mass - G

        w_r = wx * bx + wy * by + wz * bz
        wtx, wty, wtz = wx - w_r * bx, wy - w_r * by, wz - w_r * bz
        gyk = (I_t - I_a) * w_r
        tot_x = ttx + tax - gyk * (by * wtz - bz * wty)
        tot_y = tty + tay - gyk * (bz * wtx - bx * wtz)
        tot_z = ttz + taz - gyk * (bx * wty - by * wtx)
        tr_r = tot_x * bx + tot_y * by + tot_z * bz
        alx = (tot_x - tr_r * bx) / I_t + tr_r * bx / I_a
        aly = (tot_y - tr_r * by) / I_t + tr_r * by / I_a
        alz = (tot_z - tr_r * bz) / I_t + tr_r * bz / I_a

        # ---------------- telemetry ----------------
        if n_tel > 0 and step % 20 == 0 and tel_i < n_tel:
            tel[tel_i, 0] = t
            tel[tel_i, 1] = h
            tel[tel_i, 2] = vz
            tel[tel_i, 3] = x
            tel[tel_i, 4] = vx
            tel[tel_i, 5] = math.degrees(math.acos(clampf(bz, -1.0, 1.0)))
            tel[tel_i, 6] = thrust
            tel[tel_i, 7] = k_act
            tel[tel_i, 8] = srv1
            tel[tel_i, 9] = srv2
            tel[tel_i, 10] = math.degrees(math.sqrt(wx * wx + wy * wy + wz * wz))
            tel[tel_i, 11] = thr_boost
            tel[tel_i, 12] = defl[0]
            tel[tel_i, 13] = defl[1]
            tel[tel_i, 14] = math.degrees(wx * bx + wy * by + wz * bz)
            tel_i += 1

        # ---------------- integrate ----------------
        nh = h + vz * DT_PHYS
        dt_s = DT_PHYS
        touchdown = nh <= 0.0
        if touchdown and vz < -1e-9:
            # sub-step exactly onto the ground plane. `frac` is a TIME, so it is not
            # multiplied by dt again - and it can be arbitrarily small, which is why
            # the loop below exits on the touchdown flag rather than on h <= 0: a
            # zero-length final step advances neither h nor t and hangs forever.
            dt_s = h / (-vz)
            if dt_s > DT_PHYS:
                dt_s = DT_PHYS
        vx += ax * dt_s
        vy += ay * dt_s
        vz += az * dt_s
        x += vx * dt_s
        y += vy * dt_s
        h += vz * dt_s
        wx += alx * dt_s
        wy += aly * dt_s
        wz += alz * dt_s
        # attitude: rotate both body-fixed vectors, then re-orthonormalise
        bx1 = bx + (wy * bz - wz * by) * dt_s
        by1 = by + (wz * bx - wx * bz) * dt_s
        bz1 = bz + (wx * by - wy * bx) * dt_s
        gx1 = gx + (wy * gz - wz * gy) * dt_s
        gy1 = gy + (wz * gx - wx * gz) * dt_s
        gz1 = gz + (wx * gy - wy * gx) * dt_s
        bn = math.sqrt(bx1 * bx1 + by1 * by1 + bz1 * bz1)
        bx, by, bz = bx1 / bn, by1 / bn, bz1 / bn
        d = gx1 * bx + gy1 * by + gz1 * bz
        gx1, gy1, gz1 = gx1 - d * bx, gy1 - d * by, gz1 - d * bz
        gn = math.sqrt(gx1 * gx1 + gy1 * gy1 + gz1 * gz1)
        gx, gy, gz = gx1 / gn, gy1 / gn, gz1 / gn

        t += dt_s
        step += 1
        if touchdown or h <= 0.0:
            if h > 0.0:
                h = 0.0
            break

    # ---------------- touchdown gates ----------------
    vh = math.sqrt(vx * vx + vy * vy)
    tilt = math.degrees(math.acos(clampf(bz, -1.0, 1.0)))
    # The rate gate is read on the TRANSVERSE rate only. Roll about the vehicle's own
    # axis is uncontrollable by design - there is no roll actuator - and it does not
    # tip the vehicle over on landing; gating on the total |w| would just fail a third
    # of the flights for a spin the airframe was handed at separation.
    w_r = wx * bx + wy * by + wz * bz
    wtx, wty, wtz = wx - w_r * bx, wy - w_r * by, wz - w_r * bz
    om = math.degrees(math.sqrt(wtx * wtx + wty * wty + wtz * wtz))
    ok = (abs(vz) < GATE_VZ and vh < GATE_VH and tilt < GATE_TILT
          and om < GATE_OMEGA and h < GATE_H)
    out[0] = 1.0 if ok else 0.0
    out[1] = abs(vz)
    out[2] = vh
    out[3] = tilt
    out[4] = om
    out[5] = h_cmd
    out[6] = h_ign_real
    out[7] = ign_delay
    out[8] = boost_used
    out[9] = b_t_ign if b_t_ign < 1.0e5 else -1.0
    out[10] = 1.0 if (t_burn0 >= 0.0 and t - t_burn0 > mt[-1]) else 0.0
    out[11] = dv_clamp
    out[12] = k_sum / k_n if k_n > 0.0 else 0.0
    out[13] = max_tilt
    out[14] = t
    out[15] = math.degrees(abs(w_r))
    out[16] = dv_tilt
    return out, tel_i


# ======================================================================
# --- CONFIGURATION / MONTE CARLO --------------------------------------
@dataclass
class TvcConfig:
    """Everything the 3-D campaign needs. Defaults are the test plan asked for."""
    motor: str = "long"
    gross_mass: float = landsim.GROSS_MASS
    propellant: float = landsim.PROPELLANT_MASS
    cd: float = CD_AXIAL
    thrust_mult: float = 1.0
    n_boosters: int = 1
    use_booster_rule: bool = True
    booster_cant: float = D9_CANT
    booster_azimuth: float = D9_AZIMUTH
    # entry grid
    h_lo: float = 140.0
    h_hi: float = 180.0
    h_step: float = 5.0
    vx_max: float = 7.0
    vx_step: float = 1.0
    vz0: float = 0.0
    runs: int = 40
    # dispersions
    ign_delay_max: float = IGN_DELAY_MAX
    delay_pad: float = IGN_DELAY_PLAN
    thrust_scatter: float = THRUST_SCATTER
    thrust_tau: float = THRUST_TAU
    roll_max: float = ROLL_RATE_MAX
    # fins
    fins: bool = True
    fin_drift_null: bool = False     # aerodynamic drift nulling before ignition.
                                     # Off by default - measured, it buys nothing on
                                     # this airframe; see the README.
    fin_count: int = FIN_COUNT
    fin_root: float = FIN_ROOT
    fin_tip: float = FIN_TIP
    fin_span: float = FIN_SPAN
    fin_arm: float = FIN_ARM
    fin_max_deflect: float = FIN_MAX_DEFLECT
    fin_travel_time: float = FIN_TRAVEL_TIME
    fin_brake: str = "auto"          # "auto" (unlit only) | "always" | "off"
    # controller
    wn: float = WN_DEFAULT                 # bandwidth with the motor lit (TVC)
    zeta: float = ZETA_DEFAULT
    wn_fin: float = WN_FIN_DEFAULT         # bandwidth on the fins alone
    zeta_fin: float = ZETA_FIN_DEFAULT
    roll_gain: float = FIN_ROLL_GAIN
    sched_tvc: float = 0.0                 # bandwidth ~ (thrust / 100 N) ** this
    sched_fin: float = 0.0                 # bandwidth ~ (q / 700 Pa) ** this
    gyro_ff: bool = True
    seed0: int = 20260830

    def entry_grid(self):
        h = np.arange(self.h_lo, self.h_hi + 1e-9, self.h_step)
        n = int(round(self.vx_max / self.vx_step))
        vx = np.concatenate((-np.arange(n, 0, -1) * self.vx_step,
                             [0.0], np.arange(1, n + 1) * self.vx_step))
        return h, vx

    def tables(self):
        m = Motor(self.motor, propellant_mass=self.propellant,
                  thrust_multiplier=self.thrust_mult)
        b = Booster()
        return m, b

    def gains(self):
        """The seven numbers the tuner searches, in one place."""
        return (self.wn, self.zeta, self.wn_fin, self.zeta_fin, self.roll_gain,
                self.sched_tvc, self.sched_fin)

    def with_gains(self, g) -> "TvcConfig":
        from dataclasses import replace
        return replace(self, wn=g[0], zeta=g[1], wn_fin=g[2], zeta_fin=g[3],
                       roll_gain=g[4], sched_tvc=g[5], sched_fin=g[6])

    def fin_set(self) -> Fins:
        return Fins(count=self.fin_count if self.fins else 0,
                    root=self.fin_root, tip=self.fin_tip, span=self.fin_span,
                    arm=self.fin_arm, max_deflect=self.fin_max_deflect,
                    travel_time=self.fin_travel_time, enabled=self.fins)

    def fin_args(self):
        """(n, area, arm, roll arm, CL_alpha, AR, max, rate, brake mode) for the
        compiled plant, plus the free-fall drag coefficient the planner should use."""
        f = self.fin_set()
        n = float(f.count) if f.enabled else 0.0
        mode = {"off": 0.0, "auto": 1.0, "always": 2.0}.get(self.fin_brake, 1.0)
        if not f.enabled:
            mode = 0.0
        cd_free = self.cd + (f.cd_extra(f.max_deflect)
                             if (f.enabled and mode > 0.5) else
                             (f.cd_extra(0.0) if f.enabled else 0.0))
        drift = 1.0 if (f.enabled and self.fin_drift_null) else 0.0
        return ((n, f.area, f.arm, f.roll_arm, f.cl_alpha, f.aspect,
                 f.max_deflect, f.rate, mode, drift), cd_free)


def plan_ignition(cfg: TvcConfig, h0: float, vx0: float) -> float:
    """The commanded ignition altitude for one entry state (seed independent)."""
    m, b = cfg.tables()
    mt, mf, mc = motor_arrays(m)
    bt, bf, bb = booster_arrays(b)
    mass0 = cfg.gross_mass + cfg.n_boosters * b.total_mass
    ready = 1.0 if (cfg.n_boosters > 0 and cfg.use_booster_rule) else 0.0
    _, cd_free = cfg.fin_args()
    h_cmd, _, _ = find_ignition(float(h0), float(vx0), float(cfg.vz0), cfg.cd,
                                cd_free, mass0, cfg.delay_pad, ready, mt, mf, mc,
                                m.propellant_mass, m.total_impulse,
                                bt, bf * math.cos(math.radians(cfg.booster_cant)),
                                bb, b.propellant_mass)
    return float(h_cmd)


def fly_one(cfg: TvcConfig, seed: int, h0: float, vx0: float, n_tel: int = 0,
            h_cmd: float = -1.0):
    """One flight. Returns (result vector, telemetry array)."""
    m, b = cfg.tables()
    mt, mf, mc = motor_arrays(m)
    bt, bf, bb = booster_arrays(b)
    tel = np.zeros((max(n_tel, 1), 15))
    fa, cd_free = cfg.fin_args()
    out, n = fly(seed, float(h0), float(vx0), float(cfg.vz0),
                 cfg.gross_mass, cfg.cd, cfg.wn, cfg.zeta,
                 cfg.wn_fin, cfg.zeta_fin, cfg.roll_gain,
                 cfg.sched_tvc, cfg.sched_fin,
                 float(cfg.n_boosters), 1.0 if cfg.use_booster_rule else 0.0,
                 cfg.roll_max, cfg.ign_delay_max, cfg.delay_pad,
                 cfg.thrust_scatter, cfg.thrust_tau,
                 1.0 if cfg.gyro_ff else 0.0,
                 fa[0], fa[1], fa[2], fa[3], fa[4], fa[5], fa[6], fa[7], fa[8],
                 fa[9], cd_free, math.radians(cfg.booster_cant),
                 math.radians(cfg.booster_azimuth),
                 mt, mf, mc, m.propellant_mass, m.total_impulse,
                 bt, bf, bb, b.propellant_mass, b.total_mass, tel, n_tel,
                 float(h_cmd))
    return out, tel[:n]


def ignition_grid(cfg: TvcConfig):
    """The commanded ignition altitude for every cell. It depends on the vehicle and
    the entry state but NOT on the controller gains, which is what makes the tuner
    affordable: solved once, reused by every candidate."""
    h_grid, vx_grid = cfg.entry_grid()
    return np.array([[plan_ignition(cfg, float(h0), float(vx0)) for vx0 in vx_grid]
                     for h0 in h_grid])


def run_campaign(cfg: TvcConfig, on_progress=None, should_stop=None,
                 h_cmd_grid=None):
    """Monte Carlo over the entry grid under COMMON RANDOM NUMBERS.

    Every cell flies the same list of seeds, so a difference between two
    configurations is a real difference and not sampling noise.
    """
    m, b = cfg.tables()
    mt, mf, mc = motor_arrays(m)
    bt, bf, bb = booster_arrays(b)
    h_grid, vx_grid = cfg.entry_grid()
    fa, cd_free = cfg.fin_args()
    seeds = cfg.seed0 + np.arange(cfg.runs)
    res = np.zeros((len(h_grid), len(vx_grid), cfg.runs, N_OUT))
    tel = np.zeros((1, 15))
    total = len(h_grid) * len(vx_grid)
    done = 0
    for i, h0 in enumerate(h_grid):
        for j, vx0 in enumerate(vx_grid):
            h_cmd = (float(h_cmd_grid[i, j]) if h_cmd_grid is not None
                     else plan_ignition(cfg, float(h0), float(vx0)))
            for r, sd in enumerate(seeds):
                out, _ = fly(int(sd), float(h0), float(vx0), float(cfg.vz0),
                             cfg.gross_mass, cfg.cd, cfg.wn, cfg.zeta,
                             cfg.wn_fin, cfg.zeta_fin, cfg.roll_gain,
                             cfg.sched_tvc, cfg.sched_fin,
                             float(cfg.n_boosters),
                             1.0 if cfg.use_booster_rule else 0.0,
                             cfg.roll_max, cfg.ign_delay_max, cfg.delay_pad,
                             cfg.thrust_scatter, cfg.thrust_tau,
                             1.0 if cfg.gyro_ff else 0.0,
                             fa[0], fa[1], fa[2], fa[3], fa[4], fa[5], fa[6],
                             fa[7], fa[8], fa[9], cd_free,
                             math.radians(cfg.booster_cant),
                             math.radians(cfg.booster_azimuth),
                             mt, mf, mc, m.propellant_mass, m.total_impulse,
                             bt, bf, bb, b.propellant_mass, b.total_mass, tel, 0,
                             h_cmd)
                res[i, j, r] = out
            done += 1
            if on_progress is not None and (done % 5 == 0 or done == total):
                sr = res[i, j, :, 0].mean() * 100.0
                on_progress(f"  {done:4d}/{total}  h={h0:5.1f} m  vx={vx0:+.0f} m/s"
                            f"   success {sr:5.1f} %")
            if should_stop is not None and should_stop():
                raise landsim.Cancelled()
    return {"h_grid": h_grid, "vx_grid": vx_grid, "seeds": seeds, "out": res,
            "cfg": cfg}


def summarise(camp):
    """Success rates and the p95 metrics, in the shape the report and the plots want."""
    out = camp["out"]
    flat = out.reshape(-1, N_OUT)
    succ = flat[:, 0]
    grid = out[:, :, :, 0].mean(axis=2) * 100.0
    return {
        "success": succ.mean() * 100.0,
        "grid": grid,
        "by_h": out[:, :, :, 0].mean(axis=(1, 2)) * 100.0,
        "by_vx": out[:, :, :, 0].mean(axis=(0, 2)) * 100.0,
        "gate_vz": (flat[:, 1] < GATE_VZ).mean() * 100.0,
        "gate_vh": (flat[:, 2] < GATE_VH).mean() * 100.0,
        "gate_tilt": (flat[:, 3] < GATE_TILT).mean() * 100.0,
        "gate_om": (flat[:, 4] < GATE_OMEGA).mean() * 100.0,
        "p95_vz": float(np.percentile(flat[:, 1], 95)),
        "p95_vh": float(np.percentile(flat[:, 2], 95)),
        "p95_tilt": float(np.percentile(flat[:, 3], 95)),
        "p95_om": float(np.percentile(flat[:, 4], 95)),
        "boost_rate": flat[:, 8].mean() * 100.0,
        "boost_grid": out[:, :, :, 8].mean(axis=2) * 100.0,
        "mean_h_cmd": float(flat[:, 5].mean()),
        "mean_k": float(flat[:, 12].mean()),
        "dv_tilt": float(flat[:, 16].mean()),
        "dv_clamp": float(flat[:, 11].mean()),
        "burnout_rate": flat[:, 10].mean() * 100.0,
        "flat": flat,
    }


# ======================================================================
# --- GAIN TUNING ------------------------------------------------------
# The controller has seven numbers in it that no physics fixes: two bandwidths, two
# damping ratios, the roll-damper gain and the two schedule exponents. Guessing them
# is how a landing simulation ends up measuring the guess instead of the vehicle, so
# they are FITTED here, on the ground, against a small campaign - and the fit is what
# the main run then flies.
#
# Three things make it honest and affordable:
#   * COMMON RANDOM NUMBERS. Every candidate flies the identical seed list over the
#     identical entry states, so a difference between two gain sets is a difference
#     between the gain sets.
#   * The ignition altitudes are solved ONCE. They depend on the vehicle and the
#     entry state, never on the gains, and they cost more than the flights do.
#   * The score is not raw success. Success on this vehicle is dominated by the
#     propulsive |vz| gate, which the gains barely touch, so a success-only score is
#     nearly flat and the search wanders. The margin penalty below keeps the gradient
#     where the gains actually act - attitude, rate and horizontal speed.
# ======================================================================
GAIN_NAMES = ("wn", "zeta", "wn_fin", "zeta_fin", "roll_gain",
              "sched_tvc", "sched_fin")
GAIN_BOUNDS = np.array([
    (4.0, 16.0),      # wn        - TVC bandwidth [rad/s]
    (0.6, 1.6),       # zeta      - TVC damping
    (2.0, 12.0),      # wn_fin    - fin bandwidth [rad/s]
    (0.6, 1.6),       # zeta_fin  - fin damping
    (0.3, 6.0),       # roll_gain [rad/s]
    (-0.6, 0.6),      # sched_tvc - bandwidth ~ (T/100 N) ** this
    (-0.6, 0.6),      # sched_fin - bandwidth ~ (q/700 Pa) ** this
])


def gain_cost(camp):
    """Lower is better. Miss fraction plus a bounded margin penalty."""
    flat = summarise(camp)["flat"]
    miss = 1.0 - flat[:, 0].mean()
    pen = (np.minimum(flat[:, 1] / GATE_VZ, 3.0) ** 2 * 0.5
           + np.minimum(flat[:, 2] / GATE_VH, 3.0) ** 2
           + np.minimum(flat[:, 3] / GATE_TILT, 3.0) ** 2
           + np.minimum(flat[:, 4] / GATE_OMEGA, 3.0) ** 2)
    return float(miss + 0.06 * pen.mean()), float(1.0 - miss)


def tune_gains(cfg: TvcConfig, budget=60, runs=12, cells=3, seed=1,
               on_progress=None, should_stop=None):
    """Fit the seven controller gains on a reduced campaign.

    Returns (best gains tuple, report dict). `budget` is the number of candidate gain
    sets flown; each one is a `cells x cells x runs` campaign.
    """
    from dataclasses import replace
    rng = np.random.default_rng(seed)
    # reduced but representative grid: the corners and the middle of the envelope
    step_h = max(1.0, (cfg.h_hi - cfg.h_lo) / max(cells - 1, 1))
    small = replace(cfg, h_step=step_h, vx_step=max(cfg.vx_max / ((cells - 1) / 2.0
                                                                 if cells > 1 else 1),
                                                    0.5),
                    runs=runs)
    h_cmd_grid = ignition_grid(small)
    n_cells = h_cmd_grid.size

    def report(msg):
        if on_progress is not None:
            on_progress(msg)

    def evaluate(g):
        if should_stop is not None and should_stop():
            raise landsim.Cancelled()
        camp = run_campaign(small.with_gains(tuple(g)), h_cmd_grid=h_cmd_grid)
        return gain_cost(camp)

    report(f"tuning {len(GAIN_NAMES)} gains on {n_cells} entry states x {runs} runs "
           f"= {n_cells * runs} flights per candidate, {budget} candidates")

    lo, hi = GAIN_BOUNDS[:, 0], GAIN_BOUNDS[:, 1]
    pop_size = min(12, max(6, budget // 5))
    pop = rng.uniform(lo, hi, size=(pop_size, len(lo)))
    pop[0] = np.array(cfg.gains())          # the shipped default is candidate zero
    cost = np.empty(pop_size)
    succ = np.empty(pop_size)
    for i in range(pop_size):
        cost[i], succ[i] = evaluate(pop[i])
        report(f"  seed candidate {i + 1}/{pop_size}: cost {cost[i]:.4f}  "
               f"success {succ[i] * 100:5.1f} %")
    used = pop_size

    # differential evolution, the same one the 1-D optimiser uses
    while used < budget:
        for i in range(pop_size):
            if used >= budget:
                break
            a, b, c = pop[rng.choice(pop_size, 3, replace=False)]
            f = rng.uniform(0.4, 0.9)
            trial = np.clip(a + f * (b - c), lo, hi)
            mask = rng.random(len(lo)) < 0.8
            mask[rng.integers(len(lo))] = True
            trial = np.where(mask, trial, pop[i])
            t_cost, t_succ = evaluate(trial)
            used += 1
            if t_cost < cost[i]:
                pop[i], cost[i], succ[i] = trial, t_cost, t_succ
                report(f"  {used:3d}/{budget}  improved: cost {t_cost:.4f}  "
                       f"success {t_succ * 100:5.1f} %  "
                       + "  ".join(f"{n}={v:.2f}" for n, v in
                                   zip(GAIN_NAMES, trial)))
    best = int(np.argmin(cost))
    g = tuple(float(x) for x in pop[best])
    report("  best: " + "  ".join(f"{n}={v:.3f}" for n, v in zip(GAIN_NAMES, g)))
    return g, {"cost": float(cost[best]), "success": float(succ[best]),
               "budget": budget, "runs": runs, "cells": n_cells}


def save_gains(gains, report, path="tvc_gains.json"):
    import json
    with open(path, "w") as fh:
        json.dump({"gains": dict(zip(GAIN_NAMES, gains)), "report": report}, fh,
                  indent=2)
    return path


def load_gains(path="tvc_gains.json"):
    import json
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    return tuple(float(d["gains"][n]) for n in GAIN_NAMES)


# ======================================================================
# --- FIGURES ----------------------------------------------------------
# One visual system across all five: one sequential blue ramp for magnitude,
# two status colours for pass/fail, recessive grid, direct labels, and never two
# y-scales on one pair of axes.
# ======================================================================
INK = "#0b0b0b"
INK2 = "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GOOD, BAD = "#1baf7a", "#e34948"


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color="#e7e6e1", linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def make_figures(camp, outdir="figures", single=None):
    """Write the campaign's figures. Returns the list of paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(outdir, exist_ok=True)
    s = summarise(camp)
    cfg = camp["cfg"]
    h_grid, vx_grid = camp["h_grid"], camp["vx_grid"]
    flat = s["flat"]
    paths = []
    blues = LinearSegmentedColormap.from_list(
        "blues", ["#f2f6fb", "#cddff4", "#8fb9e8", "#4a90d9", "#2a78d6", "#17457c"])

    # --- 1. success envelope ------------------------------------------
    fig, ax = plt.subplots(figsize=(1.1 * len(vx_grid) + 3.2,
                                    0.52 * len(h_grid) + 2.6), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(s["grid"], origin="lower", cmap=blues, vmin=0, vmax=100,
                   aspect="auto")
    ax.set_xticks(range(len(vx_grid)))
    ax.set_xticklabels([f"{v:+.0f}" for v in vx_grid])
    ax.set_yticks(range(len(h_grid)))
    ax.set_yticklabels([f"{h:.0f}" for h in h_grid])
    for i in range(len(h_grid)):
        for j in range(len(vx_grid)):
            v = s["grid"][i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 55 else INK)
    _style(ax, f"Landing success across the entry envelope  -  "
               f"{cfg.runs} flights per cell, {int(s['success'] * len(flat) / 100)}"
               f"/{len(flat)} overall ({s['success']:.1f} %)",
           "horizontal entry speed vx [m/s]", "release altitude [m]")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("flights inside every gate [%]", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_success_envelope.png")
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths.append(p)

    # --- 2. touchdown dispersion --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ok = flat[:, 0] > 0.5
    ax = axes[0]
    ax.scatter(flat[~ok, 2], flat[~ok, 1], s=14, c=BAD, alpha=0.55,
               linewidths=0, label="failed")
    ax.scatter(flat[ok, 2], flat[ok, 1], s=14, c=GOOD, alpha=0.75,
               linewidths=0, label="landed")
    ax.axvline(GATE_VH, color=INK2, lw=1.2, ls="--")
    ax.axhline(GATE_VZ, color=INK2, lw=1.2, ls="--")
    ax.text(GATE_VH, ax.get_ylim()[1] * 0.97, f" |v_h| gate {GATE_VH} m/s",
            fontsize=8, color=INK2, va="top")
    ax.text(ax.get_xlim()[1] * 0.98, GATE_VZ, f"|v_z| gate {GATE_VZ} m/s ",
            fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_yscale("symlog", linthresh=5)
    _style(ax, "Where the vehicle actually arrives",
           "horizontal speed at touchdown [m/s]", "vertical speed [m/s]")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    ax = axes[1]
    ax.scatter(flat[~ok, 3], flat[~ok, 4], s=14, c=BAD, alpha=0.55, linewidths=0)
    ax.scatter(flat[ok, 3], flat[ok, 4], s=14, c=GOOD, alpha=0.75, linewidths=0)
    ax.axvline(GATE_TILT, color=INK2, lw=1.2, ls="--")
    ax.axhline(GATE_OMEGA, color=INK2, lw=1.2, ls="--")
    _style(ax, "Attitude at touchdown  (dashed = gates)",
           "tilt from vertical [deg]", "transverse rate [deg/s]")
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_touchdown_dispersion.png")
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths.append(p)

    # --- 3. a single flight -------------------------------------------
    if single is None:
        mid_h = float(h_grid[len(h_grid) // 2])
        single = fly_one(cfg, int(camp["seeds"][0]), mid_h,
                         float(vx_grid[-1]), n_tel=4000)
    out, tel = single
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 10.0), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    t = tel[:, 0]
    ax = axes[0][0]
    ax.plot(t, tel[:, 1], color=SERIES[0], lw=2)
    ax.axhline(0, color=INK2, lw=0.8)
    burn = tel[tel[:, 6] > 0.5]
    if len(burn):
        ax.axvspan(burn[0, 0], burn[-1, 0], color="#2a78d6", alpha=0.07)
        ax.text(burn[0, 0], ax.get_ylim()[1] * 0.9, " burn", fontsize=8,
                color=INK2)
    _style(ax, "Altitude", "time [s]", "h [m]")

    ax = axes[0][1]
    ax.plot(t, tel[:, 2], color=SERIES[0], lw=2, label="vertical")
    ax.plot(t, tel[:, 4], color=SERIES[1], lw=2, label="horizontal")
    ax.axhline(0, color=INK2, lw=0.8)
    _style(ax, f"Velocity  (touchdown {out[1]:.2f} m/s down, {out[2]:.2f} m/s across)",
           "time [s]", "v [m/s]")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[1][0]
    ax.plot(t, tel[:, 6], color=SERIES[0], lw=2, label="total thrust")
    ax.plot(t, tel[:, 11], color=SERIES[2], lw=1.6, label="D9 booster")
    ax.plot(t, tel[:, 7] * 100.0, color=SERIES[1], lw=1.6, label="clamp [%]")
    _style(ax, "Thrust and clamp", "time [s]", "N   /   clamp %")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[1][1]
    ax.plot(t, tel[:, 5], color=SERIES[0], lw=2, label="tilt [deg]")
    ax.plot(t, tel[:, 8], color=SERIES[1], lw=1.2, label="servo 1 [deg]")
    ax.plot(t, tel[:, 9], color=SERIES[2], lw=1.2, label="servo 2 [deg]")
    _style(ax, "Attitude and both gimbal channels", "time [s]", "deg")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[2][0]
    ax.plot(t, tel[:, 12], color=SERIES[0], lw=1.6, label="fin 1")
    ax.plot(t, tel[:, 13], color=SERIES[1], lw=1.6, label="fin 2")
    lim = cfg.fin_max_deflect
    ax.axhline(lim, color=INK2, lw=0.8, ls="--")
    ax.axhline(-lim, color=INK2, lw=0.8, ls="--")
    ax.text(t[0], lim, " deflection limit", fontsize=8, color=INK2, va="bottom")
    _style(ax, "Fin deflection  (opposed = airbrake, differential = steering)",
           "time [s]", "deg")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[2][1]
    ax.plot(t, tel[:, 14], color=SERIES[2], lw=2)
    ax.axhline(0, color=INK2, lw=0.8)
    _style(ax, f"Roll rate - the fins null the spin the vehicle arrived with "
               f"({out[15]:.1f} deg/s at touchdown)", "time [s]", "deg/s")
    fig.suptitle(f"One flight - release {single[1][0, 1]:.0f} m, "
                 f"ignition commanded at {out[5]:.1f} m, lit at {out[6]:.1f} m "
                 f"after {out[7] * 1000:.0f} ms, "
                 f"D9 {'used' if out[8] > 0.5 else 'not used'}",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(outdir, "fig3_single_flight.png")
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths.append(p)

    # --- 4. ignition window -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axes[0]
    out4 = camp["out"]
    for i, h0 in enumerate(h_grid):
        cell = out4[i].reshape(-1, N_OUT)
        okc = cell[:, 0] > 0.5
        ax.scatter(np.full(okc.sum(), h0), cell[okc, 6], s=12, c=GOOD,
                   alpha=0.6, linewidths=0)
        ax.scatter(np.full((~okc).sum(), h0), cell[~okc, 6], s=12, c=BAD,
                   alpha=0.5, linewidths=0)
        ax.scatter([h0], [cell[0, 5]], s=30, c=INK, marker="_")
    ax.plot([], [], marker="_", color=INK, lw=0, label="commanded altitude")
    ax.plot([], [], marker="o", color=GOOD, lw=0, label="lit - landed")
    ax.plot([], [], marker="o", color=BAD, lw=0, label="lit - failed")
    _style(ax, "Where the motor actually lit  (spread = igniter delay)",
           "release altitude [m]", "ignition altitude [m]")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    ax = axes[1]
    xs = np.arange(len(vx_grid))
    ax.bar(xs, s["boost_grid"].mean(axis=0), color=SERIES[2], width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:+.0f}" for v in vx_grid])
    for x, v in zip(xs, s["boost_grid"].mean(axis=0)):
        ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, 108)
    _style(ax, "How often the on-board rule lit the D9",
           "horizontal entry speed vx [m/s]", "flights using the booster [%]")
    fig.tight_layout()
    p = os.path.join(outdir, "fig4_ignition_and_booster.png")
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths.append(p)

    # --- 5. gates ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=140)
    fig.patch.set_facecolor("#fcfcfb")
    ax = axes[0]
    names = ["|v_z| < 3", "|v_h| < 0.75", "tilt < 10", "rate < 60", "ALL"]
    vals = [s["gate_vz"], s["gate_vh"], s["gate_tilt"], s["gate_om"],
            s["success"]]
    cols = [SERIES[0]] * 4 + [SERIES[1]]
    ax.barh(range(len(names)), vals, color=cols, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f} %", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 112)
    ax.invert_yaxis()
    _style(ax, "Which gate is actually binding", "flights passing [%]", None)

    ax = axes[1]
    ax.hist(np.clip(flat[:, 1], 0, 25), bins=40, color=SERIES[0])
    ax.axvline(GATE_VZ, color=BAD, lw=1.5, ls="--")
    ax.text(GATE_VZ + 0.3, ax.get_ylim()[1] * 0.9, f"gate {GATE_VZ} m/s",
            fontsize=8, color=INK2)
    _style(ax, f"Vertical touchdown speed  (p95 = {s['p95_vz']:.1f} m/s)",
           "|v_z| at touchdown [m/s]", "flights")
    fig.tight_layout()
    p = os.path.join(outdir, "fig5_gates.png")
    fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)
    paths.append(p)
    return paths


# ======================================================================
# --- REPORT / CLI -----------------------------------------------------
def save_campaign(camp, path):
    """Store the raw per-flight results so the figures can be redrawn without
    re-flying 5400 trajectories."""
    cfg = camp["cfg"]
    np.savez_compressed(path, out=camp["out"], h_grid=camp["h_grid"],
                        vx_grid=camp["vx_grid"], seeds=camp["seeds"],
                        cfg=np.array([repr(cfg)], dtype=object))
    return path


def load_campaign(path, cfg: "TvcConfig" = None):
    d = np.load(path, allow_pickle=True)
    if cfg is None:
        cfg = eval(str(d["cfg"][0]),                      # noqa: S307 - our own repr
                   {"TvcConfig": TvcConfig, "Motor": Motor, "Booster": Booster,
                    "Fins": Fins})
    return {"out": d["out"], "h_grid": d["h_grid"], "vx_grid": d["vx_grid"],
            "seeds": d["seeds"], "cfg": cfg}


def print_report(camp):
    s = summarise(camp)
    cfg = camp["cfg"]
    m, b = cfg.tables()
    print("=" * 78)
    print("LandSim Light - 3-D / TVC Monte Carlo")
    print("=" * 78)
    print(f"  motor             : {m.name}, {m.peak_thrust:.1f} N peak, "
          f"{m.burn_time:.2f} s, {m.total_impulse:.1f} Ns")
    print(f"  boosters carried  : {cfg.n_boosters} x {b.name} "
          f"({b.total_impulse:.1f} Ns each), lit by the on-board rule")
    if cfg.n_boosters:
        print(f"  booster mounting  : canted {cfg.booster_cant:.0f} deg through the "
              f"CG at azimuth {cfg.booster_azimuth:.0f} deg -> "
              f"{math.cos(math.radians(cfg.booster_cant)) * 100:.1f} % of its thrust "
              f"axial, {math.sin(math.radians(cfg.booster_cant)) * 100:.1f} % sideways")
    print(f"  mass / Cd         : {cfg.gross_mass:.3f} kg + boosters / {cfg.cd}")
    f = cfg.fin_set()
    if f.enabled:
        print(f"  fins              : {f.describe()}")
        print(f"  airbrake mode     : {cfg.fin_brake}  "
              f"(free-fall Cd {cfg.fin_args()[1]:.3f} against {cfg.cd:.3f} bare)")
    else:
        print("  fins              : none")
    print(f"  entry grid        : {cfg.h_lo:.0f}-{cfg.h_hi:.0f} m step "
          f"{cfg.h_step:.0f}, vx 0 +/-{cfg.vx_max:.0f} m/s step {cfg.vx_step:.0f}")
    print(f"  gains             : wn {cfg.wn:.2f} / zeta {cfg.zeta:.2f} (TVC, "
          f"schedule {cfg.sched_tvc:+.2f}), wn {cfg.wn_fin:.2f} / zeta "
          f"{cfg.zeta_fin:.2f} (fins, schedule {cfg.sched_fin:+.2f}), "
          f"roll {cfg.roll_gain:.2f}")
    print(f"  dispersions       : igniter U(0, {cfg.ign_delay_max * 1000:.0f}) ms, "
          f"thrust +/-{cfg.thrust_scatter * 100:.0f} % over {cfg.thrust_tau * 1000:.0f} ms, "
          f"roll U(0, {cfg.roll_max:.0f}) deg/s")
    print(f"  flights           : {camp['out'].shape[0] * camp['out'].shape[1]} cells"
          f" x {cfg.runs} = {camp['out'][:, :, :, 0].size}")
    print()
    print(f"  SUCCESS (all five gates)   : {s['success']:5.1f} %")
    print(f"    |v_z| < {GATE_VZ} m/s              : {s['gate_vz']:5.1f} %"
          f"   p95 {s['p95_vz']:6.2f} m/s")
    print(f"    |v_h| < {GATE_VH} m/s           : {s['gate_vh']:5.1f} %"
          f"   p95 {s['p95_vh']:6.2f} m/s")
    print(f"    tilt  < {GATE_TILT} deg             : {s['gate_tilt']:5.1f} %"
          f"   p95 {s['p95_tilt']:6.2f} deg")
    print(f"    rate  < {GATE_OMEGA} deg/s           : {s['gate_om']:5.1f} %"
          f"   p95 {s['p95_om']:6.2f} deg/s")
    print(f"  D9 lit in                  : {s['boost_rate']:5.1f} % of flights")
    print(f"  burnout before touchdown   : {s['burnout_rate']:5.1f} %")
    print(f"  mean commanded ignition    : {s['mean_h_cmd']:5.1f} m, "
          f"mean clamp {s['mean_k']:.2f}")
    print(f"  dV spent on steering       : {s['dv_tilt']:5.2f} m/s "
          f"(clamp waste {s['dv_clamp']:5.1f} m/s)")
    print()
    print("  success [%] by release altitude:")
    for h, v in zip(camp["h_grid"], s["by_h"]):
        print(f"    {h:6.1f} m : {v:5.1f}")
    print("  success [%] by horizontal entry speed:")
    for vx, v in zip(camp["vx_grid"], s["by_vx"]):
        print(f"    {vx:+5.1f} m/s : {v:5.1f}")


def verify():
    """Cheap invariants of the 3-D model - the things that are easy to get wrong
    and impossible to see in an aggregate success rate."""
    rng = np.random.default_rng(7)
    worst_basis = 0.0
    worst_inv = 0.0
    for _ in range(2000):
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        g = rng.normal(size=3)
        u1 = np.array(gimbal_basis(b[0], b[1], b[2], g[0], g[1], g[2])[:3])
        u2 = np.array(gimbal_basis(b[0], b[1], b[2], g[0], g[1], g[2])[3:])
        worst_basis = max(worst_basis, abs(u1 @ b), abs(u2 @ b), abs(u1 @ u2),
                          abs(np.linalg.norm(u1) - 1.0),
                          abs(np.linalg.norm(u2) - 1.0))
        # dynamic inversion round trip: tau -> s -> tau
        tau = rng.normal(size=3)
        tau_perp = tau - (tau @ b) * b
        T_, L_ = 87.3, L_GIMBAL
        sv = np.cross(b, tau_perp) / (T_ * L_)
        back = -L_ * T_ * np.cross(b, sv)
        worst_inv = max(worst_inv, float(np.max(np.abs(back - tau_perp))))
    # roll rate is untouched by any modelled torque
    cfg = TvcConfig(roll_max=90.0, fins=False)
    worst_roll = 0.0
    for sd in range(11, 19):
        out, tel = fly_one(cfg, sd, 150.0, 7.0, n_tel=4000)
        worst_roll = max(worst_roll, abs(tel[0, 10] - out[15]))
    # with fins the roll IS controllable, and that is the point of them
    cfg_f = TvcConfig(roll_max=90.0)
    rolls_in, rolls_out = [], []
    for sd in range(11, 19):
        out, tel = fly_one(cfg_f, sd, 150.0, 7.0, n_tel=4000)
        rolls_in.append(tel[0, 10])
        rolls_out.append(out[15])
    f = cfg_f.fin_set()
    print("verification")
    print(f"  gimbal basis orthonormal to        : {worst_basis:.2e}")
    print(f"  dynamic inversion round trip       : {worst_inv:.2e}")
    print(f"  roll drift, fins OFF, 8 flights    : {worst_roll:.2e} deg/s "
          f"(the gimbal moment is perpendicular to the body axis by construction, "
          f"so without fins the inherited spin must survive untouched)")
    print(f"  roll, fins ON  : entry {np.mean(rolls_in):5.1f} deg/s "
          f"-> touchdown {np.mean(rolls_out):4.1f} deg/s (mean of 8)")
    print(f"  fin airbrake   : free-fall Cd {cfg_f.fin_args()[1]:.3f} against "
          f"{cfg_f.cd:.3f} bare, i.e. "
          f"{cfg_f.fin_args()[1] / cfg_f.cd:.2f}x the drag")
    ok = (worst_basis < 1e-9 and worst_inv < 1e-9 and worst_roll < 0.5
          and np.mean(rolls_out) < 5.0)
    print(f"  -> {'OK' if ok else 'FAILED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="LandSim Light - 3-D / TVC campaign")
    ap.add_argument("--runs", type=int, default=40, help="flights per grid cell")
    ap.add_argument("--h-lo", type=float, default=140.0)
    ap.add_argument("--h-hi", type=float, default=180.0)
    ap.add_argument("--h-step", type=float, default=5.0)
    ap.add_argument("--vx-max", type=float, default=7.0)
    ap.add_argument("--vx-step", type=float, default=1.0)
    ap.add_argument("--motor", default="long", choices=sorted(landsim.MOTOR_TABLES))
    ap.add_argument("--mass", type=float, default=landsim.GROSS_MASS)
    ap.add_argument("--cd", type=float, default=CD_AXIAL)
    ap.add_argument("--thrust-mult", type=float, default=1.0)
    ap.add_argument("--boosters", type=int, default=1)
    ap.add_argument("--ign-delay", type=float, default=IGN_DELAY_MAX,
                    help="igniter delay is U(0, this) [s]")
    ap.add_argument("--thrust-scatter", type=float, default=THRUST_SCATTER)
    ap.add_argument("--thrust-tau", type=float, default=THRUST_TAU)
    ap.add_argument("--roll-max", type=float, default=ROLL_RATE_MAX)
    ap.add_argument("--no-fins", action="store_true",
                    help="fly without fin control (gimbal only, roll uncontrolled)")
    ap.add_argument("--fin-count", type=int, default=FIN_COUNT)
    ap.add_argument("--fin-root", type=float, default=FIN_ROOT * 1000, help="mm")
    ap.add_argument("--fin-tip", type=float, default=FIN_TIP * 1000, help="mm")
    ap.add_argument("--fin-span", type=float, default=FIN_SPAN * 1000, help="mm")
    ap.add_argument("--fin-arm", type=float, default=FIN_ARM, help="m from the CG")
    ap.add_argument("--fin-deflect", type=float, default=FIN_MAX_DEFLECT,
                    help="deg, +/-")
    ap.add_argument("--fin-travel", type=float, default=FIN_TRAVEL_TIME,
                    help="s, end stop to end stop")
    ap.add_argument("--fin-brake", choices=("auto", "always", "off"), default="auto",
                    help="when the fins are splayed as airbrakes")
    ap.add_argument("--fin-drift-null", action="store_true",
                    help="try to kill the sideways drift aerodynamically before "
                         "ignition (measured: does not pay on this airframe)")
    ap.add_argument("--booster-cant", type=float, default=D9_CANT,
                    help="deg the D9 is canted from the body axis (it is aimed "
                         "through the CG, so it makes no moment)")
    ap.add_argument("--booster-azimuth", type=float, default=D9_AZIMUTH,
                    help="deg, body-fixed azimuth of the D9 mounting")
    ap.add_argument("--wn", type=float, default=WN_DEFAULT,
                    help="TVC attitude bandwidth [rad/s]")
    ap.add_argument("--zeta", type=float, default=ZETA_DEFAULT)
    ap.add_argument("--wn-fin", type=float, default=WN_FIN_DEFAULT,
                    help="attitude bandwidth on the fins alone [rad/s]")
    ap.add_argument("--zeta-fin", type=float, default=ZETA_FIN_DEFAULT)
    ap.add_argument("--roll-gain", type=float, default=FIN_ROLL_GAIN)
    ap.add_argument("--sched-tvc", type=float, default=0.0,
                    help="TVC bandwidth ~ (real thrust / 100 N) ** this")
    ap.add_argument("--sched-fin", type=float, default=0.0,
                    help="fin bandwidth ~ (dynamic pressure / 700 Pa) ** this")
    ap.add_argument("--tune", action="store_true",
                    help="fit the seven controller gains first, then fly the campaign "
                         "with them")
    ap.add_argument("--tune-budget", type=int, default=60,
                    help="candidate gain sets to fly while tuning")
    ap.add_argument("--tune-runs", type=int, default=12,
                    help="flights per entry state while tuning")
    ap.add_argument("--gains", default="tvc_gains.json",
                    help="gain file to load if it exists (and to write with --tune)")
    ap.add_argument("--no-gains", action="store_true",
                    help="ignore the gain file and use the built-in defaults")
    ap.add_argument("--no-gyro-ff", action="store_true")
    ap.add_argument("--figures", default="figures", help="output directory")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--save", default="", help="write the raw results to an .npz")
    ap.add_argument("--load", default="",
                    help="redraw the report and figures from a saved .npz")
    ap.add_argument("--verify", action="store_true",
                    help="run the model's invariant checks and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    cfg = TvcConfig(motor=args.motor, gross_mass=args.mass, cd=args.cd,
                    thrust_mult=args.thrust_mult, n_boosters=max(0, args.boosters),
                    h_lo=args.h_lo, h_hi=args.h_hi, h_step=args.h_step,
                    vx_max=args.vx_max, vx_step=args.vx_step, runs=args.runs,
                    ign_delay_max=args.ign_delay, delay_pad=args.ign_delay,
                    thrust_scatter=args.thrust_scatter, thrust_tau=args.thrust_tau,
                    roll_max=args.roll_max, wn=args.wn, zeta=args.zeta,
                    wn_fin=args.wn_fin, zeta_fin=args.zeta_fin,
                    roll_gain=args.roll_gain, sched_tvc=args.sched_tvc,
                    sched_fin=args.sched_fin,
                    fins=not args.no_fins, fin_count=args.fin_count,
                    fin_root=args.fin_root / 1000.0, fin_tip=args.fin_tip / 1000.0,
                    fin_span=args.fin_span / 1000.0, fin_arm=args.fin_arm,
                    fin_max_deflect=args.fin_deflect,
                    fin_travel_time=args.fin_travel, fin_brake=args.fin_brake,
                    fin_drift_null=args.fin_drift_null,
                    booster_cant=args.booster_cant,
                    booster_azimuth=args.booster_azimuth,
                    gyro_ff=not args.no_gyro_ff)
    if args.tune:
        g, rep = tune_gains(cfg, budget=args.tune_budget, runs=args.tune_runs,
                            on_progress=None if args.quiet else print)
        cfg = cfg.with_gains(g)
        save_gains(g, rep, args.gains)
        print(f"  gains written to {args.gains}\n")
    elif not args.no_gains:
        g = load_gains(args.gains)
        if g is not None:
            cfg = cfg.with_gains(g)
            print(f"  gains loaded from {args.gains}\n")

    if args.load:
        camp = load_campaign(args.load)
    else:
        camp = run_campaign(cfg, on_progress=None if args.quiet else print)
    if args.save:
        save_campaign(camp, args.save)
    print_report(camp)
    if not args.no_figures:
        paths = make_figures(camp, args.figures)
        print("\n  figures written:")
        for p in paths:
            print(f"    {p}")


if __name__ == "__main__":
    main()
