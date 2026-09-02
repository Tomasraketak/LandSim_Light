"""
Read a vehicle out of an OpenRocket file.

An `.ork` is a zip archive whose only interesting member is `rocket.ork`, an XML
description of the component tree. This module walks that tree and pulls out the
things the two simulations need:

    * the airframe: total length and the largest body diameter,
    * the fins: count, root and tip chord, span and sweep, from the aft-most fin set,
    * a mass estimate, component by component,
    * the motors that are mounted, and where.

**About mass.** OpenRocket does not store the masses it computes - it recomputes them
from geometry and material density every time you open the file. So this module does
the same arithmetic: shells for tubes and nose cones, plates for fins, and the stated
mass for mass components and overrides. That is an estimate, and it is shown broken
down by component so you can see what it is made of and correct it. Any component with
its mass overridden in OpenRocket is taken at face value - if you want the import to
be exact, override the masses there.

**About the spent motor.** The vehicle in this simulation is landing, so the motor it
launched on has already burned: its propellant is gone and its casing is not. Pass
`spent_propellant` (kg) and it is subtracted from the imported mass. The parser lists
the motors it found so you know what to subtract.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field


@dataclass
class OrkComponent:
    kind: str
    name: str
    mass: float                 # kg
    source: str                 # "override", "mass", or "estimated from geometry"
    length: float = 0.0         # m, axial
    radius: float = 0.0         # m


@dataclass
class OrkRocket:
    name: str = ""
    length: float = 0.0                 # m, sum of the axial chain
    diameter: float = 0.0               # m, largest body tube
    mass: float = 0.0                   # kg, everything found
    fin_count: int = 0
    fin_root: float = 0.0               # m
    fin_tip: float = 0.0
    fin_span: float = 0.0
    fin_sweep: float = 0.0
    fin_position: float = 0.0           # m from the nose, leading edge of the root
    motors: list = field(default_factory=list)      # (designation, manufacturer)
    components: list = field(default_factory=list)  # OrkComponent
    warnings: list = field(default_factory=list)

    # ---- what the simulations want -----------------------------------
    def summary(self) -> str:
        lines = [f"{self.name or '(unnamed)'}: {self.length * 1000:.0f} mm long, "
                 f"{self.diameter * 1000:.1f} mm across, {self.mass:.3f} kg estimated"]
        if self.fin_count:
            lines.append(f"  fins: {self.fin_count} x root {self.fin_root * 1000:.0f} / "
                         f"tip {self.fin_tip * 1000:.0f} / span {self.fin_span * 1000:.0f} mm, "
                         f"sweep {self.fin_sweep * 1000:.0f} mm, "
                         f"{self.fin_position * 1000:.0f} mm from the nose")
        if self.motors:
            lines.append("  motors: " + ", ".join(f"{d} ({m})" for d, m in self.motors))
        for c in self.components:
            lines.append(f"    {c.kind:14s} {c.name[:22]:22s} {c.mass * 1000:7.1f} g"
                         f"   [{c.source}]")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    def cg_from_nose(self) -> float:
        """Rough CG, from the axial chain and the component masses."""
        num = den = 0.0
        x = 0.0
        for c in self.components:
            if c.length > 0.0:
                num += c.mass * (x + 0.5 * c.length)
                x += c.length
            else:
                num += c.mass * x
            den += c.mass
        return num / den if den > 0.0 else 0.0

    def mmoi_transverse(self, mass=None) -> float:
        """Slender-rod MMOI about the CG, from the imported length."""
        m = self.mass if mass is None else mass
        return m * self.length * self.length / 12.0

    def mmoi_roll(self, mass=None) -> float:
        m = self.mass if mass is None else mass
        return m * 0.5 * (self.diameter / 2.0) ** 2


def _f(node, tag, default=0.0):
    el = node.find(tag)
    if el is None or el.text is None:
        return default
    try:
        return float(el.text)
    except ValueError:
        return default


def _txt(node, tag, default=""):
    el = node.find(tag)
    return (el.text or default) if el is not None else default


def _density(node):
    """Bulk density [kg/m3] of a component's material, if it states one."""
    mat = node.find("material")
    if mat is None:
        return None
    try:
        return float(mat.get("density"))
    except (TypeError, ValueError):
        return None


def _override(node):
    """OpenRocket's mass override, if the component has one switched on."""
    if _txt(node, "massoverridden", "false").strip().lower() != "true":
        return None
    el = node.find("overridemass")
    if el is None or el.text is None:
        return None
    try:
        return float(el.text)
    except ValueError:
        return None


def _tube_mass(length, r_out, thickness, rho):
    r_in = max(r_out - thickness, 0.0)
    return math.pi * (r_out * r_out - r_in * r_in) * length * rho


def _cone_mass(length, r_aft, thickness, rho):
    slant = math.sqrt(r_aft * r_aft + length * length)
    return math.pi * r_aft * slant * thickness * rho


def read_ork(path) -> OrkRocket:
    """Parse an .ork (or a bare rocket.ork XML) into an OrkRocket."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            member = next((n for n in z.namelist() if n.endswith(".ork")
                           or n.endswith(".xml")), None)
            if member is None:
                raise ValueError("no rocket.ork inside the archive")
            data = z.read(member)
        root = ET.fromstring(data)
    else:
        root = ET.parse(path).getroot()

    r = OrkRocket()
    rocket = root.find(".//rocket")
    if rocket is None:
        raise ValueError("this file has no <rocket> in it - is it an OpenRocket file?")
    r.name = _txt(rocket, "name", os.path.basename(str(path)))

    fin_candidates = []
    # The axial chain is the document order of the structural components; that is how
    # OpenRocket lays a rocket out, so walking the tree in order gives the geometry.
    for node in rocket.iter():
        tag = node.tag
        rho = _density(node)
        name = _txt(node, "name", tag)
        over = _override(node)

        if tag == "nosecone":
            length = _f(node, "length")
            rad = _f(node, "aftradius") or _f(node, "radius")
            th = _f(node, "thickness", 0.002)
            m = over if over is not None else (
                _cone_mass(length, rad, th, rho) if rho else 0.0)
            r.components.append(OrkComponent(
                tag, name, m, "override" if over is not None else
                ("estimated from geometry" if rho else "no material - counted as 0"),
                length, rad))
            r.length += length
            r.diameter = max(r.diameter, 2.0 * rad)

        elif tag in ("bodytube", "transition", "innertube", "tubecoupler"):
            length = _f(node, "length")
            rad = _f(node, "radius") or _f(node, "outerradius") or _f(node, "aftradius")
            th = _f(node, "thickness", 0.002)
            m = over if over is not None else (
                _tube_mass(length, rad, th, rho) if rho else 0.0)
            r.components.append(OrkComponent(
                tag, name, m, "override" if over is not None else
                ("estimated from geometry" if rho else "no material - counted as 0"),
                length, rad))
            if tag in ("bodytube", "transition"):
                r.length += length
                r.diameter = max(r.diameter, 2.0 * rad)

        elif tag in ("trapezoidfinset", "ellipticalfinset", "freeformfinset"):
            count = int(_f(node, "fincount", 3))
            root_c = _f(node, "rootchord")
            tip_c = _f(node, "tipchord")
            span = _f(node, "height")
            sweep = _f(node, "sweeplength")
            if not sweep:
                ang = _f(node, "sweepangle")
                sweep = span * math.tan(math.radians(ang)) if ang else 0.0
            th = _f(node, "thickness", 0.003)
            area = 0.5 * (root_c + tip_c) * span
            m = over if over is not None else (
                count * area * th * rho if rho else 0.0)
            r.components.append(OrkComponent(
                tag, name, m, "override" if over is not None else
                ("estimated from geometry" if rho else "no material - counted as 0")))
            fin_candidates.append((_f(node, "position"), count, root_c, tip_c, span,
                                   sweep))

        elif tag in ("masscomponent", "shockcord", "parachute", "streamer",
                     "engineblock", "centeringring", "bulkhead", "launchlug",
                     "railbutton"):
            m = over if over is not None else _f(node, "mass",
                                                 _f(node, "packedmass", 0.0))
            if m or over is not None:
                r.components.append(OrkComponent(tag, name, m,
                                                 "override" if over is not None
                                                 else "stated mass"))

        elif tag == "motor":
            des = _txt(node, "designation").strip()
            man = _txt(node, "manufacturer").strip()
            if des:
                r.motors.append((des, man))

    if fin_candidates:
        # the aft-most set is the one that flies this vehicle
        _pos, count, root_c, tip_c, span, sweep = max(fin_candidates,
                                                      key=lambda f: f[0])
        r.fin_count, r.fin_root, r.fin_tip = count, root_c, tip_c
        r.fin_span, r.fin_sweep, r.fin_position = span, sweep, _pos
    else:
        r.warnings.append("no fin set found - fin geometry left as it was")

    r.mass = sum(c.mass for c in r.components)
    if r.mass <= 0.0:
        r.warnings.append("every component came out at zero mass - the file has no "
                          "materials and no overrides, so only geometry was imported")
    if any(c.source.startswith("estimated") for c in r.components):
        r.warnings.append("some masses are ESTIMATED from geometry x material "
                          "density; override them in OpenRocket for an exact import")
    if r.motors:
        r.warnings.append(f"motor masses are NOT in an .ork file. Found "
                          f"{len(r.motors)} motor(s): "
                          + ", ".join(d for d, _m in r.motors)
                          + ". Add the landing motor's mass yourself, and subtract the "
                            "propellant of the one that is already spent.")
    return r


def apply_to_config(rocket: OrkRocket, cfg, spent_propellant=0.0, add_motor_mass=0.0):
    """Return a copy of a TvcConfig describing the imported vehicle.

    `spent_propellant` is the propellant of the motor that has already burned by the
    time this scenario starts (it launched the rocket); its casing stays on board.
    `add_motor_mass` is the total mass of motors the .ork could not weigh.
    """
    from dataclasses import replace
    mass = rocket.mass + add_motor_mass - spent_propellant
    if mass <= cfg.propellant:
        raise ValueError(f"imported mass {mass:.3f} kg is not more than the landing "
                         f"propellant {cfg.propellant:.3f} kg - check the import")
    out = replace(cfg, gross_mass=mass)
    if rocket.diameter > 0:
        out = replace(out, diameter=rocket.diameter)
    if rocket.length > 0:
        out = replace(out, mmoi_transverse=mass * rocket.length ** 2 / 12.0,
                      mmoi_roll=mass * 0.5 * (out.diameter / 2.0) ** 2)
    if rocket.fin_count:
        out = replace(out, fin_count=rocket.fin_count, fin_root=rocket.fin_root,
                      fin_tip=rocket.fin_tip, fin_span=rocket.fin_span)
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(read_ork(p).summary())
