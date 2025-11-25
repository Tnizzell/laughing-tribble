# marine_docking/utils/hydrodynamics.py

"""
First-order hydrodynamic model for a small yacht hull.

We model:
- Surge (forward/back) drag
- Sway (sideways) drag
- Yaw (heading) damping

Using linear + quadratic drag terms:
    F = - (Cd_lin * v + Cd_quad * |v| * v)

All forces are in the hull/body frame.

This is a simplified but physically grounded model that is commonly
used for marine control and RL before full CFD.
"""

from dataclasses import dataclass
import math

from .water_surface import WaveParams, get_wave_elevation_and_velocity

@dataclass
class HullHydroParams:
    length: float          # m
    width: float           # m
    draft: float           # m (submerged height)
    mass: float            # kg

    rho: float = 1025.0    # sea water density [kg/m^3]

    # Linear drag coefficients
    cd_surge_lin: float = 800.0
    cd_sway_lin: float = 1200.0
    cd_yaw_lin: float = 300.0

    # Quadratic drag coefficients
    cd_surge_quad: float = 1500.0
    cd_sway_quad: float = 2600.0
    cd_yaw_quad: float = 700.0

def _drag_1d(v: float, cd_lin: float, cd_quad: float) -> float:
    return -(cd_lin * v + cd_quad * abs(v) * v)

def compute_hydrodynamic_forces(
    hull_params: HullHydroParams,
    wave_params: WaveParams,
    world_pos: tuple,
    body_vel: tuple,
    yaw_rate: float,
    t: float,
):
    """
    Compute hydrodynamic forces & moments on the hull.

    Args:
        hull_params: HullHydroParams
        wave_params: WaveParams
        world_pos: (x, y, z) world position of hull CG
        body_vel: (u, v, r) where:
            u = surge velocity in body frame [m/s]
            v = sway velocity in body frame [m/s]
            r = yaw rate (same as yaw_rate) [rad/s]
        yaw_rate: redundant with body_vel[2], provided for clarity
        t: simulation time [s]

    Returns:
        Fx, Fy, Mz: forces in surge/sway and yaw moment (body frame)
    """
    x, y, _ = world_pos
    u, v, r = body_vel

    # Get local water horizontal velocity from wave model
    _, wx, wy = get_wave_elevation_and_velocity(x, y, t, wave_params)

    # Relative velocity between hull and water
    u_rel = u - wx
    v_rel = v - wy
    r_rel = r  # assume waves don't impart direct angular velocity

    # Surge drag (forward/back)
    Fx = _drag_1d(u_rel, hull_params.cd_surge_lin, hull_params.cd_surge_quad)

    # Sway drag (sideways)
    Fy = _drag_1d(v_rel, hull_params.cd_sway_lin, hull_params.cd_sway_quad)

    # Yaw damping (rotation)
    Mz = _drag_1d(r_rel, hull_params.cd_yaw_lin, hull_params.cd_yaw_quad)

    return Fx, Fy, Mz
