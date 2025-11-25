# marine_docking/utils/water_surface.py

"""
Simple physically-based wave model for yacht simulation.

This is NOT just "visual waves".
We compute:
- surface elevation
- local water velocity

so hydrodynamic forces can depend on relative velocity between hull and water.
"""

import math
from dataclasses import dataclass

@dataclass
class WaveParams:
    amplitude: float       # wave height [m]
    frequency: float       # wave frequency [Hz]
    direction_rad: float   # wave direction [rad]
    phase: float = 0.0     # phase shift

    # Water properties
    water_density: float = 1025.0  # kg/m^3 (sea water)

    # "Stiffness" of wave model
    wavelength: float = 20.0       # m (approx)
    steepness: float = 0.5         # 0..1, controls sharpness

def _wave_number(params: WaveParams) -> float:
    # k = 2π / λ
    return 2.0 * math.pi / params.wavelength

def _angular_frequency(params: WaveParams) -> float:
    # ω = 2π f
    return 2.0 * math.pi * params.frequency

def get_wave_elevation_and_velocity(x: float, y: float, t: float, params: WaveParams):
    """
    Compute water surface elevation and horizontal water velocity
    at world position (x, y) and time t.

    Returns:
        eta: surface height (z) in meters
        vx, vy: horizontal water velocity components [m/s]
    """
    k = _wave_number(params)
    w = _angular_frequency(params)

    # Project position onto wave direction
    dx = math.cos(params.direction_rad)
    dy = math.sin(params.direction_rad)
    s = x * dx + y * dy

    # Phase term
    theta = k * s - w * t + params.phase

    # Gerstner-like approximation
    a = params.amplitude
    q = params.steepness

    # Surface elevation
    eta = a * math.sin(theta)

    # Horizontal water velocity (approx derivative of wave potential)
    vx = q * a * w * math.cos(theta) * dx
    vy = q * a * w * math.cos(theta) * dy

    return eta, vx, vy
