# marine_docking/utils/thruster_model.py

"""
Simple twin-thruster model for small yacht / motorboat.

We assume:
- Left (port) and right (starboard) thrusters
- Both produce forward/reverse thrust along the hull's x-axis
- Their lateral separation creates a yaw moment when throttles differ

This is standard in marine control literature and is realistic enough
for RL docking and control.
"""

from dataclasses import dataclass
import math

@dataclass
class ThrusterParams:
    max_thrust: float        # N (per thruster)
    lateral_offset: float    # m (distance from centerline to each thruster)
    response_time: float = 0.4  # s, simple first-order lag

def saturate(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))

class TwinThrusterModel:
    def __init__(self, params: ThrusterParams):
        self.params = params
        # Internal state for simple thrust lag model
        self.left_thrust = 0.0
        self.right_thrust = 0.0

    def reset(self):
        self.left_thrust = 0.0
        self.right_thrust = 0.0

    def step(self, action_left: float, action_right: float, dt: float):
        """
        Update thruster forces given control actions in [-1, 1].

        Args:
            action_left, action_right: throttle commands [-1, 1]
            dt: timestep [s]

        Returns:
            Fx, Mz: net surge force and yaw moment (body frame)
        """
        aL = saturate(action_left, -1.0, 1.0)
        aR = saturate(action_right, -1.0, 1.0)

        target_left = aL * self.params.max_thrust
        target_right = aR * self.params.max_thrust

        # First-order lag: dT/dt = (T_target - T) / tau
        tau = max(self.params.response_time, 1e-4)
        alpha = saturate(dt / tau, 0.0, 1.0)

        self.left_thrust += alpha * (target_left - self.left_thrust)
        self.right_thrust += alpha * (target_right - self.right_thrust)

        # Net surge force
        Fx = self.left_thrust + self.right_thrust

        # Yaw moment: opposite-signed thrusts create rotation
        # Mz = (T_R - T_L) * lateral_offset
        Mz = (self.right_thrust - self.left_thrust) * self.params.lateral_offset

        return Fx, Mz
