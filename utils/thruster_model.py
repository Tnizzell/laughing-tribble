# isaaclab_tasks/marine_docking/utils/thruster_model.py

import torch
from dataclasses import dataclass


# ------------------------------------------------------------
# Correct saturate for tensors (FIXED)
# ------------------------------------------------------------
def saturate(x, min_v, max_v):
    """Clamp tensor actions safely."""
    return torch.clamp(x, min_v, max_v)


# ------------------------------------------------------------
# Thruster parameter dataclass
# ------------------------------------------------------------
@dataclass
class ThrusterParams:
    max_thrust: float = 4000.0
    lateral_offset: float = 1.0  # meters from centerline


# ------------------------------------------------------------
# Twin thruster model (left + right)
# ------------------------------------------------------------
class TwinThrusterModel:
    def __init__(self, params: ThrusterParams):
        self.params = params

        # last outputs
        self.last_force = None
        self.last_moment = None

    def reset(self):
        self.last_force = None
        self.last_moment = None

    def step(self, action_left, action_right, dt):
        """
        Inputs:
            action_left  : tensor [N] in [-1,1]
            action_right : tensor [N] in [-1,1]
            dt           : float

        Returns:
            Fx : surge force   [N]
            Mz : yaw moment    [Nm]
        """
        # Ensure tensors
        device = action_left.device

        aL = saturate(action_left, -1.0, 1.0)
        aR = saturate(action_right, -1.0, 1.0)

        # Convert thrust from normalized input → Newtons
        T = self.params.max_thrust
        Fx_left = aL * T
        Fx_right = aR * T

        # Total forward force
        Fx_total = Fx_left + Fx_right

        # Yaw moment (difference in thrust * lever arm)
        d = self.params.lateral_offset
        Mz_total = (Fx_right - Fx_left) * d

        self.last_force = Fx_total
        self.last_moment = Mz_total

        return Fx_total, Mz_total
