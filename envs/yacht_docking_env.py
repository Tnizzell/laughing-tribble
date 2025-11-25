# marine_docking/envs/yacht_docking_env.py

import torch
import math

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import wrap_to_pi



from isaaclab_tasks.marine_docking.utils.water_surface import WaveParams
from isaaclab_tasks.marine_docking.utils.hydrodynamics import (
    HullHydroParams, compute_hydrodynamic_forces
)
from isaaclab_tasks.marine_docking.utils.thruster_model import (
    ThrusterParams, TwinThrusterModel
)


class YachtDockingEnv(DirectRLEnv):
    """
    Core yacht docking RL environment.

    The agent controls:
        - left thruster
        - right thruster

    The environment simulates:
        - hydrodynamic drag
        - thruster forces
        - wave-induced water velocity
        - simple dynamics model
        - sensors (lidar, radar, camera)
        - docking reward

    Goal:
        approach dock and stop within tolerance.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.cfg = cfg

        # ---------------------------
        # Physics params
        # ---------------------------
        self.wave_params = WaveParams(
            amplitude=cfg.wave_amplitude,
            frequency=cfg.wave_frequency,
            direction_rad=math.radians(cfg.wave_direction_deg),
        )

        self.hull_params = HullHydroParams(
            length=cfg.yacht_length,
            width=cfg.yacht_width,
            draft=0.6,  # assume light draft
            mass=cfg.yacht_mass,
        )

        self.thrusters = TwinThrusterModel(
            ThrusterParams(
                max_thrust=cfg.max_thrust,
                lateral_offset=cfg.yacht_width * 0.35,
            )
        )

        # ---------------------------
        # Simulation state
        # ---------------------------
        self.pos = None          # [N, 3]
        self.yaw = None          # [N]
        self.vel = None          # surge/sway
        self.yaw_rate = None     # [N]

        # Dock target
        dx, dy, dz = cfg.dock_position
        self.dock_pos = torch.tensor([dx, dy, dz], device=self.device)

    # ======================================================
    # RESET
    # ======================================================
    def reset_idx(self, env_ids):
        """
        Reset specific environments to initial random poses.
        """
        n = len(env_ids)

        # Random spawn around start area
        x = torch.zeros(n, device=self.device)
        y = torch.zeros(n, device=self.device)
        x += torch.rand(n, device=self.device) * 8.0 - 4.0
        y += torch.rand(n, device=self.device) * 8.0 - 4.0

        self.pos[env_ids, 0] = x
        self.pos[env_ids, 1] = y
        self.pos[env_ids, 2] = 0.0

        self.yaw[env_ids] = (torch.rand(n, device=self.device) * 2 - 1) * 0.5

        self.vel[env_ids, :] = 0.0
        self.yaw_rate[env_ids] = 0.0

        self.thrusters.reset()

    # ======================================================
    # OBSERVATIONS
    # ======================================================
    def get_obs(self):
        """
        Observation vector for RL policy.

        Current set (minimal):
            [relative_x, relative_y, yaw_error,
             surge_vel, sway_vel, yaw_rate]
            + lidar
            + radar
            + camera features
        """
        num_envs = self.num_envs

        # Relative position to dock
        rel = self.dock_pos.unsqueeze(0) - self.pos[:, :3]
        rel_x = rel[:, 0]
        rel_y = rel[:, 1]

        # Yaw error (desired facing dock)
        desired_yaw = torch.atan2(rel_y, rel_x)
        yaw_err = wrap_to_pi(desired_yaw - self.yaw)

        base = torch.stack([
            rel_x,
            rel_y,
            yaw_err,
            self.vel[:, 0],     # surge
            self.vel[:, 1],     # sway
            self.yaw_rate
        ], dim=-1)

        # -----------------
        # Add sensors
        # -----------------
        lidar = self.sensors["lidar"].get_obs()
        radar = self.sensors["radar"].get_obs()
        camera = self.sensors["camera"].get_obs()

        return torch.cat([base, lidar, radar, camera], dim=-1)

    # ======================================================
    # ACTIONS
    # ======================================================
    def apply_actions(self, actions):
        """
        Actions: [N, 2] in [-1, 1] each.
        Maps directly to thrusters.
        """
        left = actions[:, 0]
        right = actions[:, 1]

        dt = self.cfg.dt
        Fx, Mz = self.thrusters.step(left, right, dt)

        # Add hydrodynamic forces
        Fx_h, Fy_h, Mz_h = compute_hydrodynamic_forces(
            hull_params=self.hull_params,
            wave_params=self.wave_params,
            world_pos=self.pos,
            body_vel=self.vel,
            yaw_rate=self.yaw_rate,
            t=self.sim_time,
        )

        # Total forces in body frame
        Fx_total = Fx + Fx_h
        Fy_total = Fy_h
        Mz_total = Mz + Mz_h

        # Integrate dynamics
        m = self.hull_params.mass
        Iz = m * (self.hull_params.length**2 + self.hull_params.width**2) / 12

        # Accelerations
        u_dot = Fx_total / m
        v_dot = Fy_total / m
        r_dot = Mz_total / Iz

        # Integrate velocities
        self.vel[:, 0] += u_dot * dt
        self.vel[:, 1] += v_dot * dt
        self.yaw_rate += r_dot * dt

        # Integrate position / yaw in WORLD frame
        cy = torch.cos(self.yaw)
        sy = torch.sin(self.yaw)

        vx_world = cy * self.vel[:, 0] - sy * self.vel[:, 1]
        vy_world = sy * self.vel[:, 0] + cy * self.vel[:, 1]

        self.pos[:, 0] += vx_world * dt
        self.pos[:, 1] += vy_world * dt
        self.yaw += self.yaw_rate * dt

    # ======================================================
    # REWARDS
    # ======================================================
    def compute_reward(self):
        """
        Reward shaped toward:
        - decreasing distance to dock
        - facing the dock
        - low velocity near target
        """

        rel = self.dock_pos.unsqueeze(0) - self.pos[:, :3]
        dist = torch.norm(rel[:, :2], dim=-1)

        desired_yaw = torch.atan2(rel[:, 1], rel[:, 0])
        yaw_err = wrap_to_pi(desired_yaw - self.yaw)

        # Core reward terms
        r_approach = -dist
        r_heading = -torch.abs(yaw_err)
        r_slow = -torch.norm(self.vel, dim=-1)

        return r_approach + 0.5 * r_heading + 0.2 * r_slow

    # ======================================================
    # TERMINATION
    # ======================================================
    def compute_termination(self):
        """
        Episode ends if:
        - too far from area
        - collision (later)
        - successful docking
        """
        rel = self.dock_pos.unsqueeze(0) - self.pos[:, :3]
        dist = torch.norm(rel[:, :2], dim=-1)

        # Success when within 2m and slow
        success = (dist < 2.0) & (torch.norm(self.vel, dim=-1) < 0.2)

        # Failure: drift too far
        failure = dist > 80.0

        return success | failure
