# isaaclab_tasks/marine_docking/envs/yacht_docking_env.py

import math
from collections.abc import Sequence
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import wrap_to_pi
from isaaclab_tasks.marine_docking.utils.thruster_model import (
    ThrusterParams,
    TwinThrusterModel,
)

class YachtDockingEnv(DirectRLEnv):
    """
    Simple direct-RL yacht docking environment.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg = cfg

        # Physical parameters
        self.mass = float(cfg.yacht_mass)
        self.inertia_z = self.mass * (cfg.yacht_length**2 + cfg.yacht_width**2) / 12.0

        self.thrusters = TwinThrusterModel(
            ThrusterParams(
                max_thrust=cfg.max_thrust,
                lateral_offset=cfg.yacht_width * 0.35,
            )
        )

        # Dock target (world frame)
        dx, dy, dz = cfg.dock_position
        self.dock_pos = torch.tensor([dx, dy, dz], device=self.device, dtype=torch.float32)

        # State buffers
        n_envs = self.num_envs
        self.pos = torch.zeros(n_envs, 3, device=self.device, dtype=torch.float32)
        self.yaw = torch.zeros(n_envs, device=self.device, dtype=torch.float32)
        self.vel = torch.zeros(n_envs, 2, device=self.device, dtype=torch.float32)
        self.yaw_rate = torch.zeros(n_envs, device=self.device, dtype=torch.float32)
        self._time = torch.zeros(n_envs, device=self.device, dtype=torch.float32)

        all_ids = torch.arange(n_envs, device=self.device, dtype=torch.long)
        self._reset_idx(all_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions is None: return
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        dt = self.physics_dt
        a = self.actions
        left = a[:, 0]
        right = a[:, 1]

        Fx, Mz = self.thrusters.step(left, right, dt)
        Fy = torch.zeros_like(Fx)

        # Dynamics
        m = self.mass
        Iz = self.inertia_z

        u_dot = Fx / m
        v_dot = Fy / m
        r_dot = Mz / Iz

        self.vel[:, 0] += u_dot * dt
        self.vel[:, 1] += v_dot * dt
        self.yaw_rate += r_dot * dt

        cy = torch.cos(self.yaw)
        sy = torch.sin(self.yaw)
        u, v = self.vel[:, 0], self.vel[:, 1]
        
        vx_world = cy * u - sy * v
        vy_world = sy * u + cy * v

        self.pos[:, 0] += vx_world * dt
        self.pos[:, 1] += vy_world * dt
        self.pos[:, 2] = 0.0
        self.yaw += self.yaw_rate * dt
        self._time += dt

    def _get_observations(self):
        obs = self.get_obs()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """
        FIXED REWARD FUNCTION (Positive Bounded)
        """
        rel = self.dock_pos.unsqueeze(0) - self.pos
        dist_xy = torch.norm(rel[:, :2], dim=-1)
        
        # 1. Distance Reward: 0.0 (far) to 1.0 (perfect)
        # This replaces the old "-dist_xy" which caused the -17,000 score
        r_dist = 1.0 / (1.0 + 0.5 * dist_xy)

        # 2. Velocity Penalty (Small)
        vel_mag = torch.norm(self.vel, dim=-1)
        r_vel = -0.05 * vel_mag

        # 3. Action Penalty (Save energy)
        r_action = -0.001 * torch.sum(torch.square(self.actions), dim=-1)

        # 4. Heading Bonus (Optional, small)
        desired_yaw = torch.atan2(rel[:, 1], rel[:, 0])
        yaw_err = wrap_to_pi(desired_yaw - self.yaw)
        r_heading = 0.1 * (1.0 - torch.abs(yaw_err) / 3.14159)

        # 5. Success Bonus
        is_close = dist_xy < 1.5
        is_slow = vel_mag < 0.2
        r_success = (is_close & is_slow).float() * 2.0

        total_reward = r_dist + r_vel + r_action + r_heading + r_success
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        rel = self.dock_pos.unsqueeze(0) - self.pos
        dist_xy = torch.norm(rel[:, :2], dim=-1)
        speed = torch.norm(self.vel, dim=-1)

        # Success: Close and Slow
        success = (dist_xy < 2.0) & (speed < 0.2)
        
        # Failure: Too far away
        failure = dist_xy > 80.0

        terminated = success | failure
        time_out = self.episode_length_buf >= self.max_episode_length
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        n = env_ids.shape[0]
        # Random spawn +/- 4m
        x = torch.rand(n, device=self.device) * 8.0 - 4.0
        y = torch.rand(n, device=self.device) * 8.0 - 4.0

        self.pos[env_ids, 0] = x
        self.pos[env_ids, 1] = y
        self.pos[env_ids, 2] = 0.0
        self.yaw[env_ids] = (torch.rand(n, device=self.device) * 2.0 - 1.0) * 0.5
        self.vel[env_ids, :] = 0.0
        self.yaw_rate[env_ids] = 0.0
        self._time[env_ids] = 0.0

    def get_obs(self) -> torch.Tensor:
        rel = self.dock_pos.unsqueeze(0) - self.pos
        rel_x = rel[:, 0]
        rel_y = rel[:, 1]
        dist_xy = torch.sqrt(rel_x**2 + rel_y**2 + 1e-8)
        
        desired_yaw = torch.atan2(rel_y, rel_x)
        yaw_err = wrap_to_pi(desired_yaw - self.yaw)
        
        obs = torch.stack([
            rel_x, rel_y, yaw_err,
            self.vel[:, 0], self.vel[:, 1], self.yaw_rate,
            dist_xy,
            torch.cos(yaw_err), torch.sin(yaw_err)
        ], dim=-1)
        
        # Pad to 256 size if config expects it (Common RSL_RL requirement)
        padding = torch.zeros(self.num_envs, 256 - obs.shape[1], device=self.device)
        return torch.cat([obs, padding], dim=-1)