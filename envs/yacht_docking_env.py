# isaaclab_tasks/marine_docking/envs/yacht_docking_env.py

import math
from collections.abc import Sequence

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import wrap_to_pi

from isaaclab.sim import DistantLightCfg

from isaaclab_tasks.marine_docking.utils.thruster_model import (
    ThrusterParams,
    TwinThrusterModel,
)


class YachtDockingEnv(DirectRLEnv):
    """
    Simple direct-RL yacht docking environment.

    - Vectorized over cfg.scene.num_envs.
    - Uses a planar (x, y, yaw) kinematic model with body-frame velocities (u, v) and yaw-rate r.
    - Actions: 3D in [-1, 1]; first two are used as left/right thruster commands.
    - Observations: 9D (see get_obs()).
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Keep a typed view of the config
        self.cfg = cfg

        # ------------------------------------------------------------------
        # Physical parameters (mass / inertia + thruster model)
        # ------------------------------------------------------------------
        self.mass = float(cfg.yacht_mass)
        # crude box-approx inertia around z
        self.inertia_z = self.mass * (
            cfg.yacht_length**2 + cfg.yacht_width**2
        ) / 12.0

        self.thrusters = TwinThrusterModel(
            ThrusterParams(
                max_thrust=cfg.max_thrust,
                lateral_offset=cfg.yacht_width * 0.35,
            )
        )

        # Dock target (world frame)
        dx, dy, dz = cfg.dock_position
        self.dock_pos = torch.tensor(
            [dx, dy, dz], device=self.device, dtype=torch.float32
        )

        # ------------------------------------------------------------------
        # State buffers (all on sim device, batched over num_envs)
        # ------------------------------------------------------------------
        n_envs = self.num_envs

        # Position in world frame (x, y, z)
        self.pos = torch.zeros(
            n_envs, 3, device=self.device, dtype=torch.float32
        )
        # Heading (yaw) in radians
        self.yaw = torch.zeros(
            n_envs, device=self.device, dtype=torch.float32
        )
        # Body-frame linear velocity [u (surge), v (sway)]
        self.vel = torch.zeros(
            n_envs, 2, device=self.device, dtype=torch.float32
        )
        # Yaw rate r
        self.yaw_rate = torch.zeros(
            n_envs, device=self.device, dtype=torch.float32
        )

        # Simple time accumulator per-env if we ever want time-based effects
        self._time = torch.zeros(
            n_envs, device=self.device, dtype=torch.float32
        )

        # Initialize all envs
        all_ids = torch.arange(
            n_envs, device=self.device, dtype=torch.long
        )
        self._reset_idx(all_ids)

    # ======================================================================
    # Core DirectRLEnv hooks
    # ======================================================================

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Store incoming actions into self.actions.

        actions: (num_envs, action_dim) on *any* device; DirectRLEnv.step
        already moved it to self.device before calling this.
        """
        if actions is None:
            return

        # Clamp to [-1, 1] and store in the buffer allocated by DirectRLEnv
        self.actions[:] = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        """
        Apply current actions at physics dt.

        We treat:
            a[:, 0] -> left thruster command
            a[:, 1] -> right thruster command
            a[:, 2] -> currently unused (reserved / future bow thruster)
        """
        dt = self.physics_dt
        a = self.actions  # (N, 3)

        # Decode actions (we ignore the 3rd one for now)
        left = a[:, 0]
        right = a[:, 1]

        # Thruster model returns body-frame surge force Fx and yaw moment Mz.
        # This implementation must be tensor-friendly (torch.clamp etc.).
        Fx, Mz = self.thrusters.step(left, right, dt)

        # No sway force from thrusters (for now)
        Fy = torch.zeros_like(Fx)

        # ------------------------------------------------------------------
        # Integrate simple planar dynamics
        # ------------------------------------------------------------------
        m = self.mass
        Iz = self.inertia_z

        # Accelerations in body frame
        u_dot = Fx / m
        v_dot = Fy / m
        r_dot = Mz / Iz

        # Integrate body-frame velocities
        self.vel[:, 0] += u_dot * dt  # surge u
        self.vel[:, 1] += v_dot * dt  # sway v
        self.yaw_rate += r_dot * dt   # yaw rate r

        # Transform to world velocities and integrate position
        cy = torch.cos(self.yaw)
        sy = torch.sin(self.yaw)

        u = self.vel[:, 0]
        v = self.vel[:, 1]

        vx_world = cy * u - sy * v
        vy_world = sy * u + cy * v

        self.pos[:, 0] += vx_world * dt
        self.pos[:, 1] += vy_world * dt
        # keep z locked at 0
        self.pos[:, 2] = 0.0

        # Integrate heading
        self.yaw += self.yaw_rate * dt

        # Update time accumulator
        self._time += dt

    def _get_observations(self):
        """
        Wrap get_obs() into the dict format expected by RSL-RL.

        Returns:
            dict with key "policy" -> (num_envs, 9) tensor
        """
        obs = self.get_obs()
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # print("!!! I AM USING THE NEW CODE !!!") # You can comment this out now
        
        rel = self.dock_pos.unsqueeze(0) - self.pos
        dist_xy = torch.norm(rel[:, :2], dim=-1)
        vel_mag = torch.norm(self.vel, dim=-1)
        
        # 1. POSITIVE Distance Reward (0.0 to 1.0)
        # This prevents the -15,000 score
        r_dist = 1.0 / (1.0 + 0.5 * dist_xy)

        # 2. Small Velocity Penalty
        r_vel = -0.05 * vel_mag

        # 3. Small Action Penalty
        r_action = -0.001 * torch.sum(torch.square(self.actions), dim=-1)

        # 4. Heading Bonus
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
        """
        Termination and timeout flags.

        - Terminate if:
            * success: within 2m and slow
            * failure: too far from dock area (> 80m)
        - Timeout if episode_length_buf >= max_episode_length
        """
        rel = self.dock_pos.unsqueeze(0) - self.pos
        dist_xy = torch.norm(rel[:, :2], dim=-1)
        speed = torch.norm(self.vel, dim=-1)

        success = (dist_xy < 2.0) & (speed < 0.2)
        failure = dist_xy > 80.0

        terminated = success | failure
        time_out = self.episode_length_buf >= self.max_episode_length

        # For logging
        self.extras["success"] = success
        self.extras["failure"] = failure

        return terminated, time_out

    # ======================================================================
    # Helper methods
    # ======================================================================
    def _setup_scene(self):
        self.yacht = self.scene["yacht"]
        self.dock = self.scene["dock"]


    def _reset_idx(self, env_ids: Sequence[int]):
        """
        Reset selected environments both in the scene and our state tensors.

        This overrides DirectRLEnv._reset_idx to add our custom state reset,
        but still calls super() so the scene / events / noise get reset.
        """
        # First let the base class reset the scene / events / noise
        super()._reset_idx(env_ids)

        # Convert indices to a long tensor on the right device
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )

        n = env_ids.shape[0]

        # Random spawn near origin in a small square
        x = torch.rand(n, device=self.device) * 8.0 - 4.0
        y = torch.rand(n, device=self.device) * 8.0 - 4.0

        self.pos[env_ids, 0] = x
        self.pos[env_ids, 1] = y
        self.pos[env_ids, 2] = 0.0

        # Small random initial heading
        self.yaw[env_ids] = (torch.rand(n, device=self.device) * 2.0 - 1.0) * 0.5

        # Zero velocities and time
        self.vel[env_ids, :] = 0.0
        self.yaw_rate[env_ids] = 0.0
        self._time[env_ids] = 0.0

    # ======================================================================
    # Public utilities
    # ======================================================================

    def get_obs(self) -> torch.Tensor:
        """
        Build the 9D observation vector per env.

        Layout:
            0: rel_x          (dock_x - x)
            1: rel_y          (dock_y - y)
            2: yaw_err        (desired - current)
            3: u              (surge velocity)
            4: v              (sway velocity)
            5: yaw_rate       (r)
            6: dist_xy        (planar distance to dock)
            7: cos(yaw_err)
            8: sin(yaw_err)
        """
        rel = self.dock_pos.unsqueeze(0) - self.pos  # (N, 3)
        rel_x = rel[:, 0]
        rel_y = rel[:, 1]

        dist_xy = torch.sqrt(rel_x**2 + rel_y**2 + 1e-8)

        desired_yaw = torch.atan2(rel_y, rel_x)
        yaw_err = wrap_to_pi(desired_yaw - self.yaw)

        u = self.vel[:, 0]
        v = self.vel[:, 1]
        r = self.yaw_rate

        obs = torch.stack(
            [
                rel_x,
                rel_y,
                yaw_err,
                u,
                v,
                r,
                dist_xy,
                torch.cos(yaw_err),
                torch.sin(yaw_err),
            ],
            dim=-1,
        )

        return obs
