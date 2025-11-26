# isaaclab_tasks/marine_docking/configs/yacht_env_cfg.py

from dataclasses import dataclass, field
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab_tasks.marine_docking.configs.yacht_sensors_cfg import (
    make_lidar_cfg,
    make_radar_cfg,
    make_camera_cfg
)
from gymnasium import spaces

@dataclass
class YachtEnvCfg(DirectRLEnvCfg):
    # ---------------------------------------------------------
    # SENSORS (CORRECT - Uses Custom Factories)
    # ---------------------------------------------------------
    sensors: dict = field(
        default_factory=lambda: {
            "lidar": make_lidar_cfg(),
            "radar": make_radar_cfg(),
            "camera_front": make_camera_cfg(),
        }
    )

    decimation: int = 2

    # ---------------------------------------------------------
    # SCENE
    # ---------------------------------------------------------
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(
            num_envs=4096,
            env_spacing=32.0,
        )
    )

    # ---------------------------------------------------------
    # PHYSICS
    # ---------------------------------------------------------
    dt: float = 1 / 60
    sim_physics_engine: str = "physx"
    gravity: tuple = (0, 0, -9.81)

    # ---------------------------------------------------------
    # ASSETS
    # ---------------------------------------------------------
    # Update paths to be relative to where you run python or absolute
    yacht_usd_path: str = "marine_docking/models/yacht.usd"
    yacht_mass: float = 3200.0
    yacht_length: float = 11.0
    yacht_width: float = 3.5
    max_thrust: float = 4200.0

    use_waves: bool = True
    wave_amplitude: float = 0.3
    wave_frequency: float = 0.25
    wave_direction_deg: float = 35.0

    dock_usd_path: str = "marine_docking/models/dock.usd"
    dock_position: tuple = (20.0, 0.0, 0.0)

    # ---------------------------------------------------------
    # SPACES
    # ---------------------------------------------------------
    observation_space: spaces.Box = field(
        default_factory=lambda: spaces.Box(
            low=-1.0, high=1.0, shape=(256,), dtype=float
        )
    )

    action_space: spaces.Box = field(
        default_factory=lambda: spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=float
        )
    )

    state_space: spaces.Box = field(
        default_factory=lambda: spaces.Box(
            low=-1.0, high=1.0, shape=(256,), dtype=float
        )
    )

    episode_length_s: float = 25.0