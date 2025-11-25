from dataclasses import dataclass, field
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from gymnasium import spaces

from isaaclab_tasks.marine_docking.configs.yacht_sensors_cfg import make_lidar_cfg


@dataclass
class YachtEnvCfg(DirectRLEnvCfg):
    """High-level environment configuration for yacht docking."""

    # REQUIRED — from base class
    decimation: int = 2

    # ---------------------------------------------------------
    # SCENE BLOCK  (REQUIRED)
    # ---------------------------------------------------------
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=32.0,
    )
    )
    # PHYSICS
    dt: float = 1/60
    sim_physics_engine: str = "physx"
    gravity: tuple = (0, 0, -9.81)

    # YACHT
    yacht_usd_path: str = "marine_docking/models/yacht.usd"
    yacht_mass: float = 3200.0
    yacht_length: float = 11.0
    yacht_width: float = 3.5
    max_thrust: float = 4200.0

    # WATER
    use_waves: bool = True
    wave_amplitude: float = 0.3
    wave_frequency: float = 0.25
    wave_direction_deg: float = 35.0

    # DOCK
    dock_usd_path: str = "marine_docking/models/dock.usd"
    dock_position: tuple = (20.0, 0.0, 0.0)

    # SENSORS
    sensors: dict = field(default_factory=lambda: {
        "ray_lidar": make_lidar_cfg(),
        "camera_front": CameraCfg(
            prim_path="/World/envs/env_*/Yacht/Camera",
            update_period=1,
            width=320,
            height=240,
            data_types=["rgb"],
            spawn=None      # Valid
        ),
    })

    custom_sensors: dict = field(default_factory=lambda: {
        "lidar": "isaaclab_tasks.marine_docking.sensors.lidar_sensor.YachtLidar",
        "radar": "isaaclab_tasks.marine_docking.sensors.radar_sensor.YachtRadar",
        "camera": "isaaclab_tasks.marine_docking.sensors.camera_sensor.YachtCamera",
    })

    # REQUIRED SPACES
    observation_space: spaces.Box = field(
        default_factory=lambda: spaces.Box(
            low=-1.0, high=1.0, shape=(256,), dtype=float
        )
    )

    action_space: spaces.Box = field(
        default_factory=lambda: spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=float
        )
    )

    state_space: spaces.Box = field(
        default_factory=lambda: spaces.Box(
            low=-1.0, high=1.0, shape=(256,), dtype=float
        )
    )

    # ENVIRONMENT
    episode_length_s: float = 25.0
