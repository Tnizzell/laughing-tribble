# isaaclab_tasks/marine_docking/configs/yacht_env_cfg.py

import os
from dataclasses import dataclass, field
import gymnasium as gym

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sim import UsdFileCfg, DistantLightCfg
from isaaclab.utils import configclass

# Custom Sensor Factories
from isaaclab_tasks.marine_docking.configs.yacht_sensors_cfg import (
    make_lidar_cfg,
    make_radar_cfg,
    make_camera_cfg
)

# ---------------------------------------------------------
# PATH RESOLUTION (The Fix)
# ---------------------------------------------------------
# Get the absolute path of the directory this file is in (configs/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level (marine_docking/) and into models/
ASSET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../models"))

# ---------------------------------------------------------
# SCENE CONFIGURATION
# ---------------------------------------------------------
@configclass
class YachtSceneCfg(InteractiveSceneCfg):
    """Configuration for the yacht scene."""
    
    # 1. THE YACHT
    yacht: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Yacht",
        spawn=UsdFileCfg(
            # Use absolute path
            usd_path=f"{ASSET_DIR}/yacht.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 2. THE DOCK
    dock: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Dock",
        spawn=UsdFileCfg(
            # Use absolute path
            usd_path=f"{ASSET_DIR}/dock.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(20.0, 0.0, 0.0),
        ),
    )

    # 3. LIGHT
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=DistantLightCfg(
            intensity=3000.0, 
            color=(1.0, 1.0, 1.0)
        ),
    )


# ---------------------------------------------------------
# MAIN ENVIRONMENT CONFIG
# ---------------------------------------------------------
@dataclass
class YachtEnvCfg(DirectRLEnvCfg):
    # Use our custom scene
    scene: YachtSceneCfg = field(
        default_factory=lambda: YachtSceneCfg(
            num_envs=4096,
            env_spacing=32.0,
            replicate_physics=True,
        )
    )

    # SENSORS
    sensors: dict = field(
        default_factory=lambda: {
            "lidar": make_lidar_cfg(),
            "radar": make_radar_cfg(),
            "camera_front": make_camera_cfg(),
        }
    )

    decimation: int = 2
    dt: float = 1 / 60
    sim_physics_engine: str = "physx"
    gravity: tuple = (0, 0, -9.81)

    # ---------------------------------------------------------
    # PARAMETERS (Logic still needs these)
    # ---------------------------------------------------------
    yacht_usd_path: str = f"{ASSET_DIR}/yacht.usd"
    yacht_mass: float = 3200.0
    yacht_length: float = 11.0
    yacht_width: float = 3.5
    max_thrust: float = 4200.0

    use_waves: bool = True
    wave_amplitude: float = 0.3
    wave_frequency: float = 0.25
    wave_direction_deg: float = 35.0

    dock_usd_path: str = f"{ASSET_DIR}/dock.usd"
    dock_position: tuple = (20.0, 0.0, 0.0)

    # ---------------------------------------------------------
    # SPACES
    # ---------------------------------------------------------
    observation_space: gym.spaces.Box = field(
        default_factory=lambda: gym.spaces.Box(
            low=-1.0, high=1.0, shape=(256,), dtype=float
        )
    )

    action_space: gym.spaces.Box = field(
        default_factory=lambda: gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=float
        )
    )

    state_space: gym.spaces.Box = field(
        default_factory=lambda: gym.spaces.Box(
            low=-1.0, high=1.0, shape=(256,), dtype=float
        )
    )

    episode_length_s: float = 25.0