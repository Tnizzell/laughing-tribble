# marine_docking/configs/yacht_env_cfg.py

from dataclasses import dataclass, field

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.ray_caster import (
    RayCasterCfg,
    CircularPatternCfg,         # VALID IN YOUR VERSION
    UniformRayPatternCfg        # VALID IN YOUR VERSION
)


@dataclass
class YachtEnvCfg(DirectRLEnvCfg):
    """Stable, version-compatible environment configuration for Yacht Docking."""

    # -------------------------------------------------------------------------
    # SIMULATION SETTINGS
    # -------------------------------------------------------------------------
    dt: float = 1.0 / 60.0
    sim_physics_engine: str = "physx"
    gravity: tuple = (0.0, 0.0, -9.81)

    # -------------------------------------------------------------------------
    # YACHT PARAMETERS
    # -------------------------------------------------------------------------
    yacht_usd_path: str = "marine_docking/models/yacht.usd"
    yacht_mass: float = 3200.0
    yacht_length: float = 11.0
    yacht_width: float = 3.5
    max_thrust: float = 4200.0

    # -------------------------------------------------------------------------
    # WATER / WAVES
    # -------------------------------------------------------------------------
    use_waves: bool = True
    wave_amplitude: float = 0.3
    wave_frequency: float = 0.25
    wave_direction_deg: float = 35.0

    # -------------------------------------------------------------------------
    # DOCK
    # -------------------------------------------------------------------------
    dock_usd_path: str = "marine_docking/models/dock.usd"
    dock_position: tuple = (20.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # SENSORS (LOW-LEVEL RAYCAST + CAMERA)
    # -------------------------------------------------------------------------
    sensors: dict = field(default_factory=lambda: {
        #
        # ------------------ LIDAR ------------------
        #
        "lidar": RayCasterCfg(
            class_type="isaaclab_tasks.marine_docking.sensors.lidar_sensor.YachtLidar",

            prim_path="/World/envs/env_*/Yacht/Lidar",
            mesh_prim_paths=["/World"],      # safe default

            # Pattern supported by your build
            pattern_cfg=CircularPatternCfg(
                num_rays=64,
                horizontal_fov=180.0
            ),

            # Required
            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 2.0),
                rot=(0.0, 0.0, 0.0, 1.0)
            ),

            ray_alignment="base",
            max_distance=45.0,
        ),

        #
        # ------------------ RADAR ------------------
        #
        "radar": RayCasterCfg(
            class_type="isaaclab_tasks.marine_docking.sensors.radar_sensor.YachtRadar",

            prim_path="/World/envs/env_*/Yacht/Radar",

            pattern_cfg=UniformRayPatternCfg(
                num_rays=16
            ),

            offset=RayCasterCfg.OffsetCfg(
                pos=(0.0, 0.0, 3.0),
                rot=(0.0, 0.0, 0.0, 1.0)
            ),

            max_distance=120.0,
            ray_alignment="base",
        ),

        #
        # ------------------ CAMERA ------------------
        #
        "camera": CameraCfg(
            class_type="isaaclab_tasks.marine_docking.sensors.camera_sensor.YachtCamera",

            prim_path="/World/envs/env_*/Yacht/Camera",
            width=320,
            height=240,

            data_types=["rgb"],
            update_interval=1,
        ),
    })

    # -------------------------------------------------------------------------
    # HIGH-LEVEL SENSOR WRAPPERS
    # -------------------------------------------------------------------------
    custom_sensors: dict = field(default_factory=lambda: {
        "lidar": "isaaclab_tasks.marine_docking.sensors.lidar_sensor.YachtLidar",
        "radar": "isaaclab_tasks.marine_docking.sensors.radar_sensor.YachtRadar",
        "camera": "isaaclab_tasks.marine_docking.sensors.camera_sensor.YachtCamera",
    })

    # -------------------------------------------------------------------------
    # ENVIRONMENT SIZE
    # -------------------------------------------------------------------------
    num_envs: int = 4096
    env_spacing: float = 32.0
    episode_length_s: float = 25.0
