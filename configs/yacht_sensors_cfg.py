# marine_docking/configs/yacht_sensors_cfg.py

from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg, PinholeCameraPatternCfg
from isaaclab.utils import configclass


# --------------------------------------------------------
# LIDAR CONFIG (VALID FOR ISAAC LAB 5.1)
# --------------------------------------------------------
def make_lidar_cfg():
    return RayCasterCfg(
        class_type="isaaclab_tasks.marine_docking.sensors.lidar_sensor.YachtLidar",

        prim_path="/World/envs/env_*/Yacht/Lidar",

        pattern_cfg=LidarPatternCfg(
            channels=32,                              # vertical beams
            vertical_fov_range=(-15.0, 15.0),         # degrees
            horizontal_fov_range=(0.0, 360.0),        # degrees
            horizontal_res=0.4,                       # degree increments
        ),
    )


# --------------------------------------------------------
# RADAR CONFIG (FAKE SIMPLE LIDAR-STYLE)
# --------------------------------------------------------
def make_radar_cfg():
    return RayCasterCfg(
        class_type="isaaclab_tasks.marine_docking.sensors.radar_sensor.YachtRadar",

        prim_path="/World/envs/env_*/Yacht/Radar",

        pattern_cfg=LidarPatternCfg(
            channels=4,
            vertical_fov_range=(-3.0, 3.0),
            horizontal_fov_range=(0.0, 120.0),
            horizontal_res=1.0,
        ),
    )


# --------------------------------------------------------
# CAMERA CONFIG (RAW RGB RAYCAST)
# --------------------------------------------------------
def make_camera_cfg():
    return RayCasterCfg(
        class_type="isaaclab_tasks.marine_docking.sensors.camera_sensor.YachtCamera",

        prim_path="/World/envs/env_*/Yacht/Camera",

        pattern_cfg=PinholeCameraPatternCfg(
            width=320,
            height=240,
        ),
    )
