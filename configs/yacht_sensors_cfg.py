# marine_docking/configs/yacht_sensors_cfg.py

from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LidarPatternCfg, PinholeCameraPatternCfg

# --------------------------------------------------------
# LIDAR CONFIG
# --------------------------------------------------------
def make_lidar_cfg():
    return RayCasterCfg(
        class_type="isaaclab_tasks.marine_docking.sensors.lidar_sensor.YachtLidar",
        prim_path="/World/envs/env_*/Yacht/Lidar",
        update_period=0.1,
        
        # --- THE FIX: Tell Lidar what to hit (Everything in env) ---
        mesh_prim_paths=["/World/envs/env_.*"], 
        
        pattern_cfg=LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=0.4,
        ),
    )


# --------------------------------------------------------
# RADAR CONFIG
# --------------------------------------------------------
def make_radar_cfg():
    return RayCasterCfg(
        class_type="isaaclab_tasks.marine_docking.sensors.radar_sensor.YachtRadar",
        prim_path="/World/envs/env_*/Yacht/Radar",
        update_period=0.1,
        
        # --- THE FIX ---
        mesh_prim_paths=["/World/envs/env_.*"],
        
        pattern_cfg=LidarPatternCfg(
            channels=4,
            vertical_fov_range=(-3.0, 3.0),
            horizontal_fov_range=(0.0, 120.0),
            horizontal_res=1.0,
        ),
    )


# --------------------------------------------------------
# CAMERA CONFIG (Ray Cast)
# --------------------------------------------------------
def make_camera_cfg():
    return RayCasterCfg(
        class_type="isaaclab_tasks.marine_docking.sensors.camera_sensor.YachtCamera",
        prim_path="/World/envs/env_*/Yacht/Camera",
        update_period=0.1,
        
        # --- THE FIX ---
        mesh_prim_paths=["/World/envs/env_.*"],
        
        pattern_cfg=PinholeCameraPatternCfg(
            width=320,
            height=240,
        ),
    )