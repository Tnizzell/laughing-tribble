# gen_yacht_dock_assets.py
#
# Generates very simple USDs for:
#   - models/yacht.usd      (box hull + mast + sensor prims)
#   - models/dock.usd      (flat box dock)
#
# Units: meters.

from pxr import Usd, UsdGeom, UsdPhysics, Gf
import os

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def setup_stage(stage):
    """Set meters and up-axis."""
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)        # 1 unit = 1 meter
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)


def add_rigid_body(prim):
    """Make this prim a rigid body + collision in PhysX."""
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)


# ---------------------------------------------------------------------
# Yacht asset
# ---------------------------------------------------------------------

def create_yacht_usd(path, length=11.0, width=3.5, draft=1.0, mast_height=2.0):
    """
    Create a simple yacht asset:

    /Yacht
      /Hull   (scaled cube)
      /Mast   (cylinder)
      /Camera (sensor mount)
      /Lidar  (sensor mount)
      /Radar  (sensor mount)
    """
    stage = Usd.Stage.CreateNew(path)
    setup_stage(stage)

    # Root xform for the yacht
    yacht = UsdGeom.Xform.Define(stage, "/Yacht")
    yacht_prim = yacht.GetPrim()

    # --- Hull: cube scaled to (length, width, draft) ---
    hull = UsdGeom.Cube.Define(stage, "/Yacht/Hull")
    hull_prim = hull.GetPrim()

    # Cube has "size" (default 2) and sits centered at origin.
    # We’ll use scale so that:
    #   X ~ length, Y ~ width, Z ~ draft
    hull.AddScaleOp().Set(Gf.Vec3f(length, width, draft))
    # Lift it so bottom sits roughly at z = 0
    hull.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, draft * 0.5))

    add_rigid_body(hull_prim)

    # --- Mast: cylinder at centerline, near mid-ship ---
    mast = UsdGeom.Cylinder.Define(stage, "/Yacht/Mast")
    mast_prim = mast.GetPrim()
    mast.CreateHeightAttr(mast_height)
    mast.CreateRadiusAttr(0.1)      # 20 cm diameter pole
    # Put it near the longitudinal center, on the deck
    mast.AddTranslateOp().Set(
        Gf.Vec3f(0.0, 0.0, draft + mast_height * 0.5)
    )

    add_rigid_body(mast_prim)

    # --- Sensor mounts (just Xforms, no geometry) ---
    #
    # These must match the prim_paths in your configs:
    #   /World/envs/env_*/Yacht/Camera
    #   /World/envs/env_*/Yacht/Lidar
    #   /World/envs/env_*/Yacht/Radar
    #
    # In the scene, the Yacht will usually be instanced under /World/envs/env_0,
    # so these relative paths line up.

    # Front camera: nose of the boat, about 1.5m above water
    cam = UsdGeom.Xform.Define(stage, "/Yacht/Camera")
    cam.AddTranslateOp().Set(
        Gf.Vec3f(length * 0.5 - 1.0, 0.0, draft + 1.5)
    )

    # Lidar: on top of the mast
    lidar = UsdGeom.Xform.Define(stage, "/Yacht/Lidar")
    lidar.AddTranslateOp().Set(
        Gf.Vec3f(0.0, 0.0, draft + mast_height + 0.2)
    )

    # Radar: a bit behind the mast
    radar = UsdGeom.Xform.Define(stage, "/Yacht/Radar")
    radar.AddTranslateOp().Set(
        Gf.Vec3f(-length * 0.2, 0.0, draft + mast_height)
    )

    # Save
    stage.GetRootLayer().Save()
    print(f"[INFO] Wrote yacht USD to: {path}")


# ---------------------------------------------------------------------
# Dock asset
# ---------------------------------------------------------------------

def create_dock_usd(path, length=20.0, width=6.0, thickness=1.0):
    """
    Simple rectangular dock:

    /Dock
      /Body (scaled cube)
    """
    stage = Usd.Stage.CreateNew(path)
    setup_stage(stage)

    dock = UsdGeom.Xform.Define(stage, "/Dock")
    dock_prim = dock.GetPrim()

    body = UsdGeom.Cube.Define(stage, "/Dock/Body")
    body_prim = body.GetPrim()

    # Scale cube to dock plate
    body.AddScaleOp().Set(Gf.Vec3f(length, width, thickness))
    body.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, thickness * 0.5))

    add_rigid_body(body_prim)

    stage.GetRootLayer().Save()
    print(f"[INFO] Wrote dock USD to: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust base_dir if your task folder is somewhere else
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "..", "models")
    models_dir = os.path.abspath(models_dir)

    os.makedirs(models_dir, exist_ok=True)

    yacht_path = os.path.join(models_dir, "yacht.usd")
    dock_path = os.path.join(models_dir, "dock.usd")

    create_yacht_usd(yacht_path, length=11.0, width=3.5, draft=1.0, mast_height=2.0)
    create_dock_usd(dock_path, length=20.0, width=6.0, thickness=1.0)

    print("[INFO] Done.")
