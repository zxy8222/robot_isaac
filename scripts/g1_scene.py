"""Build and inspect the minimal G1/ground/light/camera USD scene.

This module is imported only after SimulationApp starts. It never edits source
assets or writes result files; physics stepping and persistence belong to callers.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.experimental.prims import Articulation
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

from g1_urdf_config import ASSET_PATH, ROBOT_PATH, VARIANT
GROUND_PATH = "/World/Ground"
CAMERA_PATH = "/World/OverviewCamera"


def wait_for_assets(update: Callable[[], None], timeout: float = 600.0) -> None:
    deadline = time.monotonic() + timeout
    context = omni.usd.get_context()
    while context.get_stage_loading_status()[2] > 0:
        if time.monotonic() >= deadline:
            raise TimeoutError("G1 stage assets did not finish loading")
        update()
    for _ in range(10):
        update()


def build_scene(update: Callable[[], None], asset_usd: str | None = None, *, ground_size: float = 80.0) -> tuple[Usd.Stage, Articulation, object, dict[str, object]]:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("SimulationApp did not create a USD stage")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr().Set(9.81)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    scene_api.CreateEnableGPUDynamicsAttr().Set(False)
    scene_api.CreateBroadphaseTypeAttr().Set("MBP")
    scene_api.CreateSolverTypeAttr().Set("TGS")

    GroundPlane(GROUND_PATH, sizes=ground_size, colors=[0.32, 0.36, 0.40], templates=None)
    ground_colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(GROUND_PATH))
                        if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not ground_colliders:
        raise RuntimeError("Ground plane has no collision API")
    for path in ground_colliders:
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr().Set(True)

    resolved_asset = asset_usd or str(ASSET_PATH)
    print(f"G1_ASSET_LOADING={resolved_asset}", flush=True)
    robot_prim = stage_utils.add_reference_to_stage(usd_path=resolved_asset, path=ROBOT_PATH)
    wait_for_assets(update)
    embedded_scenes = [prim for prim in Usd.PrimRange(robot_prim) if prim.IsA(UsdPhysics.Scene)]
    disabled_scenes = [str(prim.GetPath()) for prim in embedded_scenes]
    for prim in embedded_scenes:
        prim.SetActive(False)
    robot_colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(robot_prim)
                       if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not robot_colliders:
        raise RuntimeError("G1 asset has no loaded collision shapes")

    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    box = bounds.ComputeWorldBound(robot_prim).ComputeAlignedBox()
    if box.IsEmpty() or not math.isfinite(float(box.GetMin()[2])):
        raise RuntimeError("G1 asset has no finite loaded geometry bounds")
    spawn_offset = 0.08 - float(box.GetMin()[2])
    UsdGeom.Xformable(robot_prim).AddTranslateOp(opSuffix="spawn").Set(Gf.Vec3d(0, 0, spawn_offset))
    robot = Articulation(ROBOT_PATH)
    if robot.num_dofs <= 0:
        raise RuntimeError("G1 articulation has no degrees of freedom")

    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr().Set(1000.0)
    camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    camera.CreateFocalLengthAttr().Set(22.0)
    camera.CreateHorizontalApertureAttr().Set(20.955)
    camera.CreateVerticalApertureAttr().Set(11.787)
    camera.AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(3.2, 2.7, 2.0), Gf.Vec3d(0, 0, 0.6), Gf.Vec3d(0, 0, 1)).GetInverse()
    )
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No RTX viewport is available for screenshots")
    viewport.camera_path = CAMERA_PATH
    viewport.texture_resolution = (1280, 720)
    metadata = {
        "asset_usd": resolved_asset,
        "variant": VARIANT,
        "robot_path": ROBOT_PATH,
        "ground_path": GROUND_PATH,
        "ground_collision_paths": ground_colliders,
        "robot_collider_count": len(robot_colliders),
        "disabled_embedded_physics_scenes": disabled_scenes,
        "spawn_offset_m": spawn_offset,
        "ground_height_m": 0.0,
        "ground_visual_size_m": ground_size,
        "gravity_m_s2": 9.81,
        "camera_path": CAMERA_PATH,
        "physics_device": "cpu",
        "controller_enabled": False,
    }
    print(f"G1_SCENE_CREATED dofs={robot.num_dofs} colliders={len(robot_colliders)}", flush=True)
    return stage, robot, viewport, metadata


def follow_camera(stage: Usd.Stage, position) -> None:
    """Move the existing overview camera while keeping a fixed world heading."""
    x, y, _ = position
    camera = UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA_PATH))
    camera.GetOrderedXformOps()[0].Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(x + 2.6, y + 2.6, 1.6),
                              Gf.Vec3d(x, y, 0.65), Gf.Vec3d(0, 0, 1)).GetInverse())
