"""Convert the official Unitree URDF and validate it beside the NVIDIA G1 USD.

The source cache is read-only during conversion. All generated assets, the
comparison scene and verification evidence stay in a fresh output directory.
The existing G1 scene builder and walking controller are not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from fetch_g1_urdf import mesh_paths
from g1_outputs import capture_png, file_digest, write_json
from g1_urdf_config import URDF_MODELS, UrdfModel


def parse_args(model: UrdfModel):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=model.output_dir)
    args = parser.parse_args()
    allowed = Path(__file__).resolve().parents[1] / ".runtime/assets"
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == allowed or not args.output_dir.is_relative_to(allowed):
        parser.error(f"Import output must be inside {allowed}")
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing imports are preserved")
    return args


def verify_sources(model: UrdfModel) -> dict:
    if not model.manifest_path.is_file() or not model.urdf_path.is_file():
        raise RuntimeError(f"Run python scripts/fetch_g1_urdf.py --model {model.key} before importing")
    manifest = json.loads(model.manifest_path.read_text())
    if (manifest["repository"] != model.repository or manifest["commit"] != model.commit
            or file_digest(model.urdf_path) != model.urdf_sha256):
        raise ValueError("The URDF does not match the pinned official source")
    mesh_paths(model.urdf_path, model.joint_names)
    for item in manifest["files"]:
        path = (model.source_root / item["path"]).resolve()
        if not path.is_relative_to(model.source_root.resolve()) or file_digest(path) != item["sha256"]:
            raise ValueError(f"Source file changed: {item['path']}")
    return manifest


def convert_asset(output_dir: Path, model: UrdfModel) -> tuple[Path, dict]:
    """Use Isaac Sim 6.0.1's importer and publish a stable relative-reference USD."""
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    config = URDFImporterConfig(
        urdf_path=str(model.urdf_path), usd_path=str(output_dir / "converted"),
        merge_fixed_joints=True, merge_mesh=False, collision_from_visuals=False,
        allow_self_collision=False, fix_base=False, robot_type="Humanoid",
        joint_drive_type="force", joint_target_type="none",
        override_joint_stiffness=0.0, override_joint_damping=0.0,
        run_asset_transformer=True, run_multi_physics_conversion=True,
    )
    converted = Path(URDFImporter(config).import_urdf()).resolve()
    source_stage = Usd.Stage.Open(str(converted))
    if not source_stage or not source_stage.GetDefaultPrim():
        raise RuntimeError("URDF importer did not produce a valid default prim")
    if UsdGeom.GetStageUpAxis(source_stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Imported model is not Z-up")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(source_stage), 1.0):
        raise RuntimeError("Imported model is not expressed in meters")
    asset_path = output_dir / f"{model.stem}.usd"
    asset = Usd.Stage.CreateNew(str(asset_path))
    UsdGeom.SetStageUpAxis(asset, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(asset, 1.0)
    root = UsdGeom.Xform.Define(asset, f"/{model.stem}").GetPrim()
    root.GetReferences().AddReference(os.path.relpath(converted, output_dir))
    root.SetCustomDataByKey("unitree_source_repository", model.repository)
    root.SetCustomDataByKey("unitree_source_commit", model.commit)
    root.SetCustomDataByKey("unitree_source_urdf_sha256", model.urdf_sha256)
    asset.SetDefaultPrim(root)
    asset.GetRootLayer().Save()
    layers, dependencies, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(asset_path)))
    if unresolved:
        raise RuntimeError(f"Imported USD has unresolved dependencies: {unresolved}")
    external = [str(path) for path in [*[layer.realPath for layer in layers if layer.realPath], *dependencies]
                if not Path(path).resolve().is_relative_to(output_dir)]
    if external:
        raise RuntimeError(f"Imported asset references files outside its output directory: {external}")
    return asset_path, {"config": asdict(config), "converter_output": str(converted),
                        "usd_layers": [layer.realPath for layer in layers],
                        "asset_dependencies": list(dependencies), "unresolved_dependencies": []}


def build_comparison(asset_path: Path, update, model: UrdfModel):
    """Place the two independent assets side by side, retaining the existing loader."""
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from pxr import Gf, UsdGeom
    from g1_scene import build_scene, wait_for_assets

    omni.usd.get_context().new_stage()
    update()
    stage, nvidia, viewport, metadata = build_scene(update)
    root = UsdGeom.Xform.Define(stage, model.robot_path).GetPrim()
    root.GetReferences().AddReference(str(asset_path))
    wait_for_assets(update)
    bounds = UsdGeom.BBoxCache(0, ["default", "render", "proxy"])
    box = bounds.ComputeWorldBound(root).ComputeAlignedBox()
    if box.IsEmpty() or not math.isfinite(float(box.GetMin()[2])):
        raise RuntimeError("The imported robot has no visible geometry")
    height = 0.08 - float(box.GetMin()[2])
    UsdGeom.Xformable(root).AddTranslateOp(opSuffix="comparison").Set(Gf.Vec3d(0, 1.6, height))
    imported = Articulation(model.robot_path)
    camera = UsdGeom.Xformable(stage.GetPrimAtPath("/World/OverviewCamera"))
    camera.GetOrderedXformOps()[0].Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(3.8, 4.2, 2.2), Gf.Vec3d(0, 0.8, 0.65),
                              Gf.Vec3d(0, 0, 1)).GetInverse())
    return stage, nvidia, imported, viewport, metadata


def validate_model(robot, stage, model: UrdfModel) -> dict:
    """Check actual PhysX DOFs, limits, effort caps, inertia and floating-base state."""
    import numpy as np
    from isaacsim.core.experimental.utils.backend import use_backend
    from pxr import Usd, UsdGeom, UsdPhysics

    source = ET.parse(model.urdf_path).getroot()
    joints = {j.attrib["name"]: j for j in source.findall("joint") if j.attrib["type"] != "fixed"}
    names = list(robot.dof_names)
    if len(names) != len(model.joint_names) or set(names) != set(model.joint_names):
        raise RuntimeError(f"The imported joints differ from {model.stem}: {names}")
    with use_backend("tensor"):
        lower, upper = robot.get_dof_limits()
        stiffness, damping = robot.get_dof_gains()
        efforts = robot.get_dof_max_efforts()
        velocities = robot.get_dof_max_velocities()
        masses = robot.get_link_masses()
        inertias = robot.get_link_inertias()
        gravity = robot.get_link_enabled_gravities()
        com_positions, com_orientations = robot.get_link_coms()
    if (robot.num_links != len(names)+1 or robot.mass_matrix_shape != (len(names)+6,) * 2
            or not gravity.numpy().all()):
        raise RuntimeError("The imported articulation must have a floating base and enabled gravity")
    if np.any(stiffness.numpy()) or np.any(damping.numpy()):
        raise RuntimeError("The imported torque-control joints have nonzero drive gains")
    joint_results = {}
    for i, name in enumerate(names):
        limit = joints[name].find("limit").attrib
        actual = [float(lower.numpy()[0,i]), float(upper.numpy()[0,i]),
                  float(efforts.numpy()[0,i]), float(velocities.numpy()[0,i])]
        expected = [float(limit[key]) for key in ("lower", "upper", "effort", "velocity")]
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
            raise RuntimeError(f"Imported joint limits differ from URDF: {name}: {actual} != {expected}")
        joint_results[name] = {"sdk_index": i, "limits_rad": actual[:2], "effort_limit_Nm": actual[2],
                               "velocity_limit_rad_s": actual[3], "drive_stiffness": 0, "drive_damping": 0,
                               "source_parent_link": joints[name].find("parent").attrib["link"],
                               "source_child_link": joints[name].find("child").attrib["link"],
                               "source_axis": joints[name].find("axis").attrib["xyz"],
                               "source_origin": joints[name].find("origin").attrib}
    source_mass = sum(float(link.find("inertial/mass").attrib["value"])
                      for link in source.findall("link") if link.find("inertial/mass") is not None)
    actual_mass = float(masses.numpy().sum())
    if not math.isclose(actual_mass, source_mass, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"Imported mass differs from URDF: {actual_mass} != {source_mass}")
    eigenvalues = np.linalg.eigvalsh(inertias.numpy()[0].reshape(-1,3,3))
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise RuntimeError("Imported link inertias must be finite and positive")
    prims = list(Usd.PrimRange(stage.GetPrimAtPath(model.robot_path), Usd.TraverseInstanceProxies()))
    colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    spheres = [prim for prim in colliders if prim.IsA(UsdGeom.Sphere)]
    if len(colliders) != len(source.findall("link/collision")) or len(spheres) != 8:
        raise RuntimeError(f"Expected preserved collisions and eight foot spheres; got {len(colliders)}, {len(spheres)}")
    for sphere in spheres:
        if not math.isclose(UsdGeom.Sphere(sphere).GetRadiusAttr().Get(), .005, rel_tol=1e-5):
            raise RuntimeError("Foot sphere radius differs from the official URDF")
    return {"num_dofs": len(names), "dof_names": names, "num_links": robot.num_links,
            "mass_matrix_shape": list(robot.mass_matrix_shape), "floating_base": True, "gravity_enabled": True,
            "joint_properties": joint_results, "total_mass_kg": actual_mass, "source_total_mass_kg": source_mass,
            "link_names": list(robot.link_names), "link_masses_kg": masses.numpy()[0].tolist(),
            "link_inertias": inertias.numpy()[0].tolist(), "positive_inertia_eigenvalues": True,
            "link_com_positions_m": com_positions.numpy()[0].tolist(),
            "link_com_orientations_wxyz": com_orientations.numpy()[0].tolist(),
            "collision_shape_count": len(colliders), "foot_collision_sphere_count": len(spheres)}


def main(model_key: str = "unitree12") -> int:
    model = URDF_MODELS[model_key]
    args = parse_args(model)
    manifest = verify_sources(model)
    if version("isaacsim") != "6.0.1.0":
        raise RuntimeError("This import requires the configured Isaac Sim 6.0.1.0 environment")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "LICENSE.txt").open("x") as license_file:
        license_file.write((model.source_root / "LICENSE").read_text())
    write_json(args.output_dir / "source-manifest.json", manifest)
    app = timeline = None
    status = 1
    try:
        sys.argv = [sys.argv[0], "--portable-root", str(args.portable_root)]
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0,
                             "limit_cpu_threads": 4, "width": 1280, "height": 720})
        import omni.timeline
        import omni.kit.app
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.simulation_manager import SimulationManager

        def update():
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped during URDF import")
            app.update()

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        update()
        asset, conversion = convert_asset(args.output_dir, model)
        print(f"UNITREE_G1_USD_CREATED={asset}", flush=True)
        stage, nvidia, imported, viewport, nvidia_metadata = build_comparison(asset, update, model)
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        timeline.commit()
        comparison = args.output_dir / "comparison.usda"
        if not stage.GetRootLayer().Export(str(comparison)):
            raise RuntimeError("Failed to export the comparison scene")
        SimulationManager.setup_simulation(dt=.002, device="cpu")
        SimulationManager.enable_fabric(False)
        SimulationManager.initialize_physics()
        for _ in range(30):
            update()
        imported_result = validate_model(imported, stage, model)
        if nvidia.num_dofs != 37:
            raise RuntimeError("The existing NVIDIA model did not retain its 37 DOFs")
        before = SimulationManager.get_num_physics_steps()
        preview = capture_png(viewport, update, args.output_dir / "comparison.png")
        if SimulationManager.get_num_physics_steps() != before:
            raise RuntimeError("Preview rendering unexpectedly advanced physics")
        for _ in range(20):
            SimulationManager.step(steps=1)
            with use_backend("tensor"):
                for robot in (nvidia, imported):
                    positions, orientations = robot.get_world_poses()
                    if not all(math.isfinite(float(value)) for value in positions.numpy().flat):
                        raise RuntimeError("A robot has non-finite physics state")
                    if not all(math.isfinite(float(value)) for value in orientations.numpy().flat):
                        raise RuntimeError("A robot has non-finite orientation")
                    for state in (robot.get_dof_positions(), robot.get_dof_velocities()):
                        if not all(math.isfinite(float(value)) for value in state.numpy().flat):
                            raise RuntimeError("A robot has non-finite joint state")
        if SimulationManager.get_num_physics_steps()-before != 20:
            raise RuntimeError("Physics smoke-test step count mismatch")
        verify_sources(model)
        write_json(args.output_dir / "import-summary.json", {
            "status": "passed", "isaacsim_version": version("isaacsim"),
            "robot_variant": model.variant, "source_commit": model.commit,
            "source_urdf": str(model.urdf_path), "source_urdf_sha256": model.urdf_sha256,
            "source_manifest": manifest, "asset_usd": str(asset), "conversion": conversion,
            "imported_robot_prim": model.robot_path, "imported_model": imported_result,
            "retained_nvidia_model": {"asset_usd": nvidia_metadata["asset_usd"], "prim": "/World/G1", "num_dofs": 37},
            "comparison_scene": str(comparison), "preview": preview,
            "physics_smoke_steps": 20, "physics_smoke_duration_s": .04,
            "acceptance": "Asset conversion, dependency resolution, exact source joint set, limits, mass, inertia validity, floating base, collisions and bounded physics; no standing or walking acceptance.",
            "source_sha256": {name: file_digest(Path(__file__).parent / name) for name in
                              ("g1_urdf_config.py", "g1_urdf_import.py", "fetch_g1_urdf.py")},
            "files_sha256": {str(p.relative_to(args.output_dir)): file_digest(p)
                              for p in args.output_dir.rglob("*") if p.is_file()},
        })
        print(f"UNITREE_G1_IMPORT_PASSED dofs={len(model.joint_names)} mass_kg={imported_result['total_mass_kg']:.6f}", flush=True)
        print(f"UNITREE_G1_COMPARISON={comparison}", flush=True)
        status = 0
        return 0
    except Exception as error:
        traceback.print_exc()
        write_json(args.output_dir / "import-failure.json", {"error": str(error), "type": type(error).__name__})
        return 1
    finally:
        if timeline is not None:
            timeline.stop()
            timeline.commit()
        if app is not None:
            print(f"UNITREE_G1_CLOSING exit_status={status}", flush=True)
            app.close(exit_code=status)


if __name__ == "__main__":
    raise SystemExit(main())
