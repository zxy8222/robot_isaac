"""Convert the fixed Mode-15 URDF and verify USD dependencies and PhysX properties.

Application lifecycle and scene preview belong to import_g1_29dof.py. This
module imports Isaac Sim APIs only inside the operations that require them.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

from fetch_g1_urdf import mesh_paths
from g1_outputs import file_digest
import g1_urdf_config as model


def verify_sources() -> dict:
    if not model.MANIFEST_PATH.is_file() or not model.URDF_PATH.is_file():
        raise RuntimeError("Run python scripts/fetch_g1_urdf.py before importing")
    manifest = json.loads(model.MANIFEST_PATH.read_text())
    if (manifest["repository"] != model.SOURCE_REPOSITORY or manifest["commit"] != model.SOURCE_COMMIT
            or file_digest(model.URDF_PATH) != model.URDF_SHA256):
        raise ValueError("The URDF does not match the pinned official source")
    mesh_paths(model.URDF_PATH)
    for item in manifest["files"]:
        path = (model.SOURCE_ROOT / item["path"]).resolve()
        if not path.is_relative_to(model.SOURCE_ROOT.resolve()) or file_digest(path) != item["sha256"]:
            raise ValueError(f"Source file changed: {item['path']}")
    return manifest


def convert_asset(output_dir: Path) -> tuple[Path, dict]:
    """Use Isaac Sim 6.0.1's importer and publish a stable relative-reference USD."""
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    config = URDFImporterConfig(
        urdf_path=str(model.URDF_PATH), usd_path=str(output_dir / "converted"),
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
    asset_path = output_dir / f"{model.STEM}.usd"
    asset = Usd.Stage.CreateNew(str(asset_path))
    UsdGeom.SetStageUpAxis(asset, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(asset, 1.0)
    root = UsdGeom.Xform.Define(asset, f"/{model.STEM}").GetPrim()
    root.GetReferences().AddReference(os.path.relpath(converted, output_dir))
    root.SetCustomDataByKey("unitree_source_repository", model.SOURCE_REPOSITORY)
    root.SetCustomDataByKey("unitree_source_commit", model.SOURCE_COMMIT)
    root.SetCustomDataByKey("unitree_source_urdf_sha256", model.URDF_SHA256)
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


def validate_model(robot, stage) -> dict:
    """Check actual PhysX DOFs, limits, effort caps, inertia and floating-base state."""
    import numpy as np
    from isaacsim.core.experimental.utils.backend import use_backend
    from pxr import Usd, UsdGeom, UsdPhysics

    source = ET.parse(model.URDF_PATH).getroot()
    joints = {j.attrib["name"]: j for j in source.findall("joint") if j.attrib["type"] != "fixed"}
    names = list(robot.dof_names)
    if len(names) != len(model.JOINT_NAMES) or set(names) != set(model.JOINT_NAMES):
        raise RuntimeError(f"The imported joints differ from {model.STEM}: {names}")
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
    prims = list(Usd.PrimRange(stage.GetPrimAtPath(model.ROBOT_PATH), Usd.TraverseInstanceProxies()))
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
