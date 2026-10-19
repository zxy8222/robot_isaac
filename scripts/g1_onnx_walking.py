"""Isaac Sim adapter for RL Lab ONNX and the unmodified Mode-15 actuator caps."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from g1_control_outputs import ControlSample
from g1_onnx_policy import CONFIG_SHA256, G1OnnxPolicy, POLICY_SHA256, SOURCE_URL, verified_file
import g1_urdf_config as asset


def mode15_limits(names):
    """Return original URDF limits in the requested *named* order."""
    source = ET.parse(verified_file(asset.URDF_PATH, asset.URDF_SHA256)).getroot()
    joints = {j.get("name"): j for j in source.findall("joint") if j.get("type") != "fixed"}
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(joints):
        raise ValueError("The Mode-15 controller requires all 29 named joints")
    return {key: np.array([float(joints[name].find("limit").get(key)) for name in names], dtype=np.float32)
            for key in ("lower", "upper", "effort", "velocity")}


def contact_metadata(robot, ground_paths):
    """Read composed collision/material settings, including instance proxies."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = robot.prims[0].GetStage()
    colliders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(asset.ROBOT_PATH), Usd.TraverseInstanceProxies())
                 if p.HasAPI(UsdPhysics.CollisionAPI)]
    feet = [p for p in colliders if p.IsA(UsdGeom.Sphere) and "ankle_roll_link" in str(p.GetPath())]

    def properties(prim):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        physics_material = UsdPhysics.MaterialAPI(material.GetPrim()) if material else None
        physx_material = PhysxSchema.PhysxMaterialAPI(material.GetPrim()) if material else None
        return {"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "contact_offset_m": PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get(),
                "rest_offset_m": PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get(),
                "material": str(material.GetPath()) if material else None,
                "static_friction": physics_material.GetStaticFrictionAttr().Get() if physics_material else None,
                "dynamic_friction": physics_material.GetDynamicFrictionAttr().Get() if physics_material else None,
                "restitution": physics_material.GetRestitutionAttr().Get() if physics_material else None,
                "friction_combine": physx_material.GetFrictionCombineModeAttr().Get() if physx_material else None,
                "radius_m": UsdGeom.Sphere(prim).GetRadiusAttr().Get() if prim.IsA(UsdGeom.Sphere) else None}

    if len(colliders) != 35 or len(feet) != 8:
        raise RuntimeError("Mode-15 collision model changed: expected 35 shapes and eight foot spheres")
    return {"robot_collision_shapes": len(colliders), "feet": [properties(p) for p in feet],
            "ground": [properties(stage.GetPrimAtPath(path)) for path in ground_paths],
            "modified_for_walking": False,
            "null_attribute_meaning": "Unauthored USD value or unbound material; PhysX resolves defaults at runtime."}


class OnnxWalkingController:
    """All 29 joints use fresh-state explicit PD at 500 Hz, ONNX at 50 Hz."""

    def __init__(self, robot, policy_path: Path, command, *, ground_paths):
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.experimental.utils.backend import use_backend
        from pxr import PhysxSchema

        self.robot = robot
        self.names = list(robot.dof_names)
        self.policy = G1OnnxPolicy(policy_path, self.names, command)
        self.control_names = self.policy.joint_names
        self.indices = self.policy.indices
        self.deployment = self.policy.deployment
        self.limits = mode15_limits(self.control_names)
        self.pd_interval = .002
        self.decimation = round(self.deployment.step_dt / self.pd_interval)
        self.maximum_speed = np.zeros(29)
        self.velocity_exceedance_steps = np.zeros(29, dtype=int)
        self.position_exceedance_steps = np.zeros(29, dtype=int)
        self.target_exceedance_steps = np.zeros(29, dtype=int)
        self.max_effort_readback_error = 0.0
        self.runtime_checks = 0
        self.contact = contact_metadata(robot, ground_paths)
        feet_paths = [robot.link_paths[0][robot.link_names.index(f"{side}_ankle_roll_link")]
                      for side in ("left", "right")]
        self.feet = RigidPrim(feet_paths, contact_filter_paths=ground_paths, max_contact_count=64)
        self.feet_paths = feet_paths
        self.feet.set_enabled_contact_tracking([True])
        for prim in robot.prims:
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
        with use_backend("usd"):
            robot.set_dof_drive_types("force")
            robot.set_dof_gains([0.0]*29, [0.0]*29)
            robot.set_dof_max_efforts(self.limits["effort"], dof_indices=self.indices)
            robot.set_dof_max_velocities(self.limits["velocity"], dof_indices=self.indices)
            robot.set_dof_armatures([.01]*29)
            robot.set_solver_iteration_counts([8], [4])
            robot.set_enabled_self_collisions([False])

    def verify_runtime(self):
        kp, kd = self.robot.get_dof_gains()
        if np.any(kp.numpy()) or np.any(kd.numpy()):
            raise RuntimeError("Explicit PD requires zero implicit PhysX Drive gains")
        for method, key in ((self.robot.get_dof_max_efforts, "effort"),
                            (self.robot.get_dof_max_velocities, "velocity")):
            if not np.allclose(method().numpy()[0, self.indices], self.limits[key], rtol=1e-6):
                raise RuntimeError(f"Mode-15 runtime {key} limits changed")
        lower, upper = self.robot.get_dof_limits()
        for actual, key in ((lower, "lower"), (upper, "upper")):
            if not np.allclose(actual.numpy()[0, self.indices], self.limits[key], atol=1e-5):
                raise RuntimeError("Mode-15 position limits changed")
        self.runtime_checks += 1

    def initialize(self):
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.simulation_manager import SimulationManager

        with use_backend("tensor"):
            if self.robot.mass_matrix_shape != (35, 35) or not self.robot.get_link_enabled_gravities().numpy().all():
                raise RuntimeError("The Mode-15 robot must have a floating base with gravity")
            self.verify_runtime()
            self.robot.set_world_poses([[0, 0, .8]], [[1, 0, 0, 0]])
            self.robot.set_velocities([[0, 0, 0]], [[0, 0, 0]])
            self.robot.set_dof_positions(self.deployment.default, dof_indices=self.indices)
            self.robot.set_dof_velocities([0.0]*29)
        # Public PhysX Tensor API resolves shape-dependent defaults absent in USD.
        feet_view = SimulationManager.get_physics_simulation_view().create_rigid_body_view(self.feet_paths)
        material = feet_view.get_material_properties().numpy().copy()
        contact = feet_view.get_contact_offsets().numpy().copy()
        rest = feet_view.get_rest_offsets().numpy().copy()
        if (feet_view.count != 2 or feet_view.max_shapes != 4
                or not all(np.isfinite(a).all() for a in (material, contact, rest))
                or np.any(material < 0) or np.any(contact < rest)):
            raise RuntimeError("Invalid resolved foot contact properties")
        self.contact["resolved_foot_properties"] = {
            "link_paths": list(feet_view.prim_paths), "shapes_per_link": feet_view.max_shapes,
            "material_columns": ["static_friction", "dynamic_friction", "restitution"],
            "material_properties": material.tolist(), "contact_offsets_m": contact.tolist(),
            "rest_offsets_m": rest.tolist(), "source": "PhysX Tensor API; read-only"}

    def apply(self, state):
        from isaacsim.core.experimental.utils.backend import use_backend

        if state.step % self.decimation == 0:
            self.policy.update(state)
        q = np.asarray(state.joint_positions)[self.indices]
        dq = np.asarray(state.joint_velocities)[self.indices]
        raw = self.deployment.kp*(self.policy.targets-q)-self.deployment.kd*dq
        torques = np.clip(raw, -self.limits["effort"], self.limits["effort"])
        with use_backend("tensor"):
            if state.step % 500 == 0:
                self.verify_runtime()
            self.robot.set_dof_efforts(torques, dof_indices=self.indices)
            sent = self.robot.get_dof_efforts().numpy()[0, self.indices]
            error = float(np.max(np.abs(sent-torques)))
            if error > 1e-4:
                raise RuntimeError("PhysX effort readback differs from the command")
            self.max_effort_readback_error = max(self.max_effort_readback_error, error)
        self.maximum_speed = np.maximum(self.maximum_speed, np.abs(dq))
        self.velocity_exceedance_steps += np.abs(dq) > self.limits["velocity"]+.01
        self.position_exceedance_steps += (q < self.limits["lower"]-.01) | (q > self.limits["upper"]+.01)
        self.target_exceedance_steps += (self.policy.targets < self.limits["lower"]) | (self.policy.targets > self.limits["upper"])
        return ControlSample(state.step, state.simulation_time, self.policy.updates, self.policy.action.copy(),
                             self.policy.targets.copy(), q, dq, raw, torques)

    def read_feet(self):
        from isaacsim.core.experimental.utils.backend import use_backend

        with use_backend("tensor"):
            positions, _ = self.feet.get_world_poses()
            forces = self.feet.get_net_contact_forces(dt=self.pd_interval)
        return positions.numpy().copy(), forces.numpy().copy()

    def metadata(self):
        return {"name": "unitree_rl_lab_g1_29dof_onnx", "source": SOURCE_URL,
                "robot_variant": "G1_29dof_mode_15", "policy_sha256": POLICY_SHA256,
                "deploy_yaml_sha256": CONFIG_SHA256,
                "policy_updates": self.policy.updates, "policy_hz": 50, "physics_hz": 500,
                "command_m_s_rad_s": self.policy.command.tolist(), "initial_base_height_m": .8,
                "observation_dimension": 480, "observation_history": "term-major; oldest to newest; five samples",
                "recurrent_state": "none; feedforward ONNX with explicit observation history",
                "joint_mapping": [{"name": name, "policy_index": i, "isaac_dof_index": self.indices[i],
                                   "unitree_motor_id": self.deployment.motor_names.index(name)}
                                  for i, name in enumerate(self.control_names)],
                "control_joint_order": list(self.control_names), "Kp": self.deployment.kp.tolist(),
                "Kd": self.deployment.kd.tolist(), "gain_source": "unmodified official deploy.yaml; named from motor order",
                "default_joint_positions": self.deployment.default.tolist(), "action_scale": self.deployment.scale.tolist(),
                "mode15_limits": {key: array.tolist() for key, array in self.limits.items()},
                "maximum_measured_joint_speed_rad_s": self.maximum_speed.tolist(),
                "velocity_exceedance_steps_tolerance_001_rad_s": self.velocity_exceedance_steps.tolist(),
                "position_exceedance_steps_tolerance_001_rad": self.position_exceedance_steps.tolist(),
                "target_outside_position_limit_steps": self.target_exceedance_steps.tolist(),
                "all_runtime_drive_gains_zero": True, "runtime_constraint_checks": self.runtime_checks,
                "maximum_effort_command_readback_error_Nm": self.max_effort_readback_error,
                "contact_model": self.contact, "armature": .01, "solver_iterations_position_velocity": [8, 4],
                "floating_base": True, "gravity_enabled": True, "self_collisions": False,
                "effort_caps_relaxed": False, "velocity_caps_relaxed": False}
