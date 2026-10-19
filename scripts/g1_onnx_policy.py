"""Official RL Lab G1 velocity policy: named joints and term-major history.

This module owns inference and deployment configuration, without Isaac Sim.
Gains in deploy.yaml use Unitree motor order; observations and actions use the
order obtained from joint_ids_map. Both become joint names before SDK mapping.
"""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from g1_outputs import file_digest
from g1_policy import world_to_body

SOURCE_COMMIT = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
SOURCE_URL = f"https://github.com/unitreerobotics/unitree_rl_lab/tree/{SOURCE_COMMIT}"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / ".runtime/unitree-rl-lab" / SOURCE_COMMIT
POLICY_DIRECTORY = "deploy/robots/g1_29dof/config/policy/velocity/v0"
POLICY_RELATIVE = f"{POLICY_DIRECTORY}/exported/policy.onnx"
CONFIG_RELATIVE = f"{POLICY_DIRECTORY}/params/deploy.yaml"
ASSET_CONFIG_RELATIVE = "source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py"
DEFAULT_POLICY_PATH = SOURCE_ROOT / POLICY_RELATIVE
POLICY_SHA256 = "610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f"
CONFIG_SHA256 = "64b04c0596a7010f39f8ac6e9ec46dc750141063cdcf2d42c83ecc444e57bc63"
ASSET_CONFIG_SHA256 = "5bb4d9b8cabfde167b0f905d0cfa63a2d8a6ab0209077c4885997edbc25c6a62"
OBSERVATION_TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands",
                     "joint_pos_rel", "joint_vel_rel", "last_action")


def verified_file(path: Path, digest: str) -> Path:
    if not path.is_file() or file_digest(path) != digest:
        raise ValueError(f"Official RL Lab file missing or changed: {path}; run fetch_g1_onnx.py")
    return path


def named_indices(actual_names, required_names) -> list[int]:
    """Require the full named model, accepting any Isaac Sim DOF permutation."""
    if len(actual_names) != len(set(actual_names)) or set(actual_names) != set(required_names):
        raise ValueError("Expected the complete, uniquely named 29-DOF G1 joint set")
    return [actual_names.index(name) for name in required_names]


class Deployment:
    """Read the pinned YAML and safely extract the official motor-name list."""

    def __init__(self) -> None:
        self.config = yaml.safe_load(verified_file(SOURCE_ROOT / CONFIG_RELATIVE, CONFIG_SHA256).read_text())
        source = verified_file(SOURCE_ROOT / ASSET_CONFIG_RELATIVE, ASSET_CONFIG_SHA256).read_text()
        assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "UNITREE_G1_29DOF_CFG"
                                  for target in node.targets))
        self.motor_names = tuple(ast.literal_eval(next(item.value for item in assignment.value.keywords
                                                       if item.arg == "joint_sdk_names")))
        mapping = self.config["joint_ids_map"]
        if sorted(mapping) != list(range(29)) or len(set(self.motor_names)) != 29:
            raise ValueError("Invalid official joint mapping")
        self.joint_names = tuple(self.motor_names[index] for index in mapping)
        # Kp/Kd are indexed by motor ID in State_RLBase::enter, not policy index.
        self.kp = np.asarray(self.config["stiffness"], dtype=np.float32)[mapping]
        self.kd = np.asarray(self.config["damping"], dtype=np.float32)[mapping]
        self.default = np.asarray(self.config["default_joint_pos"], dtype=np.float32)
        action = self.config["actions"]["JointPositionAction"]
        self.scale = np.asarray(action["scale"], dtype=np.float32)
        self.offset = np.asarray(action["offset"], dtype=np.float32)
        if action["clip"] is not None or action["joint_ids"] is not None:
            raise ValueError("Unexpected action clipping or joint subset")
        self.step_dt = float(self.config["step_dt"])
        if tuple(self.config["observations"]) != OBSERVATION_TERMS or self.step_dt != .02:
            raise ValueError("Unexpected observation terms or policy interval")
        for array in (self.kp, self.kd, self.default, self.scale, self.offset):
            if array.shape != (29,) or not np.isfinite(array).all():
                raise ValueError("Invalid official deployment vector")
        self.observation_scales = {}
        for name, term in self.config["observations"].items():
            size = 3 if name in OBSERVATION_TERMS[:3] else 29
            if term["history_length"] != 5 or term["clip"] is not None:
                raise ValueError("Unexpected official observation history or clipping")
            scale = np.asarray(term["scale"], dtype=np.float32)
            if scale.shape != (size,) or not np.isfinite(scale).all():
                raise ValueError("Invalid official observation scale")
            self.observation_scales[name] = scale


class G1OnnxPolicy:
    """Maintain five samples per observation term, oldest to newest, then infer."""

    def __init__(self, path: Path, dof_names, command) -> None:
        import onnxruntime as ort

        self.deployment = Deployment()
        self.joint_names = self.deployment.joint_names
        self.indices = named_indices(list(dof_names), self.joint_names)
        self.command = np.asarray(command, dtype=np.float32)
        bounds = self.deployment.config["commands"]["base_velocity"]["ranges"]
        for value, name in zip(self.command, ("lin_vel_x", "lin_vel_y", "ang_vel_z"), strict=True):
            if not np.isfinite(value) or not bounds[name][0] <= value <= bounds[name][1]:
                raise ValueError(f"Command outside official deployment range: {name}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(verified_file(path, POLICY_SHA256)), options,
                                           providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].shape != [1, 480] or inputs[0].type != "tensor(float)"
                or len(outputs) != 1 or outputs[0].shape != [1, 29]):
            raise ValueError("Unexpected official ONNX input/output contract")
        self.input_name = inputs[0].name
        self.history = None
        self.action = np.zeros(29, dtype=np.float32)
        self.targets = self.deployment.offset.copy()
        self.updates = 0
        self.last_observation = None

    def observation_terms(self, state) -> dict[str, np.ndarray]:
        state.values(len(state.joint_positions))
        terms = dict(zip(OBSERVATION_TERMS, (
            world_to_body(state.orientation, state.velocities[3:]),
            world_to_body(state.orientation, [0, 0, -1]), self.command,
            np.asarray(state.joint_positions)[self.indices]-self.deployment.default,
            np.asarray(state.joint_velocities)[self.indices], self.action.copy()), strict=True))
        return {name: np.asarray(value, dtype=np.float32)*self.deployment.observation_scales[name]
                for name, value in terms.items()}

    def update(self, state) -> np.ndarray:
        terms = self.observation_terms(state)
        if self.history is None:
            self.history = {name: deque([value.copy() for _ in range(5)], maxlen=5)
                            for name, value in terms.items()}
        for name, value in terms.items():
            self.history[name].append(value.copy())
        self.last_observation = np.concatenate([np.concatenate(self.history[name]) for name in OBSERVATION_TERMS])
        if self.last_observation.shape != (480,) or not np.isfinite(self.last_observation).all():
            raise ValueError("Invalid 480-dimensional RL Lab observation")
        action = self.session.run(None, {self.input_name: self.last_observation[None]})[0][0]
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError("Invalid ONNX policy output")
        self.action = action.copy()
        self.targets = self.action*self.deployment.scale+self.deployment.offset
        self.updates += 1
        return self.targets
