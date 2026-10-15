"""Unitree's 47-observation / 12-action G1 policy, independent of Isaac Sim.

Deployment constants and observation layout follow unitree_rl_gym at SOURCE_COMMIT.
Copyright and BSD-3-Clause notice: LICENSES/Unitree-RL-Gym-BSD-3-Clause.txt.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from g1_outputs import StateSample, file_digest

SOURCE_COMMIT = "276801e46c5d433564f24658bac64f254b7d2d4b"
SOURCE_URL = f"https://github.com/unitreerobotics/unitree_rl_gym/tree/{SOURCE_COMMIT}"
POLICY_URL = f"https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/{SOURCE_COMMIT}/deploy/pre_train/g1/motion.pt"
POLICY_SHA256 = "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / ".runtime/policies/unitree-g1" / SOURCE_COMMIT / "motion.pt"
PHYSICS_DT = 0.002
DECIMATION = 10
LEG_NAMES = tuple(f"{side}_{joint}_joint" for side in ("left", "right")
                  for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=np.float32)
KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float32)
KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float32)
EFFORT_LIMITS = np.array([88, 139, 88, 139, 50, 50] * 2, dtype=np.float32)


def leg_indices(names: list[str]) -> list[int]:
    if len(names) != len(set(names)) or not set(LEG_NAMES).issubset(names):
        raise ValueError("G1 requires twelve uniquely named leg joints")
    return [names.index(name) for name in LEG_NAMES]


def world_to_body(quaternion, vector) -> np.ndarray:
    """Rotate a world vector into the body frame (quaternion order w,x,y,z)."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Expected a finite, nonzero wxyz quaternion")
    q /= norm
    w, xyz = q[0], q[1:]
    vector = np.asarray(vector, dtype=np.float64)
    return vector - 2*w*np.cross(xyz, vector) + 2*np.cross(xyz, np.cross(xyz, vector))


def observation(state: StateSample, indices: list[int], command, previous_action) -> np.ndarray:
    state.values(len(state.joint_positions))
    phase = 2 * math.pi * ((state.simulation_time % 0.8) / 0.8)
    return np.concatenate((
        world_to_body(state.orientation, state.velocities[3:]) * 0.25,
        world_to_body(state.orientation, [0, 0, -1]),
        np.asarray(command) * [2.0, 2.0, 0.25],
        np.asarray(state.joint_positions)[indices] - DEFAULT_ANGLES,
        np.asarray(state.joint_velocities)[indices] * 0.05,
        previous_action, [math.sin(phase), math.cos(phase)],
    )).astype(np.float32)


class G1Policy:
    """Own recurrent policy memory; inference produces targets in official order."""

    def __init__(self, path: Path, names: list[str], command: tuple[float, float, float]) -> None:
        import torch

        if not path.is_file() or file_digest(path) != POLICY_SHA256:
            raise ValueError(f"Official policy is missing or has an unexpected SHA256: {path}; run fetch_g1_policy.py")
        self.indices = leg_indices(names)
        self.command = command
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.model.reset_memory()
        self.action = np.zeros(12, dtype=np.float32)
        self.targets = DEFAULT_ANGLES.copy()
        self.updates = 0

    def update(self, state: StateSample) -> np.ndarray:
        import torch

        obs = observation(state, self.indices, self.command, self.action)
        if obs.shape != (47,) or not np.isfinite(obs).all():
            raise ValueError("Invalid G1 policy observation")
        with torch.inference_mode():
            action = self.model(torch.from_numpy(obs).unsqueeze(0)).numpy()[0].copy()
        if action.shape != (12,) or not np.isfinite(action).all():
            raise RuntimeError("G1 policy produced an invalid action")
        self.action = action
        self.targets = DEFAULT_ANGLES + 0.25 * action
        self.updates += 1
        return self.targets
