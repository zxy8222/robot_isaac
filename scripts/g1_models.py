"""SDK-independent model selection and walking actuator configuration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_policy import DEFAULT_ANGLES, EFFORT_LIMITS, LEG_NAMES, SOURCE_COMMIT, leg_indices

DEFAULT_UNITREE_USD = (Path(__file__).resolve().parents[1] / ".runtime/assets/unitree-g1-12dof"
                       / SOURCE_COMMIT / "g1_12dof.usd")


@dataclass(frozen=True)
class WalkingModel:
    variant: str
    initial_base_height_m: float
    positions: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limits: np.ndarray


def walking_model(names: list[str]) -> WalkingModel:
    """Resolve both supported models by exact joint names, regardless of SDK order."""
    legs = leg_indices(names)
    upper = {"torso_joint"} | {f"{side}_{joint}_joint" for side in ("left", "right")
        for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll",
                      "zero", "one", "two", "three", "four", "five", "six")}
    if set(names) == set(LEG_NAMES):
        variant, height = "G1_12dof_URDF", 0.793
    elif set(names) == set(LEG_NAMES) | upper:
        variant, height = "G1_23dof", 0.755
    else:
        raise ValueError("Walking supports the official 12-DOF G1 or NVIDIA's 37-DOF G1_23dof")
    positions = np.zeros(len(names), dtype=np.float32)
    stiffness = np.full(len(names), 200.0, dtype=np.float32)
    damping = np.full(len(names), 10.0, dtype=np.float32)
    limits = np.full(len(names), 50.0, dtype=np.float32)
    positions[legs] = DEFAULT_ANGLES
    stiffness[legs] = damping[legs] = 0.0
    limits[legs] = EFFORT_LIMITS
    return WalkingModel(variant, height, positions, stiffness, damping, limits)
