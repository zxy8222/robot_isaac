"""Pinned official URDF identities and local paths, independent of Isaac Sim."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from g1_policy import LEG_NAMES, SOURCE_COMMIT

ROOT = Path(__file__).resolve().parents[1]
MODE15_JOINTS = (LEG_NAMES + tuple(f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch"))
                 + tuple(f"{side}_{joint}_joint" for side in ("left", "right")
                         for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                       "wrist_roll", "wrist_pitch", "wrist_yaw")))


@dataclass(frozen=True)
class UrdfModel:
    key: str
    repository: str
    commit: str
    cache_name: str
    description: PurePosixPath
    stem: str
    urdf_sha256: str
    joint_names: tuple[str, ...]
    variant: str
    asset_directory: str
    manifest_name: str

    @property
    def source_root(self) -> Path:
        return ROOT / ".runtime" / self.cache_name / self.commit

    @property
    def urdf_path(self) -> Path:
        return self.source_root / self.description / f"{self.stem}.urdf"

    @property
    def manifest_path(self) -> Path:
        return self.source_root / self.manifest_name

    @property
    def output_dir(self) -> Path:
        return ROOT / ".runtime/assets" / self.asset_directory / self.commit

    @property
    def asset_path(self) -> Path:
        return self.output_dir / f"{self.stem}.usd"

    @property
    def robot_path(self) -> str:
        return "/World/UnitreeG1_12DOF" if self.key == "unitree12" else "/World/UnitreeG1_29DOF_Mode15"


URDF_MODELS = {
    "unitree12": UrdfModel(
        "unitree12", "unitreerobotics/unitree_rl_gym", SOURCE_COMMIT, "unitree-rl-gym",
        PurePosixPath("resources/robots/g1_description"), "g1_12dof",
        "52a0f729252cba764f610136f0772573e55694024b888da93abd5ad23aba0152", LEG_NAMES,
        "G1_12dof_URDF", "unitree-g1-12dof", "g1-12dof-source.json"),
    "unitree29": UrdfModel(
        "unitree29", "unitreerobotics/unitree_ros", "5994d4faef0a9cadd3287f8de0199a67eeb2a259", "unitree-ros",
        PurePosixPath("robots/g1_description"), "g1_29dof_mode_15",
        "80fc328761b0d0a502d675106c234eceabb70fa09c9f7d5cbcb40c2955629bac", MODE15_JOINTS,
        "G1_29dof_mode_15", "unitree-g1-29dof-mode-15", "g1-29dof-mode-15-source.json"),
}
