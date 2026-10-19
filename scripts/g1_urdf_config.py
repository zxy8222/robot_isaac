"""Mode-15 asset identity, official joint names, and immutable local paths."""

from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = "unitreerobotics/unitree_ros"
SOURCE_COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
SOURCE_ROOT = ROOT / ".runtime/unitree-ros" / SOURCE_COMMIT
DESCRIPTION = PurePosixPath("robots/g1_description")
STEM = "g1_29dof_mode_15"
VARIANT = "G1_29dof_mode_15"
URDF_PATH = SOURCE_ROOT / DESCRIPTION / f"{STEM}.urdf"
URDF_SHA256 = "80fc328761b0d0a502d675106c234eceabb70fa09c9f7d5cbcb40c2955629bac"
MANIFEST_PATH = SOURCE_ROOT / "g1-29dof-mode-15-source.json"
OUTPUT_DIR = ROOT / ".runtime/assets/unitree-g1-29dof-mode-15" / SOURCE_COMMIT
ASSET_PATH = OUTPUT_DIR / f"{STEM}.usd"
ROBOT_PATH = "/World/G1"

# Unitree motor order describes the supported asset; the ONNX channel order is
# derived separately from the pinned deployment YAML and official motor names.
JOINT_NAMES = (
    tuple(f"{side}_{joint}_joint" for side in ("left", "right")
          for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
    + tuple(f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch"))
    + tuple(f"{side}_{joint}_joint" for side in ("left", "right")
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                          "wrist_roll", "wrist_pitch", "wrist_yaw"))
)
