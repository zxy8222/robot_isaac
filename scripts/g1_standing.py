"""G1 static standing with PhysX implicit joint PD drives.

Posture, actuator groups, armatures and solver iterations are based on NVIDIA Isaac Lab
G1_CFG (BSD-3-Clause), rather than an assumed numeric joint order:
https://github.com/isaac-sim/IsaacLab/blob/76aeb425d214f04bd2895f5f956066c3ec716e81/source/isaaclab_assets/isaaclab_assets/robots/unitree.py#L257-L361
The upstream copyright and license are retained in LICENSES/IsaacLab-BSD-3-Clause.txt.

Drive gains and the initial height are adapted for static stance on the current
G1_23dof USD. The ankle effort cap follows the 50 Nm G1 ankle actuator limit
documented in G1_29DOF_CFG in the same official source. This implementation adds
base pose/velocity feedback; it does not run a pretrained locomotion policy.

The controller initializes the robot once, then sends joint targets only.
Gravity, the floating base and ground contact remain part of the simulation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from g1_outputs import StateSample

SOURCE_URL = "https://github.com/isaac-sim/IsaacLab/blob/76aeb425d214f04bd2895f5f956066c3ec716e81/source/isaaclab_assets/isaaclab_assets/robots/unitree.py#L257-L361"
INITIAL_BASE_HEIGHT_M = 0.72


@dataclass(frozen=True)
class JointDrive:
    position: float
    stiffness: float
    damping: float
    max_effort: float
    armature: float


def standing_drives(names: list[str]) -> list[JointDrive]:
    """Resolve the supported 37-DOF G1 configuration in the SDK's actual order."""
    settings = {"torso_joint": JointDrive(0.0, 1000.0, 40.0, 300.0, 0.01)}
    for side, sign in (("left", 1), ("right", -1)):
        for suffix, position, stiffness in (
            ("hip_pitch", -0.20, 200.0), ("hip_roll", 0.0, 150.0),
            ("hip_yaw", 0.0, 150.0), ("knee", 0.42, 200.0),
        ):
            settings[f"{side}_{suffix}_joint"] = JointDrive(position, stiffness * 5.0, 40.0, 300.0, 0.01)
        for suffix in ("ankle_pitch", "ankle_roll"):
            settings[f"{side}_{suffix}_joint"] = JointDrive(
                -0.23 if suffix == "ankle_pitch" else 0.0, 200.0, 20.0, 50.0, 0.01
            )
        for suffix, position in (
            ("shoulder_pitch", 0.35), ("shoulder_roll", sign * 0.16),
            ("shoulder_yaw", 0.0), ("elbow_pitch", 0.87), ("elbow_roll", 0.0),
            ("zero", 0.0), ("one", sign * 1.0), ("two", sign * 0.52),
            ("three", 0.0), ("four", 0.0), ("five", 0.0), ("six", 0.0),
        ):
            armature = 0.01 if suffix.startswith(("shoulder_", "elbow_")) else 0.001
            settings[f"{side}_{suffix}_joint"] = JointDrive(position, 40.0, 10.0, 300.0, armature)
    if len(names) != len(set(names)) or set(names) != set(settings):
        raise ValueError(f"Unsupported G1 joints; missing={sorted(set(settings) - set(names))}, "
                         f"unexpected={sorted(set(names) - set(settings))}")
    return [settings[name] for name in names]


@dataclass(frozen=True)
class BalanceGains:
    """Local ankle feedback added to the official nominal joint posture."""

    tilt: float = 0.5
    angular_velocity: float = 0.15
    position: float = 0.2
    linear_velocity: float = 0.1
    maximum_offset_rad: float = 0.35


def balance_offsets(state: StateSample, gains: BalanceGains) -> tuple[float, float]:
    """Return ankle pitch/roll offsets using measured base pose and velocity."""
    state.values(len(state.joint_positions))
    w, x, y, z = state.orientation
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = (value / norm for value in (w, x, y, z))
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    c, s = math.cos(yaw), math.sin(yaw)
    forward = c*state.position[0] + s*state.position[1]
    lateral = -s*state.position[0] + c*state.position[1]
    vx, vy, _, wx, wy, _ = state.velocities
    forward_speed, lateral_speed = c*vx + s*vy, -s*vx + c*vy
    roll_rate, pitch_rate = c*wx + s*wy, -s*wx + c*wy
    pitch_offset = (gains.tilt*pitch + gains.angular_velocity*pitch_rate
                    + gains.position*forward + gains.linear_velocity*forward_speed)
    roll_offset = (gains.tilt*roll + gains.angular_velocity*roll_rate
                   - gains.position*lateral - gains.linear_velocity*lateral_speed)
    return tuple(max(-gains.maximum_offset_rad, min(gains.maximum_offset_rad, value))
                 for value in (pitch_offset, roll_offset))


class StandingController:
    """Configure USD drives before physics, then refresh targets before each step."""

    def __init__(self, robot) -> None:
        from isaacsim.core.experimental.utils.backend import use_backend

        self.robot = robot
        self.names = list(robot.dof_names)
        self.drives = standing_drives(self.names)
        self.targets = [drive.position for drive in self.drives]
        self.balance_gains = BalanceGains()
        self.ankle_pitch_indices = [self.names.index(f"{side}_ankle_pitch_joint") for side in ("left", "right")]
        self.ankle_roll_indices = [self.names.index(f"{side}_ankle_roll_joint") for side in ("left", "right")]
        with use_backend("usd"):
            lower, upper = robot.get_dof_limits()
            for name, target, low, high in zip(self.names, self.targets, lower.numpy()[0], upper.numpy()[0]):
                if not low <= target <= high:
                    raise ValueError(f"Standing target exceeds joint limits: {name}={target}, [{low}, {high}]")
            robot.set_dof_drive_types("force")
            robot.set_dof_gains([drive.stiffness for drive in self.drives],
                                [drive.damping for drive in self.drives])
            robot.set_dof_max_efforts([drive.max_effort for drive in self.drives])
            robot.set_dof_armatures([drive.armature for drive in self.drives])
            robot.set_solver_iteration_counts([8], [4])
            robot.set_enabled_self_collisions([False])
            robot.set_dof_position_targets(self.targets)
            robot.set_dof_velocity_targets([0.0] * len(self.names))

    def initialize(self) -> None:
        """Set the initial state once; subsequent control never resets the base."""
        from isaacsim.core.experimental.utils.backend import use_backend

        with use_backend("tensor"):
            if self.robot.mass_matrix_shape != (len(self.names) + 6,) * 2:
                raise RuntimeError("Standing validation requires a floating-base articulation")
            if not self.robot.get_link_enabled_gravities().numpy().all():
                raise RuntimeError("Standing validation requires gravity on every robot link")
            # Set runtime gains through PhysX tensors in Nm/rad. USD angular
            # drive authoring uses a different conversion path in the SDK.
            self.robot.set_dof_gains([drive.stiffness for drive in self.drives],
                                     [drive.damping for drive in self.drives])
            self.robot.set_dof_max_efforts([drive.max_effort for drive in self.drives])
            self.robot.set_dof_armatures([drive.armature for drive in self.drives])
            actual_stiffness, actual_damping = self.robot.get_dof_gains()
            for index, drive in enumerate(self.drives):
                if (not math.isclose(float(actual_stiffness.numpy()[0, index]), drive.stiffness, rel_tol=1e-5)
                        or not math.isclose(float(actual_damping.numpy()[0, index]), drive.damping, rel_tol=1e-5)):
                    raise RuntimeError(f"PhysX drive gain mismatch: {self.names[index]}")
            print("G1_RUNTIME_PD_GAINS_VERIFIED", flush=True)
            self.robot.set_world_poses([[0.0, 0.0, INITIAL_BASE_HEIGHT_M]], [[1.0, 0.0, 0.0, 0.0]])
            self.robot.set_velocities([[0.0] * 3], [[0.0] * 3])
            self.robot.set_dof_positions(self.targets)
            self.robot.set_dof_velocities([0.0] * len(self.names))
        self.apply()

    def apply(self, state: StateSample | None = None) -> None:
        from isaacsim.core.experimental.utils.backend import use_backend

        targets = self.targets.copy()
        if state is not None:
            pitch_offset, roll_offset = balance_offsets(state, self.balance_gains)
            for index in self.ankle_pitch_indices:
                targets[index] += pitch_offset
            for index in self.ankle_roll_indices:
                targets[index] += roll_offset
        with use_backend("tensor"):
            self.robot.set_dof_position_targets(targets)
            self.robot.set_dof_velocity_targets([0.0] * len(self.names))

    def metadata(self) -> dict[str, object]:
        return {"name": "implicit_pd_standing_with_ankle_feedback", "source": SOURCE_URL,
                "floating_base": True, "gravity_enabled": True,
                "initial_base_height_m": INITIAL_BASE_HEIGHT_M,
                "balance_gains": asdict(self.balance_gains),
                "gain_adaptation": "Static stance: leg/torso stiffness x5, damping 40; ankle stiffness 200, damping 20, effort cap 50",
                "joint_drives": {name: asdict(drive) for name, drive in zip(self.names, self.drives)}}


@dataclass(frozen=True)
class StandingLimits:
    minimum_height_m: float = 0.55
    maximum_height_m: float = 0.90
    maximum_tilt_deg: float = 15.0
    maximum_drift_m: float = 0.25
    settling_time_s: float = 2.0
    maximum_joint_error_rad: float = 0.35
    maximum_settled_speed_m_s: float = 0.30
    maximum_settled_angular_speed_rad_s: float = 0.50


class StandingMonitor:
    """Fail on falls and record actual upright stability across the full run."""

    def __init__(self, initial: StateSample, targets: list[float]) -> None:
        self.origin = initial.position[:2]
        self.targets = targets
        self.limits = StandingLimits()
        self.min_height = self.limits.maximum_height_m
        self.max_height = 0.0
        self.max_tilt = self.max_drift = self.max_joint_error = 0.0
        self.max_settled_speed = self.max_settled_angular_speed = 0.0
        self.settled_samples = 0

    def observe(self, state: StateSample) -> None:
        state.values(len(self.targets))
        w, x, y, z = state.orientation
        norm_squared = w * w + x * x + y * y + z * z
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x*x + y*y) / norm_squared))))
        drift = math.hypot(state.position[0] - self.origin[0], state.position[1] - self.origin[1])
        height = state.position[2]
        self.min_height, self.max_height = min(self.min_height, height), max(self.max_height, height)
        self.max_tilt, self.max_drift = max(self.max_tilt, tilt), max(self.max_drift, drift)
        if not self.limits.minimum_height_m <= height <= self.limits.maximum_height_m:
            raise RuntimeError(f"Standing height failed at step {state.step}: {height:.4f} m")
        if tilt > self.limits.maximum_tilt_deg or drift > self.limits.maximum_drift_m:
            raise RuntimeError(f"Standing stability failed at step {state.step}: tilt={tilt:.2f} deg, drift={drift:.4f} m")
        if state.simulation_time >= self.limits.settling_time_s:
            error = max(abs(actual - target) for actual, target in zip(state.joint_positions, self.targets))
            speed = math.sqrt(sum(value * value for value in state.velocities[:3]))
            angular_speed = math.sqrt(sum(value * value for value in state.velocities[3:]))
            self.max_joint_error = max(self.max_joint_error, error)
            self.max_settled_speed = max(self.max_settled_speed, speed)
            self.max_settled_angular_speed = max(self.max_settled_angular_speed, angular_speed)
            self.settled_samples += 1
            if (error > self.limits.maximum_joint_error_rad or speed > self.limits.maximum_settled_speed_m_s
                    or angular_speed > self.limits.maximum_settled_angular_speed_rad_s):
                raise RuntimeError(f"Standing did not settle at step {state.step}: joint_error={error:.4f}, "
                                   f"speed={speed:.4f}, angular_speed={angular_speed:.4f}")

    def summary(self) -> dict[str, object]:
        if self.settled_samples == 0:
            raise RuntimeError("Standing run ended before settled stability could be verified")
        return {"status": "passed", "limits": asdict(self.limits),
                "minimum_base_height_m": self.min_height, "maximum_base_height_m": self.max_height,
                "maximum_tilt_deg": self.max_tilt, "maximum_horizontal_drift_m": self.max_drift,
                "maximum_settled_joint_error_rad": self.max_joint_error,
                "maximum_settled_speed_m_s": self.max_settled_speed,
                "maximum_settled_angular_speed_rad_s": self.max_settled_angular_speed,
                "settled_samples": self.settled_samples}
