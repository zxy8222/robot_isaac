"""Isaac Sim adapter: map G1 joints, hold the upper body, apply policy PD torques.

Only initialization writes robot state. Each subsequent step applies joint
efforts/targets; root motion comes from floating-base dynamics and foot contact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from g1_policy import DEFAULT_ANGLES, DECIMATION, EFFORT_LIMITS, G1Policy, KD, KP, LEG_NAMES, POLICY_SHA256, SOURCE_URL


class WalkingController:
    def __init__(self, robot, policy_path: Path, command=(0.5, 0.0, 0.0), *, ground_paths=None) -> None:
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.experimental.utils.backend import use_backend

        self.robot = robot
        self.names = list(robot.dof_names)
        self.policy = G1Policy(policy_path, self.names, command)
        self.legs = self.policy.indices
        expected_upper = {"torso_joint"} | {f"{side}_{joint}_joint" for side in ("left", "right")
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll",
                          "zero", "one", "two", "three", "four", "five", "six")}
        if set(self.names) != set(LEG_NAMES) | expected_upper:
            raise ValueError("Walking adapter supports the current 37-DOF G1_23dof model")
        self.targets = np.zeros(len(self.names), dtype=np.float32)
        self.targets[self.legs] = DEFAULT_ANGLES
        self.stiffness = np.full(len(self.names), 200.0, dtype=np.float32)
        self.damping = np.full(len(self.names), 10.0, dtype=np.float32)
        self.stiffness[self.legs] = self.damping[self.legs] = 0.0
        self.effort_limits = np.full(len(self.names), 50.0, dtype=np.float32)
        self.effort_limits[self.legs] = EFFORT_LIMITS
        self.feet = RigidPrim([f"/World/G1/{side}_ankle_roll_link" for side in ("left", "right")],
                              contact_filter_paths=ground_paths or ["/World/Ground/collisionPlane"], max_contact_count=64)
        self.feet.set_enabled_contact_tracking([True])
        with use_backend("usd"):
            robot.set_dof_drive_types("force")
            robot.set_dof_gains(self.stiffness, self.damping)
            robot.set_dof_max_efforts(self.effort_limits)
            robot.set_dof_armatures([0.01] * len(self.names))
            robot.set_solver_iteration_counts([8], [4])
            robot.set_enabled_self_collisions([False])

    def initialize(self) -> None:
        from isaacsim.core.experimental.utils.backend import use_backend

        with use_backend("tensor"):
            if self.robot.mass_matrix_shape != (len(self.names) + 6,) * 2:
                raise RuntimeError("Walking requires a floating base")
            if not self.robot.get_link_enabled_gravities().numpy().all():
                raise RuntimeError("Walking requires gravity on all links")
            self.robot.set_dof_gains(self.stiffness, self.damping)
            self.robot.set_dof_max_efforts(self.effort_limits)
            self.robot.set_dof_armatures([0.01] * len(self.names))
            actual_kp, actual_kd = self.robot.get_dof_gains()
            if (not np.allclose(actual_kp.numpy()[0], self.stiffness)
                    or not np.allclose(actual_kd.numpy()[0], self.damping)):
                raise RuntimeError("PhysX drive gains differ from the walking configuration")
            self.robot.set_world_poses([[0, 0, 0.755]], [[1, 0, 0, 0]])
            self.robot.set_velocities([[0, 0, 0]], [[0, 0, 0]])
            self.robot.set_dof_positions(self.targets)
            self.robot.set_dof_velocities([0.0] * len(self.names))
            self.robot.set_dof_position_targets(self.targets)
            self.robot.set_dof_velocity_targets([0.0] * len(self.names))

    def apply(self, state) -> None:
        from isaacsim.core.experimental.utils.backend import use_backend

        if state.step > 0 and state.step % DECIMATION == 0:
            self.policy.update(state)
        q = np.asarray(state.joint_positions)[self.legs]
        dq = np.asarray(state.joint_velocities)[self.legs]
        torques = np.clip(KP*(self.policy.targets-q) - KD*dq, -EFFORT_LIMITS, EFFORT_LIMITS)
        with use_backend("tensor"):
            self.robot.set_dof_efforts(torques, dof_indices=self.legs)

    def metadata(self) -> dict:
        return {"name": "unitree_g1_recurrent_walking", "source": SOURCE_URL,
                "policy_sha256": POLICY_SHA256, "policy_updates": self.policy.updates,
                "command_m_s_rad_s": list(self.policy.command), "leg_joint_order": list(LEG_NAMES),
                "physics_hz": 500, "policy_hz": 50, "floating_base": True, "gravity_enabled": True,
                "upper_body": "zero joint targets with implicit PD; leg joints use explicit PD efforts"}

    def read_feet(self):
        """Return measured ankle-link positions and total contact forces in L/R order."""
        from isaacsim.core.experimental.utils.backend import use_backend
        from g1_policy import PHYSICS_DT

        with use_backend("tensor"):
            positions, _ = self.feet.get_world_poses()
            forces = self.feet.get_net_contact_forces(dt=PHYSICS_DT)
        return positions.numpy().copy(), forces.numpy().copy()
