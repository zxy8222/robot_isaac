"""Verify named ONNX deployment, historical observations and Mode-15 caps."""

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_onnx_policy import DEFAULT_POLICY_PATH, Deployment, G1OnnxPolicy, named_indices
from g1_onnx_walking import mode15_limits
from g1_outputs import StateSample
from g1_urdf_config import JOINT_NAMES


class TestOfficialDeployment(unittest.TestCase):
    def test_motor_gains_and_policy_defaults_are_mapped_by_different_name_orders(self):
        deployment = Deployment()
        for name, kp, kd in (("left_knee_joint", 150, 4), ("right_knee_joint", 150, 4),
                             ("waist_yaw_joint", 200, 5), ("left_ankle_pitch_joint", 40, 2),
                             ("right_shoulder_pitch_joint", 40, 10)):
            index = deployment.joint_names.index(name)
            self.assertEqual((deployment.kp[index], deployment.kd[index]), (kp, kd))
        self.assertEqual(deployment.default[deployment.joint_names.index("left_shoulder_pitch_joint")], np.float32(.3))
        self.assertEqual(deployment.joint_names[1], "right_hip_pitch_joint")
        self.assertEqual(deployment.motor_names[1], "left_hip_roll_joint")

    def test_unknown_missing_or_duplicate_joints_cannot_silently_map(self):
        for names in (list(JOINT_NAMES)[:-1], list(JOINT_NAMES)+["finger_joint"],
                      list(JOINT_NAMES)[:-1]+[JOINT_NAMES[0]]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                named_indices(names, JOINT_NAMES)

    def test_original_mode15_torque_and_velocity_caps_follow_names(self):
        names = list(JOINT_NAMES)[::-1]
        limits = mode15_limits(names)
        for side in ("left", "right"):
            for joint in ("ankle_pitch", "ankle_roll"):
                index = names.index(f"{side}_{joint}_joint")
                self.assertEqual(limits["effort"][index], 35)
                self.assertEqual(limits["velocity"][index], 30)
            self.assertEqual(limits["effort"][names.index(f"{side}_hip_pitch_joint")], 139)
            self.assertEqual(limits["velocity"][names.index(f"{side}_hip_pitch_joint")], 20)


class TestOfficialOnnx(unittest.TestCase):
    def policy(self, names=JOINT_NAMES, command=(.5, 0, 0)):
        return G1OnnxPolicy(DEFAULT_POLICY_PATH, list(names), command)

    def state(self, policy):
        q = np.empty(29)
        q[policy.indices] = policy.deployment.default
        return StateSample(0, 0., [0, 0, .8], [1, 0, 0, 0], [0.]*6, q, np.zeros(29))

    def test_dof_permutation_preserves_observation_action_and_named_targets(self):
        a, b = self.policy(), self.policy(JOINT_NAMES[::-1])
        sa, sb = self.state(a), self.state(b)
        for step in range(4):
            qa, qb = np.asarray(sa.joint_positions).copy(), np.asarray(sb.joint_positions).copy()
            qa[a.indices] += np.arange(29)*.001*step
            qb[b.indices] += np.arange(29)*.001*step
            sa = replace(sa, step=step*10, simulation_time=step*.02, joint_positions=qa)
            sb = replace(sb, step=step*10, simulation_time=step*.02, joint_positions=qb)
            np.testing.assert_array_equal(a.update(sa), b.update(sb))
            np.testing.assert_array_equal(a.last_observation, b.last_observation)
            self.assertEqual(a.last_observation.shape, (480,))

    def test_term_major_history_oldest_to_newest_and_previous_raw_action(self):
        policy = self.policy()
        state = self.state(policy)
        policy.update(state)
        first = policy.last_observation.copy()
        np.testing.assert_array_equal(first[:15], np.zeros(15))
        np.testing.assert_array_equal(first[15:30], np.tile([0, 0, -1], 5))
        np.testing.assert_array_equal(first[30:45], np.tile([.5, 0, 0], 5))
        np.testing.assert_array_equal(first[335:], np.zeros(145))
        previous = policy.action.copy()
        state = replace(state, step=10, simulation_time=.02, velocities=[0, 0, 0, 1, 2, 3])
        policy.update(state)
        np.testing.assert_array_equal(policy.last_observation[:12], np.zeros(12))
        np.testing.assert_array_equal(policy.last_observation[12:15], np.array([.2,.4,.6], np.float32))
        np.testing.assert_array_equal(policy.last_observation[335:451], np.zeros(116))
        np.testing.assert_array_equal(policy.last_observation[451:], previous)
        np.testing.assert_array_equal(policy.targets, policy.deployment.offset+.25*policy.action)

    def test_new_run_resets_history_and_rejects_corrupt_policy_and_bad_command(self):
        a = self.policy()
        state = self.state(a)
        initial = a.update(state).copy()
        a.update(replace(state, step=10, simulation_time=.02))
        b = self.policy()
        np.testing.assert_array_equal(initial, b.update(self.state(b)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"policy.onnx"
            path.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                G1OnnxPolicy(path, list(JOINT_NAMES), (.5,0,0))
        with self.assertRaises(ValueError):
            self.policy(command=(.5,0,.3))


if __name__ == "__main__":
    unittest.main()
