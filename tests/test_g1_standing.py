"""Check joint-order independence, balance feedback, and false standing success."""

import math
import contextlib
import io
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_outputs import StateSample
from g1_standing import BalanceGains, StandingMonitor, balance_offsets, standing_drives
from run_g1_ground import parse_args


SDK_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_roll_joint", "right_elbow_roll_joint",
    "left_five_joint", "left_three_joint", "left_zero_joint", "right_five_joint", "right_three_joint",
    "right_zero_joint", "left_six_joint", "left_four_joint", "left_one_joint", "right_six_joint",
    "right_four_joint", "right_one_joint", "left_two_joint", "right_two_joint",
]


class TestG1Standing(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = [drive.position for drive in standing_drives(SDK_NAMES)]
        self.state = StateSample(0, 0.0, [0, 0, 0.72], [1, 0, 0, 0], [0] * 6,
                                 self.targets, [0] * len(self.targets))

    def test_named_joint_mapping_survives_sdk_reordering(self) -> None:
        for names in (SDK_NAMES, SDK_NAMES[::-1]):
            settings = dict(zip(names, standing_drives(names)))
            self.assertEqual(settings["left_hip_pitch_joint"].position, -0.20)
            self.assertEqual(settings["right_knee_joint"].position, 0.42)
            self.assertEqual(settings["left_shoulder_roll_joint"].position, 0.16)
            self.assertEqual(settings["right_shoulder_roll_joint"].position, -0.16)
            self.assertEqual(settings["left_one_joint"].position, 1.0)
            self.assertEqual(settings["right_one_joint"].position, -1.0)
            self.assertEqual(settings["left_ankle_pitch_joint"].max_effort, 50.0)

    def test_incompatible_or_duplicate_joints_fail_before_control(self) -> None:
        for names in (SDK_NAMES[:-1], SDK_NAMES + [SDK_NAMES[0]], SDK_NAMES[:-1] + ["unknown_joint"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                standing_drives(names)

    def test_feedback_restores_pitch_and_roll_and_is_bounded(self) -> None:
        self.assertEqual(balance_offsets(self.state, BalanceGains()), (0.0, 0.0))
        angle = 0.10
        for axis, quaternion in ((0, [math.cos(angle/2), 0, math.sin(angle/2), 0]),
                                 (1, [math.cos(angle/2), math.sin(angle/2), 0, 0])):
            state = replace(self.state, orientation=quaternion)
            offsets = balance_offsets(state, BalanceGains())
            self.assertAlmostEqual(offsets[axis], BalanceGains().tilt * angle)
            self.assertEqual(offsets, balance_offsets(replace(state, orientation=[-q for q in quaternion]), BalanceGains()))
        self.assertEqual(balance_offsets(replace(self.state, position=[10, -10, 0.72]), BalanceGains()), (0.35, 0.35))

    def test_feedback_accounts_for_heading_and_rate(self) -> None:
        state = replace(self.state, position=[0.1, 0, 0.72],
                        orientation=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
        pitch, roll = balance_offsets(state, BalanceGains())
        self.assertAlmostEqual(pitch, 0.0)
        self.assertAlmostEqual(roll, BalanceGains().position * 0.1)
        pitch, roll = balance_offsets(replace(self.state, velocities=[0, 0, 0, -0.1, 0.1, 0]), BalanceGains())
        self.assertGreater(pitch, 0)
        self.assertLess(roll, 0)
        with self.assertRaises(ValueError):
            balance_offsets(replace(self.state, orientation=[0] * 4), BalanceGains())

    def test_standing_summary_requires_settled_samples(self) -> None:
        monitor = StandingMonitor(self.state, self.targets)
        monitor.observe(self.state)
        with self.assertRaises(RuntimeError):
            monitor.summary()
        monitor.observe(replace(self.state, step=120, simulation_time=2.0))
        self.assertEqual(monitor.summary()["status"], "passed")
        self.assertEqual(monitor.summary()["settled_samples"], 1)

    def test_fall_tilt_and_drift_cannot_count_as_standing(self) -> None:
        angle = math.radians(20)
        failures = [replace(self.state, position=[0, 0, 0.2]),
                    replace(self.state, position=[0, 0, 1.0]),
                    replace(self.state, orientation=[math.cos(angle/2), math.sin(angle/2), 0, 0]),
                    replace(self.state, position=[0.3, 0, 0.72])]
        for state in failures:
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                StandingMonitor(self.state, self.targets).observe(state)

    def test_settled_motion_and_joint_error_are_rejected(self) -> None:
        settled = replace(self.state, step=120, simulation_time=2.0)
        failures = [replace(settled, velocities=[0.4, 0, 0, 0, 0, 0]),
                    replace(settled, velocities=[0, 0, 0, 0.6, 0, 0]),
                    replace(settled, joint_positions=[q + 0.4 for q in self.targets])]
        for state in failures:
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                StandingMonitor(self.state, self.targets).observe(state)


class TestG1CommandLine(unittest.TestCase):
    def parse(self, *extra: str):
        output = Path(__file__).resolve().parents[1] / ".runtime/isaacsim-6.0.1/outputs/argument-test"
        with patch.object(sys, "argv", ["run_g1_ground.py", "--portable-root", "unused-runtime",
                                         "--output-dir", str(output), *extra]):
            return parse_args()

    def test_standing_defaults_and_original_scene_defaults(self) -> None:
        standing = self.parse()
        scene = self.parse("--controller", "none")
        self.assertEqual((standing.controller, standing.steps, standing.dt), ("stand", 2000, 0.005))
        self.assertEqual((scene.controller, scene.steps, scene.dt), ("none", 600, 1.0 / 60.0))

    def test_reject_unsafe_or_unverifiable_standing_timing(self) -> None:
        for args in (("--dt", "0.016666666666666666"), ("--steps", "100"),
                     ("--steps", "12001"), ("--dt", "nan"), ("--dt", "0")):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.parse(*args)
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
