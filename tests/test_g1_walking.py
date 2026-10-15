"""Check policy frame/order conventions and reject false walking success."""

import contextlib
import io
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_outputs import StateSample
from g1_policy import DEFAULT_ANGLES, DEFAULT_POLICY_PATH, G1Policy, LEG_NAMES, leg_indices, observation, world_to_body
from g1_walk_validation import WalkingMonitor
from g1_video import VideoRecorder
from run_g1_ground import parse_args


class TestPolicyContract(unittest.TestCase):
    def setUp(self):
        self.names = list(LEG_NAMES)[::-1]
        self.indices = leg_indices(self.names)
        self.q = DEFAULT_ANGLES[::-1].tolist()
        self.state = StateSample(10, 0.02, [0, 0, 0.72], [1, 0, 0, 0], [0]*6, self.q, [0]*12)

    def test_observation_layout_and_reordered_joint_mapping(self):
        q = (DEFAULT_ANGLES + np.arange(12) * .1)[::-1]
        state = replace(self.state, joint_positions=q, joint_velocities=list(range(12)))
        obs = observation(state, self.indices, [.5, -.1, .2], np.arange(12))
        self.assertEqual((obs.shape, obs.dtype), ((47,), np.dtype("float32")))
        np.testing.assert_allclose(obs[:9], [0, 0, 0, 0, 0, -1, 1, -.2, .05])
        np.testing.assert_allclose(obs[9:21], np.arange(12)*.1, atol=1e-7)
        np.testing.assert_allclose(obs[21:33], np.arange(12)[::-1]*.05, atol=1e-7)
        np.testing.assert_array_equal(obs[33:45], np.arange(12))
        np.testing.assert_allclose(obs[45:], [math.sin(math.pi/20), math.cos(math.pi/20)])
        for names in (list(LEG_NAMES)[:-1], [*LEG_NAMES, LEG_NAMES[0]]):
            with self.assertRaises(ValueError):
                leg_indices(names)

    def test_world_angular_velocity_rotates_into_body_frame(self):
        yaw90 = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        np.testing.assert_allclose(world_to_body(yaw90, [1, 0, 0]), [0, -1, 0], atol=1e-7)
        pitch90 = [math.sqrt(.5), 0, math.sqrt(.5), 0]
        np.testing.assert_allclose(world_to_body(pitch90, [0, 0, -1]), [1, 0, 0], atol=1e-7)
        state = replace(self.state, orientation=yaw90, velocities=[0, 0, 0, 1, 0, 0])
        np.testing.assert_allclose(observation(state, self.indices, [0]*3, [0]*12)[:3],
                                   [0, -.25, 0], atol=1e-7)

    def test_corrupt_policy_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.pt"
            path.write_bytes(b"incorrect model")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                G1Policy(path, self.names, (.5, 0, 0))

    @unittest.skipUnless(DEFAULT_POLICY_PATH.is_file(), "Fetch the pinned policy for inference integration")
    def test_real_official_policy_resets_recurrent_memory(self):
        first = G1Policy(DEFAULT_POLICY_PATH, self.names, (.5, 0, 0))
        second = G1Policy(DEFAULT_POLICY_PATH, self.names, (.5, 0, 0))
        target = first.update(self.state).copy()
        np.testing.assert_array_equal(second.update(self.state), target)
        self.assertTrue(np.isfinite(target).all())
        self.assertEqual(first.updates, 1)
        first.update(replace(self.state, step=20, simulation_time=.04))
        self.assertEqual(first.updates, 2)


class TestWalkingAcceptance(unittest.TestCase):
    def setUp(self):
        self.state = StateSample(0, 0, [0, 0, .65], [1, 0, 0, 0], [.5, 0, 0, 0, 0, 0], [0]*12, [0]*12)

    def trajectory(self, *, swing=True, hopping=False):
        monitor = WalkingMonitor(self.state, [.5, 0, 0])
        for step in range(3001):
            t = step * .002
            state = replace(self.state, step=step, simulation_time=t, position=[.5*t, 0, .65])
            positions = np.array([[.5*t, .1, .04], [.5*t, -.1, .04]])
            forces = np.array([[0., 0., 160.], [0., 0., 160.]])
            phase = t % .8
            if swing:
                for index in range(2):
                    local = phase if hopping else (t + .4*index) % .8
                    if .1 < local < .4:
                        positions[index, 2] += .06 * math.sin(math.pi*(local-.1)/.3)
                        forces[index, 2] = 0
            monitor.observe(state, positions, forces)
        return monitor

    def test_alternating_measured_swings_and_contacts_pass(self):
        result = self.trajectory().summary()
        self.assertEqual(result["status"], "passed")
        self.assertGreaterEqual(min(result["footfalls_after_settling_left_right"]), 3)
        self.assertGreater(result["single_support_fraction"], .5)

    def test_sliding_and_hopping_cannot_count_as_walking(self):
        for options in ({"swing": False}, {"hopping": True}):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                self.trajectory(**options).summary()

    def test_fall_and_bad_velocity_tracking_fail(self):
        monitor = WalkingMonitor(self.state, [.5, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "stability"):
            monitor.observe(replace(self.state, position=[0, 0, .2]), np.zeros((2,3)), np.zeros((2,3)))
        monitor.observe(replace(self.state, simulation_time=3, velocities=[0]*6),
                        np.zeros((2,3)), np.zeros((2,3)))
        with self.assertRaisesRegex(RuntimeError, "velocity tracking"):
            monitor.summary()


class TestWalkingArguments(unittest.TestCase):
    def parse(self, *arguments):
        root = Path(__file__).resolve().parents[1]
        with patch.object(sys, "argv", ["run_g1_ground.py", "--portable-root", "unused",
            "--output-dir", str(root / ".runtime/isaacsim-6.0.1/outputs/argument-test"),
            "--controller", "walk", *arguments]):
            return parse_args()

    def test_defaults_match_official_timing(self):
        args = self.parse("--video")
        self.assertEqual((args.steps, args.dt, tuple(args.command)), (5000, .002, (.5, 0, 0)))

    def test_invalid_timing_commands_and_video_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for arguments in (("--dt", ".005"), ("--steps", "2000"),
                              ("--command", "nan", "0", "0"), ("--command", "2", "0", "0"),
                              ("--video", "--steps", "5001")):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    self.parse(*arguments)

    def test_video_uses_measured_state_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            recorder = VideoRecorder(Path(directory), .002, 100, lambda path, state: frames.append(state.step))
            state = StateSample(0, 0, [0,0,.7], [1,0,0,0], [0]*6, [], [])
            for step in range(101):
                recorder.observe(replace(state, step=step, simulation_time=step*.002))
            self.assertEqual(frames, [20, 40, 60, 80, 100])


if __name__ == "__main__":
    unittest.main()
