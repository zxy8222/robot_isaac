"""Reject falls, poor tracking, sliding and hopping from measured trajectories."""

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_outputs import StateSample
from g1_walk_validation import WalkingMonitor


class TestWalkingAcceptance(unittest.TestCase):
    def setUp(self):
        self.state = StateSample(0, 0, [0, 0, .65], [1, 0, 0, 0], [.5, 0, 0, 0, 0, 0], [0]*29, [0]*29)

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



if __name__ == "__main__":
    unittest.main()
