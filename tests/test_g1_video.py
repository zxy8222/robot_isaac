"""Ensure recording schedules frames from measured physics, without stepping."""

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_outputs import StateSample
from g1_video import VideoRecorder


class TestMeasuredVideo(unittest.TestCase):
    def test_video_uses_measured_state_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = []
            recorder = VideoRecorder(Path(directory), .002, 100, lambda path, state: frames.append(state.step))
            state = StateSample(0, 0, [0,0,.8], [1,0,0,0], [0]*6, [], [])
            for step in range(101):
                recorder.observe(replace(state, step=step, simulation_time=step*.002))
            self.assertEqual(frames, [20,40,60,80,100])


if __name__ == "__main__":
    unittest.main()
