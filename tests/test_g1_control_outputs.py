"""Validate per-step control recording and tracking/saturation summaries."""

import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_control_outputs import ControlSample, ControlWriter


class TestControlEvidence(unittest.TestCase):
    def setUp(self):
        self.initial = ControlSample(0,0.0,0,[0],[.1],[.1],[0],[0],[0])

    def test_tracking_rmse_is_not_confused_with_signed_average(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"control.csv"
            with ControlWriter(path,["ankle"]) as writer:
                writer.write(self.initial)
                writer.write(replace(self.initial,step=1,simulation_time=2.,policy_updates=1,
                                     positions=[.3],tau_raw=[100],tau_cmd=[50]))
                writer.write(replace(self.initial,step=2,simulation_time=2.02,policy_updates=2,
                                     positions=[-.1],tau_raw=[-100],tau_cmd=[-50]))
                report=writer.summary()
            joint=report["joints"]["ankle"]
            self.assertAlmostEqual(joint["mean_actual_minus_target_rad"],0)
            self.assertAlmostEqual(joint["tracking_rmse_rad"],.2)
            self.assertEqual(joint["saturated_steps"],2)
            self.assertEqual(joint["maximum_abs_tau_raw_Nm"],100)
            self.assertEqual(joint["maximum_abs_tau_cmd_Nm"],50)
            with path.open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(float(rows[1]["tau_raw_ankle"]),100)

    def test_missing_steps_bad_shapes_and_nonfinite_commands_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with ControlWriter(Path(directory)/"control.csv",["ankle"]) as writer:
                for sample in (replace(self.initial,step=1),replace(self.initial,tau_cmd=[float("nan")]),
                               replace(self.initial,targets=[]),replace(self.initial,policy_updates=2)):
                    with self.subTest(sample=sample), self.assertRaises(ValueError):
                        writer.write(sample)
                writer.write(self.initial)
                with self.assertRaises(RuntimeError):
                    writer.summary()



if __name__ == "__main__":
    unittest.main()
