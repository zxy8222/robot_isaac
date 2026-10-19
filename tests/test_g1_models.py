"""Check model compatibility and persisted control evidence without Isaac Sim."""

import contextlib
import csv
import io
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_models import DEFAULT_UNITREE_USD, walking_model
from g1_policy import DEFAULT_ANGLES, LEG_NAMES
from g1_control_outputs import ControlSample, ControlWriter
from run_g1_ground import parse_args
from test_g1_standing import SDK_NAMES
from g1_urdf_config import URDF_MODELS


class TestWalkingModels(unittest.TestCase):
    def test_twelve_joint_model_has_only_explicit_leg_drives(self):
        for names in (list(LEG_NAMES), list(LEG_NAMES)[::-1]):
            model = walking_model(names)
            self.assertEqual(model.variant, "G1_12dof_URDF")
            self.assertEqual(model.initial_base_height_m, .793)
            for index,name in enumerate(LEG_NAMES):
                self.assertEqual(model.positions[names.index(name)], DEFAULT_ANGLES[index])
            np.testing.assert_array_equal(model.stiffness, np.zeros(12))
            np.testing.assert_array_equal(model.damping, np.zeros(12))

    def test_original_nvidia_configuration_is_preserved(self):
        model = walking_model(SDK_NAMES)
        self.assertEqual(model.variant, "G1_23dof")
        self.assertEqual(model.initial_base_height_m, .755)
        for index,name in enumerate(SDK_NAMES):
            if name in LEG_NAMES:
                self.assertEqual(model.positions[index], DEFAULT_ANGLES[LEG_NAMES.index(name)])
                self.assertEqual((model.stiffness[index],model.damping[index]),(0,0))
            else:
                self.assertEqual((model.positions[index],model.stiffness[index],model.damping[index]),(0,200,10))

    def test_partial_or_unknown_robot_is_rejected(self):
        for names in (list(LEG_NAMES)[:-1], [*LEG_NAMES,"waist_joint"], SDK_NAMES[:-1],
                      [*LEG_NAMES[:-1],LEG_NAMES[0]]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                walking_model(names)


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


class TestRobotSelection(unittest.TestCase):
    def parse(self,*arguments):
        output=Path(__file__).resolve().parents[1]/".runtime/isaacsim-6.0.1/outputs/argument-test"
        with patch.object(sys,"argv",["run_g1_ground.py","--portable-root","unused",
             "--output-dir",str(output),"--controller","walk",*arguments]):
            return parse_args()

    def test_original_default_and_explicit_twelve_joint_selection(self):
        self.assertEqual(self.parse().robot,"nvidia")
        with patch.object(Path,"is_file",return_value=True):
            args=self.parse("--robot","unitree12")
        self.assertEqual(args.asset_usd,str(DEFAULT_UNITREE_USD))
        self.assertEqual((args.steps,args.dt),(5000,.002))

    def test_missing_import_or_unsupported_static_controller_fails(self):
        with contextlib.redirect_stderr(io.StringIO()), patch.object(Path,"is_file",return_value=False):
            for arguments in (("--robot","unitree12"), ("--robot","unitree12","--controller","stand")):
                with self.subTest(arguments=arguments),self.assertRaises(SystemExit):
                    self.parse(*arguments)

    def test_mode15_selects_its_own_walking_adapter_and_rejects_static_controller(self):
        with patch.object(Path,"is_file",return_value=True):
            args=self.parse("--robot","unitree29","--controller","none")
        self.assertEqual(args.asset_usd,str(URDF_MODELS["unitree29"].asset_path))
        self.assertEqual((args.steps,args.dt),(600,1/60))
        with patch.object(Path,"is_file",return_value=True):
            args=self.parse("--robot","unitree29")
        self.assertEqual((args.steps,args.dt),(5000,.002))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse("--robot","unitree29","--controller","stand","--asset-usd","unused.usd")
            with self.assertRaises(SystemExit):
                self.parse("--robot","unitree29","--command",".5","0",".3","--asset-usd","unused.usd")
            with patch.object(Path,"is_file",return_value=False),self.assertRaises(SystemExit):
                self.parse("--robot","unitree29","--controller","none")


if __name__ == "__main__":
    unittest.main()
