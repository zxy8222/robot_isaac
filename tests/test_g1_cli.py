"""Check the single-model entry point and reject obsolete or invalid inputs."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_onnx_policy import DEFAULT_POLICY_PATH
from g1_urdf_config import ASSET_PATH
from run_g1_ground import parse_args


class TestMode15CommandLine(unittest.TestCase):
    def parse(self, *arguments, asset_exists=True):
        root = Path(__file__).resolve().parents[1]
        output = root / ".runtime/isaacsim-6.0.1/outputs/argument-test"
        with patch.object(sys, "argv", ["run_g1_ground.py", "--portable-root", "unused",
                                       "--output-dir", str(output), *arguments]), \
                patch.object(Path, "is_file", return_value=asset_exists):
            return parse_args()

    def test_defaults_select_mode15_onnx_and_official_timing(self):
        args = self.parse("--video")
        self.assertEqual((args.controller, args.steps, args.dt), ("walk", 5000, .002))
        self.assertEqual(args.asset_usd, ASSET_PATH)
        self.assertEqual(args.policy_path, DEFAULT_POLICY_PATH)
        self.assertEqual(tuple(args.command), (.5, 0, 0))

    def test_loading_check_is_short_and_has_no_controller(self):
        args = self.parse("--controller", "none")
        self.assertEqual((args.steps, args.dt), (20, .002))

    def test_invalid_timing_commands_and_video_fail(self):
        cases = (("--dt", ".005"), ("--steps", "2000"), ("--steps", "30001"),
                 ("--dt", "nan"), ("--command", "nan", "0", "0"),
                 ("--command", "2", "0", "0"), ("--command", ".5", "0", ".3"),
                 ("--video", "--steps", "5001"), ("--output-dir", "/tmp/outside-results"))
        with contextlib.redirect_stderr(io.StringIO()):
            for args in cases:
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    self.parse(*args)

    def test_missing_asset_and_removed_legacy_options_fail(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse(asset_exists=False)
            for args in (("--robot", "nvidia"), ("--robot", "unitree12"), ("--controller", "stand")):
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    self.parse(*args)


if __name__ == "__main__":
    unittest.main()
