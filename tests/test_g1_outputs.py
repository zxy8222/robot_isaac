"""CPU-only tests for state validation and safe downloadable result generation."""

import csv
import json
import math
import struct
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_outputs import StateSample, StateWriter, combine_root_velocities, package_results, png_dimensions, write_json


class TestG1Outputs(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="g1-output-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.state = StateSample(0, 0.0, [0, 0, 0.8], [1, 0, 0, 0], [0] * 6, [0, 0], [0, 0])

    def test_csv_roundtrip_and_no_overwrite(self) -> None:
        path = self.directory / "states.csv"
        with StateWriter(path, ["hip", "knee"]) as writer:
            writer.write(self.state)
            writer.write(replace(self.state, step=1, simulation_time=1 / 60))
            self.assertEqual(writer.rows, 2)
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["z_m"], "0.8")
        self.assertIn("q_hip_rad", rows[0])
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            StateWriter(path, ["hip", "knee"])
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_states(self) -> None:
        for sample in [replace(self.state, position=[math.nan, 0, 1]),
                       replace(self.state, simulation_time=math.inf),
                       replace(self.state, joint_positions=[0]),
                       replace(self.state, orientation=[0, 0, 0, 0]),
                       replace(self.state, step=-1), replace(self.state, step=True)]:
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                sample.values(2)

    def test_separate_root_velocity_vectors(self) -> None:
        values = combine_root_velocities([1, 2, 3], [4, 5, 6])
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for linear, angular in [([0, 0], [0, 0, 0]), ([0, 0, 0], [math.inf, 0, 0])]:
            with self.subTest(linear=linear, angular=angular), self.assertRaises(ValueError):
                combine_root_velocities(linear, angular)

    def test_duplicate_steps_or_time_rejected(self) -> None:
        with StateWriter(self.directory / "states.csv", ["hip", "knee"]) as writer:
            writer.write(self.state)
            with self.assertRaises(ValueError):
                writer.write(replace(self.state, step=1))
            with self.assertRaises(ValueError):
                writer.write(replace(self.state, step=2, simulation_time=1))
            self.assertEqual(writer.rows, 1)

    def test_invalid_dof_names(self) -> None:
        for names in [[], ["hip", "hip"], [""]]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                StateWriter(self.directory / "states.csv", names)

    def test_json_and_archive(self) -> None:
        write_json(self.directory / "summary.json", {"status": "passed"})
        archive = package_results(self.directory, ["summary.json"])
        with zipfile.ZipFile(archive) as source:
            self.assertIsNone(source.testzip())
            self.assertEqual(json.loads(source.read("summary.json")), {"status": "passed"})
        with self.assertRaises(FileExistsError):
            package_results(self.directory, ["summary.json"])
        with self.assertRaises(ValueError):
            package_results(self.directory, ["../outside"])

    def test_nonfinite_json_leaves_no_file(self) -> None:
        path = self.directory / "invalid.json"
        with self.assertRaises(ValueError):
            write_json(path, {"value": math.nan})
        self.assertFalse(path.exists())

    def test_png_header(self) -> None:
        path = self.directory / "sample.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1280, 720))
        self.assertEqual(png_dimensions(path), (1280, 720))
        path.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            png_dimensions(path)


if __name__ == "__main__":
    unittest.main()
