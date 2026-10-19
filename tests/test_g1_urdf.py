"""Guard imported source identity, joint compatibility and mesh path boundaries."""

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_g1_urdf import git_blob_digest, mesh_paths
from g1_policy import LEG_NAMES
from g1_urdf_config import MODE15_JOINTS, URDF_MODELS


class TestG1UrdfSources(unittest.TestCase):
    def parse(self, mesh="meshes/pelvis.STL", names=LEG_NAMES, expected=LEG_NAMES):
        robot = ET.Element("robot", name="g1_12dof")
        link = ET.SubElement(robot, "link", name="pelvis")
        for purpose in ("visual", "collision"):
            geometry = ET.SubElement(ET.SubElement(link, purpose), "geometry")
            ET.SubElement(geometry, "mesh", filename=mesh)
        for name in names:
            ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(robot, "joint", name="upper_body", type="fixed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "g1_12dof.urdf"
            ET.ElementTree(robot).write(path)
            return mesh_paths(path, expected)

    def test_git_blob_hash_uses_git_object_format(self):
        self.assertEqual(git_blob_digest(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_fixed_upper_body_and_duplicate_mesh_references_are_supported(self):
        self.assertEqual(self.parse(names=LEG_NAMES[::-1]), ["meshes/pelvis.STL"])

    def test_missing_duplicate_or_extra_actuated_joints_are_rejected(self):
        for names in (LEG_NAMES[:-1], LEG_NAMES + ("waist_joint",), LEG_NAMES[:-1] + (LEG_NAMES[0],)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.parse(names=names)

    def test_mesh_paths_cannot_escape_or_use_remote_package_locations(self):
        for path in ("", "../pelvis.STL", "meshes/../../pelvis.STL", "/tmp/pelvis.STL",
                     "package://g1/meshes/pelvis.STL", "https://example.com/pelvis.STL"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.parse(mesh=path)

    def test_mode15_requires_waist_and_all_seven_arm_joints(self):
        self.assertEqual(len(MODE15_JOINTS), 29)
        self.assertEqual(self.parse(names=MODE15_JOINTS[::-1], expected=MODE15_JOINTS), ["meshes/pelvis.STL"])
        for names in (LEG_NAMES, MODE15_JOINTS[:-1], MODE15_JOINTS[:-1]+(MODE15_JOINTS[0],),
                      MODE15_JOINTS+("left_hand_finger_joint",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.parse(names=names, expected=MODE15_JOINTS)

    def test_mode15_sources_are_separate_from_twelve_joint_cache(self):
        twelve, mode15 = URDF_MODELS["unitree12"], URDF_MODELS["unitree29"]
        self.assertEqual(mode15.repository, "unitreerobotics/unitree_ros")
        self.assertEqual(mode15.urdf_path.name, "g1_29dof_mode_15.urdf")
        self.assertEqual(mode15.asset_path.name, "g1_29dof_mode_15.usd")
        self.assertNotEqual(mode15.source_root, twelve.source_root)
        self.assertNotEqual(mode15.output_dir, twelve.output_dir)


if __name__ == "__main__":
    unittest.main()
