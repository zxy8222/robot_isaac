"""Guard imported source identity, joint compatibility and mesh path boundaries."""

import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_g1_urdf import mesh_paths
from g1_urdf_config import JOINT_NAMES, STEM


class TestG1UrdfSources(unittest.TestCase):
    def parse(self, mesh="meshes/pelvis.STL", names=JOINT_NAMES):
        robot = ET.Element("robot", name=STEM)
        link = ET.SubElement(robot, "link", name="pelvis")
        for purpose in ("visual", "collision"):
            geometry = ET.SubElement(ET.SubElement(link, purpose), "geometry")
            ET.SubElement(geometry, "mesh", filename=mesh)
        for name in names:
            ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(robot, "joint", name="upper_body", type="fixed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{STEM}.urdf"
            ET.ElementTree(robot).write(path)
            return mesh_paths(path)

    def test_fixed_links_and_duplicate_mesh_references_are_supported(self):
        self.assertEqual(self.parse(names=JOINT_NAMES[::-1]), ["meshes/pelvis.STL"])

    def test_missing_duplicate_or_extra_actuated_joints_are_rejected(self):
        for names in (JOINT_NAMES[:-1], JOINT_NAMES + ("finger_joint",), JOINT_NAMES[:-1] + (JOINT_NAMES[0],)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.parse(names=names)

    def test_mesh_paths_cannot_escape_or_use_remote_package_locations(self):
        for path in ("", "../pelvis.STL", "meshes/../../pelvis.STL", "/tmp/pelvis.STL",
                     "package://g1/meshes/pelvis.STL", "https://example.com/pelvis.STL"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.parse(mesh=path)

    def test_mode15_requires_waist_and_all_seven_arm_joints(self):
        self.assertEqual(len(JOINT_NAMES), 29)
        self.assertEqual(self.parse(names=JOINT_NAMES[::-1]), ["meshes/pelvis.STL"])
        for names in (JOINT_NAMES[:12], JOINT_NAMES[:-1], JOINT_NAMES[:-1]+(JOINT_NAMES[0],),
                      JOINT_NAMES+("left_hand_finger_joint",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.parse(names=names)


if __name__ == "__main__":
    unittest.main()
