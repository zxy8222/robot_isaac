"""Check body/world conventions independently of any policy or simulator."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_math import world_to_body


class TestCoordinateTransforms(unittest.TestCase):
    def test_world_velocity_and_gravity_rotate_into_body_frame(self):
        yaw90 = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        np.testing.assert_allclose(world_to_body(yaw90, [1,0,0]), [0,-1,0], atol=1e-7)
        pitch90 = [math.sqrt(.5), 0, math.sqrt(.5), 0]
        np.testing.assert_allclose(world_to_body(pitch90, [0,0,-1]), [1,0,0], atol=1e-7)
        np.testing.assert_allclose(world_to_body([-x for x in pitch90], [0,0,-1]), [1,0,0], atol=1e-7)

    def test_invalid_root_quaternions_are_rejected(self):
        for q in ([0,0,0,0], [1,0,0], [float("nan"),0,0,0]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                world_to_body(q, [0,0,-1])


if __name__ == "__main__":
    unittest.main()
