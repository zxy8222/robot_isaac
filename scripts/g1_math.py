"""Simulator-independent coordinate transforms for observations and validation."""

import numpy as np


def world_to_body(quaternion, vector) -> np.ndarray:
    """Rotate a world vector into the body frame; quaternion order is wxyz."""
    q = np.array(quaternion, dtype=np.float64, copy=True)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Expected a finite, nonzero wxyz quaternion")
    q /= norm
    w, xyz = q[0], q[1:]
    vector = np.asarray(vector, dtype=np.float64)
    return vector - 2*w*np.cross(xyz, vector) + 2*np.cross(xyz, np.cross(xyz, vector))
