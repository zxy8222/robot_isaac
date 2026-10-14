"""Validate and persist G1 states, viewport images, and a downloadable archive.

State/JSON/archive helpers use only the Python standard library. The screenshot
helper imports Kit lazily after the caller has initialized SimulationApp.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import struct
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


def combine_root_velocities(linear: Sequence[float], angular: Sequence[float]) -> list[float]:
    """Convert the SDK's separate velocity vectors to CSV order: vx,vy,vz,wx,wy,wz."""
    if len(linear) != 3 or len(angular) != 3:
        raise ValueError("Linear and angular velocities must each have three components")
    values = [float(value) for value in (*linear, *angular)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Root velocities contain a non-finite number")
    return values


@dataclass(frozen=True)
class StateSample:
    """One measured state; quaternion order is wxyz and joint units are radians."""

    step: int
    simulation_time: float
    position: Sequence[float]
    orientation: Sequence[float]
    velocities: Sequence[float]
    joint_positions: Sequence[float]
    joint_velocities: Sequence[float]

    def values(self, num_dofs: int) -> list[float | int]:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("Step must be a non-negative integer")
        groups = (
            (self.position, 3),
            (self.orientation, 4),
            (self.velocities, 6),
            (self.joint_positions, num_dofs),
            (self.joint_velocities, num_dofs),
        )
        if any(len(values) != size for values, size in groups):
            raise ValueError("State vector dimensions do not match the robot")
        numbers = [float(self.simulation_time)]
        numbers.extend(float(value) for values, _ in groups for value in values)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("State contains a non-finite number")
        if self.simulation_time < 0:
            raise ValueError("Simulation time must not be negative")
        norm = math.sqrt(sum(float(value) ** 2 for value in self.orientation))
        if not math.isclose(norm, 1.0, abs_tol=0.02):
            raise ValueError("Root orientation is not a unit quaternion")
        return [self.step, *numbers]


class StateWriter:
    """Stream validated samples to a new CSV without overwriting prior results."""

    def __init__(self, path: Path, dof_names: Sequence[str]) -> None:
        self.dof_names = tuple(dof_names)
        if not self.dof_names or len(set(self.dof_names)) != len(self.dof_names):
            raise ValueError("DOF names must be non-empty and unique")
        if any(not isinstance(name, str) or not name.strip() for name in self.dof_names):
            raise ValueError("Each DOF needs a non-empty name")
        self._file = path.open("x", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._csv.writerow(
            ["step", "simulation_time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz",
             "vx_m_s", "vy_m_s", "vz_m_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"]
            + [f"q_{name}_rad" for name in self.dof_names]
            + [f"qd_{name}_rad_s" for name in self.dof_names]
        )
        self.rows = 0
        self._last_time = -1.0

    def write(self, sample: StateSample) -> None:
        values = sample.values(len(self.dof_names))
        if sample.step != self.rows:
            raise ValueError("Steps must start at zero and increase by one")
        if sample.simulation_time <= self._last_time:
            raise ValueError("Measured physics time must increase")
        self._csv.writerow(values)
        self._file.flush()
        self.rows += 1
        self._last_time = sample.simulation_time

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> StateWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def write_json(path: Path, data: dict[str, object]) -> None:
    """Serialize before opening, so invalid JSON does not leave a partial file."""
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded + "\n")


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width == 0 or height == 0:
        raise ValueError(f"Empty PNG image: {path}")
    return width, height


def capture_png(viewport: object, update: Callable[[], None], path: Path, timeout: float = 90.0) -> dict[str, object]:
    """Wait for the asynchronous Kit capture and its file; caller freezes time."""
    from omni.kit.viewport.utility import capture_viewport_to_file

    if path.exists():
        raise FileExistsError(path)
    capture = capture_viewport_to_file(viewport, file_path=str(path))
    if capture is None:
        raise RuntimeError("Viewport did not schedule a capture")
    future = asyncio.ensure_future(capture.wait_for_result(completion_frames=5))
    deadline = time.monotonic() + timeout
    try:
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Screenshot timed out: {path.name}")
            update()
        if not future.result():
            raise RuntimeError(f"Screenshot returned no image: {path.name}")
        while not path.is_file() or path.stat().st_size < 24:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Screenshot file was not written: {path.name}")
            update()
        width, height = png_dimensions(path)
        return {"file": path.name, "width": width, "height": height, "bytes": path.stat().st_size}
    finally:
        if not future.done():
            future.cancel()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_results(directory: Path, filenames: Sequence[str]) -> Path:
    """Archive only named result files, never runtime caches or external paths."""
    if not filenames or len(set(filenames)) != len(filenames):
        raise ValueError("Archive filenames must be non-empty and unique")
    for name in filenames:
        source = directory / name
        if Path(name).name != name or not source.is_file() or source.is_symlink():
            raise ValueError(f"Unsafe or missing archive member: {name}")
    archive = directory / "results.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for name in filenames:
            output.write(directory / name, arcname=name)
    with zipfile.ZipFile(archive) as result:
        bad_file = result.testzip()
        if bad_file is not None:
            raise RuntimeError(f"Archive CRC failed: {bad_file}")
    return archive
