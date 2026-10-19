"""Record joint commands and measured tracking, without simulator dependencies."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ControlSample:
    step: int
    simulation_time: float
    policy_updates: int
    action: Sequence[float]
    targets: Sequence[float]
    positions: Sequence[float]
    velocities: Sequence[float]
    tau_raw: Sequence[float]
    tau_cmd: Sequence[float]


class ControlWriter:
    """One row per pre-step PD command; summarize tracking after settling."""

    FIELDS = ("action", "targets", "positions", "velocities", "tau_raw", "tau_cmd")

    def __init__(self, path: Path, names: Sequence[str]) -> None:
        self.names = list(names)
        self._file = path.open("x", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "simulation_time_s", "policy_updates"] +
                              [f"{field}_{name}" for field in self.FIELDS for name in self.names])
        self.rows = self.settled_rows = 0
        self._last_time = -1.0
        self._last_updates = 0
        self._policy_times = []
        self._intervals = []
        self._metrics = [{"target_sum": 0.0, "position_sum": 0.0, "error_sum": 0.0, "error_squared_sum": 0.0,
                          "maximum_absolute_error_rad": 0.0, "maximum_abs_tau_raw_Nm": 0.0,
                          "maximum_abs_tau_cmd_Nm": 0.0, "saturated_steps": 0} for _ in self.names]

    def write(self, sample: ControlSample) -> None:
        if (sample.step != self.rows or not math.isfinite(sample.simulation_time)
                or sample.simulation_time < 0 or sample.simulation_time <= self._last_time):
            raise ValueError("Control records must have consecutive steps and increasing finite time")
        values = [list(getattr(sample, field)) for field in self.FIELDS]
        if any(len(vector) != len(self.names) or not all(math.isfinite(float(x)) for x in vector) for vector in values):
            raise ValueError("Control vectors must have matching joint counts and finite values")
        if sample.policy_updates not in (self._last_updates, self._last_updates + 1):
            raise ValueError("Policy update count must stay constant or increase by one")
        if self.rows:
            self._intervals.append(sample.simulation_time-self._last_time)
        if sample.policy_updates != self._last_updates:
            self._policy_times.append(sample.simulation_time)
        self._writer.writerow([sample.step, sample.simulation_time, sample.policy_updates,
                               *[float(x) for vector in values for x in vector]])
        for index, metric in enumerate(self._metrics):
            metric["maximum_abs_tau_raw_Nm"] = max(metric["maximum_abs_tau_raw_Nm"], abs(sample.tau_raw[index]))
            metric["maximum_abs_tau_cmd_Nm"] = max(metric["maximum_abs_tau_cmd_Nm"], abs(sample.tau_cmd[index]))
            metric["saturated_steps"] += int(abs(sample.tau_raw[index]-sample.tau_cmd[index]) > 1e-7)
            if sample.simulation_time >= 2.0:
                error = sample.positions[index]-sample.targets[index]
                metric["target_sum"] += sample.targets[index]
                metric["position_sum"] += sample.positions[index]
                metric["error_sum"] += error
                metric["error_squared_sum"] += error*error
                metric["maximum_absolute_error_rad"] = max(metric["maximum_absolute_error_rad"], abs(error))
        self.settled_rows += int(sample.simulation_time >= 2.0)
        self.rows += 1
        self._last_time, self._last_updates = sample.simulation_time, sample.policy_updates
        if self.rows % 500 == 0:
            self._file.flush()

    def summary(self) -> dict:
        if not self.settled_rows:
            raise RuntimeError("No joint tracking samples after the two-second settling interval")
        metrics = {}
        for name, item in zip(self.names, self._metrics):
            metrics[name] = {key: float(value) for key, value in item.items()
                             if not key.endswith("_sum") and key != "saturated_steps"}
            metrics[name].update({"mean_target_rad": float(item["target_sum"] / self.settled_rows),
                "mean_actual_rad": float(item["position_sum"] / self.settled_rows),
                "mean_actual_minus_target_rad": float(item["error_sum"] / self.settled_rows),
                "tracking_rmse_rad": math.sqrt(float(item["error_squared_sum"] / self.settled_rows)),
                "saturated_steps": item["saturated_steps"],
                "saturated_fraction": item["saturated_steps"] / self.rows})
        policy_intervals = [b-a for a,b in zip(self._policy_times, self._policy_times[1:])]
        return {"rows": self.rows, "settling_time_s": 2.0, "settled_rows": self.settled_rows,
                "pd_interval_range_s": [min(self._intervals), max(self._intervals)],
                "policy_updates": self._last_updates,
                "policy_interval_range_s": [min(policy_intervals), max(policy_intervals)] if policy_intervals else [],
                "joints": metrics,
                "note": "Pre-step measured angles and explicit PD commands; tau_cmd is not a torque sensor."}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._file.close()
