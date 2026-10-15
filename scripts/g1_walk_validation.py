"""SDK-independent walking acceptance from measured base and foot trajectories."""

from __future__ import annotations

import math

import numpy as np

from g1_policy import world_to_body


class WalkingMonitor:
    """Reject falls, poor velocity tracking, or motion without repeated footfalls."""

    def __init__(self, initial, command) -> None:
        self.origin = np.asarray(initial.position[:2])
        self.command = np.asarray(command)
        self.minimum_height = float("inf")
        self.maximum_tilt = 0.0
        self.velocity_sum = np.zeros(3)
        self.velocity_error_sum = np.zeros(3)
        self.samples = 0
        self.single_support_samples = 0
        self.landings = []
        self.footfalls = [0, 0]
        self.swing_started = [None, None]
        self.swing_peak = [0.0, 0.0]
        self.maximum_lift = [0.0, 0.0]
        self.stance_height = [None, None]
        self.last_state = initial

    def observe(self, state, foot_positions, foot_forces) -> None:
        state.values(len(state.joint_positions))
        foot_positions, foot_forces = np.asarray(foot_positions), np.asarray(foot_forces)
        if (foot_positions.shape != (2, 3) or foot_forces.shape != (2, 3)
                or not np.isfinite(foot_positions).all() or not np.isfinite(foot_forces).all()):
            raise ValueError("Invalid measured foot state")
        gravity = world_to_body(state.orientation, [0, 0, -1])
        tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1))))
        height = state.position[2]
        self.minimum_height = min(self.minimum_height, height)
        self.maximum_tilt = max(self.maximum_tilt, tilt)
        if not 0.48 <= height <= 0.90 or tilt > 30:
            raise RuntimeError(f"Walking stability failed at {state.simulation_time:.3f}s: height={height:.3f}, tilt={tilt:.2f}")
        self.last_state = state
        if state.simulation_time < 2.0:
            return
        body_velocity = world_to_body(state.orientation, state.velocities[:3])
        body_omega = world_to_body(state.orientation, state.velocities[3:])
        measured = np.array([*body_velocity[:2], body_omega[2]])
        self.velocity_sum += measured
        self.velocity_error_sum += np.abs(measured - self.command)
        self.samples += 1
        loads = np.abs(foot_forces[:, 2])
        if (loads[0] > 20 and loads[1] < 5) or (loads[1] > 20 and loads[0] < 5):
            self.single_support_samples += 1
        # Hysteresis avoids counting tiny contact-force oscillations as steps.
        # A valid footfall must include >=40 ms without load and >=15 mm lift.
        for index in range(2):
            height = float(foot_positions[index, 2])
            load = abs(float(foot_forces[index, 2]))
            if self.stance_height[index] is None:
                if load > 20:
                    self.stance_height[index] = height
                continue
            lift = height - self.stance_height[index]
            self.maximum_lift[index] = max(self.maximum_lift[index], lift)
            if load < 5:
                if self.swing_started[index] is None:
                    self.swing_started[index] = state.simulation_time
                    self.swing_peak[index] = lift
                self.swing_peak[index] = max(self.swing_peak[index], lift)
            elif load > 20:
                start = self.swing_started[index]
                if start is not None:
                    if state.simulation_time - start >= 0.04 and self.swing_peak[index] >= 0.015:
                        self.footfalls[index] += 1
                        self.landings.append((index, state.simulation_time))
                    self.swing_started[index] = None
                self.stance_height[index] = height

    def summary(self) -> dict:
        if self.samples == 0:
            raise RuntimeError("No walking samples after settling")
        mean_velocity = self.velocity_sum / self.samples
        mean_error = self.velocity_error_sum / self.samples
        duration = self.last_state.simulation_time
        displacement = np.asarray(self.last_state.position[:2]) - self.origin
        commanded_speed = float(np.linalg.norm(self.command[:2]))
        alternating_landings = sum(first[0] != second[0] and second[1] - first[1] >= 0.1
                                   for first, second in zip(self.landings, self.landings[1:]))
        single_support_fraction = self.single_support_samples / self.samples
        if mean_error[0] > 0.25 or mean_error[1] > 0.20 or mean_error[2] > 0.65:
            raise RuntimeError(f"Walking velocity tracking failed: mean absolute errors={mean_error}")
        if commanded_speed >= 0.1:
            required_steps = max(3, int((duration - 2) / 1.2))
            if min(self.footfalls) < required_steps:
                raise RuntimeError(f"Insufficient measured footfalls: {self.footfalls}, require {required_steps} per foot")
            if alternating_landings < 2 * required_steps - 2 or single_support_fraction < 0.20:
                raise RuntimeError("Measured contacts do not demonstrate alternating walking")
            if np.linalg.norm(mean_velocity[:2]) < 0.5 * commanded_speed:
                raise RuntimeError("Robot did not achieve the commanded walking speed")
        return {"status": "passed", "minimum_base_height_m": self.minimum_height,
                "maximum_tilt_deg": self.maximum_tilt, "displacement_xy_m": displacement.tolist(),
                "mean_body_velocity_m_s_rad_s": mean_velocity.tolist(),
                "mean_absolute_velocity_error_m_s_rad_s": mean_error.tolist(),
                "footfalls_after_settling_left_right": self.footfalls,
                "alternating_landings": alternating_landings, "single_support_fraction": single_support_fraction,
                "maximum_foot_lift_m_left_right": self.maximum_lift,
                "settling_time_s": 2.0, "measured_samples_after_settling": self.samples,
                "limits": {"base_height_m": [0.48, 0.90], "maximum_tilt_deg": 30,
                           "mean_absolute_velocity_error": [0.25, 0.20, 0.65],
                           "minimum_swing_time_s": 0.04, "minimum_foot_lift_m": 0.015,
                           "minimum_single_support_fraction": 0.20}}
