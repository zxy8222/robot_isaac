"""Run a bounded headless G1 physics smoke test and export reproducible evidence.

The shell launcher supplies the isolated runtime. Scene creation and file output
are delegated to g1_scene and g1_outputs; this file owns application lifecycle,
measured physics stepping, sampling, and failure reporting.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from g1_outputs import StateSample, StateWriter, capture_png, combine_root_velocities, file_digest, package_results, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--asset-usd", help="Optional local or remote G1_23dof USD override")
    args = parser.parse_args()
    if not 2 <= args.steps <= 3600:
        parser.error("--steps must be between 2 and 3600")
    if not math.isfinite(args.dt) or not 1.0 / 1000.0 <= args.dt <= 1.0 / 30.0:
        parser.error("--dt must be finite and between 1/1000 and 1/30 seconds")
    if args.asset_usd is not None and not args.asset_usd.strip():
        parser.error("--asset-usd must not be empty")
    allowed_root = Path(__file__).resolve().parents[1] / ".runtime/isaacsim-6.0.1/outputs"
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == allowed_root or not args.output_dir.is_relative_to(allowed_root):
        parser.error(f"Results must be inside {allowed_root}")
    return args


def main() -> int:
    args = parse_args()
    if version("isaacsim") != "6.0.1.0":
        raise RuntimeError("This scenario requires the pinned Isaac Sim 6.0.1.0 environment")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    app = None
    timeline = None
    exit_status = 1
    print(f"G1_RUN_OUTPUT={args.output_dir}", flush=True)
    try:
        # Kit receives only its own arguments, not the scenario's argparse flags.
        sys.argv = [sys.argv[0], "--portable-root", str(args.portable_root)]
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0,
                             "limit_cpu_threads": 4, "width": 1280, "height": 720})
        import omni.timeline
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.simulation_manager import SimulationManager

        from g1_scene import build_scene

        def update() -> None:
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped before scenario completion")
            app.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        timeline.commit()
        stage, robot, viewport, scene_metadata = build_scene(update, args.asset_usd)
        SimulationManager.setup_simulation(dt=args.dt, device="cpu")
        SimulationManager.enable_fabric(False)
        SimulationManager.initialize_physics()
        for _ in range(30):
            update()
        if not robot.is_physics_tensor_entity_valid():
            raise RuntimeError("G1 physics articulation failed to initialize")
        dof_names = list(robot.dof_names)
        initial_time = SimulationManager.get_simulation_time()
        initial_count = SimulationManager.get_num_physics_steps()

        def sample(step: int) -> StateSample:
            with use_backend("tensor"):
                positions, orientations = robot.get_world_poses()
                linear_velocities, angular_velocities = robot.get_velocities()
                joint_positions = robot.get_dof_positions()
                joint_velocities = robot.get_dof_velocities()
            return StateSample(step, SimulationManager.get_simulation_time() - initial_time,
                               positions.numpy()[0].tolist(), orientations.numpy()[0].tolist(),
                               combine_root_velocities(linear_velocities.numpy()[0].tolist(),
                                                       angular_velocities.numpy()[0].tolist()),
                               joint_positions.numpy()[0].tolist(),
                               joint_velocities.numpy()[0].tolist())

        screenshots = []
        initial_state = sample(0)
        last_state = initial_state
        with StateWriter(args.output_dir / "states.csv", dof_names) as writer:
            writer.write(initial_state)
            shots = {0: "initial.png", args.steps // 2: "mid.png", args.steps: "final.png"}

            def screenshot(step: int) -> None:
                count_before = SimulationManager.get_num_physics_steps()
                info = capture_png(viewport, update, args.output_dir / shots[step])
                if SimulationManager.get_num_physics_steps() != count_before:
                    raise RuntimeError("Rendering unexpectedly advanced physics during screenshot")
                info.update({"step": step, "simulation_time_s": last_state.simulation_time})
                screenshots.append(info)
                print(f"G1_SCREENSHOT_SAVED={shots[step]}", flush=True)

            screenshot(0)
            for step in range(1, args.steps + 1):
                SimulationManager.step(steps=1)
                last_state = sample(step)
                writer.write(last_state)
                if last_state.position[2] < -0.20:
                    raise RuntimeError("G1 root fell below the ground; collision smoke test failed")
                if step in shots:
                    screenshot(step)
                if step % 60 == 0:
                    print(f"G1_PHYSICS_PROGRESS step={step} time={last_state.simulation_time:.6f} z={last_state.position[2]:.4f}", flush=True)
            rows = writer.rows

        measured_steps = SimulationManager.get_num_physics_steps() - initial_count
        if measured_steps != args.steps:
            raise RuntimeError(f"Physics step count mismatch: {measured_steps} != {args.steps}")
        if not math.isclose(last_state.simulation_time, args.steps * args.dt, rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError("Measured simulation duration does not match requested steps")
        scene_path = args.output_dir / "scene.usda"
        if not stage.GetRootLayer().Export(str(scene_path)):
            raise RuntimeError("USD scene export failed")
        files = ["states.csv", "scene.usda", "initial.png", "mid.png", "final.png"]
        manifest = {name: {"bytes": (args.output_dir / name).stat().st_size,
                           "sha256": file_digest(args.output_dir / name)} for name in files}
        write_json(args.output_dir / "summary.json", {
            "status": "passed", "isaacsim_version": version("isaacsim"),
            "utc_completed": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "scene": scene_metadata,
            "dof_names": dof_names, "num_dofs": len(dof_names), "requested_steps": args.steps,
            "measured_steps": measured_steps, "dt_s": args.dt, "measured_duration_s": last_state.simulation_time,
            "state_rows": rows, "initial_root_position_m": list(initial_state.position),
            "final_root_position_m": list(last_state.position), "screenshots": screenshots,
            "files": manifest, "scene_has_external_asset_references": True,
            "acceptance": "Loading, collision configuration, finite measured states, time stepping and image capture only; not balance or walking.",
        })
        archive = package_results(args.output_dir, [*files, "summary.json"])
        print(f"G1_PHYSICS_CHECK_PASSED steps={measured_steps} rows={rows}", flush=True)
        print(f"G1_DOWNLOAD_ARCHIVE={archive}", flush=True)
        exit_status = 0
        return 0
    except KeyboardInterrupt:
        exit_status = 130
        write_json(args.output_dir / "failure.json", {
            "status": "interrupted", "error_type": "KeyboardInterrupt",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
        return exit_status
    except Exception as error:
        traceback.print_exc()
        write_json(args.output_dir / "failure.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
        return 1
    finally:
        try:
            if timeline is not None:
                timeline.stop()
                timeline.commit()
        finally:
            if app is not None:
                print(f"G1_CLOSING_APPLICATION exit_status={exit_status}", flush=True)
                # Fast shutdown otherwise exits the process with zero, masking failures.
                app.close(exit_code=exit_status)


if __name__ == "__main__":
    raise SystemExit(main())
