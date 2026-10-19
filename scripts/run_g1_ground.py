"""Run Mode-15 ONNX walking on a collision ground, recording measured evidence.

The shell launcher supplies the isolated runtime. Scene creation and file output
are delegated to g1_scene and g1_outputs; this file owns application lifecycle,
measured physics stepping, sampling, and failure reporting.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

from g1_outputs import StateSample, StateWriter, capture_png, combine_root_velocities, file_digest, package_results, write_json
from g1_onnx_policy import DEFAULT_POLICY_PATH
from g1_urdf_config import ASSET_PATH, JOINT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--controller", choices=("walk", "none"), default="walk",
                        help="ONNX walking (default), or a short model-loading check")
    parser.add_argument("--steps", type=int, help="Default: 5000 walking or 20 loading-check steps")
    parser.add_argument("--dt", type=float, default=.002, help="Physics interval; walking requires 0.002 s")
    parser.add_argument("--asset-usd", type=Path, default=ASSET_PATH, help="Local Mode-15 USD")
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH, help="Verified official policy.onnx")
    parser.add_argument("--command", nargs=3, type=float, default=(.5, 0., 0.),
                        metavar=("VX", "VY", "YAW_RATE"), help="Body-frame velocity: m/s, m/s, rad/s")
    parser.add_argument("--video", action="store_true", help="Record a measured 25 fps, 1x MP4")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 5000 if args.controller == "walk" else 20
    maximum_steps = 30000 if args.controller == "walk" else 3600
    if not 2 <= args.steps <= maximum_steps:
        parser.error(f"--steps must be between 2 and {maximum_steps}")
    if not math.isfinite(args.dt) or not .001 <= args.dt <= 1/30:
        parser.error("--dt must be finite and between 1/1000 and 1/30 seconds")
    if args.controller == "walk":
        if not math.isclose(args.dt, .002, abs_tol=1e-12) or args.steps*args.dt < 5:
            parser.error("Official walking requires --dt 0.002 and at least 5 seconds")
        if (not all(math.isfinite(value) for value in args.command)
                or not -.5 <= args.command[0] <= .8 or abs(args.command[1]) > .3 or abs(args.command[2]) > .2):
            parser.error("Walking command bounds: VX [-0.5,0.8], VY [-0.3,0.3], YAW_RATE [-0.2,0.2]")
    if args.video:
        interval = round(1/(25*args.dt))
        if interval < 1 or not math.isclose(interval*args.dt*25, 1, abs_tol=1e-9) or args.steps % interval:
            parser.error("25 fps video requires dt and steps divisible by the 0.04 s frame interval")
    args.asset_usd = args.asset_usd.resolve()
    if not args.asset_usd.is_file():
        parser.error("Mode-15 USD missing; run scripts/import_g1_29dof.py first")
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

        from g1_scene import build_scene, follow_camera

        def update() -> None:
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped before scenario completion")
            app.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        timeline.commit()
        stage, robot, viewport, scene_metadata = build_scene(update, str(args.asset_usd),
                                                           ground_size=80.0 if args.controller == "walk" else 10.0)
        if robot.num_dofs != len(JOINT_NAMES) or set(robot.dof_names) != set(JOINT_NAMES):
            raise ValueError("Expected the official Mode-15 articulation with 29 named joints")
        controller = None
        if args.controller == "walk":
            from g1_onnx_walking import OnnxWalkingController
            from g1_walk_validation import WalkingMonitor

            controller = OnnxWalkingController(robot, args.policy_path, tuple(args.command),
                                               ground_paths=scene_metadata["ground_collision_paths"])
        SimulationManager.setup_simulation(dt=args.dt, device="cpu")
        SimulationManager.enable_fabric(False)
        SimulationManager.initialize_physics()
        for _ in range(30):
            update()
        if not robot.is_physics_tensor_entity_valid():
            raise RuntimeError("G1 physics articulation failed to initialize")
        if controller is not None:
            controller.initialize()
            scene_metadata["controller_enabled"] = True
            print(f"G1_CONTROLLER_INITIALIZED={args.controller}", flush=True)
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
        monitor = None
        if args.controller == "walk":
            monitor = WalkingMonitor(initial_state, args.command)
        last_state = initial_state
        video_result = None
        joint_control_result = None

        def capture(path, state):
            if args.controller == "walk":
                follow_camera(stage, state.position)
            count_before = SimulationManager.get_num_physics_steps()
            info = capture_png(viewport, update, path)
            if SimulationManager.get_num_physics_steps() != count_before:
                raise RuntimeError("Rendering unexpectedly advanced physics")
            return info

        with ExitStack() as stack:
            writer = stack.enter_context(StateWriter(args.output_dir / "states.csv", dof_names))
            foot_writer = None
            control_writer = None
            if args.controller == "walk":
                from g1_control_outputs import ControlWriter
                control_names = controller.control_names
                control_writer = stack.enter_context(ControlWriter(args.output_dir / "joint-control.csv", control_names))
                foot_writer = csv.writer(stack.enter_context((args.output_dir / "feet.csv").open("x", newline="")))
                foot_writer.writerow(["step", "simulation_time_s"] +
                    [f"{side}_{value}" for side in ("left", "right")
                     for value in ("x_m", "y_m", "z_m", "fx_N", "fy_N", "fz_N")])

            def observe(state):
                if foot_writer is not None:
                    positions, forces = controller.read_feet()
                    foot_writer.writerow([state.step, state.simulation_time,
                                          *positions[0], *forces[0], *positions[1], *forces[1]])
                    monitor.observe(state, positions, forces)

            recorder = None
            if args.video:
                from g1_video import VideoRecorder

                recorder = VideoRecorder(args.output_dir, args.dt, args.steps, capture)
            observe(initial_state)
            writer.write(initial_state)
            shots = {0: "initial.png", args.steps // 2: "mid.png", args.steps: "final.png"}
            progress_interval = max(1, round(1.0 / args.dt))

            def screenshot(step: int) -> None:
                info = capture(args.output_dir / shots[step], last_state)
                info.update({"step": step, "simulation_time_s": last_state.simulation_time})
                screenshots.append(info)
                print(f"G1_SCREENSHOT_SAVED={shots[step]}", flush=True)

            screenshot(0)
            for step in range(1, args.steps + 1):
                if controller is not None:
                    command = controller.apply(last_state)
                    if control_writer is not None:
                        control_writer.write(command)
                SimulationManager.step(steps=1)
                last_state = sample(step)
                writer.write(last_state)
                observe(last_state)
                if recorder is not None:
                    recorder.observe(last_state)
                if last_state.position[2] < -0.20:
                    raise RuntimeError("G1 root fell below the ground; collision smoke test failed")
                if step in shots:
                    screenshot(step)
                if step % progress_interval == 0:
                    print(f"G1_PHYSICS_PROGRESS step={step} time={last_state.simulation_time:.6f} z={last_state.position[2]:.4f}", flush=True)
            rows = writer.rows
            if control_writer is not None:
                joint_control_result = control_writer.summary()
            if recorder is not None:
                video_result = recorder.finish()

        measured_steps = SimulationManager.get_num_physics_steps() - initial_count
        if measured_steps != args.steps:
            raise RuntimeError(f"Physics step count mismatch: {measured_steps} != {args.steps}")
        if not math.isclose(last_state.simulation_time, args.steps * args.dt, rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError("Measured simulation duration does not match requested steps")
        control_result = monitor.summary() if monitor is not None else None
        scene_path = args.output_dir / "scene.usda"
        if not stage.GetRootLayer().Export(str(scene_path)):
            raise RuntimeError("USD scene export failed")
        files = ["states.csv", "scene.usda", "initial.png", "mid.png", "final.png"]
        if args.controller == "walk":
            files.extend(["feet.csv", "joint-control.csv"])
        if video_result is not None:
            files.extend(["simulation.mp4", "video-frames.csv", "video-encoding.log"])
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
            "controller": controller.metadata() if controller is not None else {"name": "none"},
            "walking": control_result if args.controller == "walk" else None, "video": video_result,
            "joint_control": joint_control_result,
            "source_sha256": {path.name: file_digest(path) for path in Path(__file__).parent.glob("g1_*.py")},
            "runner_sha256": file_digest(Path(__file__)),
            "acceptance": ("Unitree policy walking: floating base, gravity, uprightness, body velocity tracking, "
                           "and repeated unloaded foot swings followed by measured contact verified."
                           if args.controller == "walk" else
                           "Scene loading, collision, finite states, time stepping and image capture only."),
        })
        archive = package_results(args.output_dir, [*files, "summary.json"])
        print(f"G1_PHYSICS_CHECK_PASSED steps={measured_steps} rows={rows}", flush=True)
        if control_result is not None:
            print(f"G1_{args.controller.upper()}_CHECK_PASSED {control_result}", flush=True)
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
