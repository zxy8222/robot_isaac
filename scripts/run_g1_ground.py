"""Run bounded G1 standing or Unitree walking on a collision ground.

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, help="Default: 5000 walking, 2000 standing, 600 scene-check steps")
    parser.add_argument("--dt", type=float, help="Default: 0.002 s walking, 0.005 s standing, 1/60 s scene check")
    parser.add_argument("--asset-usd", help="Optional local or remote G1_23dof USD override")
    parser.add_argument("--controller", choices=("walk", "stand", "none"), default="stand")
    parser.add_argument("--command", nargs=3, type=float, default=(0.5, 0.0, 0.0),
                        metavar=("VX", "VY", "YAW_RATE"), help="Walking body-frame velocity: m/s, m/s, rad/s")
    parser.add_argument("--policy-path", type=Path, help="Verified official motion.pt; default is the pinned local cache")
    parser.add_argument("--video", action="store_true", help="Record a measured 25 fps, 1x playback MP4")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = {"walk": 5000, "stand": 2000, "none": 600}[args.controller]
    if args.dt is None:
        args.dt = {"walk": 0.002, "stand": 0.005, "none": 1.0 / 60.0}[args.controller]
    maximum_steps = {"walk": 30000, "stand": 12000, "none": 3600}[args.controller]
    if not 2 <= args.steps <= maximum_steps:
        parser.error(f"--steps must be between 2 and {maximum_steps}")
    if not math.isfinite(args.dt) or not 1.0 / 1000.0 <= args.dt <= 1.0 / 30.0:
        parser.error("--dt must be finite and between 1/1000 and 1/30 seconds")
    if args.controller == "stand" and args.steps * args.dt < 5.0:
        parser.error("Standing validation requires at least 5 seconds; use --controller none for shorter scene checks")
    if args.controller == "stand" and args.dt > 0.005:
        parser.error("Standing control requires --dt <= 0.005 (at least 200 Hz physics)")
    if args.controller == "walk":
        if not math.isclose(args.dt, 0.002, abs_tol=1e-12) or args.steps * args.dt < 5:
            parser.error("Official walking policy requires --dt 0.002 and at least 5 seconds")
        if (not all(math.isfinite(value) for value in args.command)
                or not -0.5 <= args.command[0] <= 0.8 or abs(args.command[1]) > 0.3 or abs(args.command[2]) > 0.5):
            parser.error("Walking command bounds: VX [-0.5,0.8], VY [-0.3,0.3], YAW_RATE [-0.5,0.5]")
    if args.video:
        interval = round(1 / (25 * args.dt))
        if interval < 1 or not math.isclose(interval * args.dt * 25, 1, abs_tol=1e-9) or args.steps % interval:
            parser.error("25 fps video requires dt and steps divisible by the 0.04 s frame interval")
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

        from g1_scene import build_scene, follow_camera

        def update() -> None:
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped before scenario completion")
            app.update()

        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        timeline.commit()
        stage, robot, viewport, scene_metadata = build_scene(update, args.asset_usd,
                                                           ground_size=80.0 if args.controller == "walk" else 10.0)
        controller = None
        if args.controller == "stand":
            from g1_standing import StandingController, StandingMonitor

            controller = StandingController(robot)
        elif args.controller == "walk":
            from g1_policy import DEFAULT_POLICY_PATH
            from g1_walking import WalkingController
            from g1_walk_validation import WalkingMonitor

            controller = WalkingController(robot, args.policy_path or DEFAULT_POLICY_PATH, tuple(args.command),
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
            scene_metadata.update({"locomotion_or_balance_controller": True,
                                   "standing_success_not_an_acceptance_condition": False})
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
        if args.controller == "stand":
            monitor = StandingMonitor(initial_state, controller.targets)
        elif args.controller == "walk":
            monitor = WalkingMonitor(initial_state, args.command)
        last_state = initial_state
        video_result = None

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
            if args.controller == "walk":
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
                elif monitor is not None:
                    monitor.observe(state)

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
                    controller.apply(last_state)
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
            files.append("feet.csv")
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
            "standing": control_result if args.controller == "stand" else None,
            "walking": control_result if args.controller == "walk" else None, "video": video_result,
            "source_sha256": {path.name: file_digest(path) for path in Path(__file__).parent.glob("g1_*.py")},
            "runner_sha256": file_digest(Path(__file__)),
            "acceptance": ("Unitree policy walking: floating base, gravity, uprightness, body velocity tracking, "
                           "and repeated unloaded foot swings followed by measured contact verified."
                           if args.controller == "walk" else
                           "Joint PD standing on flat ground with gravity and a floating base; "
                           "height, tilt, drift, settled joint tracking and root speeds verified."
                           if controller is not None else
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
