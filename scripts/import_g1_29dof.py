"""Run a bounded Mode-15 conversion, standalone preview and physics check."""

import argparse
import math
import sys
import traceback
from importlib.metadata import version
from pathlib import Path

from g1_outputs import capture_png, file_digest, write_json
from g1_urdf_import import convert_asset, validate_model, verify_sources
import g1_urdf_config as model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=model.OUTPUT_DIR)
    args = parser.parse_args()
    allowed = model.ROOT / ".runtime/assets"
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == allowed or not args.output_dir.is_relative_to(allowed):
        parser.error(f"Import output must be inside {allowed}")
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing imports are preserved")
    return args


def main() -> int:
    args = parse_args()
    manifest = verify_sources()
    if version("isaacsim") != "6.0.1.0":
        raise RuntimeError("This import requires Isaac Sim 6.0.1.0")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "LICENSE.txt").open("x") as output:
        output.write((model.SOURCE_ROOT / "LICENSE").read_text())
    write_json(args.output_dir / "source-manifest.json", manifest)
    app = timeline = None
    status = 1
    try:
        sys.argv = [sys.argv[0], "--portable-root", str(args.portable_root)]
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0,
                             "limit_cpu_threads": 4, "width": 1280, "height": 720})
        import omni.timeline
        import omni.kit.app
        from isaacsim.core.experimental.utils.backend import use_backend
        from isaacsim.core.simulation_manager import SimulationManager
        from g1_scene import build_scene

        def update():
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped during URDF import")
            app.update()

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        update()
        asset, conversion = convert_asset(args.output_dir)
        print(f"UNITREE_G1_USD_CREATED={asset}", flush=True)
        stage, robot, viewport, _ = build_scene(update, str(asset), ground_size=10.0)
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        timeline.commit()
        preview_scene = args.output_dir / "preview.usda"
        if not stage.GetRootLayer().Export(str(preview_scene)):
            raise RuntimeError("Failed to export the preview scene")
        SimulationManager.setup_simulation(dt=.002, device="cpu")
        SimulationManager.enable_fabric(False)
        SimulationManager.initialize_physics()
        for _ in range(30):
            update()
        properties = validate_model(robot, stage)
        before = SimulationManager.get_num_physics_steps()
        preview = capture_png(viewport, update, args.output_dir / "preview.png")
        if SimulationManager.get_num_physics_steps() != before:
            raise RuntimeError("Preview rendering unexpectedly advanced physics")
        for _ in range(20):
            SimulationManager.step(steps=1)
            with use_backend("tensor"):
                values = [*robot.get_world_poses(), robot.get_dof_positions(), robot.get_dof_velocities()]
                if not all(math.isfinite(float(x)) for value in values for x in value.numpy().flat):
                    raise RuntimeError("Mode 15 has a non-finite physics state")
        if SimulationManager.get_num_physics_steps()-before != 20:
            raise RuntimeError("Physics smoke-test step count mismatch")
        verify_sources()
        source_files = (Path(__file__), *[Path(__file__).parent / name for name in
                        ("g1_urdf_config.py", "g1_urdf_import.py", "g1_source_cache.py", "fetch_g1_urdf.py", "g1_scene.py")])
        write_json(args.output_dir / "import-summary.json", {
            "status": "passed", "isaacsim_version": version("isaacsim"), "robot_variant": model.VARIANT,
            "source_commit": model.SOURCE_COMMIT, "source_urdf": str(model.URDF_PATH),
            "source_urdf_sha256": model.URDF_SHA256, "source_manifest": manifest,
            "asset_usd": str(asset), "conversion": conversion, "imported_robot_prim": model.ROBOT_PATH,
            "imported_model": properties, "preview_scene": str(preview_scene), "preview": preview,
            "physics_smoke_steps": 20, "physics_smoke_duration_s": .04,
            "acceptance": "Mode-15 conversion, dependencies, joints, limits, mass, inertia, collisions and bounded physics; no walking acceptance.",
            "source_sha256": {p.name: file_digest(p) for p in source_files},
            "files_sha256": {str(p.relative_to(args.output_dir)): file_digest(p)
                              for p in args.output_dir.rglob("*") if p.is_file()},
        })
        print(f"UNITREE_G1_IMPORT_PASSED dofs=29 mass_kg={properties['total_mass_kg']:.6f}", flush=True)
        status = 0
        return 0
    except Exception as error:
        traceback.print_exc()
        write_json(args.output_dir / "import-failure.json", {"error": str(error), "type": type(error).__name__})
        return 1
    finally:
        if timeline is not None:
            timeline.stop()
            timeline.commit()
        if app is not None:
            print(f"UNITREE_G1_CLOSING exit_status={status}", flush=True)
            app.close(exit_code=status)


if __name__ == "__main__":
    raise SystemExit(main())
