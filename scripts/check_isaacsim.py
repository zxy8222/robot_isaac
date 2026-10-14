"""Verify Isaac Sim initialization, an RTX viewport, and sustained updates.

The caller configures libraries and runtime paths. This script creates a small
in-memory scene, checks rendering updates, and saves no scene assets.
"""

import time
from importlib.metadata import version


def main() -> None:
    """Keep rendering for at least 30 seconds, then release the application."""
    print(f"ISAACSIM_VERSION={version('isaacsim')}", flush=True)
    # Omniverse/pxr imports must occur only after SimulationApp starts.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "multi_gpu": False,
            "active_gpu": 0,
            "limit_cpu_threads": 4,
        }
    )
    print("SIMULATION_APP_CREATED", flush=True)
    try:
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("SimulationApp did not create a USD stage")
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No active viewport; RTX rendering is unverified")
        cube = UsdGeom.Cube.Define(stage, "/World/StartupCheckCube")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
        light = UsdLux.DomeLight.Define(stage, "/World/StartupCheckLight")
        light.CreateIntensityAttr(500.0)
        viewport.camera_path = "/OmniverseKit_Persp"
        print("RTX_VIEWPORT_AND_STAGE_CREATED", flush=True)

        started = time.monotonic()
        next_report = 10.0
        frames = 0
        while time.monotonic() - started < 30.0 or frames < 120:
            if not app.is_running():
                raise RuntimeError("Isaac Sim stopped before verification finished")
            app.update()
            frames += 1
            elapsed = time.monotonic() - started
            if elapsed >= next_report:
                print(f"RENDER_PROGRESS seconds={elapsed:.1f} frames={frames}", flush=True)
                next_report += 10.0
        print(f"RENDER_CHECK_PASSED frames={frames}", flush=True)
    finally:
        # A normal close is attempted even if a Python-side check fails.
        print("CLOSING_SIMULATION_APP", flush=True)
        app.close()


if __name__ == "__main__":
    main()
