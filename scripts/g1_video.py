"""Render measured simulation frames and encode a verified 1x H.264 video.

The caller supplies a capture callback that keeps physics paused while rendering.
This module owns frame timing and encoding, without importing the simulation SDK.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path

from g1_outputs import file_digest


class VideoRecorder:
    def __init__(self, directory: Path, dt: float, steps: int, capture, *, fps: int = 25) -> None:
        self.interval = round(1 / (fps * dt))
        if self.interval < 1 or not math.isclose(self.interval * dt * fps, 1, abs_tol=1e-9) or steps % self.interval:
            raise ValueError("Video requires a timestep and duration divisible by the frame interval")
        self.directory, self.fps, self.steps, self.capture = directory, fps, steps, capture
        self.frames = []
        (directory / "video-frames").mkdir(exist_ok=False)

    def observe(self, state) -> None:
        if state.step == 0 or state.step % self.interval:
            return
        path = self.directory / "video-frames" / f"{len(self.frames):06d}.png"
        self.capture(path, state)
        self.frames.append((len(self.frames), state.step, state.simulation_time, path.name))
        if len(self.frames) % self.fps == 0:
            print(f"G1_VIDEO_PROGRESS frames={len(self.frames)} time={state.simulation_time:.3f}", flush=True)

    def finish(self) -> dict:
        if len(self.frames) != self.steps // self.interval:
            raise RuntimeError("Recorded video has missing simulation frames")
        with (self.directory / "video-frames.csv").open("x", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["frame", "physics_step", "simulation_time_s", "file"])
            writer.writerows(self.frames)
        env = os.environ.copy()
        env.pop("LD_LIBRARY_PATH", None)
        path = self.directory / "simulation.mp4"
        command = ["/usr/bin/ffmpeg", "-hide_banner", "-nostdin", "-n", "-framerate", str(self.fps),
                   "-start_number", "0", "-i", str(self.directory / "video-frames/%06d.png"),
                   "-frames:v", str(len(self.frames)), "-c:v", "libx264", "-preset", "fast",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        with (self.directory / "video-encoding.log").open("x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=300)
        probe = json.loads(subprocess.check_output([
            "/usr/bin/ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames:format=duration",
            "-of", "json", str(path)], env=env, text=True, timeout=30))
        stream = probe["streams"][0]
        duration = len(self.frames) / self.fps
        if (stream["codec_name"] != "h264" or (stream["width"], stream["height"]) != (1280, 720)
                or int(stream["nb_read_frames"]) != len(self.frames)
                or not math.isclose(float(probe["format"]["duration"]), duration, abs_tol=0.01)):
            raise RuntimeError(f"Encoded video verification failed: {probe}")
        return {"file": path.name, "frames": len(self.frames), "fps": self.fps,
                "duration_s": duration, "playback_speed": 1.0, "sha256": file_digest(path),
                "first_simulation_time_s": self.frames[0][2], "last_simulation_time_s": self.frames[-1][2],
                "physics_capture_interval_steps": self.interval, "rendering_did_not_advance_physics": True}
