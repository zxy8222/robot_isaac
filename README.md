# Isaac Sim 6.0.1 remote workspace

This workspace uses Python 3.12 and the pinned pip release `6.0.1.0`.
`requirements-isaacsim-6.0.1.txt` records direct dependencies. Installation and
GPU verification results are stored under `.runtime/isaacsim-6.0.1/`.
The workspace resides on HDD; `/home/xinyue027/isaac_ws` remains a compatibility
link. Installation scratch and GPU runtime caches use the approved SSD directory
`/projects/xvla/.isaacsim-6.0.1-scratch/`, without changing other project data.

## Repository scope and environment recreation

The repository stores source code, tests, the direct dependency pins, and this
setup guide. `.gitignore` excludes installed virtual environments, `.runtime/`,
`.cache/`, Python bytecode, logs, and common credential files. These exclusions
do not delete or move the existing server environment. Review every staged file
for secrets before publishing; ignore rules are not a complete secret scanner.
Generated screenshots, states, local GLU binaries, and historical job logs are
not included in a fresh clone. Historical run results mentioned below describe
the original server validation, not files bundled with the repository.

On a compatible Linux x86-64 host, clone the source and enter the new directory:

```bash
git clone https://github.com/zxy8222/robot_isaac.git
cd robot_isaac
```

Use Python 3.12 and create the expected environment only when that path does not
already exist. On this cluster, run dependency installation and package downloads
in a compute job, with scratch redirected to your own approved SSD directory:

```bash
test ! -e .venv-isaacsim-6.0.1 && python3.12 -m venv .venv-isaacsim-6.0.1
```

Stop if the command fails; do not overwrite an existing environment. Activate
the new interpreter as shown below, then follow the pinned-dependency reinstall
section. The pins reproduce direct package versions, not a complete transitive
lockfile or the host's NVIDIA driver. The tested interpreter was Python 3.12.9;
the successful G1 run used RTX 6000 Ada and driver 595.71.05. Do not assume that
arbitrary OS, driver, GPU, or package-index combinations have been validated.

The `/projects/xvla/` scratch path, `/home/xinyue027/` compatibility link, and
Slurm partition/module names describe the original cluster. Replace documented
paths with your own authorized storage locations on another host. The scripts
locate the workspace relative to themselves; preserve Slurm GPU visibility and
the existing launcher modes. A fresh clone does not recreate HDD quotas or
filesystem links. G1 assets are fetched from NVIDIA's asset service at runtime;
the repository is not an offline asset bundle.

The launcher also needs GLU and Vulkan/OpenGL support. This workspace uses a
local GLU library under `.runtime/deps/root/usr/lib/x86_64-linux-gnu/` and the
cluster's `GCCcore/13.2.0` / `libglvnd/1.7.0` modules. On Ubuntu 24.04 amd64,
the original local library can be recreated without installing a system package:

```bash
mkdir -p .runtime/deps/packages .runtime/deps/root
(cd .runtime/deps/packages && apt-get download libglu1-mesa=9.0.2-1.1build1)
dpkg-deb --extract \
  .runtime/deps/packages/libglu1-mesa_9.0.2-1.1build1_amd64.deb \
  .runtime/deps/root
```

Run that block only for a new local dependency directory in a compute job; stop
if the download fails. Package availability and system dependencies can differ
on another host: ask its administrator for compatible GLU/Vulkan/OpenGL libraries
instead of copying unrelated binaries or changing the server driver.

Before launching on a recreated environment, read and accept NVIDIA's applicable
Omniverse EULA yourself. The existing launcher defaults `OMNI_KIT_ACCEPT_EULA` to
`Y` because the original workspace owner already accepted it; cloning this
repository does not mean that a new user has accepted those terms.

## Files and responsibilities

- `.venv-isaacsim-6.0.1/`: isolated interpreter and installed dependencies.
- `.runtime/deps/`: workspace-local GLU library; OpenGL comes from cluster modules.
- `scripts/start_isaacsim.sh`: GPU checks, modules, runtime paths, and WebRTC launch.
- `scripts/check_isaacsim.py`: initialization and 30-second RTX update check.
- `scripts/run_g1_ground.py`: bounded headless G1 physics run and lifecycle.
- `scripts/g1_scene.py`: robot, collision ground, lighting, and camera setup.
- `scripts/g1_outputs.py`: validated states, screenshots, and result archive.
- `tests/test_g1_outputs.py`: CPU-only output safety and serialization checks.
- `.runtime/isaacsim-6.0.1/diagnostics-5.1/`: representative old crash evidence.

## Activate the interpreter

```bash
source .venv-isaacsim-6.0.1/bin/activate
python --version
python -m pip show isaacsim
```

Select `.venv-isaacsim-6.0.1/bin/python` in your IDE. Do not edit `pyvenv.cfg`
to change the Python or Isaac Sim version.

## Request a GPU on cluster02

From this workspace on a login node:

```bash
srun --pty --partition=cluster02 --gres=gpu:rtx5090:1 \
  --time=02:00:00 bash -l
```

The GPU model can be changed to an available supported RTX model. Let Slurm assign
the model-specific CPU and memory allocation; do not set `--cpus-per-task` or
`--mem` for these GPU jobs. Interactive jobs cannot exceed 2 hours.

## Check rendering and launch streaming

Inside the allocated GPU shell, from this workspace:

```bash
export ISAACSIM_RUNTIME_DIR=/projects/xvla/.isaacsim-6.0.1-scratch/runtime
bash scripts/start_isaacsim.sh --check
bash scripts/start_isaacsim.sh
```

The second command starts the full headless WebRTC experience. Connect using
the NVIDIA Isaac Sim WebRTC Streaming Client. Client connectivity additionally
requires a cluster-approved route to the compute node for TCP 49100 and UDP
47998; an SSH TCP tunnel alone does not carry UDP. The server being ready does
not by itself verify that your local client can reach it.

The launcher preserves Slurm GPU visibility restrictions and loads
`GCCcore/13.2.0` and `libglvnd/1.7.0`. It sets the already accepted Omniverse
EULA environment variable. Override `ISAACSIM_RUNTIME_DIR` only when you intend
to use a different approved writable runtime location. Additional Kit arguments
can be appended to the launch command.

## Reinstall the pinned dependencies

Use Python 3.12 in a new virtual environment. Run installation in a CPU batch job
or a compute-node shell, not on a login node. Redirect scratch and caches to the
approved SSD location before installing; do not use home or `/tmp` for downloads.
For concurrent jobs, give each job a distinct temporary directory.

```bash
isaac_scratch=/projects/xvla/.isaacsim-6.0.1-scratch
export TMPDIR="$isaac_scratch/tmp/${SLURM_JOB_ID:?Run inside a Slurm job}"
export PIP_CACHE_DIR="$isaac_scratch/pip"
export CONDA_PKGS_DIRS="$isaac_scratch/conda"
export HF_HOME="$isaac_scratch/hf"
export TRITON_CACHE_DIR="$isaac_scratch/triton"
export CUDA_CACHE_PATH="$isaac_scratch/cuda"
export XDG_CACHE_HOME="$isaac_scratch/cache"
export XDG_CONFIG_HOME="$isaac_scratch/runtime/config"
export XDG_DATA_HOME="$isaac_scratch/runtime/data"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' 'torchaudio==2.11.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir \
  -r requirements-isaacsim-6.0.1.txt --extra-index-url https://pypi.nvidia.com
python -m pip check
python -B -c 'import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__)'
```

All three PyTorch-family packages must use the same CUDA build. A successful
`pip check` checks dependency metadata, not whether binary operators can load.
Keep installation and validation logs under `.runtime/isaacsim-6.0.1/logs/`;
Kit's portable runtime, caches, and temporary files stay on SSD via the override
shown above. Existing crash evidence and the local GLU dependency remain on HDD.

## Headless G1 and collision ground

The scenario uses the official `Isaac/Robots/Unitree/G1_23dof/g1.usd` asset and
a 10-meter ground plane. It advances 600 physics steps at 1/60 second, using
CPU PhysX for this small scene and RTX on the allocated GPU for screenshots.
It does not start WebRTC, use ROS 2, or implement a balance/walking controller.
The floating-base robot can fall; standing is not an acceptance condition.
The asset path's name does not guarantee the total runtime DOF count: the first
local load reported 37 DOFs. Always use the actual names/count in `summary.json`
rather than assuming there are exactly 23 joints. That first run failed before
sampling due to a velocity-vector conversion error, now corrected in the script.
The authorized retry, Slurm job `194616` on `6000ada`, completed successfully on
2026-10-09 in 11 minutes 51 seconds with exit code zero. It measured 600 physics
steps and 10 seconds, saved 601 state rows and three valid 1280x720 PNGs, and
passed independent state, image, SHA-256, and ZIP CRC checks. The screenshots
show G1 and the ground; this remains a smoke test, not proof of active balance.
Its results are in `.runtime/isaacsim-6.0.1/outputs/g1-ground-194616/`.

Inside a Slurm GPU batch job, use a fresh per-job runtime and output directory:

```bash
export ISAACSIM_RUNTIME_DIR=/projects/xvla/.isaacsim-6.0.1-scratch/runtime/g1-ground-${SLURM_JOB_ID:?}
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --steps 600 --dt 0.016666666666666666 \
  --output-dir "$PWD/.runtime/isaacsim-6.0.1/outputs/g1-ground-$SLURM_JOB_ID"
```

Submit only one `6000ada` GPU, let Slurm choose CPU/memory, redirect all scratch
caches to the fresh SSD runtime, and bound the job to 30 minutes including
startup and shutdown. Keep stdout/stderr in `logs/g1-ground-<job-id>.log`.
The launcher also supports `--python <existing-script.py> [arguments]` while
preserving its existing default streaming mode and `--check` behavior.

Successful results contain `states.csv` (root pose, root velocities, DOF
positions/velocities), `summary.json` (actual times, steps, DOF names and hashes),
`initial.png`, `mid.png`, `final.png`, `scene.usda`, and `results.zip`.
An existing output directory is refused. Failures retain partial output and
`failure.json`; they do not produce a success archive. PNGs use a fixed overview
camera and are captured without advancing physics. The USD references external
assets: it is not a fully collected offline asset pack.

On Windows, download `results.zip` through your Remote SSH IDE, or use PowerShell
with the same login address you already use for SSH (not the GPU node IP):

```powershell
$g1JobId = "194616" # Replace for a later successful job.
$g1Archive = "$env:USERPROFILE\Downloads\g1-ground-$g1JobId.zip"
$g1Directory = "$env:USERPROFILE\Downloads\g1-ground-$g1JobId"
if ((Test-Path -LiteralPath $g1Archive) -or (Test-Path -LiteralPath $g1Directory)) {
    throw "Choose unused download and extraction paths; existing results are preserved."
}
scp "xinyue027@<your-SSH-login-address>:/home/xinyue027/isaac_ws/.runtime/isaacsim-6.0.1/outputs/g1-ground-$g1JobId/results.zip" $g1Archive
if ($LASTEXITCODE -ne 0) { throw "SCP failed; do not extract an incomplete download." }
Expand-Archive -LiteralPath $g1Archive -DestinationPath $g1Directory
```

Open the PNG files with Photos and the CSV with Excel or a text editor. Inspect
`summary.json` before interpreting results: the smoke test checks loading,
collision configuration, finite states, measured time stepping, and capture,
not robot balance or a trained policy. Download/viewing on Windows is a manual
handoff and is not verified by the server-side run.

CPU-only checks use the standard library, not newly installed test dependencies.
Set `TMPDIR` to the approved SSD `runtime/g1-ground-validation` directory and
run `python -B -m unittest discover -s tests -p test_g1_outputs.py -v`.

Official instructions:
https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html
