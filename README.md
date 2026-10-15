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
- `scripts/g1_standing.py`: named-joint standing PD drives, ankle feedback, and stability checks.
- `scripts/g1_policy.py`: official Unitree observation layout, recurrent inference, and named leg mapping; no simulator imports.
- `scripts/fetch_g1_policy.py`: pinned policy download, checksum verification, and provenance; no simulation.
- `scripts/g1_walking.py`: Isaac Sim joint drives, upper-body hold, leg efforts, and foot telemetry; no file output.
- `scripts/g1_walk_validation.py`: walking acceptance from measured base/foot states; no simulator or file access.
- `scripts/g1_video.py`: measured frame scheduling and MP4 encoding; simulation capture is supplied by the runner.
- `scripts/g1_outputs.py`: validated states, screenshots, and result archive.
- `tests/test_g1_outputs.py`: CPU-only output safety and serialization checks.
- `tests/test_g1_standing.py`: joint mapping, feedback direction, standing acceptance, and CLI checks.
- `tests/test_g1_walking.py`: observation frames/order, recurrent inference, false walking rejection, timing, and video scheduling.
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

## Headless G1 standing and collision ground

The default scenario now controls standing on a 10-meter ground plane using
the official `Isaac/Robots/Unitree/G1_23dof/g1.usd` asset. The asset reports
37 runtime DOFs, including fingers; the controller matches exact joint names
and rejects an incompatible model instead of assuming a numeric joint order.
CPU PhysX advances the small scene and the GPU renders its screenshots.

The nominal posture, actuator grouping, armatures, and 8/4 solver iterations
are based on NVIDIA Isaac Lab's
[G1_CFG](https://github.com/isaac-sim/IsaacLab/blob/76aeb425d214f04bd2895f5f956066c3ec716e81/source/isaaclab_assets/isaaclab_assets/robots/unitree.py#L257-L361).
Its BSD-3-Clause notice is retained in `LICENSES/IsaacLab-BSD-3-Clause.txt`.
This is a local static-standing controller, with adaptations for this USD:
leg/torso stiffness is five times the nominal configuration, their damping is
40 Nm*s/rad, and ankle stiffness/damping is 200 Nm/rad / 20 Nm*s/rad. The
50 Nm ankle effort cap follows G1_29DOF_CFG in the same source. The initial
base height is 0.72 m. Runtime PD gains are set and read back through PhysX
tensors. Base tilt, angular velocity, position drift, and linear velocity
provide bounded ankle-target corrections on every physics step.

The controller sets the initial joint/base state once. During the measured run
it sends only joint position/velocity targets: gravity stays enabled and the
base stays floating. This implementation holds a static stance on flat ground;
it does not load a learned locomotion policy. Disturbance recovery and walking
are outside the validated behavior.

Standing defaults to 2,000 steps at 0.005 s (200 Hz), giving 10 seconds.
Use `--steps 12000` for 60 seconds. Standing requires a physics step no greater
than 0.005 s and at least 5 seconds of measured simulation. The previous
60 Hz scene-check timestep is insufficient for this standing controller.

On the `/home/wl/RSS` workstation, activate its Conda environment and choose a
new output directory for each run:

```bash
conda activate isaacsim-6.0.1
cd /home/wl/RSS
g1_stand_run="g1-standing-$(date +%Y%m%d-%H%M%S)"
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --controller stand --steps 12000 --dt 0.005 \
  --output-dir "$PWD/.runtime/isaacsim-6.0.1/outputs/$g1_stand_run"
```

Standing success requires base height between 0.55 and 0.90 m, tilt at most
15 degrees, and horizontal drift at most 0.25 m throughout the run. After
2 seconds of settling, the maximum nominal joint-position error must be at
most 0.35 rad, root linear speed at most 0.30 m/s, and angular speed at most
0.50 rad/s. Every state is checked; any violation produces `failure.json`
and a nonzero exit code. `summary.json` records thresholds, measured extrema,
all joint drive parameters, and the pinned official reference.

The original loading/collision smoke test remains available with
`--controller none`. Its defaults remain 600 steps at 1/60 s, and it does not
check standing. The historical successful cluster run was Slurm job `194616`
on `6000ada` (2026-10-09); those original results were scene-check evidence.

On the original cluster, run either mode inside a GPU allocation, retain its
approved per-job SSD runtime override, and keep a 30-minute job bound. The
launcher continues to support `--python`, default streaming, and `--check`.

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
`summary.json` before interpreting results: its `controller` field identifies
standing or the original scene check. Only standing mode includes a passing
`standing` stability report. Download/viewing on Windows is a manual handoff
and is not verified by the server-side run.

CPU-only checks use unittest, NumPy, and the existing PyTorch installation.
Set `TMPDIR` to the approved SSD `runtime/g1-ground-validation` directory and
run `python -B -m unittest discover -s tests -v`.

Official instructions:
https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_python.html

## 本机 G1 行走（Isaac Sim 6.0.1）

行走使用宇树官方 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b)
的预训练 `deploy/pre_train/g1/motion.pt`，固定提交
`276801e46c5d433564f24658bac64f254b7d2d4b`。观测布局、缩放、默认腿部姿态、
PD 参数和 0.8 秒相位周期来自其 `deploy_mujoco.py` / `configs/g1.yaml`；
力矩上限来自同提交的 `g1_12dof.xml`。许可保存在
`LICENSES/Unitree-RL-Gym-BSD-3-Clause.txt`。策略 SHA256 为
`cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`，加载前强制校验。

这是对当前 37 自由度 NVIDIA G1 模型的本地适配：按名称映射双腿 12 个关节；
上身和手指以零位 PD 保持，对应官方 12 自由度模型固定上身的设定。
腿部使用显式 PD 力矩，关闭腿部隐式驱动，避免重复施力。物理步长固定 0.002 秒
（500Hz），策略每 10 个物理步更新一次（50Hz），LSTM 记忆在每次运行开始时重置。
根节点仅初始化一次；运行中保留重力、浮动基座和脚地接触。

本机仓库为 `/home/wl/RSS`，Conda 环境为
`/home/wl/miniconda3/envs/isaacsim-6.0.1`。以下命令下载经过校验的策略，然后运行
10 秒、前进速度指令 0.5m/s 的仿真，并录制 25fps、1280×720、1 倍速视频：

```bash
source /home/wl/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim-6.0.1
cd /home/wl/RSS
python -B scripts/fetch_g1_policy.py
walk_output="$PWD/.runtime/isaacsim-6.0.1/outputs/g1-walk-$(date +%Y%m%d-%H%M%S)"
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --controller walk --command 0.5 0 0 --video --output-dir "$walk_output"
```

`--command VX VY YAW_RATE` 是机器人自身坐标下的前进、侧移速度（m/s）和偏航角速度（rad/s）。
命令会一直保持到本次仿真结束。当前接口限制范围为 VX [-0.5,0.8]、VY [-0.3,0.3]、
YAW_RATE [-0.5,0.5]；这些边界是输入保护，不能代表全部组合均已实测。
默认前进命令不含绝对航向闭环，行进航向可能缓慢漂移。
`--steps 30000` 可运行 60 秒；不加 `--video` 可加快物理验证。
保留原有 `--controller stand` 和 `--controller none` 模式。

输出目录必须位于仓库 `.runtime/isaacsim-6.0.1/outputs/` 下且不存在。
`simulation.mp4` 为跟随视角视频，`states.csv` 为每个物理步的根节点和关节状态，
`feet.csv` 为左右踝链接位置及接触力，`video-frames.csv` 记录每帧对应的物理步和仿真时间。
`summary.json` 包含策略来源、代码哈希、文件哈希、实际步数和行走验收指标；
`results.zip` 包含结果及视频。渲染每帧前后核对物理计数，确保视频录制不会推进物理。

行走验收在全程检查根节点高度与倾角，忽略最初 2 秒后统计速度误差、
单脚支撑比例和交替落脚。一次有效落脚必须经历至少 40ms 无载、至少 15mm 离地，
再测得落地接触力；滑动、双脚同时跳动或仅位置变化不能满足验收。
验收通过针对当前平地场景，未验证台阶、复杂地形或真实机器人部署。

模块之间通过状态数据交互：`g1_policy` 负责“状态 → 关节目标”，
`g1_walking` 负责“目标 → 仿真关节力矩”及测量，`g1_walk_validation` 只判断测量结果，
`g1_scene` 管理场景与相机，`g1_video` 管理录像，`run_g1_ground` 负责生命周期和调度。
运行检查：`python -B -m unittest discover -s tests -v`。

本机 2026-10-15 验证：60 秒 / 30000 物理步通过，平均机体前向速度
0.5145m/s（指令 0.5m/s），最大倾角 4.8066°，最小根节点高度 0.6416m。
最初 2 秒后左右脚各 72 次有效落脚、143 次交替落脚，单脚支撑比例 89.76%。
本机结果目录为 `.runtime/isaacsim-6.0.1/outputs/g1-walk-60s-20261015-final/`；
10 秒 / 250 帧行走视频位于
`.runtime/isaacsim-6.0.1/outputs/g1-walk-video-20261015-final/simulation.mp4`。
结果目录不随 Git 克隆分发；对应 `summary.json` 和日志目录中的
`*.validation.json` 保存可核对的状态、源代码哈希及结果包校验依据。
