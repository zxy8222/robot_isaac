# Isaac Sim 6.0.1：G1 Mode 15 ONNX 行走

当前主要流程使用宇树官方 `g1_29dof_mode_15.urdf` 转换的 USD，加载
`unitree_rl_lab` 的 29-DOF `velocity/v0/policy.onnx` 控制行走。
观测、动作和 Kp/Kd 按关节名称映射到 Isaac Sim DOF；沿用官方部署增益，
保留 Mode 15 原始力矩与速度限制，四个踝关节力矩上限为 **35Nm**。
原 NVIDIA `G1_23dof`、官方 12-DOF USD 以及 `motion.pt` 路径继续保留。

本机已验证 60 秒平地行走：指令 0.5m/s，实测平均前向速度 0.4923m/s，
最大倾角 2.83°，全程未触发关节力矩限幅或速度越限。
控制细节、固定上游版本和验证范围见[Mode 15 ONNX 基线](#mode15-onnx)。

仓库提交代码、测试、依赖版本和说明。USD、STL、策略权重、Conda 环境、视频和运行结果
保存在忽略的 `.runtime/` / `.venv*` 中；克隆后按下列步骤下载固定版本并生成 USD。
文中本机历史结果路径不随 Git 分发。

<a id="mode15-quickstart"></a>

## Mode 15 快速开始

需要 Python 3.12、Isaac Sim pip 包 `6.0.1.0`、可用的 NVIDIA RTX GPU 和图形运行库。
`--video` 还需要系统中的 `ffmpeg`（含 libx264）与 `ffprobe`。
以下示例复用已安装的 Conda 环境 `isaacsim-6.0.1`：

```bash
git clone https://github.com/zxy8222/robot_isaac.git
cd robot_isaac
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate isaacsim-6.0.1

# 启动器从仓库内的这个路径查找解释器；已有环境或链接保持不变。
if [ ! -e .venv-isaacsim-6.0.1 ] && [ ! -L .venv-isaacsim-6.0.1 ]; then
  ln -s "$CONDA_PREFIX" .venv-isaacsim-6.0.1
fi
python -m pip install -r requirements-g1-onnx.txt
python -m pip check
```

在本机 `/home/wl/RSS` 中继续运行时，直接进入该目录并激活原环境，无需再次克隆或建立链接。
原 Conda 环境位于 `/home/wl/miniconda3/envs/isaacsim-6.0.1`。
新主机尚无环境时，先用 `conda create -n isaacsim-6.0.1 python=3.12` 创建，
按[固定依赖安装说明](#reinstall-the-pinned-dependencies)安装 CUDA PyTorch 和 Isaac Sim，
再安装 ONNX 依赖并建立上述解释器链接。该安装说明中 Slurm/SSD 路径仅适用于历史集群，
其他主机应使用自己的存储路径和 GPU 分配方式。

首次克隆后下载官方源文件、生成 Mode 15 USD，并下载策略及部署配置：

```bash
python -B scripts/fetch_g1_urdf.py --model unitree29
bash scripts/start_isaacsim.sh --python scripts/import_g1_29dof.py
python -B scripts/fetch_g1_onnx.py
```

已有 USD 时跳过导入命令：导入器拒绝覆盖现有输出目录。
下载器会核对固定上游提交及文件哈希，既有文件校验失败时停止。
Mode 15 USD 入口为
`.runtime/assets/unitree-g1-29dof-mode-15/5994d4faef0a9cadd3287f8de0199a67eeb2a259/g1_29dof_mode_15.usd`。
该入口引用同目录的 `converted/`，移动模型时需保留整个资产目录。

运行 10 秒并录制 25fps、1280×720、1 倍速行走视频：

```bash
mode15_output="$PWD/.runtime/isaacsim-6.0.1/outputs/g1-mode15-$(date +%Y%m%d-%H%M%S)"
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --robot unitree29 --controller walk --command 0.5 0 0 --video \
  --output-dir "$mode15_output"
```

添加 `--steps 30000` 可运行 60 秒；去掉 `--video` 可加快物理验证。
`--robot unitree29` 会选择 Mode 15 和官方 ONNX 策略，不传 `--robot` 时仍选择原 NVIDIA 模型。
输出目录中 `simulation.mp4` 为视频，`states.csv` 为测量状态，`joint-control.csv` 为
29 关节逐步动作、目标/实际角及 `tau_raw` / `tau_cmd`，`feet.csv` 为脚部测量，
`summary.json` 为映射、控制/接触参数、哈希及验收结果，`results.zip` 为结果包。
失败时保存 `failure.json` 并返回非零退出码；每次运行需选择新的输出目录。

完成上述下载后可运行 CPU 测试；若要同时执行旧 TorchScript 策略的集成测试，先下载它：

```bash
python -B scripts/fetch_g1_policy.py
python -B -m unittest discover -s tests -v
```

当前完整测试为 47 项，覆盖 DOF 顺序打乱、历史观测、官方增益与 Mode 15 限制、
旧模型兼容性和行走验收。无需启动 Isaac Sim 即可执行这些 CPU 测试。

## Historical cluster setup

The following sections retain the original remote-cluster setup and its run
history. The current workstation workflow is documented above. Both use Python
3.12 and the pinned Isaac Sim pip release `6.0.1.0`.
`requirements-isaacsim-6.0.1.txt` records the direct simulation dependencies.
On the original cluster, the workspace resided on HDD, `/home/xinyue027/isaac_ws`
was a compatibility link, and runtime caches used
`/projects/xvla/.isaacsim-6.0.1-scratch/` on SSD.

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
- `scripts/g1_models.py`: exact joint-set recognition and separate 12/37-DOF actuator/initial-state settings.
- `scripts/g1_control_outputs.py`: per-step policy actions, target/actual angles, PD torques, and tracking summaries.
- `scripts/g1_urdf_config.py`: pinned official 12-DOF and 29-DOF mode-15 source identities and separate asset paths.
- `scripts/fetch_g1_urdf.py`: verified URDF/mesh/license download; `--model unitree29` selects mode 15.
- `scripts/g1_urdf_import.py`: shared URDF conversion, model checks, and comparison preview.
- `scripts/import_g1_12dof.py`, `scripts/import_g1_29dof.py`: model-specific import entry points.
- `scripts/g1_onnx_policy.py`: official 29-DOF ONNX configuration, named mapping, and observation history; no simulator dependency.
- `scripts/g1_onnx_walking.py`: Mode-15 explicit PD, original actuator caps, and contact/constraint telemetry.
- `scripts/fetch_g1_onnx.py`: pinned RL Lab policy, deployment YAML, source references and notices.
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

## 单独导入宇树官方 G1 12-DOF USD

保留现有 NVIDIA `Isaac/Robots/Unitree/G1_23dof/g1.usd`。另提供
`scripts/fetch_g1_urdf.py` 下载固定提交 `276801e46c5d433564f24658bac64f254b7d2d4b`
中的官方 `resources/robots/g1_description/g1_12dof.urdf` 及其引用的 27 个 STL 网格。
下载文件按 Git blob SHA1 校验，并记录 SHA256；已有文件校验失败时会停止，避免覆盖。
来源、许可与缓存保存在 `.runtime/unitree-rl-gym/<commit>/`。

`scripts/import_g1_12dof.py` 使用 Isaac Sim 6.0.1 的 `URDFImporter` 转换：合并固定关节，
保留 12 个可动腿关节、原始碰撞和惯性定义，开启重力并保留浮动基座。
关节使用 force 类型、无位置目标驱动，stiffness/damping 为零。
不从可视网格重新生成碰撞，不添加密度覆盖。转换产物、来源清单与许可存入独立目录。

```bash
source /home/wl/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim-6.0.1
cd /home/wl/RSS
python -B scripts/fetch_g1_urdf.py
bash scripts/start_isaacsim.sh --python scripts/import_g1_12dof.py
```

默认输出目录：
`.runtime/assets/unitree-g1-12dof/276801e46c5d433564f24658bac64f254b7d2d4b/`。
已有输出目录会使导入命令停止；如需重新转换，通过 `--output-dir` 指定
`.runtime/assets/` 下的新目录。无需重新生成时，可以直接打开以下文件：

- `g1_12dof.usd`：新模型的独立入口，引用本目录内 `converted/` 的 USD 层。
- `comparison.usda`：两模型并排场景，原 NVIDIA 模型为 `/World/G1`，
  新模型为 `/World/UnitreeG1_12DOF`。
- `comparison.png`：并排预览，左侧为 NVIDIA 模型，右侧为官方 URDF 转换模型。
- `import-summary.json`：来源、转换参数、实际 PhysX 关节/质量/惯性与碰撞检查记录。
- `LICENSE.txt`、`source-manifest.json`：上游许可和经过校验的源文件清单。

单独移动新模型时，应保留入口 USD 和整个 `converted/` 目录；转换后的几何不再依赖源 STL。
并排场景仍引用 NVIDIA 原资产服务和本机新模型路径。

2026-10-19 本机验证：新模型为 12 个 DOF、13 个刚体，浮动基座质量矩阵维度为 18×18，
总质量 32.10685kg，与源 URDF 一致。关节位置/速度/力矩限位与源文件逐项核对通过；
保留 8 个半径 0.005m 的脚底碰撞球，资产依赖全部在输出目录内且无缺失。
两套模型同时完成 20 个物理步的加载与有限状态检查。

资产导入阶段的预览和短时物理检查不作为站立或行走验收。
同一运行入口现已支持通过 `--robot unitree12` 选择新模型，具体行走命令见下节。

## 使用 motion.pt 控制官方 12-DOF USD 行走

策略文件、47 维观测、默认腿部角度、0.25 动作缩放和官方 PD 参数沿用原实现。
`g1_models.py` 根据实际关节名称识别两套模型，12-DOF 模型的初始根节点高度为
0.793m，取自同一固定提交的官方 `g1_12dof.xml`。上身已在源 URDF 中固定并合并，
无需额外上身驱动；12 个腿关节的隐式 stiffness/damping 均为零。
控制器根据实际 link 路径定位双脚，不假设两套 USD 的层级一致。
运行时在场景覆盖层补齐 PhysX articulation 配置，独立资产文件保持不变。

物理频率为 500Hz，策略频率为 50Hz。每个物理步读取刚完成的仿真状态，
重新计算并发送显式 PD 力矩；策略周期之间保持目标角，力矩仍逐步更新。
策略开始前 20ms 使用官方默认角，随后保持 LSTM 状态连续更新。

```bash
source /home/wl/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim-6.0.1
cd /home/wl/RSS
walk_output="$PWD/.runtime/isaacsim-6.0.1/outputs/g1-unitree12-$(date +%Y%m%d-%H%M%S)"
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --controller walk --robot unitree12 --command 0.5 0 0 --video \
  --output-dir "$walk_output"
```

默认运行 10 秒，生成 25fps、1280×720、1 倍速 `simulation.mp4`；
`--steps 30000` 可验证 60 秒。原 NVIDIA 模型通过 `--robot nvidia` 选择，
不传 `--robot` 时仍默认使用它。12-DOF 模型支持 `walk` 和 `none` 模式，
原有 `stand` 控制器继续用于 NVIDIA 模型。

两套模型的行走模式均新增 `joint-control.csv`，记录每次物理步进前的
`action`、`targets`、`positions`、`velocities`、`tau_raw`、`tau_cmd` 和策略更新次数。
关节角单位为 rad，速度为 rad/s，力矩为 Nm，关节字段按官方策略的左右腿顺序排列。
`summary.json` 的 `joint_control` 包含实际更新间隔、稳定阶段跟踪 RMSE、
目标/实际角均值和力矩限幅统计，`controller` 保存读回确认的腿部零 Drive 增益。

10 秒运行包含 5001 行状态（含初始与最终状态）、5000 行控制命令及 499 次策略更新。
最终状态之后不再执行无用的策略推理。`tau_cmd` 为发送给 PhysX 的命令，
这些记录不应解释为物理力矩传感器测量。行走验收仍检查浮动基座、倾角、速度跟踪、
脚部离地与交替落脚，未扩展到复杂地形或真实机器人。

2026-10-19 本机验证：12-DOF 模型连续运行 60 秒 / 30000 物理步通过，
平均机体前向速度 0.4593m/s（指令 0.5m/s），最大倾角 4.0547°，
最小根节点高度 0.7641m。最初 2 秒后左右脚各 72 次有效落脚、143 次交替落脚，
单脚支撑比例 92.20%。12 个关节全程未触发显式 PD 力矩限幅。
稳态左右膝实际均角为 36.22° / 37.29°，目标均角为 33.60° / 34.48°；
仍有负载下的跟踪误差，左右踝 pitch 的支撑期 RMSE 均约 20.98°。
同次 NVIDIA 10 秒回归的膝实际均角为 56.95° / 55.86°，说明换用官方模型后
同一策略的屈膝程度有所降低。速度命令仍不包含绝对航向闭环。

本机结果位于 `.runtime/isaacsim-6.0.1/outputs/`：

- `g1-unitree12-walk-60s-20261019-final/`：60 秒状态、逐步控制记录、汇总和结果包。
- `g1-unitree12-walk-video-20261019-final/simulation.mp4`：10 秒、250 帧 H.264 视频。
- `g1-nvidia-walk-regression-20261019/`：原 NVIDIA 模型的 10 秒行走回归。
- `g1-stand-regression-20261019/`、`g1-scene-regression-20261019/`：原站立和场景加载回归。

控制审计脚本及结果保存在 `.runtime/audits/g1-unitree12-walk-20261019/`。
逐行核对了控制记录与最新状态、动作缩放、PD 公式、限幅、20ms / 2ms 周期，
并确认导入资产文件未被运行修改。日志中的 `*.validation.json` 另行核对了结果包、
文件哈希、视频帧数和从状态重算的行走指标。38 项 CPU 测试通过。

## 与实机型号对应的 G1 29-DOF mode 15 USD

新增模型取自宇树官方
[unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description)
固定提交 `5994d4faef0a9cadd3287f8de0199a67eeb2a259` 的
`robots/g1_description/g1_29dof_mode_15.urdf`。官方型号表列明 mode_machine=15、
髋 pitch/roll 减速比 22.5/22.5、新版 5010 腕电机、腰部未锁定。
模型包含双腿 12、腰部 3、双臂 14 个可动关节，手掌为固定橡胶手，无可动手指。
这些是描述文件中的型号参数；本次没有连接实机读取型号或校准数据。

源 URDF SHA256：`80fc328761b0d0a502d675106c234eceabb70fa09c9f7d5cbcb40c2955629bac`。
34 个引用网格逐项核对 Git blob SHA1 / SHA256，源文件、官方型号说明和许可缓存在
`.runtime/unitree-ros/<commit>/`。转换保持源碰撞、质量和惯性，合并固定关节，
保留浮动基座、重力与 29 个可动关节；全部隐式 Drive stiffness/damping 为零。
源关节上下限、速度上限和力矩上限均按名称与实际 PhysX 数据核对。

```bash
source /home/wl/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim-6.0.1
cd /home/wl/RSS
python -B scripts/fetch_g1_urdf.py --model unitree29
bash scripts/start_isaacsim.sh --python scripts/import_g1_29dof.py
```

新资产入口为
`.runtime/assets/unitree-g1-29dof-mode-15/5994d4faef0a9cadd3287f8de0199a67eeb2a259/g1_29dof_mode_15.usd`。
同目录下的 `comparison.usda` / `comparison.png` 将原 NVIDIA 模型放在左侧、mode 15
放在右侧；`import-summary.json` 记录关节实际索引与源关节轴/原点、质量、惯性和加载检查。
移动资产时需保留入口 USD 和整个 `converted/` 目录。已有 12-DOF 与 NVIDIA 模型继续保留。

单独加载 mode 15 到现有地面场景进行 20 步 / 40ms 检查：

```bash
mode15_output="$PWD/.runtime/isaacsim-6.0.1/outputs/g1-mode15-load-$(date +%Y%m%d-%H%M%S)"
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --robot unitree29 --controller none --steps 20 --dt 0.002 \
  --output-dir "$mode15_output"
```

`--robot unitree29 --controller none` 用于模型加载和物理检查；
`--robot unitree29 --controller walk` 选择下节的官方 29 动作 ONNX 策略。
原 `motion.pt` 仍用于 12-DOF 与 NVIDIA 模型，Mode 15 不使用它。
Mode 15 的髋 pitch 力矩上限为 139Nm、踝 pitch/roll 为 35Nm，
控制器逐项保留该型号的原始上限。

2026-10-19 本机验证通过：29 DOF、30 刚体、35×35 浮动基座质量矩阵，
总质量 33.740044kg（源 URDF 为 33.740043kg）、35 个碰撞形状和 8 个脚底球。
全部关节轴方向与零位链接变换独立对照源 URDF，通过；最大零位位置分量误差
2.33e-8m，关节轴向量最大分量误差 2.25e-7。资产共有 10 个 USD 层，依赖均在
资产目录内且无缺失。原 12-DOF 导入回归的物理属性与先前结果逐项一致，原资产哈希未变。

单模型加载结果位于
`.runtime/isaacsim-6.0.1/outputs/g1-mode15-load-20261019-final/`，
包含 `scene.usda`、`initial.png`、21 行状态与 `summary.json`，完成 20 个 2ms 物理步。
41 项 CPU 测试通过，独立验证记录为
`.runtime/audits/g1-mode15-20261019/validation.json`，日志保存于
`.runtime/isaacsim-6.0.1/logs/g1-mode15-*.log`。

<a id="mode15-onnx"></a>

## Mode 15：官方 unitree_rl_lab ONNX 行走基线

使用宇树官方 `unitree_rl_lab` 提交
`4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` 的
`deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx`，
SHA256 为 `610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f`。
`params/deploy.yaml` SHA256 为
`64b04c0596a7010f39f8ac6e9ec46dc750141063cdcf2d42c83ecc444e57bc63`。
下载器保存来源、Git blob SHA1、SHA256 和上游参考文件；运行前强制核对策略、部署配置和
官方电机名称列表的哈希。许可来源见 `LICENSES/Unitree-RL-Lab-NOTICE.txt`。

该模型是前馈网络：输入 `obs` 为 `[1,480]`，输出 `actions` 为 `[1,29]`。
输入依次为机体系角速度、投影重力、机体系速度指令、相对默认关节角、关节速度和上一帧原始动作。
每项保存 5 帧，按“观测项优先，每项从旧到新”拼接；角速度缩放 0.2，关节速度缩放 0.05，
指令不使用旧策略的 2 倍缩放。无 gait phase、无 LSTM；运行开始时用初始测量填满历史，
上一动作置零。动作保持官方 `target = offset + 0.25 * action`，不增加裁剪或目标角修饰。

关节映射分两步：先用官方 `joint_sdk_names` 和 `joint_ids_map` 得到每个策略通道的关节名，
再按名称查找本次 Isaac Sim 的 `dof_names`。Kp/Kd 则按官方 `State_RLBase::enter()` 的
电机顺序转成具名参数，然后映射到控制通道，不能直接沿用 YAML 数组下标。
缺失、重复、额外关节会导致停止。`summary.json` 的 `controller.joint_mapping` 保存
关节名、策略索引、Isaac DOF 索引和宇树电机编号，29 个关节全部由策略控制。

基线逐项使用官方部署 YAML 的 Kp/Kd：

| 关节组 | Kp (Nm/rad) | Kd (Nm·s/rad) |
| --- | ---: | ---: |
| hip pitch/roll/yaw | 100 | 2 |
| knee | 150 | 4 |
| ankle pitch/roll | 40 | 2 |
| waist yaw/roll/pitch | 200 | 5 |
| shoulder/elbow/wrist | 40 | 10 |

这是官方部署参数；上游训练资产配置中的部分臂部 Kd 为 1，与部署 YAML 不同。
本基线使用部署 YAML 的值，不在两套配置间混用。初始骨盆高度为官方配置的 0.8m，
默认关节角来自部署 YAML。根节点仅初始化一次，重力和浮动基座持续开启。

PPO 每 20ms 推理（首个物理步前即推理），显式 PD 每 2ms 根据最新 q/dq 重算，
然后通过 `Articulation.set_dof_efforts(..., dof_indices=具名映射)` 发送。
29 个 PhysX Drive 的 stiffness/damping 全为零，运行中每秒读回检查。
显式力矩按 Mode 15 原 URDF 上限裁剪，PhysX 最大力矩/速度也按同一份 URDF 设置并读回；
四个踝关节上限始终为 35Nm、30rad/s。没有添加力矩补偿、根节点扶持或提高踝部力矩上限。
`tau_raw`、`tau_cmd`、实际角度和速度逐物理步保存在 `joint-control.csv`，
发送后的 Tensor 力矩读回误差、目标/实际角越限和实际速度越限计数写入汇总。
速度/实际角越限计数使用 0.01rad/s / 0.01rad 容差；原始值仍完整保存。

接触基线保留导入资产的 35 个碰撞形状和 8 个半径 5mm 的脚底球，
保留现有地面与材质设置。汇总区分未显式设置的 USD 接触属性与 PhysX Tensor API
读回的脚部实际摩擦、接触偏移和静止偏移，避免把空属性当作零。
自碰撞保持导入配置的关闭状态，armature 为官方配置的 0.01，求解器迭代数为 8/4。
该基线与上游训练资产的型号、部分执行器上限、材质随机化和自碰撞配置存在差异，
不应把本机平地结果解释为实机部署验收或完整电机扭矩—速度曲线验证。

```bash
source /home/wl/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim-6.0.1
cd /home/wl/RSS
python -m pip install -r requirements-g1-onnx.txt
python -B scripts/fetch_g1_onnx.py
mode15_walk_output="$PWD/.runtime/isaacsim-6.0.1/outputs/g1-mode15-onnx-$(date +%Y%m%d-%H%M%S)"
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --robot unitree29 --controller walk --command 0.5 0 0 --video \
  --output-dir "$mode15_walk_output"
```

默认 10 秒，`--steps 30000` 为 60 秒。ONNX 推理使用 CPUExecutionProvider，
依赖单独列在 `requirements-g1-onnx.txt`。每次运行新建策略实例，清空历史；
原 NVIDIA / 12-DOF 的 TorchScript 路径继续保留。速度命令范围目前取两套入口的交集：
VX [-0.5,0.8]、VY [-0.3,0.3]、YAW_RATE [-0.2,0.2]。

2026-10-19 的 60 秒 / 30000 步基线通过：指令前向速度 0.5m/s，实测机体平均前向速度
0.492254m/s，最大倾角 2.8253°，最低骨盆高度 0.777070m。
最初 2 秒后有效落脚左右 102 / 97 次，交替落脚 193 次，单脚支撑比例 67.06%。
所有关节的力矩限幅、实际角越限和速度越限计数均为零；四个踝关节保持 35Nm 上限。

| 踝关节 | 全程原始 PD / 命令力矩峰值 (Nm) | 最大实际速度 (rad/s) |
| --- | ---: | ---: |
| left ankle pitch | 14.430 / 14.430 | 4.056 |
| right ankle pitch | 18.014 / 18.014 | 4.175 |
| left ankle roll | 9.048 / 9.048 | 4.456 |
| right ankle roll | 9.791 / 9.791 | 3.765 |

脚部 8 个碰撞球的 PhysX 读回值一致：静/动摩擦均 0.5，恢复系数 0，
contact offset 0.0002m，rest offset 0。上述参数未为行走而修改。
左右膝的稳态实际平均角约 23.82° / 24.08°；踝 pitch 的目标/实际跟踪 RMSE
约 6.25° / 10.06°，仍存在承重时的跟踪偏差。该偏差未通过改变官方 Kp/Kd 或放大力矩上限隐藏。
本次验证针对 0.5m/s 平地前进；命令不含绝对航向闭环，60 秒世界坐标位移约
(29.33, 1.73)m，存在缓慢航向漂移。

60 秒记录：`.runtime/isaacsim-6.0.1/outputs/g1-mode15-onnx-60s-20261019-final-02/`。
审计脚本和结果位于 `.runtime/audits/g1-mode15-onnx-20261019/`：
从状态独立重建 3000 次 ONNX 输入并推理，与记录动作的最大误差为 0；
逐行核对了具名关节映射、目标缩放、PD、限幅、实际状态与 20ms / 2ms 周期，
重算行走指标并核对源代码哈希、结果包和导入资产哈希。原 Mode-15 资产文件未被修改。
47 项 CPU 测试通过，依赖检查 `pip check` 通过。

10 秒 / 250 帧 / 1280×720 / 25fps 视频：
`.runtime/isaacsim-6.0.1/outputs/g1-mode15-onnx-video-20261019-final/simulation.mp4`。
视频完整解码与帧时序检查通过，录像运行的状态、脚部数据和控制命令与 60 秒运行的前 10 秒逐行一致。
踝部目标/实际角、原始/命令力矩及速度曲线：
`.runtime/audits/g1-mode15-onnx-20261019/ankle-tracking.png`。
原 12-DOF 与 NVIDIA 模型各运行 10 秒回归，其状态、脚部数据和控制命令与先前结果逐行一致。
