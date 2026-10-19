# Isaac Sim 6.0.1：G1 Mode 15 ONNX 行走

使用宇树官方 `g1_29dof_mode_15.urdf` 转换的 USD，加载
`unitree_rl_lab` 的 29-DOF `velocity/v0/policy.onnx` 控制行走。
运行入口默认选择这套模型和策略。观测、动作和官方 Kp/Kd 按关节名称映射到 Isaac Sim DOF；
保留 Mode 15 原始关节限位、力矩与速度上限，四个踝关节力矩上限始终为 **35Nm**。

仓库保存源码、测试、依赖版本与上游许可说明。USD、STL、策略权重、环境、视频和运行记录
由脚本下载或生成，保存在忽略的 `.runtime/` / `.venv*` 中。
源码清理不会删除本机已生成的资产或历史结果，历史实现仍可从 Git 提交记录查阅。

<a id="mode15-quickstart"></a>

## 环境与快速开始

需要 Linux x86-64、可用的 NVIDIA RTX GPU、Python 3.12 和 Isaac Sim pip 包 `6.0.1.0`。
视频需要系统中的 `ffmpeg`（含 libx264）与 `ffprobe`。
首次使用前，新用户需自行阅读并接受 [NVIDIA Omniverse EULA](https://docs.omniverse.nvidia.com/eula)。
本机已经完成环境配置与启动验证。

```bash
git clone https://github.com/zxy8222/robot_isaac.git
cd robot_isaac
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate isaacsim-6.0.1

# 启动器从这个路径查找解释器；已有环境或链接保持不变。
if [ ! -e .venv-isaacsim-6.0.1 ] && [ ! -L .venv-isaacsim-6.0.1 ]; then
  ln -s "$CONDA_PREFIX" .venv-isaacsim-6.0.1
fi
python -m pip install -r requirements-g1-onnx.txt
python -m pip check
```

本机仓库为 `/home/wl/RSS`，环境为 `/home/wl/miniconda3/envs/isaacsim-6.0.1`。
继续使用本机已有仓库时直接进入 `/home/wl/RSS`，无需再次克隆。
新主机尚无环境时，先按下节创建并安装基础仿真依赖，再执行上面的解释器链接和 ONNX 依赖安装。

### 新环境安装

仅在尚无同名 Conda 环境时执行创建命令。以下是已验证的直接依赖版本，
不包含 NVIDIA 驱动或全部间接依赖的锁定：

```bash
conda create -n isaacsim-6.0.1 python=3.12
conda activate isaacsim-6.0.1
python -m pip install \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' 'torchaudio==2.11.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-isaacsim-6.0.1.txt --extra-index-url https://pypi.nvidia.com
python -m pip install -r requirements-g1-onnx.txt
python -m pip check
```

Isaac Sim 需要兼容的 Vulkan/OpenGL/GLU 运行库。系统依赖和 GPU 驱动应按本机环境配置；
安装包版本见 `requirements-isaacsim-6.0.1.txt`，ONNX 推理依赖见 `requirements-g1-onnx.txt`。
启动器使用仓库相对路径；在有 Slurm 的集群中应先申请 GPU 作业再运行。
`ISAACSIM_RUNTIME_DIR` 可覆盖默认 `.runtime/isaacsim-6.0.1` 运行缓存位置。

## 下载并生成 Mode 15 USD

```bash
python -B scripts/fetch_g1_urdf.py
bash scripts/start_isaacsim.sh --python scripts/import_g1_29dof.py
python -B scripts/fetch_g1_onnx.py
```

已有 USD 时跳过导入命令。导入器拒绝覆盖现有目录；需要重新转换时，
通过 `--output-dir "$PWD/.runtime/assets/新目录名"` 指定新目录。
下载器核对固定上游提交、Git blob SHA1 和 SHA256，既有文件校验失败时停止并保留原文件。

模型入口为：

```text
.runtime/assets/unitree-g1-29dof-mode-15/5994d4faef0a9cadd3287f8de0199a67eeb2a259/g1_29dof_mode_15.usd
```

该入口通过相对路径引用同目录的 `converted/`，移动模型时需要保留整个资产目录。
新导入目录还包含 `preview.usda`、`preview.png`、`import-summary.json`、许可和源文件清单。
导入过程只加载 Mode 15，检查 29 DOF、30 刚体、35×35 浮动基座质量矩阵、
质量/惯性、源关节限位和 35 个碰撞形状，并完成 20 个物理步检查。
导入检查不作为站立或行走验收。

## 运行与输出

默认运行 10 秒，机体系速度指令为 `(0.5, 0, 0)`：

```bash
walk_output="$PWD/.runtime/isaacsim-6.0.1/outputs/g1-mode15-$(date +%Y%m%d-%H%M%S)"
bash scripts/start_isaacsim.sh --python scripts/run_g1_ground.py \
  --command 0.5 0 0 --video --output-dir "$walk_output"
```

- `--steps 30000`：运行 60 秒；默认 5000 步。去掉 `--video` 可加快物理验证。
- `--command VX VY YAW_RATE`：机体系前进、侧移速度（m/s）与偏航角速度（rad/s）。
  当前入口范围分别为 `[-0.5,0.8]`、`[-0.3,0.3]`、`[-0.2,0.2]`，并非所有组合都已实测。
- `--asset-usd /完整路径/g1_29dof_mode_15.usd`：选择重新导入的本地 Mode 15 USD，仍校验具名关节与运行限制。
- `--policy-path /完整路径/policy.onnx`：选择同一官方权重的本地副本，仍强制核对 SHA256。
- `--controller none`：只做模型加载与物理检查，默认 20 步 / 40ms。

输出目录必须是仓库 `.runtime/isaacsim-6.0.1/outputs/` 内尚不存在的子目录。
`--video` 录制 25fps、1280×720、1 倍速 H.264 视频，采样帧对应实际物理时间，渲染不推进物理。

| 输出 | 内容 |
| --- | --- |
| `simulation.mp4` | 跟随视角行走视频 |
| `states.csv` | 每个物理步的根节点位姿/速度与 29 关节角度/速度 |
| `joint-control.csv` | 每步动作、目标/实际角、关节速度、`tau_raw`、`tau_cmd` 与策略更新次数 |
| `feet.csv` | 左右踝链接位置与测得的接触力 |
| `summary.json` | 具名映射、官方增益、关节限制、接触属性、验收指标与文件/源码哈希 |
| `scene.usda`、PNG | 运行场景及初始/中间/最终截图 |
| `results.zip` | 结果包；录像时包含 MP4 |

失败时保存部分结果和 `failure.json`，进程返回非零退出码，不生成成功结果包。
10 秒行走包括 5001 行状态、5000 行控制命令与 500 次策略推理。
`tau_cmd` 是发送给 PhysX 的命令，读回检查也针对命令，不是力矩传感器测量。

## 文件职责与依赖边界

| 文件 | 单一职责 |
| --- | --- |
| `g1_urdf_config.py` | 固定 Mode 15 来源、29 个关节名和资产路径 |
| `g1_source_cache.py` | 上游文件校验、原子缓存写入与来源清单 |
| `fetch_g1_urdf.py` | 下载 Mode 15 URDF、引用网格、说明和许可 |
| `g1_urdf_import.py` | URDF 转换、USD 依赖和实际 PhysX 属性校验 |
| `import_g1_29dof.py` | 导入应用生命周期、单模型预览与短时物理检查 |
| `fetch_g1_onnx.py` | 下载官方策略、部署 YAML 与参考文件 |
| `g1_onnx_policy.py` | 部署配置解析、关节名称映射、历史观测与 ONNX 推理；不依赖仿真 SDK |
| `g1_math.py` | 世界/机体坐标变换；不依赖策略或仿真 SDK |
| `g1_onnx_walking.py` | PhysX 适配、显式 PD、原始限制和接触属性读取；不写结果文件 |
| `g1_scene.py` | Mode 15、地面、灯光与相机的场景创建 |
| `g1_walk_validation.py` | 根据测量状态和脚部接触判断行走；不调用仿真 SDK 或写文件 |
| `g1_control_outputs.py` | 逐步控制记录与跟踪/限幅统计 |
| `g1_outputs.py` | 状态校验、截图、JSON 与结果打包 |
| `g1_video.py` | 视频帧调度和编码；截图回调由运行入口提供 |
| `run_g1_ground.py` | 运行生命周期、物理步进、采样与模块调度 |
| `start_isaacsim.sh`、`check_isaacsim.py` | 隔离运行环境、启动器和 GPU/RTX 启动检查 |

运行数据通过 `StateSample` 和 `ControlSample` 传递。
策略负责“状态 → 具名关节目标”，适配器负责“目标与最新状态 → 关节力矩”，
验证器只读测量结果，文件与视频输出不参与控制计算。

<a id="mode15-onnx"></a>

## 策略与控制基线

| 固定来源 | 版本 / SHA256 |
| --- | --- |
| [Mode 15 URDF](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description/g1_29dof_mode_15.urdf) | 提交 `5994d4faef0a9cadd3287f8de0199a67eeb2a259`；URDF `80fc328761b0d0a502d675106c234eceabb70fa09c9f7d5cbcb40c2955629bac` |
| [ONNX 与部署配置](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/deploy/robots/g1_29dof/config/policy/velocity/v0) | 提交 `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` |
| `policy.onnx` | `610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f` |
| `deploy.yaml` | `64b04c0596a7010f39f8ac6e9ec46dc750141063cdcf2d42c83ecc444e57bc63` |

模型包含双腿 12、腰部 3、双臂 14 个可动关节，手掌固定，无可动手指。
合并固定关节，保留源质量、惯性、碰撞、重力与浮动基座。上游许可说明见 `LICENSES/`，
完整参考文件与源许可随下载保存在 `.runtime/unitree-ros/` 和 `.runtime/unitree-rl-lab/`。

ONNX 是前馈网络，输入 `obs=[1,480]`，输出 `actions=[1,29]`。
观测依次为机体系角速度、投影重力、速度指令、相对默认关节角、关节速度、上一帧原始动作。
每项保留 5 帧，按“观测项优先、每项从旧到新”拼接；角速度缩放 0.2，关节速度缩放 0.05。
开始时用初始测量填满历史，上一动作置零。无 LSTM 或 gait phase。
关节目标保持官方 `target = offset + 0.25 * action`，不增加动作裁剪或目标角修饰。

官方 `joint_sdk_names` 和 `joint_ids_map` 先确定各策略通道的关节名，
再按名称查找本次 Isaac Sim 的 `dof_names`。Kp/Kd 则从官方电机顺序转换成具名参数，
不能直接将部署数组下标当作策略或 Isaac DOF 下标。缺失、重复或额外关节会使运行停止。
`summary.json` 保存关节名、策略索引、Isaac DOF 索引和宇树电机编号。

| 关节组 | Kp (Nm/rad) | Kd (Nm·s/rad) |
| --- | ---: | ---: |
| hip pitch/roll/yaw | 100 | 2 |
| knee | 150 | 4 |
| ankle pitch/roll | 40 | 2 |
| waist yaw/roll/pitch | 200 | 5 |
| shoulder/elbow/wrist | 40 | 10 |

使用官方部署 YAML 的原始增益，未混入训练配置中不同的臂部 Kd。
初始骨盆高度为 0.8m，默认关节角来自部署 YAML；根节点只初始化一次。
策略每 20ms 推理，首步前即推理；PD 每 2ms 根据最新 q/dq 重算，
通过 `Articulation.set_dof_efforts(..., dof_indices=具名映射)` 发送。
29 个隐式 PhysX Drive 增益均为零，运行中每秒读回检查。

显式力矩和 PhysX 最大力矩/速度使用同一份 Mode 15 原始 URDF 限制。
四个踝关节上限为 35Nm、30rad/s；运行不添加根节点扶持、力矩补偿或放宽上限。
位置/速度越限计数使用 0.01rad / 0.01rad/s 容差，原始值完整保存。
接触保留 35 个碰撞形状和 8 个半径 5mm 的脚底球，材质与碰撞参数不为行走修改；
PhysX Tensor API 读回实际脚部摩擦、contact offset 和 rest offset。
自碰撞关闭、armature=0.01、求解器位置/速度迭代数为 8/4。

## 验证与适用范围

完成 URDF 和 ONNX 下载后，无需启动 Isaac Sim 即可执行 CPU 测试：

```bash
python -B -m unittest discover -s tests -v
```

测试覆盖坐标变换、DOF 顺序打乱、历史观测、官方增益、Mode 15 限制、缓存完整性、
控制记录、命令行、视频帧时序，以及对跌倒、滑行和双脚跳动的拒绝。
行走验收检查根节点高度/倾角、速度跟踪、实测脚部离地与接触、交替落脚和单脚支撑。

本次源码清理后，35 项 CPU 测试通过，在单独导出的源码目录中再次通过；
重新导入的 Mode 15 物理属性与既有资产一致。60 秒行走与 10 秒录像均通过验收，
状态、脚部接触和关节控制 CSV 与清理前 ONNX 基线逐字节一致；
3000 次策略推理离线重放动作误差为零，10 秒录像为 250 帧 / 25fps。

本机 60 秒基线：平均前向速度 0.4923m/s（指令 0.5m/s）、最大倾角 2.83°、
最低骨盆高度 0.7771m；踝关节命令力矩峰值 18.014Nm，无力矩限幅或关节速度越限。
脚部静/动摩擦为 0.5、恢复系数 0、contact offset=0.0002m、rest offset=0。
这些是本机平地仿真结果；尚未验证复杂地形、实机校准或完整电机扭矩—速度曲线。
当前速度命令没有绝对航向闭环，仍存在缓慢航向漂移；承重时踝部也有目标跟踪偏差。
