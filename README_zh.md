# AMP_mjlab

[English README](README.md)

部署集成代码位于 [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) 项目中的 `MJAmp State`。

基于 mjlab + rsl_rl 的 G1 / BXI ELF3 AMP 运动控制项目。

本项目的核心特点是：

- 使用同一个 policy 同时学习 locomotion（走/跑）与 recovery（跌倒恢复）
- 通过 AMP 判别器约束动作风格与运动先验
- 在训练与导出链路中保持一致，支持直接导出 ONNX policy

## 核心思路

传统做法常把“走跑策略”和“恢复策略”分开训练并做切换；本项目将两类能力放入一个策略中统一学习。

实现要点：

- 运动数据分组：
	- Walk/Run 数据目录：`src/assets/motions/g1/amp/WalkandRun`
	- Recovery 数据目录：`src/assets/motions/g1/amp/Recovery`
- 延迟重置机制（Delayed Termination）：
	- 一部分环境在触发终止后不立即 reset，而是给定恢复窗口
	- 该子集环境优先从 Recovery 片段采样 reset 状态
- 统一 AMP 训练：
	- 单一 actor-critic + 单一 AMP discriminator
	- 在同一训练过程中学习速度跟踪、抗扰动与恢复能力

这样可以减少策略切换带来的状态不连续问题，得到更一致的行为。

## 环境要求

- Linux
- Python 3.11（建议）
- 已可用的 MuJoCo / GPU 驱动环境

本项目在 ELF3 训练机上的精确软硬件版本、独立环境创建命令、安装顺序和部署检查见
[docs/ELF3_DEPLOYMENT_ZH.md](docs/ELF3_DEPLOYMENT_ZH.md)。建议新机器优先按该文档安装，
不要直接复用系统 Python。

## 快速开始

本节只列当前使用的 ELF3 V3.1 和 V4 任务。所有命令在仓库根目录运行；
Git 仓库不包含训练权重或 SwanLab 凭据。

### 1. 创建独立环境并安装

```bash
git clone https://github.com/JKYovo/Amp_mjlab.git
cd Amp_mjlab
conda create --prefix ./.venv python=3.11.16 pip -y
.venv/bin/python -m pip install -r requirements-elf3.txt
.venv/bin/python -m pip install --no-deps -e ./rsl_rl
.venv/bin/python -m pip install --no-deps -e .
```

必须使用仓库内的 AMP `rsl_rl` fork，覆盖上游包。已验证的本机版本、
CUDA 安装注意事项见[环境复现文档](docs/ELF3_DEPLOYMENT_ZH.md)。

### 2. 应用必需的历史观测补丁

```bash
MJLAB_DIR="$(.venv/bin/python -c 'import pathlib, mjlab; print(pathlib.Path(mjlab.__file__).parent)')"
cp mjlab_patch/mjlab/managers/observation_manager.py \
  "$MJLAB_DIR/managers/observation_manager.py"
.venv/bin/python scripts/list_envs.py --keyword ELF3
```

部署接口为单帧 96 维、按时间排列的 4 帧历史（384 维输入）、29 维动作。
不能通过删除 `history_ordering` 来跳过补丁，否则不再保持相同的部署接口。

### 3. 选择训练任务

| 任务名 | 地形与训练内容 | AMP 数据集 | 默认最终检查点 |
| --- | --- | --- | --- |
| `BXI-ELF3-AMP-Flat-V3-1` | 平地；走跑 + 倒地起身 | `src/assets/motions/elf3/amp_v3_1` | `model_100000.pt` |
| `BXI-ELF3-AMP-Rough-V4` | GRAVEL 崎岖地形；走路 + 倒地起身 | `src/assets/motions/elf3/amp_v4` | `model_200000.pt` |
| `BXI-ELF3-AMP-Rough-V4-Loco` | GRAVEL 崎岖地形；走路，不训练起身 | `src/assets/motions/elf3/amp_v4/WalkandRun` | `model_200000.pt` |

带起身的两个任务均有 40% 恢复环境，倒地后最长保留 5 秒恢复窗口；
运动和起身在同一次训练中联合学习，不要求先后训练两个阶段。
V4-Loco 不加载 AMP 起身参考、不从起身动作重置，倒地立即终止；
其余 V4 物理容量配置保留。目前没有独立注册的 V3.1-Loco 或平地 V4 任务。
两个 V4 任务均不输入 `terrain_scan`，地形和起步专项细节见 [V4 文档](docs/ELF3_V4_ZH.md)。

### 4. 从头训练

以下命令按需要选择一个运行，不要同时启动三个训练：

```bash
# V3.1：平地走跑与起身联合训练
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Flat-V3-1 \
  --env.scene.num-envs 4096 --agent.resume False \
  --target-iteration 100001 --enable-nan-guard True

# V4：崎岖地形走路与起身联合训练
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4 \
  --env.scene.num-envs 4096 --agent.resume False \
  --target-iteration 200001 --enable-nan-guard True

# V4：崎岖地形走路，不训练起身
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4-Loco \
  --env.scene.num-envs 4096 --agent.resume False \
  --target-iteration 200001 --enable-nan-guard True
```

轮次从 0 计数，所以目标 100001 / 200001 对应最终保存 100000 / 200000 轮。
每 100 轮保存一次模型，默认日志目录分别为：

- V3.1：`logs/rsl_rl/elf3_amp_locomotion_v3_1/<RUN_DIR>/`
- V4 带起身：`logs/rsl_rl/elf3_amp_locomotion_v4/<RUN_DIR>/`
- V4 不带起身：`logs/rsl_rl/elf3_amp_locomotion_v4_loco/<RUN_DIR>/`

本机 RTX 4090 实测 4096 环境的 V4 正式训练约占 20 GiB 显存。
显存不足可降低环境数；不带起身不意味着物理预分配缓冲自动大幅缩小。

### 5. Play 与 ONNX 导出

将 `<RUN_DIR>`、`<ITER>` 替换成实际目录和已保存轮次，play 任务与训练任务一致。
训练正运行到的轮次不一定已保存为文件。

```bash
.venv/bin/python scripts/play.py BXI-ELF3-AMP-Flat-V3-1 \
  --checkpoint-file "logs/rsl_rl/elf3_amp_locomotion_v3_1/<RUN_DIR>/model_<ITER>.pt" \
  --num-envs 20 --export-onnx False

.venv/bin/python scripts/play.py BXI-ELF3-AMP-Rough-V4 \
  --checkpoint-file "logs/rsl_rl/elf3_amp_locomotion_v4/<RUN_DIR>/model_<ITER>.pt" \
  --num-envs 20 --export-onnx False

.venv/bin/python scripts/play.py BXI-ELF3-AMP-Rough-V4-Loco \
  --checkpoint-file "logs/rsl_rl/elf3_amp_locomotion_v4_loco/<RUN_DIR>/model_<ITER>.pt" \
  --num-envs 20 --export-onnx False
```

需要导出时改为 `--export-onnx True`（默认值）。Play 导出路径为
`<RUN_DIR>/export/<TASK_ID>_model_<ITER>.onnx`，模型已包含观测 normalizer，
部署端不要再次归一化输入。关闭导出的参数是 `--export-onnx False`，
不是 `--no-export-onnx`。带起身任务的 play 保留 40% 恢复环境；
V4 play 默认关闭自动起步专项，保持普通/手动指令。
看完用 `Ctrl+C` 退出，避免多个 GPU play 进程挤占训练显存。

### 6. 续训与 SwanLab

从同一个 V4 带起身实验续训，保留优化器、normalizer 和已保存的课程计数。
`<RUN_DIR>` 在 `--agent.load-run` 中仅填目录名，不填完整路径：

```bash
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4 \
  --env.scene.num-envs 4096 --agent.resume True \
  --agent.load-run '<RUN_DIR>' --agent.load-checkpoint 'model_<ITER>.pt' \
  --log-dir "logs/rsl_rl/elf3_amp_locomotion_v4/<RUN_DIR>" \
  --resume-optimizer True --target-iteration 200001 --enable-nan-guard True
```

V3.1 / V4-Loco 续训时换成上表对应任务和日志目录；V3.1 的目标为
`--target-iteration 100001`。目标是训练结束的总轮次，不是额外训练轮次。
跨任务微调还需用 `--agent.experiment-name` 指定源实验目录，不能只替换任务名。

```bash
# 每台机器单独登录，不要把 API key 写入仓库。
.venv/bin/swanlab login
```

新建实验时在训练命令后追加 `--swanlab-project locomotion
--swanlab-experiment-name YOUR_EXPERIMENT --swanlab-resume never`；接回已有实验时追加
`--swanlab-project locomotion --swanlab-id YOUR_RUN_ID --swanlab-resume must`。
如果回退到了已上传曲线之前的检查点，应该新建 SwanLab 实验，避免较小 step 被拒收。

## 训练


```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

ELF3 平地与粗糙地形训练：

```bash
source .venv/bin/activate
python scripts/train.py BXI-ELF3-AMP-Flat-Loco --env.scene.num-envs=4096
python scripts/train.py BXI-ELF3-AMP-Flat --env.scene.num-envs=4096
python scripts/train.py BXI-ELF3-AMP-Rough --env.scene.num-envs=4096
```


日志默认在：

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`
- `logs/rsl_rl/elf3_amp_locomotion/<time_stamp_run>/`

## ELF3 适配说明

- 机器人模型来自 HoloMotion 的 `assets/robots/elf3/29dof/sim2sim/elf3.xml`，质量、关节限位、执行器增益、转子惯量和原生动作尺度均按该模型配置。
- BXI 控制器/MuJoCo 的 29 关节顺序是“腰、左腿、右腿、左臂、右臂”，Isaac 顺序则交错排列。训练 action、AMP 文件和导出的 ONNX metadata 使用前一种顺序；HoloMotion 转换、AMP reset/判别器加载和动作预览均按关节名重排，不按输入列号猜测顺序。
- ELF3 的物理浮动根是 `torso_link`；与 G1 pelvis 对应的策略语义根是 `waist_z_link`。仿真 reset、根状态和终止判断使用前者，AMP 身体结构语义使用后者，不能互换。
- `src/assets/motions/elf3/amp/WalkandRun` 中的 17 段参考动作直接由 HoloMotion ELF3 数据转换，并用 ELF3 MuJoCo 模型重新计算全身 FK 和速度。
- `src/assets/motions/elf3/amp/Recovery` 中的起身动作目前由 G1 数据按关节语义临时映射，并做了 ELF3 网格贴地和关节限位裁剪。它可以用于链路验证和初始实验，但正式训练最好替换为 ELF3 原生起身数据。
- ELF3 保持 G1 的奖励项、权重、跟踪宽度和碰撞阈值不变；只替换机器人 body/site/geom 映射，并针对 ELF3 的高位物理 root 单独设置倒地高度终止阈值。参数入口位于 `src/tasks/amp_loco/config/elf3/env_cfgs.py`。

### 训练顺序建议

代码默认采用最终形态：同一个 actor-critic 和 AMP discriminator 联合训练走跑与恢复，其中 40% 环境启用延迟终止/恢复窗口。对于新的 ELF3 机器人，建议按阶段调试，但最终仍联合训练：

1. 先使用 `BXI-ELF3-AMP-Flat-Loco` 训练站立、速度跟踪、足端接触和动作尺度。该入口不会加载 `Recovery`，也不会启用延迟终止。
2. 有可靠的 ELF3 起身参考后，切换到 `BXI-ELF3-AMP-Flat` 并从平地 checkpoint 继续训练；初次迁移时可先把 `delay_reset_env_ratio` 从 0.1 逐步提高到默认的 0.4。
3. 平地联合策略稳定后，再在 `BXI-ELF3-AMP-Rough` 上继续训练地形和抗扰动能力。

不要长期训练两个独立策略再做 locomotion/recovery 硬切换；本项目的 delayed termination、奖励屏蔽和 AMP 数据混合本来就是为单策略联合能力设计的。

从第一阶段 checkpoint 进入联合训练的示例：

```bash
python scripts/train.py BXI-ELF3-AMP-Flat \
  --env.scene.num-envs=4096 \
  --agent.resume True \
  --agent.load-run <loco_run_dir> \
  --agent.load-checkpoint 'model_<iter>.pt'
```

## 训练曲线说明（重要）

- 在约 `2w` 轮（约 20k iterations）附近，策略通常会突然学会“跌倒后恢复”行为。
- 对应地，`logs` 中多个指标会出现明显突变（阶跃式变化），这是正常现象，不一定是训练异常。

![训练日志突变示例](logs.png)

## 评估与可视化

使用已训练权重回放：

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
	--checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt 
```

说明：训练与回放阶段都支持 ONNX 导出（默认开启）。

## 运动数据准备

仓库提供 CSV 到 NPZ 的转换脚本：

```bash
python scripts/csv_to_npz.py --help
```

推荐目录组织：

- 原始 CSV：`motion_data_csv/amp`
- 转换后 NPZ：`src/assets/motions/g1/amp/WalkandRun` 与 `src/assets/motions/g1/amp/Recovery`

只要上述目录中存在可用 NPZ，训练配置会自动加载。

HoloMotion ELF3 NPZ 转换：

```bash
python scripts/convert_elf3_amp_motion.py \
  --input-path /path/to/holomotion/elf3_npz \
  --output-dir src/assets/motions/elf3/amp/WalkandRun \
  --source-format holomotion
```

转换器会写入并校验关节名、body 名和物理 root，避免依赖隐含数组顺序。`g1_amp` 格式只建议作为缺少 ELF3 原生 recovery 时的临时迁移工具。

直接检查转换后的 AMP 动作（桌面窗口）：

```bash
python scripts/play_elf3_amp_motion.py \
  --input-path src/assets/motions/elf3/amp/WalkandRun
```

无桌面环境时录制 MP4：

```bash
MUJOCO_GL=egl python scripts/play_elf3_amp_motion.py \
  --mode video \
  --input-path src/assets/motions/elf3/amp/WalkandRun \
  --output artifacts/elf3_amp_walkandrun_all.mp4 \
  --max-seconds-per-clip 3
```

播放器同样按 NPZ 中的 `joint_names` 和 `root_body_name` 映射；顺序、root 或 NaN 不合法时会直接报错。

## 目录说明

- `src/tasks/amp_loco`：AMP locomotion/recovery 任务实现
- `src/tasks/amp_loco/config/g1`：G1 任务注册、环境与 RL 配置
- `src/tasks/amp_loco/config/elf3`：ELF3 任务注册、环境与 RL 配置
- `src/tasks/amp_loco/mdp`：奖励、观测、事件、终止逻辑
- `scripts/train.py`：训练入口
- `scripts/play.py`：回放入口
- `scripts/csv_to_npz.py`：动作数据转换工具
- `scripts/convert_elf3_amp_motion.py`：HoloMotion/G1 AMP 到 ELF3 AMP 的转换工具
- `scripts/play_elf3_amp_motion.py`：按关节名播放或录制 ELF3 AMP 原始动作
- `mjlab_patch`：依赖的 mjlab 本地补丁

## 项目亮点总结

- 单一策略统一覆盖走跑与跌倒恢复
- AMP + 速度任务联合优化，兼顾风格与任务性能
- 延迟重置与 recovery 采样机制，显式强化恢复能力
- 训练到部署链路完整，支持 ONNX 导出

## 致谢

- 感谢 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 项目的开源工作与启发。
- 感谢 [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)，本项目在 rsl_rl 的 AMP 部分参考了该实现。
