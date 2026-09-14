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

### 1. 安装仓库

```bash
conda activate mjlab
cd AMP_mjlab
python -m pip install -e .
cd rsl_rl
python -m pip install -e .
```

### 2. 应用 mjlab 补丁（可选）

如果不打这个补丁，则需要在代码中去掉 `history_ordering` 配置。

补丁作用说明：

- 增加了历史观测的展开方式选项，可选择按时间维(`time`)或按观测项(`term`)展开。
- mjlab 默认仅支持按 `term` 展开。

补丁文件：

- `mjlab_patch/mjlab/managers/observation_manager.py`

示例覆盖命令：

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
	/home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

### 3. 查看可用任务

```bash
python scripts/list_envs.py --keyword AMP
```

主要任务：

- `BXI-ELF3-AMP-Flat-V3-1`（使用修正后的 V3.1 数据集，从头联合训练平地走跑与起身）
- `BXI-ELF3-AMP-Rough-V4`（V4 带起身：老天工 GRAVEL 地形盲走与倒地起身，不含跑步参考；[V4 文档](docs/ELF3_V4_ZH.md)）
- `BXI-ELF3-AMP-Rough-V4-Loco`（相同 V4 地形与指令，不使用起身重置或 AMP 起身参考）
- `BXI-ELF3-AMP-Rough-V4-Delay`（V4 带起身＋0～40 ms 位置目标延迟；[延迟训练文档](docs/ELF3_V4_DELAY_ZH.md)）

所有 V4 任务统一使用老 GRAVEL 地形，不输入 `terrain_scan`。V4 平地任务已取消，
V3.1 仍保留平地。不带 `Delay` 的任务保持无新增动作延迟，供对照使用。

## 训练


```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

ELF3 平地与粗糙地形训练：

```bash
source .venv/bin/activate
python scripts/train.py BXI-ELF3-AMP-Flat-V3-1 --env.scene.num-envs=4096
# V4 rough 带起身
python scripts/train.py BXI-ELF3-AMP-Rough-V4 --env.scene.num-envs=4096 --enable-nan-guard True
# V4 rough 不带起身
python scripts/train.py BXI-ELF3-AMP-Rough-V4-Loco --env.scene.num-envs=4096 --enable-nan-guard True
# V4 rough 带起身＋动作延迟
python scripts/train.py BXI-ELF3-AMP-Rough-V4-Delay --env.scene.num-envs=4096 --enable-nan-guard True
```

以上命令均从头训练（`resume=False`，初始学习率 `1e-3`），V4 目标保存到
`model_200000.pt`。动作延迟在 reset 时采样，29 个关节共用，本回合固定：
0～8 个物理步 × 5 ms。不改观测、PD 内环反馈或 384 输入／29 输出部署接口。

从已有 V4 检查点续训带延迟版本（`run_dir` 是 V4 日志根目录下的实验目录名）：

```bash
python scripts/train.py BXI-ELF3-AMP-Rough-V4-Delay \
  --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run <run_dir> \
  --agent.load-checkpoint 'model_<iter>.pt' \
  --resume-optimizer True --target-iteration 200001 \
  --enable-nan-guard True \
  --swanlab-project locomotion --swanlab-experiment-name v4_delay_resume
```

加载模型、归一化、优化器和课程时钟；目标是最终轮次，不是额外训练 200000 轮。
接同一个 SwanLab 实验时追加 `--swanlab-id <id> --swanlab-resume must`；
从旧检查点分支训练则新建实验，避免较小的 step 与原曲线冲突。


日志默认在：

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`
- V3.1：`logs/rsl_rl/elf3_amp_locomotion_v3_1/<time_stamp_run>/`
- V4 / V4-Delay：`logs/rsl_rl/elf3_amp_locomotion_v4/<time_stamp_run>/`
- V4-Loco：`logs/rsl_rl/elf3_amp_locomotion_v4_loco/<time_stamp_run>/`

## ELF3 适配说明

- 机器人模型来自 HoloMotion 的 `assets/robots/elf3/29dof/sim2sim/elf3.xml`，质量、关节限位、执行器增益、转子惯量和原生动作尺度均按该模型配置。
- BXI 控制器/MuJoCo 的 29 关节顺序是“腰、左腿、右腿、左臂、右臂”，Isaac 顺序则交错排列。训练 action、AMP 文件和导出的 ONNX metadata 使用前一种顺序；HoloMotion 转换、AMP reset/判别器加载和动作预览均按关节名重排，不按输入列号猜测顺序。
- ELF3 的物理浮动根是 `torso_link`；与 G1 pelvis 对应的策略语义根是 `waist_z_link`。仿真 reset、根状态和终止判断使用前者，AMP 身体结构语义使用后者，不能互换。
- `src/assets/motions/elf3/amp/WalkandRun` 中的 17 段参考动作直接由 HoloMotion ELF3 数据转换，并用 ELF3 MuJoCo 模型重新计算全身 FK 和速度。
- `src/assets/motions/elf3/amp/Recovery` 中的起身动作目前由 G1 数据按关节语义临时映射，并做了 ELF3 网格贴地和关节限位裁剪。它可以用于链路验证和初始实验，但正式训练最好替换为 ELF3 原生起身数据。
- ELF3 基础配置保持 G1 奖励，替换 body/site/geom 映射和倒地高度阈值；V4 另有步行跟踪、脚部安全和起步奖励。增加 Delay 不修改这些奖励。参数入口位于 `src/tasks/amp_loco/config/elf3/env_cfgs.py`。

带起身任务联合训练运动和恢复，40% 环境使用倒地后的延迟终止／恢复窗口。
这个恢复窗口与 Delay 任务的“动作位置目标延迟”是两回事，不能混为一谈。

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

ELF3（按模型选择对应任务）：

```bash
python scripts/play.py BXI-ELF3-AMP-Flat-V3-1 \
  --checkpoint-file logs/rsl_rl/elf3_amp_locomotion_v3_1/<run_dir>/model_<iter>.pt \
  --num-envs 20 --export-onnx True
python scripts/play.py BXI-ELF3-AMP-Rough-V4-Delay \
  --checkpoint-file logs/rsl_rl/elf3_amp_locomotion_v4/<run_dir>/model_<iter>.pt \
  --num-envs 20 --export-onnx True
python scripts/play.py BXI-ELF3-AMP-Rough-V4-Loco \
  --checkpoint-file logs/rsl_rl/elf3_amp_locomotion_v4_loco/<run_dir>/model_<iter>.pt \
  --num-envs 20 --export-onnx True
```

Delay 任务播放时也模拟 0～40 ms 延迟；同一检查点的零额外延迟对照，将任务名
换成 `BXI-ELF3-AMP-Rough-V4` 即可。ONNX 不包含延迟缓冲，真机不要人为加延迟。
训练正在占用 GPU 时，建议先用 `--num-envs 1` 播放。

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
