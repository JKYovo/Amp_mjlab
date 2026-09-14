# ELF3 V4：恢复旧 GRAVEL 地形盲走 + walking-only AMP

任务：`BXI-ELF3-AMP-Rough-V4`。默认从头训练，`resume=False`，初始学习率
`1e-3`、自适应调度，目标保存至 `model_200000.pt`。没有写入本机 V3.1
续训路径，也不会自动停止或替换正在运行的任务。

2026-09-14：标准 Rough-V4 / Rough-V4-Loco 恢复旧实验
`2026-09-12_17-22-39_v4_startup_recovery_fresh` 的 GRAVEL 地形。
带起身任务的其余训练配置保持旧实验设置，包括修复后的起步判定；
不修改观测归一化/ONNX 导出实现。旧实验保存的 `resume=True` 是后来
续训留下的状态，重新训练仍使用 `resume=False`，不加载旧模型。

## V4 三种任务

| 任务 | 地形 | 起身 reset / AMP 起身参考 | 日志根目录（`logs/rsl_rl/` 下） |
| --- | --- | --- | --- |
| `BXI-ELF3-AMP-Rough-V4` | 旧 GRAVEL 条带地形 | 有 / 有 | `elf3_amp_locomotion_v4` |
| `BXI-ELF3-AMP-Rough-V4-Loco` | 旧 GRAVEL 条带地形 | 无 / 无 | `elf3_amp_locomotion_v4_loco` |
| `BXI-ELF3-AMP-Rough-V4-Delay` | 旧 GRAVEL 条带地形 | 有 / 有；另加0～40 ms动作延迟 | `elf3_amp_locomotion_v4` |

三个任务均默认从头训练（`resume=False`、初始 LR `1e-3`），目标保存到
`model_200000.pt`。V4 平地入口已取消，包括临时的 V3.1 碰撞对照任务；
历史模型和日志没有删除，V3.1 平地任务保持可用。
Delay 仅在 reset 时采样0～8个5 ms物理步的位置目标延迟，29关节共用、
本回合固定，不修改观测、PD、奖励或归一化实现。详见[延迟训练文档](ELF3_V4_DELAY_ZH.md)。

```bash
# 选择一个任务训练，不要同时启动多个 4096 环境任务。
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4 \
  --env.scene.num-envs 4096 --agent.resume False \
  --agent.seed 42 --enable-nan-guard True \
  --agent.run-name v4_old_gravel_repro_fresh \
  --swanlab-project locomotion \
  --swanlab-experiment-name elf3_v4_old_gravel_repro_fresh
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4-Loco --env.scene.num-envs 4096
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4-Delay --env.scene.num-envs 4096

# Play 的任务名应与模型训练任务一致。
.venv/bin/python scripts/play.py BXI-ELF3-AMP-Rough-V4-Delay \
  --checkpoint-file <模型路径> --num-envs 20 --export-onnx False
.venv/bin/python scripts/play.py BXI-ELF3-AMP-Rough-V4-Loco \
  --checkpoint-file <模型路径> --num-envs 20 --export-onnx False
```

## 质心随机化

沿用已有 `base_com` 事件，每个环境启动时独立随机化 ELF3 骨盆
`ELF3_POLICY_ROOT` 与 torso `ELF3_PHYSICAL_ROOT` 的局部惯性质心位置：

| 局部轴 | 相对模型标称值的均匀偏移 |
| --- | --- |
| X | −0.025～0.025 m（不变） |
| Y | −0.05～0.05 m |
| Z | −0.05～0.05 m |

这是两个刚体的局部 COM 偏移，不是把整机质心直接平移相同距离；
质量、关节几何与默认姿态不因此改写。随机化在 startup 生效，
同一个环境内跨 episode 保持固定，不在运动中瞬间改变惯性参数。

## 地形来源与引擎适配

当前 Rough V4 与 V4-Loco 使用旧实验的 `TienKungGravelTerrainCfg`，
配置入口为 `mdp/tienkung_terrain.py::tienkung_gravel_cfg`。
这是 TienKung-Lab ELF3 GRAVEL 的 MuJoCo 高度场适配，不是直接使用 PhysX 地形。

| 参数 | V4 |
| --- | --- |
| 地块大小 / 训练网格 | 8×8 m / 10×20 地块 |
| 外边界 | 20 m |
| 高度采样范围 / 间隔 | −2～4 cm / 2 cm |
| 水平 / 竖直离散精度 | 0.1 m / 0.005 m |
| 每条带宽度 | 2 个水平栅格，20 cm |
| 地块内边界参数 | 0.25 m |
| curriculum | False |
| 子地形权重 | 0.2（唯一子地形，归一化后 100% GRAVEL） |

每个 8×8 m 地块为 81×81 端点包含栅格，沿 X 分为 40 个共享边界
顶点的高度场条带；训练 200 个地块、8000 个高度场，Play 为 25 个
地块、1000 个高度场。条带用来减少单个 geom-heightfield 配对的三角候选，
不修改高度、水平精度或机器人碰撞资产。恢复配置不保证随机布局与旧 run 完全相同。

旧条带地形的 CPU 转向诊断曾返回约 −106 mm 的异常接触距离，
而相同姿态标准 MuJoCo 的接触距离约为 −8 mm。旧实现仍保留在
`tienkung_terrain.py`，当前 Rough V4 已重新引用它。
原生随机地形工厂仍保留在 `terrain.py` 供对照，但不是当前 Rough V4 的默认地形。
回退地形不能等同于消除碰撞异常、NaN 或本机长期死机风险。

保留 CCD 50、接触容量每环境 128（跨环境共享池）、约束容量每环境 768、
接触传感器匹配容量 1024，以及出生位置远离外边界的保护。
独立检查脚本 `scripts/check_elf3_v4.py` 支持 `--device cpu` 或 `cuda:0`，
逐物理子步验证原始物理状态、接触容量及有限性；接触距离不是实测穿模深度。
正式训练仍默认 4096 环境，不会自动停止或热更新已有训练。

### 原生地形替换后的验证（2026-09-13）

本节为回退前的历史验证，不代表当前恢复的 GRAVEL 已重新完成动力学压力测试。

29 项 CPU 回归测试通过；对比替换前配置，V4/V4-Loco 的 train/play
都只有 `scene.terrain` 改变，奖励、指令、观测、随机化和 PPO 设置不变。
16 环境、100% 起身数据 reset 的 GPU 动力学压力测试完成 1000 控制步，
逐物理子步的原始物理状态、接触距离/法线、观测和奖励均有限，无容量溢出。
这不是 AMP/PPO 更新测试，也不是完整 4096 环境学习验证。

**残余碰撞异常尚未解决：**同种 GPU 测试的 300 步回查捕获约 −193 mm
接触距离及向下法线；相同姿态标准 MuJoCo 接触距离约 −23 mm，
网格顶点相对地面的最大垂直穿入约 27 mm。V4-Loco 的 16 环境、400 步
CPU 测试也捕获 −210 mm，对应标准 MuJoCo 约 −24 mm、顶点约 35 mm。
因此异常不只存在于旧条带地形，不能宣称换自带地形已消除碰撞错误或长期 NaN。
接触距离与顶点垂直探测均不能单独作为精确穿模深度。

复查命令（小批量独立测试，不停止现有训练）：

```bash
.venv/bin/python scripts/check_elf3_v4.py \
  --device cuda:0 --num-envs 16 --steps 300 \
  --recovery-fraction 1.0 --verify-contacts
```

Actor/critic 不输入 `terrain_scan`；Actor 仍为单帧 96 维、4 帧历史 384 维。
仅仿真内部有一个 3×3 点局部地面高度探针 `terrain_height`，用于身高奖励
和低身高终止判定；不送进任何 observation，不要求真机有深度相机/雷达。
身高改为相对局部地面，避免世界坐标 Z 随地形抬高/降低导致误判。

## 命令和训练保留项

```python
lin_vel_x = (-0.6, 1.0)
lin_vel_y = (-1.0, 1.0)
ang_vel_z = (-1.0, 1.0)
heading = (-math.pi / 2, math.pi / 2)
turning_max_abs_ang_vel = 2.0
```

训练与 play 一致；取消旧 `command_vel` 扩速课程，不会在 5000 轮后
恢复到跑步范围。原地转向限制在 ±2.0 rad/s（最小绝对值 0.3）。普通指令
采样为 5% 静止、15% 原地转向、20% 纯横移、60% 混合运动；纯横移左右
等概率，绝对速度在 0.2～1.0 m/s，vx=wz=0，不受 heading 覆盖。
Heading 模式开启，只对相应的混合运动环境启用，航向误差换算出的 yaw
命令同样受 ±1.0 限制；原地转向保持直接角速度命令。

保留网络、AMP/PPO 参数和 V4 的 40% 起身环境。V4 专用奖励与起步训练
见下节；V3/V3.1 的奖励、指令和重置不受这些覆盖影响。

## V4 脚部安全与静止起步训练

实现入口：`src/tasks/amp_loco/config/elf3/env_cfgs.py`，独立实现位于
`mdp/v4_rewards.py`、`mdp/v4_command.py`、`mdp/v4_events.py`。
V4 与 V4-Loco 都使用这些配置，Actor 仍为 96×4=384 维。
修改代码不会热更新已经运行的训练，需要新启动的进程才会采用。

### 随指令幅度变化的速度容差

水平速度在躯干 yaw 坐标系计算，仅跟踪 x/y；转向仅跟踪世界 z 轴角速度。
每轴使用 `sigma = absolute_tolerance + relative_tolerance * abs(command)`：

| 轴 | 容差公式 | 对应示例 |
| --- | --- | --- |
| vx | `0.25 + 0.35*abs(vx_cmd)` | 后退 −0.6 → 0.46；前进 +1 → 0.60 m/s |
| vy | `0.20 + 0.35*abs(vy_cmd)` | ±0.2 → 0.27；±0.5 → 0.375；±1 → 0.55 m/s |
| wz | `0.35 + 0.40*abs(wz_cmd)` | ±0.5 → 0.55；±1 → 0.75；±2 → 1.15 rad/s |

水平奖励为 `exp(-sum((error_xy/sigma_xy)^2))`，偏航奖励同理；权重都保持 1。
容差只随目标指令变化，实际速度不能通过增大自身来放宽误差标准。
例如忽略 vy=0.5 的静止得分由约 0.779 降至 0.169；忽略 wz=0.5 的
原地转向得分由约 0.939 降至 0.438。大指令使用更宽容差以避免奖励完全饱和。
原线速度奖励中的竖向约束拆成独立 `vertical_velocity`（`vz^2`，权重 −0.25），
身体滚转/俯仰稳定项保留，不再混入偏航跟踪误差。

### 新增腿部奖励

- `feet_safe_distance`，权重 −0.5：用 ELF3 实际脚部网格的旋转包围盒，
  通过 15 个分离轴计算保守接近程度。脚部局部盒为长 24 cm、宽 8 cm、
  高 5.45 cm，安全余量 2 cm；平行同高时约在中心横向间距小于 10 cm
  才开始扣分，不是 27 cm 的硬限制，也不是固定步宽目标。
  会考虑脚的 yaw/roll/pitch、前后错开和竖直净空；专项起步前 2 秒乘以 1.5。
  保留原全身自碰撞项，并增加专用左右脚接触传感器（maxforce、4 槽、4 子步历史）
  记录实际脚间碰撞比例。包围盒是几何近似，不保证所有碰撞都能提前避免。
- `legs_stand_pose`，权重 −0.5：12 个腿部关节相对默认站姿的平均绝对偏差。
  只有 |vx|、|vy|、|wz| 都小于 0.02 才生效；纯原地转向时解除，不约束手臂和腰。
- `feet_stumble`，权重 −0.1：脚对地面的水平净接触力同时大于 20 N、
  以及竖直力的 5 倍时扣分。用世界坐标净力，不误用自碰撞传感器的接触坐标。
  它是辅助地面绊脚检测，不能代替脚间距离或脚间碰撞项。

新奖励沿用延迟起身期间的屏蔽语义。验证过的四个横移 AMP 片段、两段
原地转向片段均未触发脚盒安全距离惩罚；整套 5089 帧中仅 10 帧触发轻微接近惩罚。
这些参数经过实现验证，实际步态改善仍需要训练后用专项指标比较。

### 起步序列

训练时，25% 的非起身重置选择默认站姿、零根速度和零关节速度；其余仍从
原 AMP 动作帧初始化。V4 的恢复环境不被此序列替换；V4-Loco 无恢复环境。

1. 保持零指令 1～2 秒，此准备阶段免除随机推扰。对水平速度向量和总角速度
   向量做时间常数 0.10 秒的指数平滑；平滑后阈值仍为 0.08 m/s、0.15 rad/s。
   最近 0.2 秒窗口内至少 80% 的帧符合要求，并且当前帧也符合，才允许起步。
   双脚各 >20 N、重力投影 z <−0.94 是硬条件；原始水平速度必须 <0.20 m/s、
   原始总角速度必须 <1.0 rad/s。任一硬条件失败立即清空窗口，不能靠平滑
   掩盖失去支撑或明显失稳；轻微平滑速度超限只损失该帧，不清空整个窗口。
2. 站稳后直接给阶跃指令，75% 是左右纯横移（|vy|=0.2～1.0），25% 是
   左右原地转向（|wz|=0.3～2.0），左右等概率；保持 4 秒后回到普通采样。
3. 3 秒仍未满足站稳条件则计为准备超时，退出专项并恢复普通采样，不把它
   计为成功起步。每次重置清空相应环境的阶段和计时，不污染其他环境。

等待、窗口、超时和运动持续时间均按整数控制步判断，时长向上取整；50 Hz
下 0.2 秒恰为 10 步，避免 float32 累加造成额外一帧等待。每环境单独清空
平滑器和窗口；未重置的环境不受影响。

此修复针对旧门槛在探索噪声下无法触发的问题：24500 检查点的小规模对照中，
旧门槛确定性动作 8/64 起步，采样动作 0/64 起步，瓶颈是瞬时角速度与连续
计时清零的组合。不能只验证确定性 play，也不能超时后强制发起起步。
相同检查点、16 环境和 4 组 reset 种子的修复后对照：确定性动作 64/64 起步，
采样动作 35/64 起步、29/64 超时，两组准备阶段均无终止。该结果证明专项
可以获得训练样本，不代表起步后已经不会绊脚；需继续训练并按方向复测。
修复通过 31 项 CPU 回归测试。25000 检查点另做 16 环境、400 控制步的
带噪声检查：8 次起步、8 次准备超时，无重置、观测和奖励有限；左右横移
前两秒分别采到 300/100 个环境控制步，未记录到 >10 N 脚间接触或终止。
这是小样本实现检查，不是实机安全保证或收敛结论。

起步序列期间的 heading 被关闭。阶段通过原生命令观测历史进入策略，
不重写历史命令帧、不在起步时修改机器人状态、不增加部署观测。
Play 默认 `startup_fraction=0`，不自动接管手动指令。
SwanLab/日志沿用原训练器记录，增加 `Metrics/v4/*` 的横移/偏航误差、
脚间接近与接触比例、纯横移/起步准备占比，以及每 episode 起步数和超时数。
`Metrics/twist/startup_attempts` 记录专项尝试数，`startup_prepare_steps` 和
各 `startup_*_fail_steps` 记录准备步数及各条件失败步数。`Metrics/v4/startup_pos_y_*`
和 `startup_neg_y_*` 单独记录左右横移起步前两秒的样本数、脚间接触、接近与
终止比例；样本数为 0 时的零比例不代表安全。全程平均不能代替专项评价。

小规模仿真验证（不训练、不连接 SwanLab，可选已有 PT 验证真实策略起步）：

```bash
.venv/bin/python -m scripts.check_elf3_v4_startup --num-envs 8 --steps 400 \
  --checkpoint /absolute/path/to/model.pt

# 必须同时检查训练时的带噪声动作；不更新模型、不连接日志服务。
.venv/bin/python -m scripts.check_elf3_v4_startup --num-envs 16 --steps 400 \
  --checkpoint /absolute/path/to/model.pt --sample-actions
```

## amp_v4 数据

路径 `src/assets/motions/elf3/amp_v4`，来源是修复后的 contact-IK V3.1。
移除 8 段跑步（含弧线跑步及镜像），保留 16 段、7664 帧：

- 4 段直线走路，4 段弧线走路。
- 4 段侧移，2 段原地转向。
- 1 段静止，1 段倒地起身。

保留的 NPZ 与源文件 SHA-256 一致；旧数据集不动，没有裁剪、慢放或新姿态编辑。
`WalkandRun` 只是兼容目录名，内部已没有跑步。AMP 和 reset 同时切换到 V4，
不会只删 AMP 示范而让 reset 继续从跑步帧初始化。
AMP 各段期望采样概率变为 1/16：保留动作的相对占比会上升，
并不是将跑步权重重分配后再保持旧的 1/24。
移除跑步示范和降低速度上限不等于数学上禁止腾空步态，训练效果仍需验证。

## 操作命令

尚未生成 V4 数据时（已存在则拒绝覆盖）：

```bash
.venv/bin/python scripts/build_elf3_amp_v4_dataset.py
```

测试（GPU smoke check 不训练、不连接 SwanLab）：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m scripts.check_elf3_v4 --num-envs 16 --steps 120
# 全尺寸倒地接触容量测试，不加载策略、不训练、不写 SwanLab：
.venv/bin/python -m scripts.check_elf3_v4 --num-envs 4096 --steps 1000 \
  --recovery-fraction 1.0
```

新训练（执行前先决定是否停止同 GPU 上的 V3.1，避免同时跑两个 4096 环境）：

```bash
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4 \
  --env.scene.num-envs 4096 \
  --enable-nan-guard True \
  --swanlab-project locomotion \
  --swanlab-experiment-name elf3_v4_gravel_walk_fresh
```

模型写入 `logs/rsl_rl/elf3_amp_locomotion_v4/<新实验目录>/`。没有配置自动续训。

## 独立的无起身任务：V4-Loco

任务名：`BXI-ELF3-AMP-Rough-V4-Loco`。保留带起身的 V4 任务，二者共享上面的
腿部安全奖励和起步专项实现，不自动替换正在运行的训练。

- 起身 reset 比例为 0，`recovery_dir=None`，不安装延迟终止管理器。
  触发现有倒地终止条件后在正常控制步 reset，不再等待 250 步尝试起身。
- AMP 专家路径只指向 `amp_v4/WalkandRun`：15 段、5089 帧，包含走路、
  弧线走路、侧移、转向和静止；不再把 Recovery 的起身动作作为专家。
  动作帧 reset 同样只使用这些非起身数据；起步专项另用默认零速度站姿。
  现有 `amp_v4/Recovery` 文件保留不动。
- 地形、COM 随机化、速度范围、观测接口、奖励定义/权重、网络及 PPO 设置均与 V4 相同。
  由于没有延迟起身环境，原有起身环境奖励掩码不再生效；AMP 剩余片段期望权重为 1/15。
- 默认从头训练，`resume=False`、初始 LR `1e-3`、目标 `model_200000.pt`，
  独立保存到 `logs/rsl_rl/elf3_amp_locomotion_v4_loco/`。未写入本机续训参数。

**显存说明：**不带起身并不等于少运行 40% 的机器人，仍然是 4096 个并行环境。
当前版本保留 CCD 50、接触容量 128、约束容量 768，以及相同旧 GRAVEL 地形。
这些预分配数组、PPO rollout 和网络不会因实际倒地接触减少而自动缩小。
直接省下的主要是起身动作张量及采样缓存（MiB 量级），不能据此宣称省下数 GiB。
无起身可能允许进一步缩减碰撞容量/分块开销，但必须独立做跌倒压力测试；
短控制步中仍可能发生身体着地，不能直接关闭身体碰撞或移除高度场溢出防护。
本次腿部安全/起步改动通过 26 项 CPU 回归测试，以及 V4、V4-Loco 各 8 环境、
400 控制步的 GPU 检查；使用已有 22600 模型验证起步状态切换与原生命令历史。
V4-Loco 有 2 次成功启动、6 次准备超时；带起身 V4 有 2 次成功启动、3 次
准备超时，另外 3 个恢复环境未被专项重置覆盖。两组观测/奖励均有限、无重置。
测试没有训练或连接 SwanLab，也没有打断已有训练；它验证实现行为，不证明
改动后的策略已经收敛或不会踩脚。新配置尚未进行完整 4096 环境学习评估。

新训练命令（不要与本机当前 4096 环境训练同时启动）：

```bash
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4-Loco \
  --env.scene.num-envs 4096 \
  --enable-nan-guard True \
  --swanlab-project locomotion \
  --swanlab-experiment-name elf3_v4_loco_gravel_walk_fresh
```

Play 使用同名任务：

```bash
.venv/bin/python scripts/play.py BXI-ELF3-AMP-Rough-V4-Loco \
  --checkpoint-file <模型路径> --num-envs 20 --export-onnx False
```

从带起身 checkpoint 续训需要显式传入 resume 参数，而且保留的策略/判别器
参数可能仍带有起身训练历史；该任务配置本身不会自动清除已学到的行为。
