# ELF3 V4：GRAVEL 盲走 + walking-only AMP

任务：`BXI-ELF3-AMP-Rough-V4`。默认从头训练，`resume=False`，初始学习率
`1e-3`、自适应调度，目标保存至 `model_200000.pt`。没有写入本机 V3.1
续训路径，也不会自动停止或替换正在运行的任务。

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

按照用户选择，使用 TienKung-Lab 的 ELF3 `walk_cfg.py` 当前启用的
**GRAVEL_TERRAINS_CFG**，不是同文件中的 ROUGH 综合台阶/坑地形。
核对版本为 `c4e7f0974b0eef90023a014ff21407cfa79bbfec`：

- [地形参数](https://github.com/MelodyAI/TienKung-Lab/blob/c4e7f0974b0eef90023a014ff21407cfa79bbfec/legged_lab/terrains/terrain_generator_cfg.py)
- [ELF3 任务选择](https://github.com/MelodyAI/TienKung-Lab/blob/c4e7f0974b0eef90023a014ff21407cfa79bbfec/legged_lab/envs/elf3/walk_cfg.py)

| 参数 | V4 |
| --- | --- |
| 地块大小 / 训练网格 | 8×8 m / 10×20 地块 |
| 外边界 | 20 m |
| 高度取值 | −2、0、2、4 cm |
| 水平 / 竖直离散精度 | 0.1 m / 0.005 m |
| 地块内边界参数 | 0.25 m |
| curriculum | False，与源 GRAVEL 一致 |
| 子地形权重 | 原值 0.2；只有一种类型，归一化后占 100% |

`tienkung_terrain.py` 独立实现 mjlab/MuJoCo 高度场，保留 IsaacLab 2.1
端点网格（81×81）、边界离散方式（上述参数对应 3 格）、负高度基准和
中心 2×2 m 最高点作为出生高度。最大相邻高度差 6 cm / 网格 10 cm，
小于源 `slope_threshold=0.75`，此配置不触发竖直面修正。
MuJoCo 使用 hfield，源仓库使用 PhysX 三角网格；随机布局和接触求解不承诺逐位一致。
倒地接触测试发现 MuJoCo-Warp 单个 geom-heightfield 配对最多收集 50 个
三角接触。为避免 ELF3 torso 倒地时溢出，每个 8 m 地块沿 X 分成
40 个 0.2 m 宽高度场条带，共 8000 个碰撞高度场。条带共享边界顶点，
高度值和网格精度不变，不修改机器人碰撞体或引擎常量。每个条带独立设置
高度偏置和幅值，以适应 MuJoCo 编译时的独立归一化；恒高边界仍严格为 0 m。
测试同时核对编译后的 `hfield_data`，而不只检查编译前的输入数组。
V4 接触缓冲为每环境 128（跨环境共享池）、约束容量每环境 768、
接触传感器匹配容量 1024。这是引擎适配，不改变地形采样。
CCD 迭代上限采用现有 ELF3 flat 的 50 次。rough 原来的 500 次配合上述
最初的 256 接触容量，会使 4096 环境仅一个 Warp EPA 临时数组就申请约 31.5 GB，
无法在本机 24 GB 显存运行。
即使只降 CCD 到 50，256 / 1500 的接触/约束容量仍会在创建 CUDA Graph
时显存不足。容量测试逐物理子步检查接触总数、宽相碰撞候选总数与单环境
约束数，并检查日志中的 CCD、高度场和传感器溢出；不能仅以程序未崩溃作为通过标准。
没有把 4096 环境降到 1024，也没有通过减小每轮采样量来绕过显存问题。
本机 RTX 4090 24 GB 实测：最终地形以 4096 环境、100% 起身 reset
运行 400 控制步（逐物理子步检查），接触池最高 36245/524288、
宽相候选池 105195/524288、单环境约束 124/768；无溢出或非有限数值。
该独立测试总显存峰值约 17.0 GiB。包含 AMP/PPO 的正式 4096 环境续训
启动后约 19.9 GiB，已完成首批更新和 checkpoint 保存。这是启动验证，
不代表保证任何未来姿态或并行进程都不会超容量；不要同 GPU 再开大批量 play。
没有引入原仓库的缓存或 IsaacLab 运行依赖。Play 缩小为 5×5 地块，地形分布不变。

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

1. 保持零指令 1～2 秒，并要求连续 0.2 秒双脚各有 >20 N 接触、水平速度
   <0.08 m/s、总角速度 <0.15 rad/s、身体接近直立。此准备阶段免除随机推扰。
2. 站稳后直接给阶跃指令，75% 是左右纯横移（|vy|=0.2～1.0），25% 是
   左右原地转向（|wz|=0.3～2.0），左右等概率；保持 4 秒后回到普通采样。
3. 3 秒仍未满足站稳条件则计为准备超时，退出专项并恢复普通采样，不把它
   计为成功起步。每次重置清空相应环境的阶段和计时，不污染其他环境。

起步序列期间的 heading 被关闭。阶段通过原生命令观测历史进入策略，
不重写历史命令帧、不在起步时修改机器人状态、不增加部署观测。
Play 默认 `startup_fraction=0`，不自动接管手动指令。
SwanLab/日志沿用原训练器记录，增加 `Metrics/v4/*` 的横移/偏航误差、
脚间接近与接触比例、纯横移/起步准备占比，以及每 episode 起步数和超时数。

小规模仿真验证（不训练、不连接 SwanLab，可选已有 PT 验证真实策略起步）：

```bash
.venv/bin/python -m scripts.check_elf3_v4_startup --num-envs 8 --steps 400 \
  --checkpoint /absolute/path/to/model.pt
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
当前版本保留已验证的 CCD 50、接触容量 128、约束容量 768，以及相同地形分块。
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
