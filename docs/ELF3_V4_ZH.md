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
lin_vel_y = (-0.5, 0.5)
ang_vel_z = (-1.57, 1.57)
heading = (-math.pi, math.pi)
```

训练与 play 一致；取消旧 `command_vel` 扩速课程，不会在 5000 轮后
恢复到跑步范围。原地转向也限制在 ±1.57 rad/s（最小绝对值 0.3），
保留 5% 静止、15% 原地转向、80% 混合运动的采样分配。
Heading 模式开启，只对相应的混合运动环境启用，航向误差换算出的 yaw
命令同样受 ±1.57 限制；原地转向保持直接角速度命令。

保留 V3.1 的奖励权重、跟踪宽度、网络、AMP/PPO 参数和 40% 起身环境。
仅 V4 的质心范围、地形、数据、速度范围和地形高度参考按要求调整。

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

任务名：`BXI-ELF3-AMP-Rough-V4-Loco`。原 V4 任务不变，也不自动替换正在运行的训练。

- 起身 reset 比例为 0，`recovery_dir=None`，不安装延迟终止管理器。
  触发现有倒地终止条件后在正常控制步 reset，不再等待 250 步尝试起身。
- AMP 专家路径只指向 `amp_v4/WalkandRun`：15 段、5089 帧，包含走路、
  弧线走路、侧移、转向和静止；不再把 Recovery 的起身动作作为专家。
  reset 同样只使用这些非起身数据。现有 `amp_v4/Recovery` 文件保留不动。
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
当前仅做了 CPU 配置、真实动作加载与采样回归测试，未为测试打断已有 GPU 训练。

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
