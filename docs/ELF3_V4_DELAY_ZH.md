# V4 天工 GRAVEL + 起身：动作延迟对照

任务：`BXI-ELF3-AMP-Rough-V4-Delay`。
基线 `BXI-ELF3-AMP-Rough-V4` 和 V4-Loco 保持无新增动作延迟，
所有 V4 任务使用老 GRAVEL 地形。V4 平地任务已取消，V3.1 保持不变。

## 唯一训练配置改动

- 位置目标延迟范围：0～40 ms（含两端，离散步长 5 ms）。
- 每个环境在 reset 时独立采样一次 `lag ∈ {0,…,8}`，本回合保持不变。
- 29 个关节共用该环境的 lag；按物理子步而非策略步存储、读取目标。
- 不添加观测延迟，不延迟 PD 内环反馈，不改变 PD 增益、动作尺度、
  编码器偏置逻辑、奖励、命令、AMP 数据、地形、碰撞或归一化实现。
- reset 清空该环境旧目标；首次新目标填充历史，避免跨回合重放。
- 策略接口仍为原始 384 维输入、29 维输出；上一动作仍是生成动作。
  ONNX 不执行模拟延迟，不需要部署端再人为增加延迟。

本次 0～40 ms 是基于策略敏感性测试选择的对照范围，不是真机实测延迟。
它不覆盖通信丢包、每步随机抖动、观测延迟或驱动器反馈延迟。

## 本机从旧 22800 续训

```bash
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4-Delay \
  --agent.resume True \
  --agent.load-run 2026-09-12_17-22-39_v4_startup_recovery_fresh \
  --agent.load-checkpoint model_22800.pt \
  --agent.run-name v4_gravel_recovery_delay_0_40ms_from22800 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 200001 \
  --target-iteration 200001 \
  --resume-optimizer True \
  --enable-nan-guard True \
  --swanlab-project locomotion \
  --swanlab-experiment-name elf3_v4_gravel_recovery_delay_0_40ms_from22800
```

使用单独日志目录和 SwanLab 新实验；不覆盖旧模型或原实验曲线。
继续加载 actor/critic、AMP 判别器与归一化、观测归一化、优化器及课程时钟。
源检查点的迭代为 22800，学习率为 `5.0625e-5`，
`common_step_counter=547248`，`sim_step_counter=2188992`。
目标仍为保存 `model_200000.pt`；不是额外训练 200000 轮。
任务默认仍 `resume=False`，续训只由上述命令指定。

本机本次日志目录：
`logs/rsl_rl/elf3_amp_locomotion_v4/2026-09-14_17-26-16_v4_gravel_recovery_delay_from22800`。
SwanLab：[locomotion / 0～40 ms 延迟续训](https://swanlab.cn/@Rainbowow/locomotion/runs/qt4s57gb)。
后台服务为 `amp-mjlab-elf3-v4-gravel-delay.service`，关闭终端不会结束训练。
故障不自动重启；再次续训需改为本次的新检查点并指定此 SwanLab ID，
不能重新执行服务初始命令而回退到旧 22800。

启动前验证：39 项 CPU 单元测试通过；32 个 GPU 环境运行 120 个策略步
（480 个物理子步），qpos/qvel/qacc、观测、动作、奖励均为有限值。
这只验证短时运行和接口，不保证长期数值稳定或真机抖动已经改善。

## 播放

```bash
.venv/bin/python scripts/play.py BXI-ELF3-AMP-Rough-V4-Delay \
  --checkpoint-file <本次日志目录>/model_<轮次>.pt \
  --num-envs 1 --export-onnx True
```

默认播放也启用 0～40 ms 延迟。零额外延迟对照使用无延迟基线任务，
加载相同检查点（`play.py` 不接受嵌套环境参数）：

```bash
.venv/bin/python scripts/play.py BXI-ELF3-AMP-Rough-V4 \
  --checkpoint-file <本次日志目录>/model_<轮次>.pt \
  --num-envs 1 --export-onnx False
```

验证应同时关注零指令站立、左右原地转向、前后走及停止后的抖动，
以及同一平地测试中的脚掌倾角和脚跟支撑，不能只依据训练 loss。
