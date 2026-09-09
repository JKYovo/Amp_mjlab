# ELF3 部署与环境复现

本文记录当前 ELF3 AMP 训练机的可用环境，并给出在另一台 Linux 机器上创建独立环境的步骤。仓库不包含 `.venv`、训练日志、SwanLab 凭据或 checkpoint。

## 已验证环境

记录日期：2026-09-09。

| 项目 | 当前训练机版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.1 LTS (Jammy) |
| Kernel | 6.8.0-138-generic |
| glibc | 2.35 |
| CPU | Intel Core i9-14900K，32 logical CPUs |
| 内存 | 62 GB 可见内存 |
| GPU | NVIDIA GeForce RTX 4090，24 GB，SM 8.9 |
| NVIDIA Driver | 595.71.05 |
| `nvidia-smi` CUDA capability | 13.2 |
| Python | 3.11.16 |
| pip | 26.2.1 |
| PyTorch | 2.14.0（`torch.version.cuda == 13.0`） |
| cuDNN | 9.24.0 |
| mjlab | 1.2.0 |
| MuJoCo | 3.12.0 |
| mujoco-warp | 3.8.1 |
| warp-lang | 1.12.0 |
| SwanLab | 0.8.5 |

其余直接依赖的锁定版本在仓库根目录的 `requirements-elf3.txt`。

不要求新机器与上述 CPU/GPU 完全一致，但建议使用 Linux、Python 3.11、支持所安装 PyTorch CUDA runtime 的 NVIDIA 驱动，并为 4096 个平地环境预留至少约 9 GB 显存。减少 `--env.scene.num-envs` 可以降低显存占用。

## 1. 克隆并创建独立环境

以下使用 Conda。环境安装在项目自己的 `.venv`，不会污染 base 环境。

```bash
git clone https://github.com/JKYovo/Amp_mjlab.git
cd Amp_mjlab

conda create --prefix ./.venv python=3.11.16 pip -y
conda activate ./.venv
python -m pip install --upgrade pip
```

如果目标机器不使用 Conda，也可以使用 Python 3.11 的 `venv`，但 CUDA/PyTorch wheel 仍需与目标机器驱动兼容。

## 2. 安装依赖与本项目 AMP fork

```bash
python -m pip install -r requirements-elf3.txt

# 必须覆盖 mjlab 自动安装的官方 rsl_rl：本项目使用仓库内 AMP fork。
python -m pip install --no-deps -e ./rsl_rl
python -m pip install --no-deps -e .
```

`mjlab==1.2.0` 的包元数据要求 `rsl-rl-lib==5.0.1`，而本仓库内 AMP fork 的包版本是 `2.3.1`。因此 `pip check` 会报告这一项版本冲突，这是当前项目有意使用本地 fork 造成的；不要再执行 `pip install rsl-rl-lib==5.0.1` 覆盖它。

如果 `torch==2.14.0` 在目标机器的默认 pip index 没有合适的 CUDA wheel，请按 PyTorch 对应 CUDA wheel 的官方安装方式先安装同版本 PyTorch，再执行其余依赖安装。最终用下面的命令确认 CUDA 可用。

## 3. 应用 mjlab observation-history 补丁

当前任务使用 `history_ordering="time"`，需要把仓库里的补丁复制到已安装的 mjlab：

```bash
MJLAB_DIR="$(python -c 'import pathlib, mjlab; print(pathlib.Path(mjlab.__file__).parent)')"
cp mjlab_patch/mjlab/managers/observation_manager.py \
  "$MJLAB_DIR/managers/observation_manager.py"
```

检查补丁是否一致：

```bash
cmp mjlab_patch/mjlab/managers/observation_manager.py \
  "$MJLAB_DIR/managers/observation_manager.py"
```

## 4. 安装后自检

```bash
python - <<'PY'
from importlib.metadata import version

import torch, mujoco, mujoco_warp, warp

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("mujoco:", mujoco.__version__)
print("mujoco-warp:", mujoco_warp.__version__)
print("warp:", warp.__version__)
print("mjlab:", version("mjlab"))
PY

python scripts/list_envs.py --keyword ELF3
```

任务列表应至少包含：

- `BXI-ELF3-AMP-Flat`
- `BXI-ELF3-AMP-Flat-Loco`
- `BXI-ELF3-AMP-Rough`

仓库已经包含训练所需的 ELF3 MJCF/mesh 和 17 段走跑 AMP 数据。`Recovery` 中包含一段从 G1 语义映射得到的临时起身数据；正式部署前仍建议替换成 ELF3 原生起身动作。

## 5. 数据播放检查

有桌面环境时：

```bash
python scripts/play_elf3_amp_motion.py \
  --input-path src/assets/motions/elf3/amp/WalkandRun
```

无桌面环境时：

```bash
MUJOCO_GL=egl python scripts/play_elf3_amp_motion.py \
  --mode video \
  --input-path src/assets/motions/elf3/amp/WalkandRun \
  --output artifacts/elf3_amp_walkandrun.mp4 \
  --max-seconds-per-clip 3
```

## 6. 训练、续训与 SwanLab

联合训练走跑和恢复：

```bash
python scripts/train.py BXI-ELF3-AMP-Flat \
  --env.scene.num-envs 4096 \
  --enable-nan-guard True
```

使用 SwanLab 前在新机器单独登录，不要把 API key 写进仓库：

```bash
swanlab login

python scripts/train.py BXI-ELF3-AMP-Flat \
  --env.scene.num-envs 4096 \
  --swanlab-project locomotion \
  --swanlab-experiment-name elf3_amp_flat
```

从本地 checkpoint 续训到指定总迭代数：

```bash
python scripts/train.py BXI-ELF3-AMP-Flat \
  --log-dir logs/rsl_rl/elf3_amp_locomotion/<run_dir> \
  --target-iteration 100001 \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run '<run_dir>' \
  --agent.load-checkpoint 'model_.*.pt'
```

若还要续接已有 SwanLab run，同时添加：

```text
--swanlab-project <project> --swanlab-id <run_id> --swanlab-resume must
```

`--target-iteration` 表示总目标迭代数。服务或进程重新启动后，它会从 checkpoint 中记录的 iteration 计算剩余轮数，避免每次额外再训练完整的 `max_iterations`。

## 7. Policy 回放

```bash
python scripts/play.py BXI-ELF3-AMP-Flat \
  --checkpoint-file logs/rsl_rl/elf3_amp_locomotion/<run_dir>/model_<iter>.pt \
  --num-envs 20 \
  --export-onnx False
```

注意：当前机器曾在 4096 环境 GPU 训练期间并行运行 native viewer，触发 X11/NVIDIA UVM 内核故障。单 GPU 机器上不要同时运行大规模训练和 `--viewer native` 的 play；先停止训练，或者换另一张 GPU/另一台机器播放。

## 8. 不随 Git 仓库分发的内容

- `.venv/`：本机虚拟环境
- `logs/`：TensorBoard、SwanLab 本地数据、checkpoint 和 ONNX
- `.swanlab/`：登录凭据
- `artifacts/`：预览视频与接触表

如需在另一台机器继续当前训练，应另外复制所需的 `model_<iter>.pt` 和对应 run 目录。Git 仓库负责源码、机器人资产、AMP 数据及环境说明，不把持续变化的训练输出作为源码提交。
