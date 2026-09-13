# AMP_mjlab

[中文 README](README_zh.md)

Deployment integration code is in [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm), under `MJAmp State`.

G1 / BXI ELF3 AMP motion control project built on top of mjlab + rsl_rl.

Key features of this repository:

- A single policy learns both locomotion (walk/run) and recovery (fall-and-get-up)
- AMP discriminator regularizes motion style and priors
- Training and deployment pipelines are consistent, with direct ONNX policy export support

## Core Idea

Instead of training separate policies for locomotion and recovery and switching between them, this project learns both capabilities in one unified policy.

Implementation highlights:

- Motion data split:
  - Walk/Run data: `src/assets/motions/g1/amp/WalkandRun`
  - Recovery data: `src/assets/motions/g1/amp/Recovery`
- Delayed termination/reset:
  - A subset of environments does not reset immediately after termination
  - These environments receive a recovery window and reset states sampled from recovery clips
- Unified AMP training:
  - One actor-critic + One AMP discriminator
  - Velocity tracking, perturbation robustness, and recovery are learned together

This reduces discontinuities caused by policy switching and yields more consistent behavior.

## Requirements

- Linux
- Python 3.11 (recommended)
- Working MuJoCo and GPU driver setup

See [docs/ELF3_DEPLOYMENT_ZH.md](docs/ELF3_DEPLOYMENT_ZH.md) for the exact
versions used on the ELF3 training host, isolated-environment setup, dependency
override order, and deployment checks.

## Quick Start

This quick start documents only the current ELF3 V3.1 and V4 tasks. Run commands
from the repository root. Checkpoints and credentials are not included in Git.

### 1. Install in an isolated environment

```bash
git clone https://github.com/JKYovo/Amp_mjlab.git
cd Amp_mjlab
conda create --prefix ./.venv python=3.11.16 pip -y
.venv/bin/python -m pip install -r requirements-elf3.txt
.venv/bin/python -m pip install --no-deps -e ./rsl_rl
.venv/bin/python -m pip install --no-deps -e .
```

The repository's AMP `rsl_rl` fork must override the upstream package. Exact
host versions and CUDA installation caveats are in the
[environment guide](docs/ELF3_DEPLOYMENT_ZH.md).

### 2. Apply the required observation-history patch

```bash
MJLAB_DIR="$(.venv/bin/python -c 'import pathlib, mjlab; print(pathlib.Path(mjlab.__file__).parent)')"
cp mjlab_patch/mjlab/managers/observation_manager.py \
  "$MJLAB_DIR/managers/observation_manager.py"
.venv/bin/python scripts/list_envs.py --keyword ELF3
```

Keep this patch: the policy contract is 96 values per frame, four frames in
time order (384 inputs), and 29 joint actions. Removing `history_ordering`
does not preserve the deployment interface.

### 3. Choose a task

| Task ID | Terrain / skills | AMP dataset | Default final checkpoint |
| --- | --- | --- | --- |
| `BXI-ELF3-AMP-Flat-V3-1` | Flat; walk/run + recovery | `src/assets/motions/elf3/amp_v3_1` | `model_100000.pt` |
| `BXI-ELF3-AMP-Rough-V4` | GRAVEL; walking + recovery | `src/assets/motions/elf3/amp_v4` | `model_200000.pt` |
| `BXI-ELF3-AMP-Rough-V4-Loco` | GRAVEL; walking, no recovery training | `src/assets/motions/elf3/amp_v4/WalkandRun` | `model_200000.pt` |

The two joint tasks use 40% recovery environments with a maximum five-second
delayed-termination window. Locomotion and recovery train together, not as two
mandatory stages. V4-Loco excludes recovery references and recovery resets and
uses immediate fall termination; it otherwise keeps the V4 physics capacities.
There is currently no registered V3.1-Loco or flat V4 task. Both V4 tasks are
blind GRAVEL locomotion without `terrain_scan`; see the [V4 guide](docs/ELF3_V4_ZH.md).

### 4. Start fresh training

Run **one** of these commands, not all three simultaneously:

```bash
# V3.1: flat locomotion and recovery
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Flat-V3-1 \
  --env.scene.num-envs 4096 --agent.resume False \
  --target-iteration 100001 --enable-nan-guard True

# V4: GRAVEL walking and recovery
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4 \
  --env.scene.num-envs 4096 --agent.resume False \
  --target-iteration 200001 --enable-nan-guard True

# V4: GRAVEL walking without recovery training
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4-Loco \
  --env.scene.num-envs 4096 --agent.resume False \
  --target-iteration 200001 --enable-nan-guard True
```

Iterations are zero-indexed: targets 100001 / 200001 save through iteration
100000 / 200000. Checkpoints are saved every 100 iterations. Default log roots:

- V3.1: `logs/rsl_rl/elf3_amp_locomotion_v3_1/<RUN_DIR>/`
- V4 joint: `logs/rsl_rl/elf3_amp_locomotion_v4/<RUN_DIR>/`
- V4-Loco: `logs/rsl_rl/elf3_amp_locomotion_v4_loco/<RUN_DIR>/`

4096-env V4 training uses approximately 20 GiB on the tested RTX 4090.
Reduce the environment count if necessary; removing recovery does not
automatically shrink the preallocated physics buffers.

### 5. Play a saved policy / export ONNX

Replace `<RUN_DIR>` and `<ITER>` with an existing run and checkpoint. Use the
same task as training; the current iteration is not necessarily a saved file.

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

To export, change to `--export-onnx True` (the default). Play writes
`<RUN_DIR>/export/<TASK_ID>_model_<ITER>.onnx`, including observation normalization;
do not normalize its inputs twice. Use `--export-onnx False`, not
`--no-export-onnx`. Joint-task play retains 40% recovery environments; V4 play
disables automatic startup trials and retains ordinary/manual commands.
Exit with `Ctrl+C`; avoid multiple GPU play processes alongside training.

### 6. Resume training and log to SwanLab

Resume the same V4 joint run, preserving the optimizer, normalizers and saved
curriculum clock. `<RUN_DIR>` is the directory name, not the full path:

```bash
.venv/bin/python scripts/train.py BXI-ELF3-AMP-Rough-V4 \
  --env.scene.num-envs 4096 --agent.resume True \
  --agent.load-run '<RUN_DIR>' --agent.load-checkpoint 'model_<ITER>.pt' \
  --log-dir "logs/rsl_rl/elf3_amp_locomotion_v4/<RUN_DIR>" \
  --resume-optimizer True --target-iteration 200001 --enable-nan-guard True
```

For V3.1 or V4-Loco, use the corresponding task and log root above; V3.1 uses
`--target-iteration 100001`. The target is the total final iteration count,
not additional iterations. Cross-task fine-tuning also requires selecting the
source experiment root via `--agent.experiment-name`; do not merely change the task ID.

```bash
# Log in separately on each machine; never put an API key in the repository.
.venv/bin/swanlab login
```

For a new experiment, append `--swanlab-project locomotion
--swanlab-experiment-name YOUR_EXPERIMENT --swanlab-resume never` to a training
command. To continue an existing experiment, append `--swanlab-project locomotion
--swanlab-id YOUR_RUN_ID --swanlab-resume must`. Do not resume the same SwanLab
experiment after rolling back below already uploaded steps; create a new one.

## Training

```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

ELF3:

```bash
source .venv/bin/activate
python scripts/train.py BXI-ELF3-AMP-Flat-Loco --env.scene.num-envs=4096
python scripts/train.py BXI-ELF3-AMP-Flat --env.scene.num-envs=4096
python scripts/train.py BXI-ELF3-AMP-Flat-V3-1 --env.scene.num-envs=4096
python scripts/train.py BXI-ELF3-AMP-Rough --env.scene.num-envs=4096
```

Logs are saved by default to:

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`
- `logs/rsl_rl/elf3_amp_locomotion/<time_stamp_run>/`

## ELF3 Notes

- The robot definition is HoloMotion's 29-DoF sim2sim ELF3 model. Native joint limits, controller gains, armatures, and action scales are preserved.
- BXI controller/MuJoCo uses waist, left leg, right leg, left arm, right arm order, while the Isaac layout is interleaved. Training actions, AMP files, and exported ONNX metadata use the former. Conversion and AMP loading always reorder by joint name.
- ELF3's physical floating root is `torso_link`; its G1-pelvis-equivalent semantic body is `waist_z_link`. Reset/root state uses the former, while the AMP policy body scheme uses the latter.
- The 17 walk/run clips are converted directly from native HoloMotion ELF3 references, with body FK and velocities recomputed using the canonical MuJoCo model.
- The bundled recovery clip is a provisional semantic conversion from G1 with ELF3 mesh ground alignment and joint-limit clipping. Replace it with native ELF3 get-up data before production training.
- ELF3 keeps the G1 reward terms, weights, tracking widths, and collision thresholds unchanged. Only robot body/site/geom mappings are replaced; the fall-height termination is adjusted for ELF3's higher physical root.
- The final policy is trained jointly on locomotion and recovery. For a new robot, use `BXI-ELF3-AMP-Flat-Loco` to stabilize flat-ground locomotion, then add native recovery data and ramp the delayed-recovery environment ratio from 0.1 to 0.4, and finally fine-tune on rough terrain.

The ELF3-specific reward and root settings are in `src/tasks/amp_loco/config/elf3/env_cfgs.py`.
The isolated V3.1 motion set is in `src/assets/motions/elf3/amp_v3_1`; rebuild it
from immutable V3 data with `scripts/build_elf3_amp_v3_1_dataset.py`.

Resume the joint task from the stage-one checkpoint with `--agent.resume True`,
`--agent.load-run <loco_run_dir>`, and `--agent.load-checkpoint 'model_<iter>.pt'`.

## Training Curve Note (Important)

- Around `2w` iterations (about 20k), the policy often suddenly learns fall-recovery behavior.
- As a result, multiple metrics in `logs` may show abrupt jumps. This is expected and not necessarily a training failure.

![Training log transition example](logs.png)

## Evaluation and Visualization

Replay with a trained checkpoint:

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt
```

Note: ONNX export is enabled by default in both training and play workflows.

## Motion Data Preparation

CSV-to-NPZ conversion script:

```bash
python scripts/csv_to_npz.py --help
```

Recommended data layout:

- Raw CSV: `motion_data_csv/amp`
- Converted NPZ: `src/assets/motions/g1/amp/WalkandRun` and `src/assets/motions/g1/amp/Recovery`

If valid NPZ files exist in these folders, training config loads them automatically.

Convert HoloMotion ELF3 data with:

```bash
python scripts/convert_elf3_amp_motion.py \
  --input-path /path/to/holomotion/elf3_npz \
  --output-dir src/assets/motions/elf3/amp/WalkandRun \
  --source-format holomotion
```

Preview converted AMP references in a desktop window:

```bash
python scripts/play_elf3_amp_motion.py \
  --input-path src/assets/motions/elf3/amp/WalkandRun
```

Record a headless directory preview:

```bash
MUJOCO_GL=egl python scripts/play_elf3_amp_motion.py \
  --mode video \
  --input-path src/assets/motions/elf3/amp/WalkandRun \
  --output artifacts/elf3_amp_walkandrun_all.mp4 \
  --max-seconds-per-clip 3
```

The player resolves joints and the physical root by name and rejects invalid order metadata or non-finite values.

## Repository Structure

- `src/tasks/amp_loco`: AMP locomotion/recovery task implementation
- `src/tasks/amp_loco/config/g1`: G1 task registration, env configs, RL configs
- `src/tasks/amp_loco/config/elf3`: ELF3 task registration, env configs, RL configs
- `src/tasks/amp_loco/mdp`: rewards, observations, events, termination logic
- `scripts/train.py`: training entry point
- `scripts/play.py`: playback entry point
- `scripts/csv_to_npz.py`: motion data conversion tool
- `scripts/convert_elf3_amp_motion.py`: HoloMotion/G1 AMP to ELF3 AMP converter
- `scripts/play_elf3_amp_motion.py`: name-safe ELF3 AMP reference player/recorder
- `mjlab_patch`: required local patch for mjlab

## Highlights

- One policy unifies walk/run and recovery skills
- AMP + velocity objective jointly optimize style and task performance
- Delayed reset with recovery sampling explicitly improves recovery ability
- End-to-end pipeline supports ONNX export for deployment

## Acknowledgements

- Thanks to [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) for open-sourcing their work and inspiration.
- Thanks to [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab); the rsl_rl AMP part in this project references their implementation.
