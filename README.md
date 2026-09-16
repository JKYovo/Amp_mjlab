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

### 1. Install

```bash
conda activate mjlab
cd AMP_mjlab
python -m pip install -e .
```

### 2. Apply mjlab Patch (Optional)

If you do not apply this patch, remove `history_ordering` configuration from the code.

What this patch does:

- It adds an option for how observation history is flattened: by time (`time`) or by term (`term`).
- Default mjlab behavior supports only `term` ordering.

Patch file:

- `mjlab_patch/mjlab/managers/observation_manager.py`

Example command:

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  /home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

### 3. List Available Tasks

```bash
python scripts/list_envs.py --keyword AMP
```

Main tasks:

- `BXI-ELF3-AMP-Rough-V4` (recommended ELF3 task)

The recommended V4 task uses TienKung GRAVEL without `terrain_scan`, the
walking-only `amp_v4/WalkandRun` reference set (no running clips), no recovery
reset or recovery AMP data, and randomized 0--40 ms position-target delay.

## Training

```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

ELF3:

```bash
source .venv/bin/activate
python scripts/train.py BXI-ELF3-AMP-Rough-V4 --env.scene.num-envs=4096 --enable-nan-guard True
```

This command starts fresh (`resume=False`, initial LR `1e-3`) and targets
`model_200000.pt`. Delay is sampled at reset, shared by all 29 joints and held
for the episode: 0--8 physics steps x 5 ms. Observations, PD feedback and the
384-input/29-output deployment interface are unchanged.

Resume a V4 checkpoint (run name is a directory under the V4 log root):

```bash
python scripts/train.py BXI-ELF3-AMP-Rough-V4 \
  --env.scene.num-envs 4096 \
  --agent.resume True --agent.load-run <run_dir> \
  --agent.load-checkpoint 'model_<iter>.pt' \
  --resume-optimizer True --target-iteration 200001 \
  --enable-nan-guard True \
  --swanlab-project locomotion --swanlab-experiment-name v4_delay_resume
```

This restores the model, normalization, optimizer and curriculum clock. The
target is the final iteration, not an additional 200000 iterations. For an
existing SwanLab run, use `--swanlab-id <id> --swanlab-resume must`; use a new
experiment when branching from an older checkpoint.

Logs are saved by default to:

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`
- ELF3 V4: `logs/rsl_rl/elf3_amp_locomotion_v4/<time_stamp_run>/`

## ELF3 Notes

- The robot definition is HoloMotion's 29-DoF sim2sim ELF3 model. Native joint limits, controller gains, armatures, and action scales are preserved.
- BXI controller/MuJoCo uses waist, left leg, right leg, left arm, right arm order, while the Isaac layout is interleaved. Training actions, AMP files, and exported ONNX metadata use the former. Conversion and AMP loading always reorder by joint name.
- ELF3's physical floating root is `torso_link`; its G1-pelvis-equivalent semantic body is `waist_z_link`. Reset/root state uses the former, while the AMP policy body scheme uses the latter.
- V4 uses 15 walking, turning, lateral and idle reference clips. Running clips and the bundled recovery clip are excluded from both AMP expert sampling and motion resets.
- The base ELF3 configuration preserves the mapped reward structure and fall-height threshold. V4 additionally uses its walking tracking, foot-safety and startup terms; action delay changes none of those rewards.

The ELF3-specific reward and root settings are in `src/tasks/amp_loco/config/elf3/env_cfgs.py`.
The V4 motion set is in `src/assets/motions/elf3/amp_v4/WalkandRun`.

## Evaluation and Visualization

Replay with a trained checkpoint:

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt
```

Note: ONNX export is enabled by default in both training and play workflows.

ELF3 V4:

```bash
python scripts/play.py BXI-ELF3-AMP-Rough-V4 \
  --checkpoint-file logs/rsl_rl/elf3_amp_locomotion_v4/<run_dir>/model_<iter>.pt \
  --num-envs 20 --export-onnx True
```

Playback also samples 0--40 ms latency. The exported ONNX contains no delay
buffer; do not add artificial delay on the real robot. Use `--num-envs 1` when
training is already occupying the GPU.

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
