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

- `Unitree-G1-AMP-Rough`
- `Unitree-G1-AMP-Flat`
- `BXI-ELF3-AMP-Rough`
- `BXI-ELF3-AMP-Flat`
- `BXI-ELF3-AMP-Flat-Loco` (stage-one walk/run-only pretraining)

## Training

```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

ELF3:

```bash
source .venv/bin/activate
python scripts/train.py BXI-ELF3-AMP-Flat-Loco --env.scene.num-envs=4096
python scripts/train.py BXI-ELF3-AMP-Flat --env.scene.num-envs=4096
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
