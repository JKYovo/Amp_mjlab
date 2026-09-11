"""RL configuration for the BXI ELF3 AMP task."""

import os

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

from src.assets.robots.elf3.elf3_constants import (
  ELF3_AMP_BODY_NAMES,
  ELF3_ANCHOR_BODY,
  ELF3_JOINT_NAMES,
)
from src.tasks.amp_loco.config.g1.rl_cfg import RslRlAmpRunnerCfg


_MOTION_DATA_DIR = os.path.normpath(
  os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    "src",
    "assets",
    "motions",
    "elf3",
    "amp",
  )
)

_MOTION_DATA_V2_DIR = os.path.join(
  os.path.dirname(_MOTION_DATA_DIR), "amp_v2_balanced"
)
_MOTION_DATA_V3_DIR = os.path.join(
  os.path.dirname(_MOTION_DATA_DIR), "amp_v3"
)
_MOTION_DATA_V3_1_DIR = os.path.join(
  os.path.dirname(_MOTION_DATA_DIR), "amp_v3_1"
)


def elf3_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create the native ELF3 AMP/PPO runner configuration."""
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMPPPO",
    ),
    experiment_name="elf3_amp_locomotion",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100001,
    amp_reward_coef=0.1,
    amp_motion_files=_MOTION_DATA_DIR,
    amp_num_preload_transitions=200000,
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=[1024, 512, 256],
    min_normalized_std=[0.05] * len(ELF3_JOINT_NAMES),
    amp_body_names=ELF3_AMP_BODY_NAMES,
    amp_anchor_name=ELF3_ANCHOR_BODY,
  )


def elf3_amp_loco_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create the stage-one runner using only native walk/run references."""
  cfg = elf3_amp_ppo_runner_cfg()
  cfg.run_name = "loco_pretrain"
  cfg.amp_motion_files = os.path.join(_MOTION_DATA_DIR, "WalkandRun")
  return cfg


def elf3_amp_v2_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create a fresh-run configuration for the isolated balanced-v2 dataset."""
  cfg = elf3_amp_ppo_runner_cfg()
  cfg.experiment_name = "elf3_amp_locomotion_v2"
  cfg.run_name = "balanced_fresh"
  cfg.amp_motion_files = _MOTION_DATA_V2_DIR
  return cfg


def elf3_amp_v3_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create the V3 fine-tuning configuration backed by V3 motions."""
  cfg = elf3_amp_ppo_runner_cfg()
  cfg.experiment_name = "elf3_amp_locomotion_v3"
  cfg.run_name = "v3_from_v2"
  cfg.amp_motion_files = _MOTION_DATA_V3_DIR
  return cfg


def elf3_amp_v3_1_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Continue V3 training with the support-corrected V3.1 motions."""
  cfg = elf3_amp_v3_ppo_runner_cfg()
  cfg.run_name = "v3_1_from_v3"
  cfg.amp_motion_files = _MOTION_DATA_V3_1_DIR
  return cfg
