from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  elf3_amp_flat_env_cfg,
  elf3_amp_flat_loco_env_cfg,
  elf3_amp_flat_v2_env_cfg,
  elf3_amp_flat_v3_env_cfg,
  elf3_amp_flat_v3_1_env_cfg,
  elf3_amp_flat_v4_env_cfg,
  elf3_amp_flat_v4_loco_env_cfg,
  elf3_amp_rough_env_cfg,
  elf3_amp_rough_v4_env_cfg,
  elf3_amp_rough_v4_loco_env_cfg,
)
from .rl_cfg import (
  elf3_amp_loco_ppo_runner_cfg,
  elf3_amp_ppo_runner_cfg,
  elf3_amp_v2_ppo_runner_cfg,
  elf3_amp_v3_ppo_runner_cfg,
  elf3_amp_v3_1_ppo_runner_cfg,
  elf3_amp_v4_ppo_runner_cfg,
  elf3_amp_v4_loco_ppo_runner_cfg,
  elf3_amp_flat_v4_ppo_runner_cfg,
  elf3_amp_flat_v4_loco_ppo_runner_cfg,
)


register_mjlab_task(
  task_id="BXI-ELF3-AMP-Rough-V4",
  env_cfg=elf3_amp_rough_v4_env_cfg(),
  play_env_cfg=elf3_amp_rough_v4_env_cfg(play=True),
  rl_cfg=elf3_amp_v4_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Rough-V4-Loco",
  env_cfg=elf3_amp_rough_v4_loco_env_cfg(),
  play_env_cfg=elf3_amp_rough_v4_loco_env_cfg(play=True),
  rl_cfg=elf3_amp_v4_loco_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Flat-V4",
  env_cfg=elf3_amp_flat_v4_env_cfg(),
  play_env_cfg=elf3_amp_flat_v4_env_cfg(play=True),
  rl_cfg=elf3_amp_flat_v4_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Flat-V4-Loco",
  env_cfg=elf3_amp_flat_v4_loco_env_cfg(),
  play_env_cfg=elf3_amp_flat_v4_loco_env_cfg(play=True),
  rl_cfg=elf3_amp_flat_v4_loco_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Flat-Loco",
  env_cfg=elf3_amp_flat_loco_env_cfg(),
  play_env_cfg=elf3_amp_flat_loco_env_cfg(play=True),
  rl_cfg=elf3_amp_loco_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Flat-V3",
  env_cfg=elf3_amp_flat_v3_env_cfg(),
  play_env_cfg=elf3_amp_flat_v3_env_cfg(play=True),
  rl_cfg=elf3_amp_v3_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Flat-V3-1",
  env_cfg=elf3_amp_flat_v3_1_env_cfg(),
  play_env_cfg=elf3_amp_flat_v3_1_env_cfg(play=True),
  rl_cfg=elf3_amp_v3_1_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)


register_mjlab_task(
  task_id="BXI-ELF3-AMP-Rough",
  env_cfg=elf3_amp_rough_env_cfg(),
  play_env_cfg=elf3_amp_rough_env_cfg(play=True),
  rl_cfg=elf3_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Flat",
  env_cfg=elf3_amp_flat_env_cfg(),
  play_env_cfg=elf3_amp_flat_env_cfg(play=True),
  rl_cfg=elf3_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="BXI-ELF3-AMP-Flat-V2",
  env_cfg=elf3_amp_flat_v2_env_cfg(),
  play_env_cfg=elf3_amp_flat_v2_env_cfg(play=True),
  rl_cfg=elf3_amp_v2_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
