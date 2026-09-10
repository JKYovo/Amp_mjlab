"""AMP-specific velocity command generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class TurningVelocityCommand(UniformVelocityCommand):
  """Uniform velocity commands with a dedicated in-place turning subset."""

  cfg: "TurningVelocityCommandCfg"

  def __init__(self, cfg: "TurningVelocityCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.is_turning_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _update_metrics(self) -> None:
    super()._update_metrics()
    turning_fraction = self.is_turning_env.float().mean()
    turning_count = self.is_turning_env.sum()
    turning_yaw_error = torch.sum(
      torch.abs(
        self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]
      )
      * self.is_turning_env
    ) / torch.clamp(turning_count, min=1)
    self._env.extras["log"]["Metrics/command_turning_fraction"] = turning_fraction
    self._env.extras["log"]["Metrics/turning_error_vel_yaw"] = turning_yaw_error

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Preserve upstream sampling behavior, including heading targets and the
    # optional initial-velocity assignment, before selecting command modes.
    super()._resample_command(env_ids)

    # These are absolute, mutually exclusive fractions of all environments.
    selector = torch.rand(len(env_ids), device=self.device)
    standing = selector < self.cfg.rel_standing_envs
    turning = (
      (selector >= self.cfg.rel_standing_envs)
      & (
        selector
        < self.cfg.rel_standing_envs + self.cfg.rel_turning_envs
      )
    )

    self.is_standing_env[env_ids] = standing
    self.is_turning_env[env_ids] = turning

    # Heading control would overwrite yaw every simulation step. Dedicated
    # standing and turning environments therefore use direct rate commands.
    self.is_heading_env[env_ids[standing | turning]] = False

    turning_ids = env_ids[turning]
    if len(turning_ids) > 0:
      self.vel_command_b[turning_ids, :2] = 0.0

      # Sample left and right equally while excluding ineffective near-zero yaw.
      direction = torch.where(
        torch.rand(len(turning_ids), device=self.device) < 0.5,
        -torch.ones(len(turning_ids), device=self.device),
        torch.ones(len(turning_ids), device=self.device),
      )
      magnitude = torch.empty(len(turning_ids), device=self.device).uniform_(
        self.cfg.turning_min_abs_ang_vel,
        self.cfg.turning_max_abs_ang_vel,
      )
      self.vel_command_b[turning_ids, 2] = direction * magnitude


@dataclass(kw_only=True)
class TurningVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for explicit in-place turning command sampling."""

  rel_turning_envs: float = 0.0
  turning_min_abs_ang_vel: float = 0.3
  turning_max_abs_ang_vel: float = 1.0

  def build(self, env: "ManagerBasedRlEnv") -> TurningVelocityCommand:
    return TurningVelocityCommand(self, env)

  def __post_init__(self) -> None:
    super().__post_init__()
    if not 0.0 <= self.rel_turning_envs <= 1.0:
      raise ValueError("rel_turning_envs must be in [0, 1]")
    if self.rel_standing_envs + self.rel_turning_envs > 1.0:
      raise ValueError("standing and turning command fractions must sum to at most 1")
    if self.turning_min_abs_ang_vel <= 0.0:
      raise ValueError("turning_min_abs_ang_vel must be positive")
    if self.turning_max_abs_ang_vel < self.turning_min_abs_ang_vel:
      raise ValueError(
        "turning_max_abs_ang_vel must be at least turning_min_abs_ang_vel"
      )
