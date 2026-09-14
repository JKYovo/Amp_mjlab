"""Physics-step position-target latency without changing ELF3's PD or policy IO."""

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.buffers import CircularBuffer


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
  """Sample one common joint-command lag per environment at each reset.

  Lags are in physics steps, not policy steps. With dt=0.005, 0..8 is
  0..40 ms. No observation, velocity-feedback or torque delay is added.
  """

  delay_min_lag: int = 0
  delay_max_lag: int = 8

  def __post_init__(self):
    super().__post_init__()
    if not 0 <= self.delay_min_lag <= self.delay_max_lag:
      raise ValueError("Expected 0 <= delay_min_lag <= delay_max_lag")

  def build(self, env):
    return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
  """Delay the complete 29-joint target coherently, once per physics substep."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._target_history = CircularBuffer(
      max_len=cfg.delay_max_lag + 1,
      batch_size=self.num_envs,
      device=self.device,
    )
    self._delay_lags = torch.randint(
      cfg.delay_min_lag, cfg.delay_max_lag + 1,
      (self.num_envs,), device=self.device,
    )

  @property
  def delay_lags(self):
    return self._delay_lags

  def apply_actions(self):
    # Keep the base action's scaling, nominal offset, encoder-bias correction
    # and actuator ordering. The policy's raw/previous actions stay generated
    # actions, not delayed targets (the existing deployment contract).
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    self._target_history.append(target)
    delayed = self._target_history[self._delay_lags]
    self._entity.set_joint_position_target(delayed, joint_ids=self._target_ids)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._target_history.reset(ids)
    self._delay_lags[ids] = torch.randint(
      self.cfg.delay_min_lag, self.cfg.delay_max_lag + 1,
      self._delay_lags[ids].shape, device=self.device,
    )
    # CircularBuffer backfills reset rows with their FIRST NEW target. Never
    # replay commands from the previous episode or send an all-zero pose.
