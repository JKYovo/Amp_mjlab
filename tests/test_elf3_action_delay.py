"""CPU tests for latency units, episode isolation and the unchanged IO contract."""

from dataclasses import asdict, fields
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
import yaml

import src.tasks  # noqa: F401
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.assets.robots.elf3.elf3_constants import ELF3_JOINT_NAMES
from src.tasks.amp_loco.mdp.delayed_action import DelayedJointPositionActionCfg


def mock_env(n=3):
  data = SimpleNamespace(
    default_joint_pos=torch.arange(29).expand(n, 29).float() * .01,
    encoder_bias=torch.full((n, 29), .005),
  )
  robot = SimpleNamespace(
    data=data,
    find_joints_by_actuator_names=lambda _: (list(range(29)), list(ELF3_JOINT_NAMES)),
    set_joint_position_target=Mock(),
  )
  return SimpleNamespace(num_envs=n, device='cpu', scene={'robot': robot})


class ActionDelayTest(unittest.TestCase):
  def test_only_action_config_changes(self):
    for play in (False, True):
      base = load_env_cfg('BXI-ELF3-AMP-Rough-V4', play=play)
      delayed = load_env_cfg('BXI-ELF3-AMP-Rough-V4-Delay', play=play)
      for f in fields(base):
        if f.name == 'scene':
          # Terrain constructs a fresh empty-spec callable. Compare its
          # serialized configuration, not that callable's object identity.
          self.assertEqual(yaml.dump(asdict(delayed.scene)), yaml.dump(asdict(base.scene)))
        elif f.name != 'actions':
          self.assertEqual(getattr(delayed, f.name), getattr(base, f.name), f.name)
      action = delayed.actions['joint_pos']
      for f in fields(base.actions['joint_pos']):
        self.assertEqual(getattr(action, f.name), getattr(base.actions['joint_pos'], f.name))
      self.assertEqual((action.delay_min_lag, action.delay_max_lag), (0, 8))
      self.assertAlmostEqual(action.delay_max_lag * delayed.sim.mujoco.timestep, .04)
      self.assertEqual(delayed.decimation, 4)
    self.assertEqual(load_rl_cfg('BXI-ELF3-AMP-Rough-V4-Delay'),
                     load_rl_cfg('BXI-ELF3-AMP-Rough-V4'))

  def test_zero_lag_exactly_matches_base_action(self):
    env = mock_env()
    cfg = DelayedJointPositionActionCfg(
      entity_name='robot', actuator_names=('.*',), scale=.23, delay_max_lag=0)
    delayed = cfg.build(env)
    base = JointPositionAction(cfg, env)
    for _ in range(5):
      actions = torch.randn(3, 29)
      base.process_actions(actions)
      delayed.process_actions(actions)
      for _ in range(4):
        base.apply_actions()
        expected = env.scene['robot'].set_joint_position_target.call_args.args[0].clone()
        delayed.apply_actions()
        actual = env.scene['robot'].set_joint_position_target.call_args.args[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
      torch.testing.assert_close(delayed.raw_action, actions)

  def test_physics_substep_delay_shared_across_joints_and_held(self):
    env = mock_env()
    cfg = DelayedJointPositionActionCfg(
      entity_name='robot', actuator_names=('.*',), scale=1.)
    term = cfg.build(env)
    term.delay_lags[:] = torch.tensor([0, 1, 8])
    initial_lags = term.delay_lags.clone()
    targets = []
    for physics_step in range(28):
      # A new policy command only every four 5-ms physics steps.
      if physics_step % 4 == 0:
        term.process_actions(torch.full((3, 29), float(physics_step // 4)))
      targets.append(term._processed_actions.clone() - env.scene['robot'].data.encoder_bias)
      term.apply_actions()
      output = env.scene['robot'].set_joint_position_target.call_args.args[0]
      expected = torch.stack([targets[max(0, physics_step - lag)][i]
                              for i, lag in enumerate((0, 1, 8))])
      torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(term.delay_lags, initial_lags)

  def test_reset_never_replays_previous_episode_targets(self):
    env = mock_env()
    term = DelayedJointPositionActionCfg(
      entity_name='robot', actuator_names=('.*',), delay_min_lag=8).build(env)
    for _ in range(12):
      term.process_actions(torch.ones(3, 29))
      term.apply_actions()
    term.reset(torch.tensor([1]))
    term.process_actions(torch.full((3, 29), 5.))
    term.apply_actions()
    output = env.scene['robot'].set_joint_position_target.call_args.args[0]
    nominal = env.scene['robot'].data.default_joint_pos
    bias = env.scene['robot'].data.encoder_bias
    torch.testing.assert_close(output[1], nominal[1] + 5. - bias[1])
    torch.testing.assert_close(output[0], nominal[0] + 1. - bias[0])
    term.reset(slice(None))
    term.process_actions(torch.full((3, 29), 7.))
    term.apply_actions()
    output = env.scene['robot'].set_joint_position_target.call_args.args[0]
    torch.testing.assert_close(output, nominal + 7. - bias)

  def test_reset_sampling_includes_zero_and_max_lag(self):
    term = DelayedJointPositionActionCfg(
      entity_name='robot', actuator_names=('.*',)).build(mock_env(256))
    for _ in range(3):
      term.reset()
      self.assertEqual(set(term.delay_lags.tolist()), set(range(9)))
    with self.assertRaises(ValueError):
      DelayedJointPositionActionCfg(
        entity_name='robot', actuator_names=('.*',), delay_min_lag=-1)


if __name__ == '__main__':
  unittest.main()
