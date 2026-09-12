"""CPU behavior tests for V4 tracking, geometric safety and startup sequencing."""
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import src.tasks  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from src.tasks.amp_loco.mdp.v4_command import StartupVelocityCommand
from src.tasks.amp_loco.mdp.v4_events import reset_with_startup_v4, push_except_startup_preparation_v4
from src.tasks.amp_loco.mdp.v4_rewards import (
    command_scaled_tracking, foot_box_separation, feet_safe_distance_v4,
    legs_stand_pose_v4, feet_stumble_v4, track_horizontal_velocity_v4, track_yaw_velocity_v4,
    vertical_velocity_v4,
)


def mock_env(n=4):
    zero = torch.zeros(n, 3)
    quat = torch.tensor([1., 0., 0., 0.]).expand(n, 4).clone()
    data = SimpleNamespace(
        root_link_lin_vel_w=zero.clone(), root_link_lin_vel_b=zero.clone(),
        root_link_ang_vel_w=zero.clone(), root_link_ang_vel_b=zero.clone(),
        projected_gravity_b=torch.tensor([0., 0., -1.]).expand(n, 3).clone(),
        heading_w=torch.zeros(n),
        body_link_quat_w=quat[:, None].clone(),
        body_link_lin_vel_w=zero[:, None].clone(), body_link_ang_vel_w=zero[:, None].clone(),
        joint_pos=torch.ones(n, 12) * .2, default_joint_pos=torch.zeros(n, 12),
        default_joint_vel=torch.zeros(n, 12),
        default_root_state=torch.cat((torch.tensor([0., 0., 1.05]).expand(n, 3), quat, torch.zeros(n, 6)), dim=-1),
    )
    robot = SimpleNamespace(data=data, write_root_state_to_sim=Mock(), write_joint_state_to_sim=Mock())
    class Scene(dict):
        pass
    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(n, 3)
    force = torch.tensor([[0., 0., 210.], [0., 0., 210.]]).expand(n, 2, 3).clone()
    scene['feet_ground_contact'] = SimpleNamespace(data=SimpleNamespace(force=force))
    scene['feet_self_contact'] = SimpleNamespace(data=SimpleNamespace(force_history=torch.zeros(n, 4, 4, 3)))
    env = SimpleNamespace(scene=scene, num_envs=n, device='cpu', step_dt=.02,
                          extras={'log': {}}, termination_manager=SimpleNamespace())
    return env


def mock_command(env, **overrides):
    cfg = load_env_cfg('BXI-ELF3-AMP-Rough-V4-Loco').commands['twist']
    for name, value in overrides.items():
        setattr(cfg, name, value)
    cfg.__post_init__()
    command = StartupVelocityCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command, get_command=lambda _: command.command)
    return command


class TrackingTest(unittest.TestCase):
    def test_signed_commands_and_error_scale(self):
        c = torch.tensor([[0., .2], [0., .5], [0., 1.], [0., -.5], [0., -1.]])
        perfect = command_scaled_tracking(c, c, (.25, .20), (.35, .35))
        idle = command_scaled_tracking(c, torch.zeros_like(c), (.25, .20), (.35, .35))
        wrong = command_scaled_tracking(c, -c, (.25, .20), (.35, .35))
        torch.testing.assert_close(perfect, torch.ones(5))
        self.assertTrue(torch.all(wrong < idle))
        torch.testing.assert_close(idle[1:3], idle[3:])
        self.assertLess(idle[1], .2)  # Old std=1 paid .779 for ignoring vy=.5.
        yaw = torch.tensor([[.5], [1.], [2.], [-2.]])
        r = command_scaled_tracking(yaw, torch.zeros_like(yaw), .35, .40)
        self.assertTrue(torch.all((r > .01) & (r < .5)))
        torch.testing.assert_close(r[2], r[3])

    def test_tracking_decouples_vertical_and_roll_pitch_and_is_yaw_invariant(self):
        env = mock_env(1)
        command = mock_command(env)
        command.vel_command_b[:] = torch.tensor([[0., .5, 1.]])
        data = env.scene['robot'].data
        data.body_link_lin_vel_w[:] = torch.tensor([[[0., .5, 2.]]])
        data.body_link_ang_vel_w[:] = torch.tensor([[[3., -2., 1.]]])
        anchor = SceneEntityCfg('robot', body_ids=[0])
        torch.testing.assert_close(track_horizontal_velocity_v4(env, 'twist', anchor), torch.ones(1))
        torch.testing.assert_close(track_yaw_velocity_v4(env, 'twist', anchor), torch.ones(1))
        torch.testing.assert_close(vertical_velocity_v4(env, anchor), torch.tensor([4.]))
        data.body_link_quat_w[:] = torch.tensor([[[math.sqrt(.5), 0., 0., math.sqrt(.5)]]])
        data.body_link_lin_vel_w[:] = torch.tensor([[[-.5, 0., 2.]]])
        torch.testing.assert_close(track_horizontal_velocity_v4(env, 'twist', anchor), torch.ones(1))

    def test_stand_pose_released_for_any_nonzero_axis_and_recovery(self):
        env = mock_env(5)
        command = mock_command(env)
        command.vel_command_b[:] = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1], [0., 0., 0.]])
        env.termination_manager._delay_env_mask = torch.tensor([False, False, False, False, True])
        env.termination_manager._delay_counters = torch.tensor([0, 0, 0, 0, 1])
        result = legs_stand_pose_v4(env, SceneEntityCfg('robot', joint_ids=list(range(12))))
        torch.testing.assert_close(result, torch.tensor([.2, 0., 0., 0., 0.]))


class FeetTest(unittest.TestCase):
    def test_lateral_amp_references_are_not_penalized(self):
        root = Path(__file__).resolve().parents[1] / 'src/assets/motions/elf3/amp_v4/WalkandRun'
        paths = sorted(root.glob('*side_step*.npz'))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                names = data['body_names'].tolist()
                ids = [names.index(name) for name in ('l_ankle_x_link', 'r_ankle_x_link')]
                gap = foot_box_separation(torch.tensor(data['body_pos_w'][:, ids]),
                                          torch.tensor(data['body_quat_w'][:, ids]),
                                          (.03, 0., -.01375), (.12, .04, .02725))
                self.assertTrue(torch.all(gap > .02), path.name)

    def boxes(self):
        p = torch.zeros(5, 2, 3)
        p[:, 0, 1] = torch.tensor([.27, .15, .09, .03, .03])
        p[3, 0, 2] = .15  # A safely raised foot is not penalized for XY overlap.
        p[4, 0, 0] = .4   # Nor are feet sufficiently separated fore/aft.
        q = torch.tensor([1., 0., 0., 0.]).expand(5, 2, 4).clone()
        return p, q

    def test_box_clearance_not_fixed_27cm_and_yaw_invariance(self):
        p, q = self.boxes()
        args = ((.03, 0., -.01375), (.12, .04, .02725))
        gaps = foot_box_separation(p, q, *args)
        self.assertTrue(torch.all(gaps[[0, 1, 3, 4]] > .02))
        self.assertAlmostEqual(gaps[2].item(), .01, places=6)
        rotation = torch.tensor([math.cos(.7), 0., 0., math.sin(.7)]).expand_as(q)
        transformed = foot_box_separation(quat_apply(rotation, p), quat_mul(rotation, q), *args)
        torch.testing.assert_close(transformed, gaps)
        swapped = foot_box_separation(p.flip(1), q.flip(1), *args)
        torch.testing.assert_close(swapped, gaps)

    def test_rotated_toe_proximity_and_startup_emphasis(self):
        env = mock_env(2)
        cmd = mock_command(env)
        p = torch.zeros(2, 2, 3)
        p[:, 0, 1] = .15
        q = torch.tensor([1., 0., 0., 0.]).expand(2, 2, 4).clone()
        # Turn the left toe toward the right foot, not away from it.
        q[:, 0] = torch.tensor([math.sqrt(.5), 0., 0., -math.sqrt(.5)])
        env.scene['robot'].data.body_link_pos_w = p
        env.scene['robot'].data.body_link_quat_w = q
        cmd.startup_phase[1] = 2
        value = feet_safe_distance_v4(env, SceneEntityCfg('robot', body_ids=[0, 1]))
        self.assertGreater(value[0], 0.)
        torch.testing.assert_close(value[1], value[0] * 1.5)
        cmd.startup_motion_time[1] = 3.
        value2 = feet_safe_distance_v4(env, SceneEntityCfg('robot', body_ids=[0, 1]))
        torch.testing.assert_close(value2[0], value2[1])

    def test_stumble_rejects_force_noise_and_normal_load(self):
        env = mock_env(3)
        f = env.scene['feet_ground_contact'].data.force
        f.zero_()
        f[0, 0] = torch.tensor([.1, 0., 0.])
        f[1, 0] = torch.tensor([100., 0., 400.])
        f[2, 0] = torch.tensor([40., 0., 2.])
        torch.testing.assert_close(feet_stumble_v4(env), torch.tensor([0., 0., 1.]))


class StartupTest(unittest.TestCase):
    def test_modes_and_left_right_command_envelopes(self):
        torch.manual_seed(42)
        env = mock_env(20000)
        cmd = mock_command(env)
        cmd.reset(torch.arange(env.num_envs))
        self.assertTrue(torch.all(cmd.command[cmd.is_standing_env] == 0.))
        for mask, expected in ((cmd.is_standing_env, .05), (cmd.is_turning_env, .15), (cmd.is_lateral_env, .20)):
            self.assertAlmostEqual(mask.float().mean().item(), expected, delta=.01)
        lateral = cmd.command[cmd.is_lateral_env]
        self.assertTrue(torch.all(lateral[:, [0, 2]] == 0.))
        self.assertTrue(torch.all((lateral[:, 1].abs() >= .2) & (lateral[:, 1].abs() <= 1.)))
        self.assertAlmostEqual((lateral[:, 1] > 0).float().mean().item(), .5, delta=.03)
        self.assertFalse(cmd.is_heading_env[cmd.is_lateral_env].any())
        turning = cmd.command[cmd.is_turning_env]
        self.assertTrue(torch.all(turning[:, :2] == 0.))
        self.assertTrue(torch.all((turning[:, 2].abs() >= .3) & (turning[:, 2].abs() <= 2.)))

    def test_wait_stable_launch_and_resume_normal_commands(self):
        env = mock_env(4)
        cmd = mock_command(env, startup_fraction=1., startup_wait_range=(1., 1.), startup_move_duration=.5)
        ids = torch.arange(4)
        cmd.prepare_reset(ids, torch.ones(4, dtype=torch.bool))
        cmd.reset(ids)
        self.assertTrue(torch.all(cmd.command == 0.))
        env.scene['feet_ground_contact'].data.force[1, 0] = 0.  # Not double support.
        env.scene['robot'].data.root_link_lin_vel_w[2, 0] = .5
        for _ in range(40):
            cmd.compute(.02)
        self.assertTrue(torch.all(cmd.command == 0.))
        for _ in range(20):
            cmd.compute(.02)
        self.assertEqual(cmd.startup_phase.tolist(), [2, 1, 1, 2])
        torch.testing.assert_close(cmd.command[[0, 3]], cmd.startup_target[[0, 3]])
        self.assertFalse(cmd.is_heading_env.any())
        for _ in range(110):
            cmd.compute(.02)
        self.assertTrue(torch.all(cmd.startup_phase == 0))
        self.assertEqual(cmd.metrics['startup_timeouts'].tolist(), [0., 1., 1., 0.])
        self.assertEqual(cmd.metrics['startup_launches'].tolist(), [1., 0., 0., 1.])
        self.assertTrue(torch.all(cmd.time_left < 9.))

    def test_reset_preserves_recovery_and_clears_only_reset_environments(self):
        env = mock_env(4)
        cmd = mock_command(env, startup_fraction=1.)
        env.termination_manager._delay_env_mask = torch.tensor([False, True, False, True])
        env.scene.env_origins[:, 2] = 2.
        with patch('src.tasks.amp_loco.mdp.v4_events.reset_from_motion_data') as original:
            reset_with_startup_v4(env, torch.arange(4), 'motion', SceneEntityCfg('robot'))
            original.assert_called_once()
        self.assertEqual(cmd.startup_phase.tolist(), [1, 0, 1, 0])
        state, ids = env.scene['robot'].write_root_state_to_sim.call_args.args
        self.assertEqual(ids.tolist(), [0, 2])
        torch.testing.assert_close(state[:, 2], torch.tensor([3.05, 3.05]))
        self.assertTrue(torch.all(state[:, 7:] == 0.))
        cmd.startup_phase[0] = 2
        cmd.startup_motion_time[0] = .7
        cmd.prepare_reset(torch.tensor([2]), torch.tensor([False]))
        self.assertEqual(cmd.startup_phase.tolist(), [2, 0, 0, 0])
        self.assertAlmostEqual(cmd.startup_motion_time[0].item(), .7, places=6)

    def test_push_only_exempts_preparation_and_play_does_not_force_startup(self):
        env = mock_env(4)
        cmd = mock_command(env, startup_fraction=0.)
        self.assertEqual(len(cmd.prepare_reset(torch.arange(4), torch.ones(4, dtype=torch.bool))), 0)
        cmd.startup_phase[:] = torch.tensor([0, 1, 2, 0])
        with patch('src.tasks.amp_loco.mdp.v4_events.envs_mdp.push_by_setting_velocity') as push:
            push_except_startup_preparation_v4(env, None, {'x': (-1., 1.)})
            self.assertEqual(push.call_args.args[1].tolist(), [0, 2, 3])

    def test_config_isolation_and_v4_only_reward_overrides(self):
        v3 = load_env_cfg('BXI-ELF3-AMP-Flat-V3-1')
        for task in ('BXI-ELF3-AMP-Rough-V4', 'BXI-ELF3-AMP-Rough-V4-Loco'):
            cfg = load_env_cfg(task)
            self.assertEqual(cfg.commands['twist'].startup_fraction, .25)
            self.assertEqual(load_env_cfg(task, play=True).commands['twist'].startup_fraction, 0.)
            self.assertEqual(cfg.observations, v3.observations)
            self.assertIs(cfg.events['reset_from_motion'].func, reset_with_startup_v4)
            self.assertIs(cfg.rewards['track_anchor_linear_velocity'].func, track_horizontal_velocity_v4)
            self.assertIs(cfg.rewards['track_anchor_angular_velocity'].func, track_yaw_velocity_v4)
            for name in ('feet_safe_distance', 'legs_stand_pose', 'feet_stumble'):
                self.assertIn(name, cfg.rewards)
                self.assertNotIn(name, v3.rewards)
        self.assertEqual(v3.rewards['track_anchor_linear_velocity'].params['std'], 1.)
        self.assertEqual(v3.rewards['track_anchor_angular_velocity'].params['turning_std'], 2.)


if __name__ == '__main__':
    unittest.main()
