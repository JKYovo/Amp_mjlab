import json
import contextlib
import io
import math
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from scripts.build_elf3_amp_v4_dataset import SOURCE, OUTPUT, REMOVED, sha256
from scripts.check_elf3_v4 import expected_heightfield_count
from src.tasks.amp_loco.mdp.rough_height import (
    assign_interior_terrain_origins,
    root_clearance,
    root_height_below_terrain,
    root_outside_terrain_interior,
)
from src.tasks.amp_loco.mdp.tienkung_terrain import TienKungGravelTerrainCfg
from src.tasks.amp_loco.mdp.delayed_action import DelayedJointPositionActionCfg
from src.tasks.amp_loco.mdp.terrain import elf3_v4_rough_terrain_cfg
from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from rsl_rl.utils.motion_loader import AMPLoader


class V4Test(unittest.TestCase):
    def test_capacity_check_uses_gravel_strip_count(self):
        for task in ('BXI-ELF3-AMP-Rough-V4', 'BXI-ELF3-AMP-Rough-V4-Loco',
                     'BXI-ELF3-AMP-Rough-V4-Delay'):
            self.assertEqual(expected_heightfield_count(load_env_cfg(task).scene.terrain.terrain_generator),
                             8000)
            self.assertEqual(expected_heightfield_count(load_env_cfg(task, play=True).scene.terrain.terrain_generator),
                             1000)
        self.assertEqual(expected_heightfield_count(elf3_v4_rough_terrain_cfg()), 200)

    def test_v4_tasks_use_gravel_and_flat_entries_are_retired(self):
        tasks = [task for task in list_tasks() if task.startswith('BXI-ELF3-AMP-')
                 and '-V4' in task]
        self.assertEqual(set(tasks), {'BXI-ELF3-AMP-Rough-V4',
                                     'BXI-ELF3-AMP-Rough-V4-Loco',
                                     'BXI-ELF3-AMP-Rough-V4-Delay'})
        for task in tasks:
            self.assertFalse(load_rl_cfg(task).resume)
            self.assertEqual(load_rl_cfg(task).algorithm.learning_rate, .001)
            for play in (False, True):
                cfg = load_env_cfg(task, play=play)
                self.assertEqual(cfg.scene.terrain.terrain_type, 'generator')
                self.assertIs(type(cfg.scene.terrain.terrain_generator.sub_terrains['random_rough']),
                              TienKungGravelTerrainCfg)
        self.assertEqual(load_env_cfg('BXI-ELF3-AMP-Flat-V3-1').scene.terrain.terrain_type,
                         'plane')

    def test_v4_loco_config_and_immediate_termination(self):
        for play in (False, True):
            full = load_env_cfg('BXI-ELF3-AMP-Rough-V4', play=play)
            cfg = load_env_cfg('BXI-ELF3-AMP-Rough-V4-Loco', play=play)
            init = cfg.events['init_motion_loader']
            self.assertIsNone(init.params['recovery_dir'])
            self.assertEqual(init.params['delay_reset_env_ratio'], 0.)
            self.assertEqual(init.params['max_delay_steps'], 0)
            for attr in ('sim', 'commands', 'rewards', 'terminations', 'observations', 'curriculum'):
                self.assertEqual(getattr(cfg, attr), getattr(full, attr))
            self.assertEqual(cfg.events['base_com'], full.events['base_com'])
            for attr in ('terrain_type', 'terrain_generator', 'max_init_terrain_level'):
                self.assertEqual(getattr(cfg.scene.terrain, attr), getattr(full.scene.terrain, attr))
            sentinel = object()
            env = SimpleNamespace(num_envs=16, termination_manager=sentinel)
            with patch('src.tasks.amp_loco.mdp.events.MotionResetManager.get') as manager:
                init.func(env, None, **init.params)
                self.assertIsNone(manager.return_value.init.call_args.kwargs['recovery_dir'])
            self.assertIs(env.termination_manager, sentinel)
            self.assertEqual(full.events['init_motion_loader'].params['delay_reset_env_ratio'], 0.)
        runner = load_rl_cfg('BXI-ELF3-AMP-Rough-V4-Loco')
        full_runner = load_rl_cfg('BXI-ELF3-AMP-Rough-V4')
        self.assertEqual(runner.amp_motion_files, str(OUTPUT/'WalkandRun'))
        self.assertEqual(runner.experiment_name, 'elf3_amp_locomotion_v4_loco')
        self.assertFalse(runner.resume)
        self.assertEqual(runner.max_iterations, 200001)
        self.assertEqual(runner.algorithm, full_runner.algorithm)
        self.assertEqual(runner.actor, full_runner.actor)
        self.assertEqual(runner.critic, full_runner.critic)
        self.assertEqual(runner.num_steps_per_env, full_runner.num_steps_per_env)
        self.assertEqual(full_runner.amp_motion_files, str(OUTPUT/'WalkandRun'))

    def test_v4_loco_actual_cpu_motion_loaders_exclude_recovery(self):
        runner = load_rl_cfg('BXI-ELF3-AMP-Rough-V4-Loco')
        normal_paths = sorted((OUTPUT/'WalkandRun').glob('*.npz'))
        with np.load(normal_paths[0], allow_pickle=False) as data:
            body_names = data['body_names'].tolist()
        reset = MotionLoader(str(OUTPUT/'WalkandRun'), [], 0, 0, device='cpu')
        self.assertEqual(len(reset.motion_data), 15)
        self.assertEqual(reset.motion_data_recovery, [])
        self.assertEqual(sum(m['dof_pos'].shape[0] for m in reset.motion_data), 5089)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            expert = AMPLoader(runner.amp_motion_files, runner.amp_body_names,
                               runner.amp_anchor_name, body_names, device='cpu')
        self.assertEqual(set(expert.motion_names), {p.stem for p in normal_paths})
        current, following = next(expert.feed_forward_generator(1, 1024))
        self.assertTrue(torch.isfinite(current).all() and torch.isfinite(following).all())
        self.assertEqual(current.shape[1], expert.observation_dim)
        # The original dataset still includes its recovery clip.
        self.assertEqual(len(list(OUTPUT.rglob('*.npz'))), 16)

    def test_config_and_fresh_defaults(self):
        v3 = load_env_cfg('BXI-ELF3-AMP-Flat-V3-1')
        for play in (False, True):
            cfg = load_env_cfg('BXI-ELF3-AMP-Rough-V4', play=play)
            command = cfg.commands['twist']
            self.assertEqual(command.ranges.lin_vel_x, (-0.6, 1.0))
            self.assertEqual(command.ranges.lin_vel_y, (-1.0, 1.0))
            self.assertEqual(command.ranges.ang_vel_z, (-1.0, 1.0))
            self.assertEqual(command.ranges.heading, (-math.pi / 2, math.pi / 2))
            self.assertEqual(command.turning_max_abs_ang_vel, 2.0)
            self.assertEqual(command.rel_turning_envs, .15)
            self.assertEqual(cfg.curriculum, {})
            self.assertEqual(cfg.scene.terrain.terrain_type, 'generator')
            self.assertEqual(cfg.sim.mujoco.ccd_iterations, 50)
            self.assertEqual(cfg.sim.nconmax, 128)
            self.assertEqual(cfg.sim.njmax, 768)
            terrain = cfg.scene.terrain.terrain_generator
            self.assertFalse(terrain.curriculum)
            self.assertEqual(set(terrain.sub_terrains), {'random_rough'})
            gravel = terrain.sub_terrains['random_rough']
            self.assertIs(type(gravel), TienKungGravelTerrainCfg)
            self.assertEqual(gravel.proportion, .2)
            self.assertEqual(gravel.noise_range, (-.02, .04))
            self.assertEqual(gravel.noise_step, .02)
            self.assertEqual(gravel.horizontal_scale, .1)
            self.assertEqual(gravel.vertical_scale, .005)
            self.assertEqual(gravel.strip_cells, 2)
            self.assertEqual(gravel.border_width, .25)
            self.assertEqual(cfg.sim.contact_sensor_maxmatch, 1024)
            self.assertEqual(cfg.events['init_motion_loader'].params['max_delay_steps'], 0)
            self.assertEqual(terrain.num_rows, 5 if play else 10)
            self.assertEqual(terrain.num_cols, 5 if play else 20)
            self.assertNotIn('terrain_scan', [s.name for s in cfg.scene.sensors])
            for group in ('actor', 'critic', 'amp'):
                self.assertEqual(list(cfg.observations[group].terms), list(v3.observations[group].terms))
                self.assertNotIn('height_scan', cfg.observations[group].terms)
            self.assertEqual(cfg.observations['actor'].history_length, 4)
            self.assertEqual(cfg.events['base_com'].params['ranges'],
                             {0: (-.025, .025), 1: (-.05, .05), 2: (-.05, .05)})
            self.assertEqual(cfg.events['init_motion_loader'].params['delay_reset_env_ratio'], 0.)
            self.assertIsNone(cfg.events['init_motion_loader'].params['recovery_dir'])
            action = cfg.actions['joint_pos']
            self.assertIs(type(action), DelayedJointPositionActionCfg)
            self.assertEqual(action.delay_min_lag, 0)
            self.assertEqual(action.delay_max_lag, 8)
            self.assertEqual(Path(cfg.events['init_motion_loader'].params['motion_dir']), OUTPUT/'WalkandRun')
            self.assertEqual(cfg.events['reset_from_motion'].params['motion_dir'], str(OUTPUT/'WalkandRun'))
            interior = cfg.events['interior_terrain_origins']
            margin = 1 if play else 2
            self.assertEqual(interior.mode, 'startup')
            self.assertEqual(interior.params, {'margin_rows': margin, 'margin_cols': margin})
            self.assertNotIn('randomize_terrain', cfg.events)
            boundary = cfg.terminations['terrain_outer_boundary']
            self.assertTrue(boundary.time_out)
            half_extent = 20.0 if play else 40.0
            self.assertEqual(boundary.params['half_extent_x'], half_extent)
            self.assertEqual(boundary.params['half_extent_y'], half_extent if play else 80.0)
            self.assertEqual(boundary.params['safety_margin'], 1.0)
            for name, reward in cfg.rewards.items():
                if name in ('track_anchor_linear_velocity', 'track_anchor_angular_velocity',
                            'feet_safe_distance', 'legs_stand_pose', 'feet_stumble', 'vertical_velocity'):
                    continue
                self.assertEqual(reward.weight, v3.rewards[name].weight)
                self.assertEqual(reward.params, v3.rewards[name].params)
        runner = load_rl_cfg('BXI-ELF3-AMP-Rough-V4')
        self.assertFalse(runner.resume)
        self.assertEqual(runner.max_iterations, 200001)
        self.assertEqual(runner.algorithm.learning_rate, .001)
        self.assertEqual(runner.amp_motion_files, str(OUTPUT/'WalkandRun'))
        self.assertEqual(runner.experiment_name, 'elf3_amp_locomotion_v4')
        self.assertEqual(runner.run_name, 'v4_gravel_walk_delay_fresh')
        # V4 must not mutate the registered V3.1 configurations.
        self.assertEqual(v3.commands['twist'].turning_max_abs_ang_vel, 2.)
        self.assertEqual(v3.scene.terrain.terrain_type, 'plane')

    def test_dataset_subset_and_hashes(self):
        manifest = json.loads((OUTPUT/'manifest.json').read_text())
        actual = {p.relative_to(OUTPUT).as_posix() for p in OUTPUT.rglob('*.npz')}
        self.assertEqual(len(actual), 16)
        self.assertEqual(manifest['kept_frames'], 7664)
        for entry in manifest['entries']:
            relative = entry['file']
            self.assertEqual(entry['kept'], Path(relative).name not in REMOVED)
            self.assertEqual(sha256(SOURCE/relative), entry['sha256'])
            self.assertEqual(relative in actual, entry['kept'])
            if entry['kept']:
                self.assertEqual(sha256(OUTPUT/relative), entry['sha256'])

    def test_native_rough_single_heightfield_geometry(self):
        cfg = elf3_v4_rough_terrain_cfg().sub_terrains['random_rough']
        cfg.size = (8., 8.)  # TerrainGenerator supplies the parent tile size.
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name='terrain')
        output = cfg.function(.5, spec, np.random.default_rng(42))
        model = spec.compile()
        self.assertEqual(len(output.geometries), 1)
        self.assertEqual(model.nhfield, 1)
        self.assertEqual(model.hfield_nrow[0], 40)
        self.assertEqual(model.hfield_ncol[0], 40)
        self.assertEqual(model.geom_type[0], mujoco.mjtGeom.mjGEOM_HFIELD)
        heights = model.hfield_data * model.hfield_size[0, 2] + model.geom_pos[0, 2]
        np.testing.assert_allclose(np.unique(heights), [0., .02, .04, .06], atol=1e-8)
        self.assertAlmostEqual(output.origin[2], .03)

    def test_legacy_gravel_geometry_signed_heights_and_origin(self):
        cfg = TienKungGravelTerrainCfg(size=(8., 8.), proportion=.2)
        grid = cfg.height_grid(np.random.default_rng(42))
        self.assertEqual(grid.shape, (81, 81))
        np.testing.assert_allclose(np.unique(grid), [-.02, 0, .02, .04])
        self.assertTrue(np.all(grid[:3] == 0) and np.all(grid[-3:] == 0))
        self.assertTrue(np.all(grid[:, :3] == 0) and np.all(grid[:, -3:] == 0))
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name='terrain')
        output = cfg.function(.5, spec, np.random.default_rng(42))
        model = spec.compile()
        strips = []
        for i, geom in enumerate(output.geometries):
            # Check the compiled data: MuJoCo renormalizes each hfield,
            # so validating only MjSpec.userdata misses constant-border bugs.
            offset = model.hfield_adr[i]
            strip = model.hfield_data[offset:offset + 81 * 3].reshape(81, 3) * geom.hfield.size[2] + geom.geom.pos[2]
            strips.append(strip if i == 0 else strip[:, 1:])
        restored = np.concatenate(strips, axis=1)
        np.testing.assert_allclose(restored, grid, atol=1e-8)
        self.assertEqual(model.nhfield, 40)
        self.assertAlmostEqual(output.origin[2], .04)

    def test_terrain_relative_height_and_missing_ray_fallback(self):
        class Scene(dict):
            env_origins = torch.tensor([[0., 0., 2.], [0., 0., -1.]])
        scene = Scene()
        scene['robot'] = SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([[0., 0., 3.], [0., 0., -.5]])))
        scene['terrain_height'] = SimpleNamespace(data=SimpleNamespace(
            hit_pos_w=torch.tensor([[[0., 0., 2.], [0., 0., 2.]],
                                    [[0., 0., float('nan')], [0., 0., 100.]]]),
            distances=torch.tensor([[1., 1.], [-1., -1.]])))
        env = SimpleNamespace(scene=scene)
        torch.testing.assert_close(root_clearance(env), torch.tensor([1., .5]))
        self.assertEqual(root_height_below_terrain(env, .62).tolist(), [False, True])

    def test_interior_origins_and_outer_boundary_timeout(self):
        class Scene(dict):
            terrain = None

        origins = torch.zeros(10, 20, 3)
        origins[..., 0] = torch.arange(10).view(-1, 1) * 8 - 36
        origins[..., 1] = torch.arange(20).view(1, -1) * 8 - 76
        scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.tensor([
                [0., 0., 1.], [38.9, 0., 1.], [39., 0., 1.],
                [0., -79.1, 1.], [float('nan'), 0., 1.],
            ])
        )))
        scene.terrain = SimpleNamespace(
            terrain_origins=origins,
            terrain_levels=torch.zeros(4096, dtype=torch.long),
            terrain_types=torch.zeros(4096, dtype=torch.long),
            env_origins=torch.zeros(4096, 3),
        )
        env = SimpleNamespace(scene=scene, num_envs=4096, device='cpu')
        torch.manual_seed(42)
        assign_interior_terrain_origins(env, margin_rows=2, margin_cols=2)
        self.assertTrue(torch.all((scene.terrain.terrain_levels >= 2)
                                  & (scene.terrain.terrain_levels < 8)))
        self.assertTrue(torch.all((scene.terrain.terrain_types >= 2)
                                  & (scene.terrain.terrain_types < 18)))
        expected = origins[scene.terrain.terrain_levels, scene.terrain.terrain_types]
        torch.testing.assert_close(scene.terrain.env_origins, expected)
        result = root_outside_terrain_interior(
            env, half_extent_x=40., half_extent_y=80., safety_margin=1.)
        self.assertEqual(result.tolist(), [False, False, True, True, True])


if __name__ == '__main__':
    unittest.main()
