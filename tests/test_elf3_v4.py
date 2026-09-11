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
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from scripts.build_elf3_amp_v4_dataset import SOURCE, OUTPUT, REMOVED, sha256
from src.tasks.amp_loco.mdp.rough_height import root_clearance, root_height_below_terrain
from src.tasks.amp_loco.mdp.tienkung_terrain import TienKungGravelTerrainCfg
from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from rsl_rl.utils.motion_loader import AMPLoader


class V4Test(unittest.TestCase):
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
            self.assertEqual(full.events['init_motion_loader'].params['delay_reset_env_ratio'], .4)
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
            self.assertEqual(terrain.sub_terrains['random_rough'].noise_range, (-.02, .04))
            self.assertNotIn('terrain_scan', [s.name for s in cfg.scene.sensors])
            for group in ('actor', 'critic', 'amp'):
                self.assertEqual(list(cfg.observations[group].terms), list(v3.observations[group].terms))
                self.assertNotIn('height_scan', cfg.observations[group].terms)
            self.assertEqual(cfg.observations['actor'].history_length, 4)
            self.assertEqual(cfg.events['base_com'].params['ranges'],
                             {0: (-.025, .025), 1: (-.05, .05), 2: (-.05, .05)})
            self.assertEqual(cfg.events['init_motion_loader'].params['delay_reset_env_ratio'], .4)
            self.assertEqual(Path(cfg.events['init_motion_loader'].params['motion_dir']), OUTPUT/'WalkandRun')
            self.assertEqual(cfg.events['reset_from_motion'].params['motion_dir'], str(OUTPUT/'WalkandRun'))
            for name, reward in cfg.rewards.items():
                self.assertEqual(reward.weight, v3.rewards[name].weight)
                self.assertEqual(reward.params, v3.rewards[name].params)
        runner = load_rl_cfg('BXI-ELF3-AMP-Rough-V4')
        self.assertFalse(runner.resume)
        self.assertEqual(runner.max_iterations, 200001)
        self.assertEqual(runner.algorithm.learning_rate, .001)
        self.assertEqual(runner.amp_motion_files, str(OUTPUT))
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

    def test_gravel_geometry_signed_heights_and_origin(self):
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


if __name__ == '__main__':
    unittest.main()
