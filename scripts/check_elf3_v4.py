"""Bounded GPU smoke check; no training service/checkpoint/SwanLab changes."""
import argparse
import json
import os
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import torch
import warp as wp
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.assets.robots.elf3.elf3_constants import ELF3_POLICY_ROOT, ELF3_PHYSICAL_ROOT
from src.tasks.amp_loco.mdp.rough_height import root_clearance


def check(num_envs=16, steps=120, nconmax=None, njmax=None, ccd_iterations=None,
          recovery_fraction=None):
    cfg = load_env_cfg('BXI-ELF3-AMP-Rough-V4')
    cfg.scene.num_envs = num_envs
    if nconmax is not None:
        cfg.sim.nconmax = nconmax
    if njmax is not None:
        cfg.sim.njmax = njmax
    if ccd_iterations is not None:
        cfg.sim.mujoco.ccd_iterations = ccd_iterations
    if recovery_fraction is not None:
        cfg.events['init_motion_loader'].params['delay_reset_env_ratio'] = recovery_fraction
    started = time.monotonic()
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        peak = dict(contacts=0, broadphase_pairs=0, constraints_per_world=0,
                    contacts_per_world=0, gpu_used_mib=0)
        data = env.sim.wp_data
        def capacity_check():
            contacts = int(wp.to_torch(data.nacon).item())
            pairs = int(wp.to_torch(data.ncollision).item())
            constraints = int(wp.to_torch(data.nefc).max().item())
            assert contacts < data.naconmax, (contacts, data.naconmax)
            assert pairs < data.naconmax, (pairs, data.naconmax)
            assert constraints < cfg.sim.njmax, (constraints, cfg.sim.njmax)
            per_world = (torch.bincount(wp.to_torch(data.contact.worldid)[:contacts].long(),
                                       minlength=num_envs).max().item() if contacts else 0)
            used_mib = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**20
            for name, value in zip(peak, (contacts, pairs, constraints, per_world, used_mib)):
                peak[name] = max(peak[name], value)
        # Inspect every physics substep, not just the final decimated state.
        original_step = env.sim.step
        def measured_step():
            original_step()
            capacity_check()
        env.sim.step = measured_step
        obs, _ = env.reset()
        capacity_check()
        assert obs['actor'].shape == (num_envs, 384), obs['actor'].shape
        model = env.sim.mj_model
        ids = [model.body('robot/' + name).id for name in (ELF3_POLICY_ROOT, ELF3_PHYSICAL_ROOT)]
        actual = env.sim.model.body_ipos[:, ids].detach().cpu()
        nominal = torch.tensor(model.body_ipos[ids], dtype=actual.dtype)
        delta = actual - nominal
        limits = torch.tensor([.025, .05, .05])
        assert torch.all(delta.abs() <= limits + 1e-6)
        assert torch.all(delta.std(dim=0) > .001), 'COM randomization not varying per world'
        assert model.nhfield == 8000, model.nhfield
        # Probe hits must be terrain, not robot geometry.
        sensor = env.scene['terrain_height']
        # Rays exactly on hfield-strip seams may miss through float rounding;
        # height estimation masks those rays and uses the other local hits.
        assert torch.all((sensor.data.distances >= 0).any(dim=-1))
        command = env.command_manager.get_term('twist')
        seen = []
        for _ in range(100):
            env.command_manager.reset(torch.arange(num_envs, device=env.device))
            env.command_manager.compute(env.step_dt)
            seen.append(command.command.clone())
        commands = torch.cat(seen)
        assert torch.all(commands[:, 0] >= -.60001) and torch.all(commands[:, 0] <= 1.00001)
        assert torch.all(commands[:, 1].abs() <= .50001)
        assert torch.all(commands[:, 2].abs() <= 1.57001)
        for step in range(steps):
            result = env.step(torch.randn(num_envs, env.action_manager.total_action_dim,
                                          device=env.device) * .03)
            assert all(torch.isfinite(v).all() for v in result[0].values())
            assert torch.isfinite(result[1]).all()
            assert torch.isfinite(root_clearance(env)).all()
            if (step + 1) % 100 == 0:
                print('V4_CAPACITY_PROGRESS ' + json.dumps({'steps': step + 1, **peak}), flush=True)
        report = {'num_envs': num_envs, 'steps': steps,
                  'actor_shape': list(obs['actor'].shape), 'terrain_tiles': 200,
                  'collision_hfield_strips': model.nhfield,
                  'com_offset_min_xyz_m': delta.amin(dim=(0, 1)).tolist(),
                  'com_offset_max_xyz_m': delta.amax(dim=(0, 1)).tolist(),
                  'sampled_command_min': commands.amin(dim=0).tolist(),
                  'sampled_command_max': commands.amax(dim=0).tolist(),
                  'finite_observations_rewards': True,
                  'nconmax': cfg.sim.nconmax, 'njmax': cfg.sim.njmax,
                  'ccd_iterations': cfg.sim.mujoco.ccd_iterations,
                  'capacity_peak': peak, 'elapsed_s': time.monotonic() - started}
        print('V4_SMOKE_RESULT ' + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--nconmax', type=int)
    parser.add_argument('--njmax', type=int)
    parser.add_argument('--ccd-iterations', type=int)
    parser.add_argument('--recovery-fraction', type=float)
    args = parser.parse_args()
    check(**vars(args))
