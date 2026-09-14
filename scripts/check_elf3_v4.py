"""Bounded GPU smoke check; no training service/checkpoint/SwanLab changes."""
import argparse
import json
import os
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
import torch
import warp as wp
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.assets.robots.elf3.elf3_constants import ELF3_POLICY_ROOT, ELF3_PHYSICAL_ROOT
from src.tasks.amp_loco.mdp.rough_height import root_clearance
from src.tasks.amp_loco.mdp.tienkung_terrain import TienKungGravelTerrainCfg


def expected_heightfield_count(terrain):
    """Account for old GRAVEL's narrow strips, not just its tile count."""
    sub = terrain.sub_terrains['random_rough']
    fields_per_tile = 1
    if isinstance(sub, TienKungGravelTerrainCfg):
        cells = int(terrain.size[1] / sub.horizontal_scale)
        fields_per_tile = (cells + sub.strip_cells - 1) // sub.strip_cells
    return terrain.num_rows * terrain.num_cols * fields_per_tile


def check(num_envs=16, steps=120, nconmax=None, njmax=None, ccd_iterations=None,
          recovery_fraction=None, device='cuda:0', seed=42,
          task='BXI-ELF3-AMP-Rough-V4', verify_contacts=False):
    if device == 'cpu':
        torch.set_num_threads(2)
    cfg = load_env_cfg(task)
    cfg.seed = seed
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
    if device.startswith('cuda') and torch.cuda.mem_get_info()[0] < 2 * 2**30:
        raise RuntimeError('Need at least 2 GiB free GPU memory for the isolated check')
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        peak = dict(contacts=0, broadphase_pairs=0, constraints_per_world=0,
                    contacts_per_world=0, gpu_used_mib=0,
                    foot_ground_contact_depth_mm=0.)
        data = env.sim.wp_data
        model = env.sim.mj_model
        feet = torch.tensor([g for g in range(model.ngeom)
                             if 'ankle_x_link_collision' in (model.geom(g).name or '')],
                            device=env.device)
        ground = torch.tensor([g for g in range(model.ngeom) if model.geom_group[g] == 0],
                              device=env.device)
        worst_case = None
        before_qpos = None
        def capacity_check():
            nonlocal worst_case
            # Check raw physics before the environment can reset a bad state.
            for field in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'sensordata'):
                assert torch.isfinite(wp.to_torch(getattr(data, field))).all(), field
            contacts = int(wp.to_torch(data.nacon).item())
            pairs = int(wp.to_torch(data.ncollision).item())
            constraints = int(wp.to_torch(data.nefc).max().item())
            assert contacts < data.naconmax, (contacts, data.naconmax)
            assert pairs < data.naconmax, (pairs, data.naconmax)
            assert constraints < cfg.sim.njmax, (constraints, cfg.sim.njmax)
            per_world = (torch.bincount(wp.to_torch(data.contact.worldid)[:contacts].long(),
                                       minlength=num_envs).max().item() if contacts else 0)
            used_mib = ((torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**20
                        if str(env.device).startswith('cuda') else 0.)
            for name, value in zip(('contacts', 'broadphase_pairs', 'constraints_per_world',
                                    'contacts_per_world', 'gpu_used_mib'),
                                   (contacts, pairs, constraints, per_world, used_mib)):
                peak[name] = max(peak[name], value)
            if contacts:
                geoms = wp.to_torch(data.contact.geom)[:contacts].long()
                distances = wp.to_torch(data.contact.dist)[:contacts]
                assert torch.isfinite(distances).all(), 'contact distance'
                assert torch.isfinite(wp.to_torch(data.contact.frame)[:contacts]).all(), 'contact frame'
                mask = ((torch.isin(geoms[:, 0], feet) | torch.isin(geoms[:, 1], feet))
                        & (torch.isin(geoms[:, 0], ground) | torch.isin(geoms[:, 1], ground)))
                if mask.any():
                    # This is solver contact distance, NOT measured physical penetration.
                    depth_mm = float((-distances[mask].min()).clamp_min(0) * 1000)
                    if verify_contacts and depth_mm > peak['foot_ground_contact_depth_mm']:
                        index = torch.where(mask)[0][distances[mask].argmin()]
                        world = int(wp.to_torch(data.contact.worldid)[index])
                        qpos = before_qpos if before_qpos is not None else wp.to_torch(data.qpos)
                        worst_case = dict(
                            world=world, warp_distance_mm=-depth_mm,
                            geom_ids=geoms[index].cpu().tolist(),
                            warp_normal=wp.to_torch(data.contact.frame)[index, 0].cpu().tolist(),
                            qpos=qpos[world].cpu().numpy().copy())
                    peak['foot_ground_contact_depth_mm'] = max(peak['foot_ground_contact_depth_mm'], depth_mm)
        # Inspect every physics substep, not just the final decimated state.
        original_step = env.sim.step
        def measured_step():
            nonlocal before_qpos
            if verify_contacts:
                before_qpos = wp.to_torch(data.qpos).clone()
            original_step()
            capacity_check()
        env.sim.step = measured_step
        obs, _ = env.reset()
        capacity_check()
        assert obs['actor'].shape == (num_envs, 384), obs['actor'].shape
        env_origins = env.scene.env_origins
        # Training keeps two generated rough-tile rings between spawn origins
        # and the outer border boxes: x origins +/-20 m, y origins +/-60 m.
        assert torch.all(env_origins[:, 0].abs() <= 20.00001), env_origins[:, 0]
        assert torch.all(env_origins[:, 1].abs() <= 60.00001), env_origins[:, 1]
        boundary = env.termination_manager.get_term_cfg('terrain_outer_boundary')
        assert boundary.time_out, 'terrain boundary must not be a failure termination'
        model = env.sim.mj_model
        ids = [model.body('robot/' + name).id for name in (ELF3_POLICY_ROOT, ELF3_PHYSICAL_ROOT)]
        actual = env.sim.model.body_ipos[:, ids].detach().cpu()
        nominal = torch.tensor(model.body_ipos[ids], dtype=actual.dtype)
        delta = actual - nominal
        limits = torch.tensor([.025, .05, .05])
        assert torch.all(delta.abs() <= limits + 1e-6)
        assert torch.all(delta.std(dim=0) > .001), 'COM randomization not varying per world'
        terrain = cfg.scene.terrain.terrain_generator
        tiles = terrain.num_rows * terrain.num_cols
        expected_fields = expected_heightfield_count(terrain)
        assert model.nhfield == expected_fields, (model.nhfield, expected_fields)
        # Probe hits must be terrain, not robot geometry.
        sensor = env.scene['terrain_height']
        assert torch.all((sensor.data.distances >= 0).any(dim=-1))
        command = env.command_manager.get_term('twist')
        seen = []
        for _ in range(100):
            env.command_manager.reset(torch.arange(num_envs, device=env.device))
            env.command_manager.compute(env.step_dt)
            seen.append(command.command.clone())
        commands = torch.cat(seen)
        assert torch.all(commands[:, 0] >= -.60001) and torch.all(commands[:, 0] <= 1.00001)
        assert torch.all(commands[:, 1].abs() <= 1.00001)
        # Mixed locomotion uses +/-1 rad/s; the dedicated in-place-turn subset
        # preserves V3.1's wider +/-2 rad/s envelope.
        assert torch.all(commands[:, 2].abs() <= 2.00001)
        for step in range(steps):
            result = env.step(torch.randn(num_envs, env.action_manager.total_action_dim,
                                          device=env.device) * .03)
            assert all(torch.isfinite(v).all() for v in result[0].values())
            assert torch.isfinite(result[1]).all()
            assert torch.isfinite(root_clearance(env)).all()
            if (step + 1) % 100 == 0:
                print('V4_CAPACITY_PROGRESS ' + json.dumps({'steps': step + 1, **peak}), flush=True)
        report = {'task': task, 'seed': seed, 'num_envs': num_envs, 'steps': steps,
                  'actor_shape': list(obs['actor'].shape), 'terrain_tiles': tiles,
                  'collision_heightfields': model.nhfield, 'device': device,
                  'com_offset_min_xyz_m': delta.amin(dim=(0, 1)).tolist(),
                  'com_offset_max_xyz_m': delta.amax(dim=(0, 1)).tolist(),
                  'sampled_command_min': commands.amin(dim=0).tolist(),
                  'sampled_command_max': commands.amax(dim=0).tolist(),
                  'env_origin_min_xyz_m': env_origins.amin(dim=0).tolist(),
                  'env_origin_max_xyz_m': env_origins.amax(dim=0).tolist(),
                  'finite_observations_rewards': True,
                  'finite_raw_physics_every_substep': True,
                  'nconmax': cfg.sim.nconmax, 'njmax': cfg.sim.njmax,
                  'ccd_iterations': cfg.sim.mujoco.ccd_iterations,
                  'capacity_peak': peak, 'elapsed_s': time.monotonic() - started}
        if worst_case is not None:
            native = mujoco.MjData(model)
            native.qpos[:] = worst_case.pop('qpos')
            mujoco.mj_forward(model, native)
            foot = next(g for g in worst_case['geom_ids'] if g in feet.tolist())
            contacts = [c for c in native.contact
                        if foot in c.geom and any(model.geom_group[g] == 0 for g in c.geom)]
            worst_case['native_min_distance_mm'] = min((c.dist * 1000 for c in contacts), default=None)
            worst_case['geom_names'] = [model.geom(g).name for g in worst_case['geom_ids']]
            # Vertex-to-ground vertical depths are an independent geometry probe,
            # not an exact convex-volume penetration calculation.
            mesh = model.geom_dataid[foot]
            start = model.mesh_vertadr[mesh]
            count = model.mesh_vertnum[mesh]
            vertices = (model.mesh_vert[start:start + count]
                        @ native.geom_xmat[foot].reshape(3, 3).T + native.geom_xpos[foot])
            groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
            geom_id = np.zeros(1, dtype=np.int32)
            depths = []
            ray_z = max(2., float(vertices[:, 2].max()) + 1.)
            for vertex in vertices:
                distance = mujoco.mj_ray(model, native,
                    np.array([vertex[0], vertex[1], ray_z]), np.array([0., 0., -1.]),
                    groups, True, model.geom_bodyid[foot], geom_id)
                if distance >= 0:
                    depths.append((ray_z - distance - vertex[2]) * 1000)
            worst_case['max_vertex_ground_depth_mm'] = max(depths, default=None)
            report['worst_contact_native_geometry_check'] = worst_case
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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', choices=['BXI-ELF3-AMP-Rough-V4', 'BXI-ELF3-AMP-Rough-V4-Loco'],
                        default='BXI-ELF3-AMP-Rough-V4')
    parser.add_argument('--verify-contacts', action='store_true')
    args = parser.parse_args()
    check(**vars(args))
