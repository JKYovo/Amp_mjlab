"""Small headless V4 startup/reward smoke test; never trains or opens a logger."""
import argparse
from dataclasses import asdict
import json
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import torch
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def check(task='BXI-ELF3-AMP-Rough-V4-Loco', checkpoint=None, num_envs=8, steps=400):
    torch.set_num_threads(2)
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.seed = 42
    cfg.sim.nan_guard.enabled = False
    # Exercise the sequence in every eligible environment; recovery stays excluded.
    cfg.commands['twist'].startup_fraction = 1.
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        if torch.cuda.mem_get_info()[0] < 400 * 2**20:
            raise RuntimeError('Not enough memory headroom for this diagnostic')
        agent = load_rl_cfg(task)
        policy = None
        if checkpoint:
            wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            runner = load_runner_cls(task)(wrapper, asdict(agent), log_dir=None, device='cuda:0')
            runner.load(checkpoint, load_optimizer=False)
            policy = runner.get_inference_policy(device='cuda:0')
        raw, _ = env.reset(seed=42)
        command = env.command_manager.get_term('twist')
        recovery = getattr(env.termination_manager, '_delay_env_mask', torch.zeros(num_envs, dtype=torch.bool, device=env.device))
        assert not (command.startup_phase[recovery] > 0).any()
        assert (command.startup_phase[~recovery] == 1).all()
        assert (command.command[~recovery] == 0).all()
        launches = timeouts = resets = 0
        peaks = {name: 0. for name in ('feet_safe_distance', 'legs_stand_pose', 'feet_stumble')}
        with torch.inference_mode():
            for _ in range(steps):
                assert raw['actor'].shape == (num_envs, 384)
                torch.testing.assert_close(raw['actor'].reshape(num_envs, 4, 96)[:, -1, 6:9], command.command)
                phase = command.startup_phase.clone()
                if policy is None:
                    action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
                else:
                    action = policy(raw['actor'])
                    if agent.clip_actions is not None:
                        action = action.clamp(-agent.clip_actions, agent.clip_actions)
                old_timeouts = command.metrics['startup_timeouts'].clone()
                raw, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                launches += int(((phase == 1) & (command.startup_phase == 2) & ~done).sum())
                timeouts += int((command.metrics['startup_timeouts'] > old_timeouts).sum())
                resets += int(done.sum())
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in raw.values())
                assert (command.command[:, 0] >= -.60001).all()
                assert (command.command[:, 0] <= 1.00001).all()
                assert (command.command[:, 1].abs() <= 1.00001).all()
                assert (command.command[:, 2].abs() <= 2.00001).all()
                for name in peaks:
                    term = cfg.rewards[name]
                    value = term.func(env, **term.params)
                    assert torch.isfinite(value).all() and (value >= 0).all()
                    peaks[name] = max(peaks[name], float(value.max()))
        if checkpoint and (~recovery).any():
            assert launches > 0, 'No settled startup launch was exercised with this checkpoint'
        print('V4_STARTUP_SMOKE_RESULT ' + json.dumps({
            'task': task, 'num_envs': num_envs, 'steps': steps, 'actor_dim': 384,
            'checkpoint': checkpoint, 'startup_launches': launches,
            'startup_timeouts': timeouts, 'resets': resets, 'max_raw_cost': peaks,
            'finite_observations_rewards': True, 'training_performed': False,
        }), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['BXI-ELF3-AMP-Rough-V4', 'BXI-ELF3-AMP-Rough-V4-Loco'],
                        default='BXI-ELF3-AMP-Rough-V4-Loco')
    parser.add_argument('--checkpoint')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=400)
    check(**vars(parser.parse_args()))
