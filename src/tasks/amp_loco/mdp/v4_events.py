"""V4 startup reset sequencing; other tasks keep their original reset events."""
import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .events import reset_from_motion_data


def reset_with_startup_v4(env, env_ids, motion_dir, asset_cfg: SceneEntityCfg, command_name='twist'):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    reset_from_motion_data(env, env_ids, motion_dir, asset_cfg)
    command = env.command_manager.get_term(command_name)
    eligible = torch.ones(len(env_ids), dtype=torch.bool, device=env.device)
    recovery = getattr(env.termination_manager, '_delay_env_mask', None)
    if recovery is not None:
        eligible &= ~recovery[env_ids]
    ids = command.prepare_reset(env_ids, eligible)
    if len(ids) == 0:
        return
    robot = env.scene[asset_cfg.name]
    state = robot.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, 7:] = 0.
    robot.write_root_state_to_sim(state, ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids],
                                  torch.zeros_like(robot.data.default_joint_vel[ids]), env_ids=ids)


def push_except_startup_preparation_v4(env, env_ids, velocity_range, command_name='twist'):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    command = env.command_manager.get_term(command_name)
    ids = env_ids[command.startup_phase[env_ids] != 1]
    if len(ids):
        envs_mdp.push_by_setting_velocity(env, ids, velocity_range=velocity_range)
