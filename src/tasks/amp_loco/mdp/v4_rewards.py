"""V4-only command tracking and leg safety; no actor observation changes."""
from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse, yaw_quat

from .rewards import _apply_delay_env_reward_scaling


def command_scaled_tracking(command, actual, absolute_tolerance, relative_tolerance):
    """Exponential tracking with per-axis absolute + relative error tolerances.

    Tolerances depend on the *command*, never the achieved velocity, so moving
    faster in the wrong direction cannot enlarge the accepted error.
    """
    absolute = command.new_tensor(absolute_tolerance)
    relative = command.new_tensor(relative_tolerance)
    sigma = absolute + relative * command.abs()
    return torch.exp(-((actual - command) / sigma).square().sum(dim=-1))


def track_horizontal_velocity_v4(
    env, command_name: str, anchor_cfg: SceneEntityCfg,
    absolute_tolerance: tuple[float, float] = (.25, .20),
    relative_tolerance: tuple[float, float] = (.35, .35),
):
    robot = env.scene[anchor_cfg.name]
    body = anchor_cfg.body_ids[0]
    actual = quat_apply_inverse(yaw_quat(robot.data.body_link_quat_w[:, body]),
                                robot.data.body_link_lin_vel_w[:, body])[:, :2]
    command = env.command_manager.get_command(command_name)[:, :2]
    reward = command_scaled_tracking(command, actual, absolute_tolerance, relative_tolerance)
    env.extras['log']['Metrics/v4/lateral_abs_error'] = (actual[:, 1] - command[:, 1]).abs().mean()
    return _apply_delay_env_reward_scaling(env, reward, True, 0.)


def track_yaw_velocity_v4(
    env, command_name: str, anchor_cfg: SceneEntityCfg,
    absolute_tolerance: float = .35, relative_tolerance: float = .40,
):
    robot = env.scene[anchor_cfg.name]
    actual = robot.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0], 2:3]
    command = env.command_manager.get_command(command_name)[:, 2:3]
    reward = command_scaled_tracking(command, actual, absolute_tolerance, relative_tolerance)
    env.extras['log']['Metrics/v4/yaw_abs_error'] = (actual - command).abs().mean()
    return _apply_delay_env_reward_scaling(env, reward, True, 0.)


def vertical_velocity_v4(env, anchor_cfg: SceneEntityCfg):
    """Retain mild vertical-motion regularization outside the XY tracking score."""
    robot = env.scene[anchor_cfg.name]
    cost = robot.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], 2].square()
    return _apply_delay_env_reward_scaling(env, cost, True, 0.)


def foot_box_separation(positions, quaternions, center_offset, half_extents):
    """Conservative mesh-box separation using all 15 OBB separating axes.

    Positive means separated; zero/negative means touching/overlapping boxes.
    This is a proximity proxy, not exact mesh distance. Full foot rotations
    account for yaw, roll, pitch and safe vertical clearance on rough ground.
    """
    centers = positions + quat_apply(quaternions, positions.new_tensor(center_offset).expand_as(positions))
    rotations = matrix_from_quat(quaternions)
    left, right = rotations[:, 0].transpose(-1, -2), rotations[:, 1].transpose(-1, -2)
    cross = torch.cross(left[:, :, None, :].expand(-1, -1, 3, -1),
                        right[:, None, :, :].expand(-1, 3, -1, -1), dim=-1).flatten(1, 2)
    axes = torch.cat((left, right, cross), dim=1)
    lengths = axes.norm(dim=-1)
    axes = axes / lengths.clamp_min(1.e-6).unsqueeze(-1)
    extents = positions.new_tensor(half_extents)
    radius_left = (torch.matmul(axes, rotations[:, 0]).abs() * extents).sum(-1)
    radius_right = (torch.matmul(axes, rotations[:, 1]).abs() * extents).sum(-1)
    distance = ((centers[:, 0] - centers[:, 1]).unsqueeze(1) * axes).sum(-1).abs()
    gaps = distance - radius_left - radius_right
    gaps = gaps.masked_fill(lengths < 1.e-6, -torch.inf)
    return gaps.max(dim=-1).values


def feet_safe_distance_v4(
    env, asset_cfg: SceneEntityCfg, command_name: str = 'twist',
    center_offset: tuple[float, float, float] = (.03, 0., -.01375),
    half_extents: tuple[float, float, float] = (.12, .04, .02725),
    safety_margin: float = .02, startup_multiplier: float = 1.5,
):
    """Soft foot-box clearance, with extra emphasis on the first startup steps.

    ELF3 mesh bounds: x [-.09,.15], y [-.04,.04], z [-.041,.0135].
    Parallel feet at the same height begin paying a cost below 10 cm lateral
    center spacing, not 27 cm. No fixed-width target is imposed on safe steps.
    """
    robot = env.scene[asset_cfg.name]
    positions = robot.data.body_link_pos_w[:, asset_cfg.body_ids]
    quaternions = robot.data.body_link_quat_w[:, asset_cfg.body_ids]
    separation = foot_box_separation(positions, quaternions, center_offset, half_extents)
    cost = ((safety_margin - separation) / safety_margin).clamp(0., 1.)
    command = env.command_manager.get_term(command_name)
    startup = (command.startup_phase == 2) & (command.startup_motion_time < 2.)
    cost = cost * torch.where(startup, startup_multiplier, 1.)
    env.extras['log']['Metrics/v4/foot_box_close_fraction'] = (separation < safety_margin).float().mean()
    contact = env.scene['feet_self_contact'].data.force_history.norm(dim=-1).amax(dim=(1, 2))
    env.extras['log']['Metrics/v4/foot_self_contact_fraction'] = (contact > 10.).float().mean()
    # Conditional first-two-second metrics; counts distinguish no samples from no failures.
    terminated = getattr(env, 'reset_terminated', torch.zeros_like(startup))
    for direction, sign in (('pos_y', 1.), ('neg_y', -1.)):
        mask = startup & (command.startup_target[:, 1] * sign > 0.)
        count = mask.sum()
        prefix = f'Metrics/v4/startup_{direction}'
        env.extras['log'][f'{prefix}_samples'] = count.float()
        for name, event in (('foot_contact', contact > 10.),
                            ('foot_close', separation < safety_margin), ('terminated', terminated)):
            env.extras['log'][f'{prefix}_{name}_fraction'] = (mask & event).sum() / count.clamp(min=1)
    return _apply_delay_env_reward_scaling(env, cost, True, 0.)


def legs_stand_pose_v4(
    env, asset_cfg: SceneEntityCfg, command_name: str = 'twist',
    linear_threshold: float = .02, angular_threshold: float = .02,
):
    """Mean leg-joint deviation only for effectively zero translation AND yaw."""
    command = env.command_manager.get_command(command_name)
    standing = (command[:, :2].abs().amax(-1) < linear_threshold) & (command[:, 2].abs() < angular_threshold)
    robot = env.scene[asset_cfg.name]
    error = (robot.data.joint_pos[:, asset_cfg.joint_ids]
             - robot.data.default_joint_pos[:, asset_cfg.joint_ids]).abs().mean(-1)
    return _apply_delay_env_reward_scaling(env, error * standing, True, 0.)


def feet_stumble_v4(
    env, sensor_name: str = 'feet_ground_contact',
    horizontal_vertical_ratio: float = 5., minimum_horizontal_force: float = 20.,
):
    """Auxiliary ground-scuff cost, rejecting tiny airborne force noise.

    Uses the ground sensor's world-frame net force, NOT contact-frame vectors
    from the self-collision sensor. Foot-foot safety is handled separately.
    """
    force = env.scene[sensor_name].data.force
    horizontal = force[..., :2].norm(dim=-1)
    hit = (horizontal > minimum_horizontal_force) & (horizontal > horizontal_vertical_ratio * force[..., 2].abs())
    return _apply_delay_env_reward_scaling(env, hit.float().sum(-1), True, 0.)
