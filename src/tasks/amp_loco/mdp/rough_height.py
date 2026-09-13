"""Terrain-relative height for V4; simulator-only, never a policy observation."""
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .rewards import _apply_delay_env_reward_mask_only

_ROBOT = SceneEntityCfg('robot')


def assign_interior_terrain_origins(env, env_ids=None, margin_rows=2,
                                    margin_cols=2):
    """Place environments away from the generator's outer border boxes.

    All V4 tiles use the same randomized roughness distribution, so excluding the
    outer tile rings does not remove a terrain class.  It only leaves generated
    rough terrain as a guard band between normal episode motion and the flat border
    boxes surrounding the complete terrain grid.
    """
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]

    num_rows, num_cols = terrain.terrain_origins.shape[:2]
    if num_rows <= 2 * margin_rows or num_cols <= 2 * margin_cols:
        raise ValueError(
            f'Terrain grid {num_rows}x{num_cols} is too small for interior '
            f'margins {margin_rows}x{margin_cols}'
        )
    rows = torch.randint(
        margin_rows, num_rows - margin_rows, (len(env_ids),), device=env.device
    )
    cols = torch.randint(
        margin_cols, num_cols - margin_cols, (len(env_ids),), device=env.device
    )
    terrain.terrain_levels[env_ids] = rows
    terrain.terrain_types[env_ids] = cols
    terrain.env_origins[env_ids] = terrain.terrain_origins[rows, cols]


def root_outside_terrain_interior(env, half_extent_x, half_extent_y,
                                  safety_margin=1.0, asset_cfg=_ROBOT):
    """Truncate before a robot can contact the terrain generator border boxes."""
    root_xy = env.scene[asset_cfg.name].data.root_link_pos_w[:, :2]
    finite = torch.isfinite(root_xy).all(dim=-1)
    outside = (
        (root_xy[:, 0].abs() >= half_extent_x - safety_margin)
        | (root_xy[:, 1].abs() >= half_extent_y - safety_margin)
    )
    return outside | ~finite


def root_clearance(env, sensor_name='terrain_height', asset_cfg=_ROBOT):
    sensor = env.scene[sensor_name]
    hits = sensor.data.hit_pos_w[..., 2]
    valid = (sensor.data.distances >= 0) & torch.isfinite(hits)
    count = valid.sum(dim=-1)
    ground = torch.where(valid, hits, 0.0).sum(dim=-1) / count.clamp(min=1)
    ground = torch.where(count > 0, ground, env.scene.env_origins[:, 2])
    return env.scene[asset_cfg.name].data.root_link_pos_w[:, 2] - ground


def root_height_below_terrain(env, minimum_height, sensor_name='terrain_height',
                            asset_cfg=_ROBOT):
    return root_clearance(env, sensor_name, asset_cfg) < minimum_height


def track_root_height_terrain(env, std, mask_delay=False, delay_env_rew_ratio=1.0,
                             sensor_name='terrain_height', asset_cfg=_ROBOT):
    desired = env.scene[asset_cfg.name].data.default_root_state[:, 2]
    error = (desired - root_clearance(env, sensor_name, asset_cfg)).square()
    return _apply_delay_env_reward_mask_only(
        env, torch.exp(-error / std**2), mask_delay, delay_env_rew_ratio)
