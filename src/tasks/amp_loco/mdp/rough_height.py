"""Terrain-relative height for V4; simulator-only, never a policy observation."""
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .rewards import _apply_delay_env_reward_mask_only

_ROBOT = SceneEntityCfg('robot')


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
