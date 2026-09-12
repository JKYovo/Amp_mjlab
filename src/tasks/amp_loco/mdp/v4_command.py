"""V4 commands with explicit pure lateral coverage and settled startup trials."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .command import TurningVelocityCommand, TurningVelocityCommandCfg


class StartupVelocityCommand(TurningVelocityCommand):
    cfg: 'StartupVelocityCommandCfg'

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.is_lateral_env = torch.zeros_like(self.is_standing_env)
        self.startup_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.startup_prepare_time = torch.zeros_like(self.time_left)
        self.startup_stable_time = torch.zeros_like(self.time_left)
        self.startup_motion_time = torch.zeros_like(self.time_left)
        self.startup_wait_time = torch.zeros_like(self.time_left)
        self.startup_target = torch.zeros_like(self.vel_command_b)
        for name in ('startup_launches', 'startup_timeouts'):
            self.metrics[name] = torch.zeros_like(self.time_left)

    def prepare_reset(self, env_ids, eligible):
        """Called by the reset event BEFORE CommandManager.reset samples commands."""
        self.startup_phase[env_ids] = 0
        self.startup_prepare_time[env_ids] = 0.
        self.startup_stable_time[env_ids] = 0.
        self.startup_motion_time[env_ids] = 0.
        self.startup_target[env_ids] = 0.
        selected = eligible & (torch.rand(len(env_ids), device=self.device) < self.cfg.startup_fraction)
        ids = env_ids[selected]
        self.startup_phase[ids] = 1
        self.startup_wait_time[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.startup_wait_range)
        turning = torch.rand(len(ids), device=self.device) < self.cfg.startup_turning_fraction
        magnitude = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.startup_lateral_range)
        turn_magnitude = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.startup_yaw_range)
        sign = torch.where(torch.rand(len(ids), device=self.device) < .5, -1., 1.)
        self.startup_target[ids, 1] = torch.where(turning, 0., sign * magnitude)
        self.startup_target[ids, 2] = torch.where(turning, sign * turn_magnitude, 0.)
        return ids

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        remaining = ~self.is_standing_env[env_ids] & ~self.is_turning_env[env_ids]
        denominator = 1. - self.cfg.rel_standing_envs - self.cfg.rel_turning_envs
        probability = self.cfg.rel_lateral_envs / denominator if denominator > 0 else 0.
        lateral = remaining & (torch.rand(len(env_ids), device=self.device) < probability)
        self.is_lateral_env[env_ids] = lateral
        ids = env_ids[lateral]
        sign = torch.where(torch.rand(len(ids), device=self.device) < .5, -1., 1.)
        limit = torch.where(sign > 0, self.cfg.ranges.lin_vel_y[1], -self.cfg.ranges.lin_vel_y[0])
        magnitude = self.cfg.lateral_min_abs_velocity + torch.rand(len(ids), device=self.device) * (limit - self.cfg.lateral_min_abs_velocity)
        self.vel_command_b[ids, 0] = 0.
        self.vel_command_b[ids, 1] = sign * magnitude
        self.vel_command_b[ids, 2] = 0.
        self.is_heading_env[ids] = False
        # Reset observations are computed before the next compute() call.
        self.vel_command_b[env_ids[self.is_standing_env[env_ids]]] = 0.
        self._apply_startup_command()

    def _update_command(self):
        super()._update_command()
        self._apply_startup_command()

    def _apply_startup_command(self):
        preparing = self.startup_phase == 1
        moving = self.startup_phase == 2
        active = preparing | moving
        self.time_left[active] = 1.e6
        self.is_heading_env[active] = False
        self.is_standing_env[active] = preparing[active]
        self.is_turning_env[active] = moving[active] & (self.startup_target[active, 2] != 0.)
        self.is_lateral_env[active] = moving[active] & (self.startup_target[active, 1] != 0.)
        self.vel_command_b[preparing] = 0.
        self.vel_command_b[moving] = self.startup_target[moving]

    def compute(self, dt):
        super().compute(dt)
        preparing = self.startup_phase == 1
        moving = self.startup_phase == 2
        force = self._env.scene[self.cfg.feet_sensor_name].data.force.norm(dim=-1)
        upright = self.robot.data.projected_gravity_b[:, 2] < -.94
        stable = ((force > self.cfg.startup_contact_force).all(-1)
                  & (self.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1) < self.cfg.startup_linear_threshold)
                  & (self.robot.data.root_link_ang_vel_w.norm(dim=-1) < self.cfg.startup_angular_threshold)
                  & upright)
        self.startup_prepare_time += preparing * dt
        self.startup_stable_time = torch.where(preparing & stable, self.startup_stable_time + dt, 0.)
        self.startup_motion_time += moving * dt
        launch = (preparing & (self.startup_prepare_time >= self.startup_wait_time)
                  & (self.startup_stable_time >= self.cfg.startup_stable_duration))
        timeout = preparing & ~launch & (self.startup_prepare_time >= self.cfg.startup_timeout)
        finished = moving & (self.startup_motion_time >= self.cfg.startup_move_duration)
        self.metrics['startup_launches'] += launch
        self.metrics['startup_timeouts'] += timeout
        self.startup_phase[launch] = 2
        self.startup_motion_time[launch] = 0.
        self.startup_phase[timeout | finished] = 0
        # Resume ordinary sampling on the following step, not a second command
        # jump on the same physics state. Failed preparation is not a launch.
        self.time_left[timeout | finished] = 0.
        self._apply_startup_command()
        self._env.extras['log']['Metrics/v4/startup_preparing_fraction'] = (self.startup_phase == 1).float().mean()
        self._env.extras['log']['Metrics/v4/pure_lateral_fraction'] = self.is_lateral_env.float().mean()


@dataclass(kw_only=True)
class StartupVelocityCommandCfg(TurningVelocityCommandCfg):
    rel_lateral_envs: float = .20
    lateral_min_abs_velocity: float = .2
    startup_fraction: float = .25
    startup_turning_fraction: float = .25
    startup_lateral_range: tuple[float, float] = (.2, 1.)
    startup_yaw_range: tuple[float, float] = (.3, 2.)
    startup_wait_range: tuple[float, float] = (1., 2.)
    startup_stable_duration: float = .2
    startup_timeout: float = 3.
    startup_move_duration: float = 4.
    startup_contact_force: float = 20.
    startup_linear_threshold: float = .08
    startup_angular_threshold: float = .15
    feet_sensor_name: str = 'feet_ground_contact'

    def build(self, env):
        return StartupVelocityCommand(self, env)

    def __post_init__(self):
        super().__post_init__()
        if not 0. <= self.rel_lateral_envs <= 1. - self.rel_standing_envs - self.rel_turning_envs:
            raise ValueError('standing, turning and lateral fractions must sum to at most one')
        for fraction in (self.startup_fraction, self.startup_turning_fraction):
            if not 0. <= fraction <= 1.:
                raise ValueError('startup fractions must be in [0, 1]')
        if not 0. < self.lateral_min_abs_velocity <= min(-self.ranges.lin_vel_y[0], self.ranges.lin_vel_y[1]):
            raise ValueError('pure lateral sampling requires nonzero ranges in both directions')
        for low, high in (self.startup_wait_range, self.startup_lateral_range, self.startup_yaw_range):
            if not 0. < low <= high:
                raise ValueError('startup ranges must be positive and ordered')
        if self.startup_lateral_range[1] > min(-self.ranges.lin_vel_y[0], self.ranges.lin_vel_y[1]):
            raise ValueError('startup lateral commands exceed the configured envelope')
        if self.startup_yaw_range[1] > self.turning_max_abs_ang_vel:
            raise ValueError('startup yaw commands exceed the turning envelope')
        if self.startup_timeout <= self.startup_wait_range[1]:
            raise ValueError('startup timeout must exceed the preparation wait')
        if min(self.startup_stable_duration, self.startup_move_duration, self.startup_contact_force,
               self.startup_linear_threshold, self.startup_angular_threshold) <= 0:
            raise ValueError('startup durations and stability thresholds must be positive')
