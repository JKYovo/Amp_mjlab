"""BXI ELF3 AMP locomotion environment configurations."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots.elf3.elf3_constants import (
  ELF3_ACTION_SCALE,
  ELF3_AMP_BODY_NAMES,
  ELF3_ANCHOR_BODY,
  ELF3_FOOT_BODIES,
  ELF3_FOOT_SITES,
  ELF3_PHYSICAL_ROOT,
  ELF3_POLICY_ROOT,
  get_elf3_robot_cfg,
)
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from src.tasks.amp_loco.mdp.command import TurningVelocityCommandCfg


def elf3_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the ELF3 rough-terrain AMP configuration."""
  cfg = make_amp_env_cfg()

  # ELF3 uses mesh collisions on a 43 kg model. Keep generous contact buffers
  # for rough terrain while bounding Warp's per-world memory use.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 512
  cfg.sim.nconmax = 64

  cfg.scene.entities = {"robot": get_elf3_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = ELF3_PHYSICAL_ROOT

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(l_ankle_x_link|r_ankle_x_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=ELF3_PHYSICAL_ROOT, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=ELF3_PHYSICAL_ROOT, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ELF3_ACTION_SCALE

  cfg.viewer.body_name = ELF3_ANCHOR_BODY
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.3

  foot_geom_names = tuple(f"{name}_collision_0" for name in ELF3_FOOT_BODIES)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (
    ELF3_POLICY_ROOT,
    ELF3_PHYSICAL_ROOT,
  )
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  motion_base = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "..",
      "..",
      "assets",
      "motions",
      "elf3",
      "amp",
    )
  )
  motion_dir = os.path.join(motion_base, "WalkandRun")
  recovery_dir = os.path.join(motion_base, "Recovery")
  cfg.events["init_motion_loader"].params["motion_dir"] = motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = motion_dir

  for group_name in ("critic", "amp"):
    for term_name in ("body_pos_b", "body_ori_b"):
      term = cfg.observations[group_name].terms[term_name]
      term.params["anchor_cfg"].body_names = (ELF3_ANCHOR_BODY,)
      term.params["body_cfg"].body_names = ELF3_AMP_BODY_NAMES
  for term_name in ("body_lin_vel_b", "body_ang_vel_b"):
    term = cfg.observations["amp"].terms[term_name]
    term.params["anchor_cfg"].body_names = (ELF3_ANCHOR_BODY,)
    term.params["body_cfg"].body_names = ELF3_AMP_BODY_NAMES

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (
    ELF3_ANCHOR_BODY,
  )
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (
    ELF3_ANCHOR_BODY,
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (
    ELF3_POLICY_ROOT,
  )
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = ELF3_FOOT_SITES

  # Keep all reward weights, tracking widths, and collision thresholds identical
  # to the shared configuration used by G1.  Only ELF3 entity mappings differ.

  # The physical root is ELF3's high torso rather than G1's pelvis. A 0.62 m
  # threshold represents a comparable fallen/collapsed state.
  cfg.terminations["bad_base_height"].params["minimum_height"] = 0.62

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    # Match training: 40% of environments use delayed termination and
    # Recovery resets, while the remaining 60% start from WalkandRun frames.
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def elf3_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the ELF3 flat-terrain AMP configuration."""
  cfg = elf3_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 768
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 320
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # Match the stage-zero training curriculum.  Early checkpoints have not yet
    # been exposed to the full command envelope used by the base config.
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg


def elf3_amp_flat_loco_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a flat-ground locomotion-only pretraining configuration."""
  cfg = elf3_amp_flat_env_cfg(play=play)
  cfg.events["init_motion_loader"].params["recovery_dir"] = None
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 0
  return cfg


def elf3_amp_flat_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat ELF3 task backed by the isolated balanced-v2 motions."""
  cfg = elf3_amp_flat_env_cfg(play=play)

  # Explicitly cover pure-yaw commands instead of relying on the very small
  # chance that two independently sampled linear velocities are both near zero.
  # Modes are mutually exclusive: 5% standing, 15% in-place turning, and 80%
  # mixed translation/turning. Existing reward weights and widths are unchanged.
  base_twist = cfg.commands["twist"]
  assert isinstance(base_twist, UniformVelocityCommandCfg)
  cfg.commands["twist"] = TurningVelocityCommandCfg(
    entity_name=base_twist.entity_name,
    resampling_time_range=base_twist.resampling_time_range,
    heading_command=base_twist.heading_command,
    heading_control_stiffness=base_twist.heading_control_stiffness,
    rel_standing_envs=0.05,
    rel_heading_envs=base_twist.rel_heading_envs,
    init_velocity_prob=base_twist.init_velocity_prob,
    rel_turning_envs=0.15,
    turning_min_abs_ang_vel=0.3,
    turning_max_abs_ang_vel=1.0,
    debug_vis=base_twist.debug_vis,
    ranges=base_twist.ranges,
    viz=base_twist.viz,
  )

  motion_base = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "..",
      "..",
      "assets",
      "motions",
      "elf3",
      "amp_v2_balanced",
    )
  )
  motion_dir = os.path.join(motion_base, "WalkandRun")
  recovery_dir = os.path.join(motion_base, "Recovery")
  cfg.events["init_motion_loader"].params["motion_dir"] = motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = motion_dir
  return cfg


def elf3_amp_flat_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Fine-tune V2 with balanced lateral motions and stronger pure turns."""
  cfg = elf3_amp_flat_v2_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, TurningVelocityCommandCfg)
  # Mixed walking/running commands keep the V2 yaw envelope in ranges.  Only
  # the dedicated zero-linear-velocity subset is expanded to +/-2 rad/s.
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.turning_max_abs_ang_vel = 2.0

  # Preserve the original G1 width for mixed locomotion.  A moderately narrower
  # width is selected inside the existing reward only for dedicated pure turns;
  # this does not add a second reward or a second dashboard series.
  cfg.rewards["track_anchor_angular_velocity"].params["turning_std"] = 2.0

  motion_base = os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      "..",
      "..",
      "..",
      "..",
      "assets",
      "motions",
      "elf3",
      "amp_v3",
    )
  )
  motion_dir = os.path.join(motion_base, "WalkandRun")
  recovery_dir = os.path.join(motion_base, "Recovery")
  cfg.events["init_motion_loader"].params["motion_dir"] = motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = motion_dir
  return cfg
