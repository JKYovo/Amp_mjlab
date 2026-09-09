"""ELF3 29-DoF robot configuration for mjlab.

The dynamics, controller gains, limits, and action scales are derived from the
canonical HoloMotion ELF3 training asset.  The physical floating root is
``torso_link``; ``waist_z_link`` is the pelvis-semantic body used by policies.
"""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH


ELF3_XML: Path = SRC_PATH / "assets" / "robots" / "elf3" / "xmls" / "elf3.xml"
assert ELF3_XML.exists()

ELF3_PHYSICAL_ROOT = "torso_link"
ELF3_POLICY_ROOT = "waist_z_link"
ELF3_ANCHOR_BODY = "torso_link"
ELF3_FOOT_BODIES = ("l_ankle_x_link", "r_ankle_x_link")
ELF3_FOOT_SITES = ("l_foot", "r_foot")

ELF3_AMP_BODY_NAMES = (
  "waist_z_link",
  "l_hip_z_link",
  "l_knee_y_link",
  "l_ankle_x_link",
  "r_hip_z_link",
  "r_knee_y_link",
  "r_ankle_x_link",
  "l_shoulder_z_link",
  "l_elbow_y_link",
  "l_wrist_z_link",
  "r_shoulder_z_link",
  "r_elbow_y_link",
  "r_wrist_z_link",
)

ELF3_JOINT_NAMES = (
  "waist_y_joint",
  "waist_x_joint",
  "waist_z_joint",
  "l_hip_y_joint",
  "l_hip_x_joint",
  "l_hip_z_joint",
  "l_knee_y_joint",
  "l_ankle_y_joint",
  "l_ankle_x_joint",
  "r_hip_y_joint",
  "r_hip_x_joint",
  "r_hip_z_joint",
  "r_knee_y_joint",
  "r_ankle_y_joint",
  "r_ankle_x_joint",
  "l_shoulder_y_joint",
  "l_shoulder_x_joint",
  "l_shoulder_z_joint",
  "l_elbow_y_joint",
  "l_wrist_x_joint",
  "l_wrist_y_joint",
  "l_wrist_z_joint",
  "r_shoulder_y_joint",
  "r_shoulder_x_joint",
  "r_shoulder_z_joint",
  "r_elbow_y_joint",
  "r_wrist_x_joint",
  "r_wrist_y_joint",
  "r_wrist_z_joint",
)


def _get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, ELF3_XML.parent / "robot_meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Load the canonical model and remove standalone-scene components.

  HoloMotion's sim2sim MJCF is self-contained and therefore includes a floor
  and torque motors.  mjlab owns the terrain and creates position actuators
  from the articulation config, so both are removed before attachment.
  """
  spec = mujoco.MjSpec.from_file(str(ELF3_XML))
  spec.assets = _get_assets(spec.meshdir)

  for actuator in list(spec.actuators):
    spec.delete(actuator)
  floor = spec.geom("floor")
  if floor:
    spec.delete(floor)

  # The canonical sim2sim model has foot bodies but no sites.  Sites provide
  # stable, low-noise foot velocities for the slip reward.
  for side in ("l", "r"):
    ankle = spec.body(f"{side}_ankle_x_link")
    ankle.add_site(
      name=f"{side}_foot",
      pos=(0.05, 0.0, -0.05),
      size=(0.01,),
    )

  # Match mjlab's built-in robot sensor contract. Unlike G1, ELF3's physical
  # root is the torso, so the IMU site is attached directly to torso_link.
  torso = spec.body(ELF3_PHYSICAL_ROOT)
  torso.add_site(name="imu_in_torso", pos=(0.0, 0.0, 0.0), size=(0.01,))
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_torso",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_torso",
  )
  spec.add_sensor(
    name="imu_lin_acc",
    type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_in_torso",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname=ELF3_PHYSICAL_ROOT,
  )
  return spec


def _actuator(
  target_names_expr: tuple[str, ...],
  *,
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=armature,
  )


def _actuator_params(joint_name: str) -> tuple[float, float, float, float]:
  """Return Kp, Kd, effort limit, and armature for one native joint."""
  if joint_name == "waist_y_joint":
    return 108.448, 6.904, 100.0, 0.027470199309110616
  if joint_name == "waist_x_joint":
    return 162.672, 10.356, 100.0, 0.041205298963665926
  if (
    joint_name == "waist_z_joint"
    or "_hip_y_joint" in joint_name
    or "_hip_x_joint" in joint_name
    or "_knee_y_joint" in joint_name
  ):
    return 176.421, 11.231, 150.0, 0.04468796134841218
  if "_ankle_y_joint" in joint_name:
    return 33.493, 2.132, 50.0, 0.008484129312291153
  if "_ankle_x_joint" in joint_name:
    return 21.771, 1.386, 20.0, 0.00551468405298925
  if (
    "_shoulder_y_joint" in joint_name
    or "_shoulder_x_joint" in joint_name
    or "_elbow_y_joint" in joint_name
    or "_hip_z_joint" in joint_name
  ):
    return 54.224, 3.452, 50.0, 0.013735099654555308
  if "_shoulder_z_joint" in joint_name or "_wrist_" in joint_name:
    return 16.747, 1.066, 25.0, 0.0042420646561455765
  raise KeyError(f"No ELF3 actuator parameters for {joint_name!r}")


# One config per joint preserves the MJCF joint order in the policy action
# vector. Grouping equal-gain joints would silently reorder the actuators.
ELF3_ACTUATORS = tuple(
  _actuator(
    (joint_name,),
    stiffness=params[0],
    damping=params[1],
    effort_limit=params[2],
    armature=params[3],
  )
  for joint_name in ELF3_JOINT_NAMES
  for params in (_actuator_params(joint_name),)
)

ELF3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=ELF3_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)

ELF3_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.05),
  joint_pos={
    "waist_[yxz]_joint": 0.0,
    ".*_hip_y_joint": -0.3,
    ".*_hip_[xz]_joint": 0.0,
    ".*_knee_y_joint": 0.6,
    ".*_ankle_y_joint": -0.3,
    ".*_ankle_x_joint": 0.0,
    ".*_shoulder_y_joint": 0.2,
    "l_shoulder_x_joint": 0.2,
    "r_shoulder_x_joint": -0.2,
    ".*_shoulder_z_joint": 0.0,
    ".*_elbow_y_joint": 0.6,
    ".*_wrist_[xyz]_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

_FOOT_GEOM_EXPR = r"^[lr]_ankle_x_link_collision_0$"
ELF3_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision_.*",),
  condim={_FOOT_GEOM_EXPR: 3, r".*_collision_.*": 1},
  priority={_FOOT_GEOM_EXPR: 1},
  friction={_FOOT_GEOM_EXPR: (0.6,)},
)


def get_elf3_robot_cfg() -> EntityCfg:
  """Return a fresh ELF3 entity configuration."""
  return EntityCfg(
    init_state=ELF3_HOME_KEYFRAME,
    collisions=(ELF3_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ELF3_ARTICULATION,
  )


# Native ELF3 action coordinates.  These are deliberately not the G1 actor
# migration scales/signs from HoloMotion.
ELF3_ACTION_SCALE: dict[str, float] = {
  ".*_shoulder_[yx]_joint": 0.231,
  ".*_shoulder_z_joint": 0.373,
  ".*_elbow_y_joint": 0.231,
  ".*_wrist_[xyz]_joint": 0.373,
  "waist_y_joint": 0.231,
  "waist_x_joint": 0.154,
  "waist_z_joint": 0.213,
  ".*_hip_[yx]_joint": 0.213,
  ".*_hip_z_joint": 0.231,
  ".*_knee_y_joint": 0.213,
  ".*_ankle_y_joint": 0.373,
  ".*_ankle_x_joint": 0.230,
}
