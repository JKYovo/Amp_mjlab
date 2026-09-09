"""Convert HoloMotion ELF3 references into this project's AMP NPZ format.

Two source formats are supported:

* ``holomotion``: ``ref_dof_pos`` plus physical torso root pose in XYZW.
* ``g1_amp``: an existing G1 AMP NPZ. This is intended for recovery clips;
  it maps semantic joints and uses G1's torso pose as ELF3's physical root.

Body poses and velocities are recomputed with the canonical ELF3 MuJoCo model,
so output ordering exactly matches the runtime entity rather than relying on an
Isaac Lab or Isaac Gym body order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro

from src.assets.robots.elf3.elf3_constants import (
  ELF3_JOINT_NAMES,
  ELF3_PHYSICAL_ROOT,
  ELF3_XML,
)


G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

G1_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)

# target ELF3 joint -> (source G1 joint, sign). The waist chain runs in the
# opposite direction because ELF3 is rooted at the torso.
G1_TO_ELF3 = {
  "waist_y_joint": ("waist_pitch_joint", -1.0),
  "waist_x_joint": ("waist_roll_joint", -1.0),
  "waist_z_joint": ("waist_yaw_joint", -1.0),
  "l_hip_y_joint": ("left_hip_pitch_joint", 1.0),
  "l_hip_x_joint": ("left_hip_roll_joint", 1.0),
  "l_hip_z_joint": ("left_hip_yaw_joint", 1.0),
  "l_knee_y_joint": ("left_knee_joint", 1.0),
  "l_ankle_y_joint": ("left_ankle_pitch_joint", 1.0),
  "l_ankle_x_joint": ("left_ankle_roll_joint", 1.0),
  "r_hip_y_joint": ("right_hip_pitch_joint", 1.0),
  "r_hip_x_joint": ("right_hip_roll_joint", 1.0),
  "r_hip_z_joint": ("right_hip_yaw_joint", 1.0),
  "r_knee_y_joint": ("right_knee_joint", 1.0),
  "r_ankle_y_joint": ("right_ankle_pitch_joint", 1.0),
  "r_ankle_x_joint": ("right_ankle_roll_joint", 1.0),
  "l_shoulder_y_joint": ("left_shoulder_pitch_joint", 1.0),
  "l_shoulder_x_joint": ("left_shoulder_roll_joint", 1.0),
  "l_shoulder_z_joint": ("left_shoulder_yaw_joint", 1.0),
  "l_elbow_y_joint": ("left_elbow_joint", 1.0),
  "l_wrist_x_joint": ("left_wrist_roll_joint", 1.0),
  "l_wrist_y_joint": ("left_wrist_pitch_joint", 1.0),
  "l_wrist_z_joint": ("left_wrist_yaw_joint", 1.0),
  "r_shoulder_y_joint": ("right_shoulder_pitch_joint", 1.0),
  "r_shoulder_x_joint": ("right_shoulder_roll_joint", 1.0),
  "r_shoulder_z_joint": ("right_shoulder_yaw_joint", 1.0),
  "r_elbow_y_joint": ("right_elbow_joint", 1.0),
  "r_wrist_x_joint": ("right_wrist_roll_joint", 1.0),
  "r_wrist_y_joint": ("right_wrist_pitch_joint", 1.0),
  "r_wrist_z_joint": ("right_wrist_yaw_joint", 1.0),
}


def _string_list(data: np.lib.npyio.NpzFile, key: str) -> tuple[str, ...] | None:
  if key not in data.files:
    return None
  return tuple(str(value) for value in data[key].tolist())


def _continuous_unit_quaternions(quaternions: np.ndarray) -> np.ndarray:
  result = np.asarray(quaternions, dtype=np.float64).copy()
  norms = np.linalg.norm(result, axis=1, keepdims=True)
  if np.any(norms < 1.0e-8):
    raise ValueError("Root motion contains a zero quaternion")
  result /= norms
  for index in range(1, len(result)):
    if np.dot(result[index - 1], result[index]) < 0.0:
      result[index] *= -1.0
  return result


def _load_holomotion(
  source: Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict]:
  with np.load(source, allow_pickle=False) as data:
    metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
    fps = float(metadata.get("motion_fps", 50.0))
    source_joint_names = tuple(metadata.get("output_dof_order", ()))
    if not source_joint_names:
      raise ValueError(f"{source}: metadata.output_dof_order is required")

    source_pos = np.asarray(data["ref_dof_pos"], dtype=np.float64)
    indexes = [source_joint_names.index(name) for name in ELF3_JOINT_NAMES]
    joint_pos = source_pos[:, indexes]
    root_pos = np.asarray(data["ref_global_translation"], dtype=np.float64)
    root_xyzw = _continuous_unit_quaternions(data["ref_global_rotation_quat"])
    root_wxyz = root_xyzw[:, [3, 0, 1, 2]]
  return fps, root_pos, root_wxyz, joint_pos, metadata


def _load_g1_amp(
  source: Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, dict]:
  with np.load(source, allow_pickle=False) as data:
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    source_joint_names = _string_list(data, "joint_names") or G1_JOINT_NAMES
    source_body_names = _string_list(data, "body_names") or G1_BODY_NAMES
    source_joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)

    joint_columns = []
    for target_name in ELF3_JOINT_NAMES:
      source_name, sign = G1_TO_ELF3[target_name]
      joint_columns.append(sign * source_joint_pos[:, source_joint_names.index(source_name)])
    joint_pos = np.stack(joint_columns, axis=1)

    torso_index = source_body_names.index("torso_link")
    root_pos = np.asarray(data["body_pos_w"][:, torso_index], dtype=np.float64)
    root_wxyz = _continuous_unit_quaternions(data["body_quat_w"][:, torso_index])

  metadata = {
    "source_robot": "unitree_g1_29dof",
    "source_physical_root": "pelvis",
    "source_pose_used_as_target_root": "torso_link",
    "waist_mapping_sign": -1.0,
  }
  return fps, root_pos, root_wxyz, joint_pos, metadata


def _portable_source_metadata(metadata: dict) -> dict:
  """Keep useful provenance without embedding machine-local absolute paths."""
  result = {
    key: metadata[key]
    for key in (
      "contract_version",
      "motion_fps",
      "motion_key",
      "output_dof_order",
      "root_quaternion_order",
      "source_robot",
      "source_physical_root",
      "source_pose_used_as_target_root",
      "waist_mapping_sign",
    )
    if key in metadata
  }
  selection = metadata.get("selection")
  if isinstance(selection, dict):
    result["selection"] = {
      key: selection[key]
      for key in ("category", "motion_id", "motion_name", "source_motion_id")
      if key in selection
    }
  if "source_sha256" in metadata:
    result["source_sha256"] = metadata["source_sha256"]
  return result


def _differentiate_qpos(
  model: mujoco.MjModel,
  qpos: np.ndarray,
  dt: float,
) -> np.ndarray:
  qvel = np.empty((len(qpos), model.nv), dtype=np.float64)
  if len(qpos) < 2:
    qvel.fill(0.0)
    return qvel
  mujoco.mj_differentiatePos(model, qvel[0], dt, qpos[0], qpos[1])
  mujoco.mj_differentiatePos(model, qvel[-1], dt, qpos[-2], qpos[-1])
  for index in range(1, len(qpos) - 1):
    mujoco.mj_differentiatePos(
      model,
      qvel[index],
      2.0 * dt,
      qpos[index - 1],
      qpos[index + 1],
    )
  return qvel


def _minimum_collision_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  """Return the lowest point of the robot's collision geometry."""
  minimum = float("inf")
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name == "floor" or model.geom_group[geom_id] != 3:
      continue
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh_id = model.geom_dataid[geom_id]
      vertex_start = model.mesh_vertadr[mesh_id]
      vertex_count = model.mesh_vertnum[mesh_id]
      vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
      world_z = (
        vertices @ data.geom_xmat[geom_id].reshape(3, 3)[2]
        + data.geom_xpos[geom_id, 2]
      )
      minimum = min(minimum, float(world_z.min()))
    else:
      minimum = min(
        minimum,
        float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id]),
      )
  if not np.isfinite(minimum):
    raise ValueError("ELF3 model has no collision geometry for ground alignment")
  return minimum


def _ground_align_recovery(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  qpos: np.ndarray,
  clearance: float = 0.005,
) -> np.ndarray:
  """Place every recovery pose on the floor using actual ELF3 mesh geometry."""
  shifts = np.empty(len(qpos), dtype=np.float64)
  data.qvel.fill(0.0)
  for frame_index, frame_qpos in enumerate(qpos):
    data.qpos[:] = frame_qpos
    mujoco.mj_forward(model, data)
    shifts[frame_index] = clearance - _minimum_collision_z(model, data)
  qpos[:, 2] += shifts
  return shifts


def convert_motion(
  source: Path,
  destination: Path,
  source_format: Literal["holomotion", "g1_amp"],
) -> dict:
  """Convert one motion file and return summary metadata."""
  if source_format == "holomotion":
    fps, root_pos, root_wxyz, joint_pos, source_metadata = _load_holomotion(source)
  else:
    fps, root_pos, root_wxyz, joint_pos, source_metadata = _load_g1_amp(source)

  spec = mujoco.MjSpec.from_file(str(ELF3_XML))
  model = spec.compile()
  data = mujoco.MjData(model)
  body_names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
    for index in range(1, model.nbody)
  )
  joint_names = tuple(
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
    for index in range(1, model.njnt)
  )
  if body_names[0] != ELF3_PHYSICAL_ROOT:
    raise ValueError(f"Unexpected physical root: {body_names[0]}")
  if joint_names != ELF3_JOINT_NAMES:
    raise ValueError("ELF3 MJCF joint order no longer matches ELF3_JOINT_NAMES")
  if not (len(root_pos) == len(root_wxyz) == len(joint_pos)):
    raise ValueError("Root and joint arrays have different frame counts")

  # Keep references inside the physical model. Report any source overshoot so
  # clipping is visible rather than silently corrupting the motion contract.
  joint_ranges = model.jnt_range[1:]
  clipped_joint_pos = np.clip(joint_pos, joint_ranges[:, 0], joint_ranges[:, 1])
  max_joint_overshoot = float(np.max(np.abs(clipped_joint_pos - joint_pos)))
  clipped_value_count = int(np.count_nonzero(clipped_joint_pos != joint_pos))
  joint_pos = clipped_joint_pos

  qpos = np.empty((len(joint_pos), model.nq), dtype=np.float64)
  qpos[:, :3] = root_pos
  qpos[:, 3:7] = root_wxyz
  qpos[:, 7:] = joint_pos
  ground_shifts = None
  if source_format == "g1_amp":
    ground_shifts = _ground_align_recovery(model, data, qpos)
  qvel = _differentiate_qpos(model, qpos, 1.0 / fps)

  body_pos_w = np.empty((len(qpos), len(body_names), 3), dtype=np.float32)
  body_quat_w = np.empty((len(qpos), len(body_names), 4), dtype=np.float32)
  body_lin_vel_w = np.empty_like(body_pos_w)
  body_ang_vel_w = np.empty_like(body_pos_w)
  object_velocity = np.empty(6, dtype=np.float64)

  for frame_index, (frame_qpos, frame_qvel) in enumerate(zip(qpos, qvel)):
    data.qpos[:] = frame_qpos
    data.qvel[:] = frame_qvel
    mujoco.mj_forward(model, data)
    body_pos_w[frame_index] = data.xpos[1:]
    body_quat_w[frame_index] = data.xquat[1:]
    for body_index in range(1, model.nbody):
      mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_index,
        object_velocity,
        0,
      )
      body_ang_vel_w[frame_index, body_index - 1] = object_velocity[:3]
      body_lin_vel_w[frame_index, body_index - 1] = object_velocity[3:]

  arrays = (
    joint_pos,
    qvel[:, 6:],
    body_pos_w,
    body_quat_w,
    body_lin_vel_w,
    body_ang_vel_w,
  )
  if not all(np.isfinite(array).all() for array in arrays):
    raise ValueError(f"{source}: converted output contains NaN or Inf")

  conversion_metadata = {
    "source_file": source.name,
    "source_format": source_format,
    "fps": fps,
    "frames": len(qpos),
    "physical_root_body": ELF3_PHYSICAL_ROOT,
    "joint_names": joint_names,
    "body_names": body_names,
    "clipped_joint_value_count": clipped_value_count,
    "max_joint_overshoot_rad": max_joint_overshoot,
    "ground_alignment": None
    if ground_shifts is None
    else {
      "clearance_m": 0.005,
      "min_shift_m": float(ground_shifts.min()),
      "max_shift_m": float(ground_shifts.max()),
    },
    "source_metadata": _portable_source_metadata(source_metadata),
  }
  destination.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    destination,
    fps=np.asarray([fps], dtype=np.float64),
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=qvel[:, 6:].astype(np.float32),
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    joint_names=np.asarray(joint_names),
    body_names=np.asarray(body_names),
    root_body_name=np.asarray(ELF3_PHYSICAL_ROOT),
    metadata=np.asarray(json.dumps(conversion_metadata, sort_keys=True)),
  )
  return conversion_metadata


def main(
  input_path: str,
  output_dir: str,
  source_format: Literal["holomotion", "g1_amp"] = "holomotion",
  limit: int | None = None,
) -> None:
  """Convert one NPZ or every NPZ below a directory."""
  source = Path(input_path)
  if source.is_file():
    files = [source]
  elif source.is_dir():
    files = sorted(source.rglob("*.npz"))
  else:
    raise FileNotFoundError(source)
  if limit is not None:
    files = files[:limit]
  if not files:
    raise ValueError(f"No NPZ files found below {source}")

  destination_dir = Path(output_dir)
  total_frames = 0
  for index, motion_file in enumerate(files, start=1):
    destination = destination_dir / motion_file.name
    metadata = convert_motion(motion_file, destination, source_format)
    total_frames += int(metadata["frames"])
    print(
      f"[{index}/{len(files)}] {motion_file.name}: "
      f"{metadata['frames']} frames, max clip "
      f"{metadata['max_joint_overshoot_rad']:.6g} rad"
    )
  print(f"Converted {len(files)} clips / {total_frames} frames into {destination_dir}")


if __name__ == "__main__":
  tyro.cli(main)
