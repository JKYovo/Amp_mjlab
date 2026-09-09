"""Build mirrored motions and a neutral stand for the ELF3 balanced AMP set.

The mirror plane is the robot sagittal X-Z plane.  Output arrays retain the
native ELF3 joint/body ordering expected by the AMP task.  The script writes
only to ``amp_v2_balanced``; the original ``amp`` dataset is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from src.assets.robots.elf3.elf3_constants import (
  ELF3_JOINT_NAMES,
  ELF3_PHYSICAL_ROOT,
  get_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = PROJECT_ROOT / "src/assets/motions/elf3/amp_v2_balanced/WalkandRun"
MIRROR_SOURCES = (
  MOTION_DIR / "neutral_idle_turn_360_001__A103.npz",
  MOTION_DIR / "walk_sideway_right_loop_002__A023.npz",
  MOTION_DIR / "walk_sideway_right_loop_003__A025.npz",
)
STAND_OUTPUT = MOTION_DIR / "step_rotate_idle_000_002__A026.npz"
MIRROR_SUFFIX = "__mirror_left"


def _mirrored_name(name: str) -> str:
  if name.startswith("l_"):
    return "r_" + name[2:]
  if name.startswith("r_"):
    return "l_" + name[2:]
  return name


def _joint_mirror_sign(name: str) -> float:
  """Return the axial-vector sign for reflection across Y=0."""
  axis = name.rsplit("_", 2)[-2]
  if axis not in {"x", "y", "z"}:
    raise ValueError(f"Cannot infer joint axis from {name!r}")
  return 1.0 if axis == "y" else -1.0


def _metadata(data: np.lib.npyio.NpzFile) -> dict[str, object]:
  if "metadata" not in data.files:
    return {}
  value = data["metadata"]
  return json.loads(str(value.item() if value.shape == () else value))


def mirror_clip(source: Path, output: Path) -> None:
  with np.load(source, allow_pickle=False) as data:
    arrays = {name: data[name].copy() for name in data.files}
    joint_names = tuple(str(name) for name in arrays["joint_names"].tolist())
    body_names = tuple(str(name) for name in arrays["body_names"].tolist())

    joint_source_ids = np.asarray(
      [joint_names.index(_mirrored_name(name)) for name in joint_names],
      dtype=np.int64,
    )
    joint_signs = np.asarray(
      [_joint_mirror_sign(name) for name in joint_names], dtype=np.float32
    )
    body_source_ids = np.asarray(
      [body_names.index(_mirrored_name(name)) for name in body_names],
      dtype=np.int64,
    )

    arrays["joint_pos"] = (
      arrays["joint_pos"][:, joint_source_ids] * joint_signs
    ).astype(np.float32)
    arrays["joint_vel"] = (
      arrays["joint_vel"][:, joint_source_ids] * joint_signs
    ).astype(np.float32)

    body_pos = arrays["body_pos_w"][:, body_source_ids].copy()
    body_pos[..., 1] *= -1.0
    arrays["body_pos_w"] = body_pos.astype(np.float32)

    # Quaternions are WXYZ.  For S=diag(1,-1,1), R'=S R S maps
    # (w,x,y,z) -> (w,-x,y,-z).
    body_quat = arrays["body_quat_w"][:, body_source_ids].copy()
    body_quat[..., 1] *= -1.0
    body_quat[..., 3] *= -1.0
    arrays["body_quat_w"] = body_quat.astype(np.float32)

    body_lin_vel = arrays["body_lin_vel_w"][:, body_source_ids].copy()
    body_lin_vel[..., 1] *= -1.0
    arrays["body_lin_vel_w"] = body_lin_vel.astype(np.float32)

    # Angular velocity is an axial vector: det(S) S = diag(-1,1,-1).
    body_ang_vel = arrays["body_ang_vel_w"][:, body_source_ids].copy()
    body_ang_vel[..., 0] *= -1.0
    body_ang_vel[..., 2] *= -1.0
    arrays["body_ang_vel_w"] = body_ang_vel.astype(np.float32)

    metadata = _metadata(data)
    metadata.update(
      {
        "augmentation": "sagittal_mirror",
        "mirrored_from": source.name,
        "mirror_plane": "world_y_zero",
      }
    )
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))

  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **arrays)


def _nominal_joint_positions() -> np.ndarray:
  values = []
  for name in ELF3_JOINT_NAMES:
    if name in {"l_hip_y_joint", "r_hip_y_joint"}:
      value = -0.3
    elif name in {"l_knee_y_joint", "r_knee_y_joint"}:
      value = 0.6
    elif name in {"l_ankle_y_joint", "r_ankle_y_joint"}:
      value = -0.3
    elif name in {"l_shoulder_y_joint", "r_shoulder_y_joint"}:
      value = 0.2
    elif name == "l_shoulder_x_joint":
      value = 0.2
    elif name == "r_shoulder_x_joint":
      value = -0.2
    elif name in {"l_elbow_y_joint", "r_elbow_y_joint"}:
      value = 0.6
    else:
      value = 0.0
    values.append(value)
  return np.asarray(values, dtype=np.float32)


def _foot_mesh_min_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  minimum = np.inf
  for geom_name in (
    "l_ankle_x_link_collision_0",
    "r_ankle_x_link_collision_0",
  ):
    geom = model.geom(geom_name)
    mesh_id = int(model.geom_dataid[geom.id])
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
    rotation = data.geom_xmat[geom.id].reshape(3, 3)
    world_vertices = vertices @ rotation.T + data.geom_xpos[geom.id]
    minimum = min(minimum, float(world_vertices[:, 2].min()))
  return float(minimum)


def build_nominal_stand(output: Path, *, frames: int, fps: float) -> None:
  if frames < 2:
    raise ValueError("A standing clip needs at least two frames")
  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)

  root_joint = model.joint("retwist_floating_root")
  root_qpos_adr = int(root_joint.qposadr[0])
  data.qpos[root_qpos_adr : root_qpos_adr + 7] = (
    0.0,
    0.0,
    1.05,
    1.0,
    0.0,
    0.0,
    0.0,
  )
  joint_pos = _nominal_joint_positions()
  for name, value in zip(ELF3_JOINT_NAMES, joint_pos, strict=True):
    data.qpos[int(model.joint(name).qposadr[0])] = float(value)

  mujoco.mj_forward(model, data)
  desired_clearance = 0.005
  data.qpos[root_qpos_adr + 2] += desired_clearance - _foot_mesh_min_z(model, data)
  mujoco.mj_forward(model, data)

  body_names = tuple(model.body(body_id).name for body_id in range(1, model.nbody))
  if body_names[0] != ELF3_PHYSICAL_ROOT:
    raise ValueError(f"Unexpected physical root: {body_names[0]!r}")
  body_pos = data.xpos[1:].astype(np.float32)
  body_quat = data.xquat[1:].astype(np.float32)

  root_body_id = int(model.body(ELF3_PHYSICAL_ROOT).id)
  left_foot_id = int(model.site("l_foot").id)
  right_foot_id = int(model.site("r_foot").id)
  foot_midpoint = 0.5 * (data.site_xpos[left_foot_id] + data.site_xpos[right_foot_id])
  com = data.subtree_com[root_body_id]

  metadata = {
    "body_names": list(body_names),
    "fps": float(fps),
    "frames": int(frames),
    "joint_names": list(ELF3_JOINT_NAMES),
    "physical_root_body": ELF3_PHYSICAL_ROOT,
    "source_format": "retwist_nominal",
    "source_reference": "TWIST ELF3_DEFAULT_JOINT_ANGLES",
    "construction": "one nominal FK frame repeated with zero velocities",
    "foot_mesh_clearance_m": desired_clearance,
    "whole_body_com_w": com.tolist(),
    "foot_site_midpoint_w": foot_midpoint.tolist(),
    "com_minus_foot_site_midpoint_w": (com - foot_midpoint).tolist(),
  }
  arrays = {
    "fps": np.asarray([fps], dtype=np.float32),
    "joint_pos": np.repeat(joint_pos[None, :], frames, axis=0),
    "joint_vel": np.zeros((frames, len(ELF3_JOINT_NAMES)), dtype=np.float32),
    "body_pos_w": np.repeat(body_pos[None, :, :], frames, axis=0),
    "body_quat_w": np.repeat(body_quat[None, :, :], frames, axis=0),
    "body_lin_vel_w": np.zeros((frames, len(body_names), 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((frames, len(body_names), 3), dtype=np.float32),
    "joint_names": np.asarray(ELF3_JOINT_NAMES),
    "body_names": np.asarray(body_names),
    "root_body_name": np.asarray(ELF3_PHYSICAL_ROOT),
    "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output, **arrays)


def validate_clip(path: Path) -> None:
  with np.load(path, allow_pickle=False) as data:
    required = (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
      "joint_names",
      "body_names",
      "root_body_name",
    )
    missing = [name for name in required if name not in data.files]
    if missing:
      raise ValueError(f"{path}: missing fields {missing}")
    frame_count = int(data["joint_pos"].shape[0])
    for name in required[:6]:
      if data[name].shape[0] != frame_count:
        raise ValueError(f"{path}: inconsistent frame count for {name}")
      if not np.isfinite(data[name]).all():
        raise ValueError(f"{path}: non-finite values in {name}")
    quat_norm = np.linalg.norm(data["body_quat_w"], axis=-1)
    if float(np.max(np.abs(quat_norm - 1.0))) > 1e-4:
      raise ValueError(f"{path}: body quaternion norm error")
    if tuple(data["joint_names"].tolist()) != ELF3_JOINT_NAMES:
      raise ValueError(f"{path}: unexpected joint order")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stand-frames", type=int, default=500)
  parser.add_argument("--fps", type=float, default=50.0)
  args = parser.parse_args()

  outputs = []
  for source in MIRROR_SOURCES:
    output = source.with_name(source.stem + MIRROR_SUFFIX + ".npz")
    mirror_clip(source, output)
    outputs.append(output)
  build_nominal_stand(STAND_OUTPUT, frames=args.stand_frames, fps=args.fps)
  outputs.append(STAND_OUTPUT)
  for output in outputs:
    validate_clip(output)
    print(output)


if __name__ == "__main__":
  main()
