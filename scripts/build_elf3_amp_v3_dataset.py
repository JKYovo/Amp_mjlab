"""Build the isolated ELF3 AMP V3 dataset from V2 and retargeted side steps.

V3 keeps V2 immutable, replaces the crossing side-step references, trims the
long idle tails from the in-place turn, and mirrors every one-direction arc.
Side-step collision height is corrected against the canonical ELF3 MJCF before
all body states and velocities are recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from scripts.augment_elf3_amp_motions import mirror_clip, validate_clip
from scripts.convert_elf3_amp_motion import (
  _differentiate_qpos,
  _minimum_collision_z,
)
from src.assets.robots.elf3.elf3_constants import (
  ELF3_JOINT_NAMES,
  ELF3_PHYSICAL_ROOT,
  ELF3_XML,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_DIR = PROJECT_ROOT / "src/assets/motions/elf3/amp_v2_balanced"
DEFAULT_OUTPUT = PROJECT_ROOT / "src/assets/motions/elf3/amp_v3"
OLD_SIDE_FILES = (
  "walk_sideway_right_loop_002__A023.npz",
  "walk_sideway_right_loop_002__A023__mirror_left.npz",
  "walk_sideway_right_loop_003__A025.npz",
  "walk_sideway_right_loop_003__A025__mirror_left.npz",
)
ARC_MIRRORS = {
  "arc_jog_left_loop_002__A029.npz": "arc_jog_left_loop_002__A029__mirror_right.npz",
  "arc_walk_left_loop_001__A029.npz": "arc_walk_left_loop_001__A029__mirror_right.npz",
  "jog_arc_cw_loop_004__A045.npz": "jog_arc_cw_loop_004__A045__mirror_ccw.npz",
  "walk_arc_cw_loop_002__A046.npz": "walk_arc_cw_loop_002__A046__mirror_ccw.npz",
}
TURN_SOURCE = "neutral_idle_turn_360_001__A103.npz"
TURN_MIRROR = "neutral_idle_turn_360_001__A103__mirror_left.npz"


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
  value = arrays.get("metadata")
  if value is None:
    return {}
  return json.loads(str(value.item() if value.shape == () else value))


def _load(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    return {name: data[name].copy() for name in data.files}


def _save(path: Path, arrays: dict[str, np.ndarray]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path, **arrays)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _slice_clip(source: Path, output: Path, start: int, stop: int) -> None:
  arrays = _load(source)
  frame_count = int(arrays["joint_pos"].shape[0])
  if not 0 <= start < stop <= frame_count:
    raise ValueError(f"Invalid slice [{start}:{stop}] for {source} ({frame_count})")
  for name in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    arrays[name] = arrays[name][start:stop].copy()
  metadata = _metadata(arrays)
  metadata.update(
    {
      "v3_trim": {"source_frames": frame_count, "start": start, "stop": stop},
      "frames": stop - start,
    }
  )
  arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
  _save(output, arrays)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
  if window < 1 or window % 2 == 0:
    raise ValueError("Smoothing window must be a positive odd number")
  radius = window // 2
  padded = np.pad(values, (radius, radius), mode="edge")
  return np.convolve(padded, np.ones(window) / window, mode="valid")


def _replay_elf3(
  arrays: dict[str, np.ndarray],
  *,
  ground_clearance: float,
  smoothing_window: int,
) -> dict[str, float]:
  """Ground-align root Z and rebuild velocities/body state using ELF3 FK."""
  joint_names = tuple(str(name) for name in arrays["joint_names"].tolist())
  body_names = tuple(str(name) for name in arrays["body_names"].tolist())
  if joint_names != ELF3_JOINT_NAMES:
    raise ValueError("Side-step input does not use the ELF3 joint order")
  root_index = body_names.index(ELF3_PHYSICAL_ROOT)
  fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])

  model = mujoco.MjSpec.from_file(str(ELF3_XML)).compile()
  data = mujoco.MjData(model)
  model_body_names = tuple(model.body(index).name for index in range(1, model.nbody))
  if body_names != model_body_names:
    raise ValueError("Side-step input does not use the canonical ELF3 body order")

  frame_count = int(arrays["joint_pos"].shape[0])
  qpos = np.empty((frame_count, model.nq), dtype=np.float64)
  qpos[:, :3] = arrays["body_pos_w"][:, root_index]
  qpos[:, 3:7] = arrays["body_quat_w"][:, root_index]
  qpos[:, 7:] = arrays["joint_pos"]

  raw_shift = np.empty(frame_count, dtype=np.float64)
  data.qvel.fill(0.0)
  for frame_index, frame_qpos in enumerate(qpos):
    data.qpos[:] = frame_qpos
    mujoco.mj_forward(model, data)
    raw_shift[frame_index] = ground_clearance - _minimum_collision_z(model, data)
  smooth_shift = _moving_average(raw_shift, smoothing_window)
  # Preserve the requested minimum clearance after smoothing.
  smooth_shift += max(0.0, float(np.max(raw_shift - smooth_shift)))
  qpos[:, 2] += smooth_shift
  qvel = _differentiate_qpos(model, qpos, 1.0 / fps)

  body_pos = np.empty((frame_count, len(body_names), 3), dtype=np.float32)
  body_quat = np.empty((frame_count, len(body_names), 4), dtype=np.float32)
  body_lin_vel = np.empty_like(body_pos)
  body_ang_vel = np.empty_like(body_pos)
  object_velocity = np.empty(6, dtype=np.float64)
  min_collision_z = float("inf")
  for frame_index, (frame_qpos, frame_qvel) in enumerate(zip(qpos, qvel, strict=True)):
    data.qpos[:] = frame_qpos
    data.qvel[:] = frame_qvel
    mujoco.mj_forward(model, data)
    min_collision_z = min(min_collision_z, _minimum_collision_z(model, data))
    body_pos[frame_index] = data.xpos[1:]
    body_quat[frame_index] = data.xquat[1:]
    for body_index in range(1, model.nbody):
      mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_index,
        object_velocity,
        0,
      )
      body_ang_vel[frame_index, body_index - 1] = object_velocity[:3]
      body_lin_vel[frame_index, body_index - 1] = object_velocity[3:]

  arrays["joint_pos"] = qpos[:, 7:].astype(np.float32)
  arrays["joint_vel"] = qvel[:, 6:].astype(np.float32)
  arrays["body_pos_w"] = body_pos
  arrays["body_quat_w"] = body_quat
  arrays["body_lin_vel_w"] = body_lin_vel
  arrays["body_ang_vel_w"] = body_ang_vel
  return {
    "clearance_m": ground_clearance,
    "smoothing_window_frames": smoothing_window,
    "min_shift_m": float(smooth_shift.min()),
    "max_shift_m": float(smooth_shift.max()),
    "verified_min_collision_z_m": min_collision_z,
  }


def _prepare_side(
  source: Path,
  output: Path,
  *,
  start: int,
  stop: int,
  direction: str,
) -> None:
  arrays = _load(source)
  source_frames = int(arrays["joint_pos"].shape[0])
  if not 0 <= start < stop <= source_frames:
    raise ValueError(f"Invalid side-step slice [{start}:{stop}] for {source}")
  for name in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    arrays[name] = arrays[name][start:stop].copy()
  alignment = _replay_elf3(
    arrays, ground_clearance=0.005, smoothing_window=9
  )
  metadata = _metadata(arrays)
  metadata.update(
    {
      "v3_role": f"amass_side_step_{direction}",
      "v3_source_sha256": _sha256(source),
      "v3_trim": {"source_frames": source_frames, "start": start, "stop": stop},
      "v3_ground_alignment": alignment,
      "frames": stop - start,
    }
  )
  arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
  _save(output, arrays)


def _write_readme(output: Path, left: Path, right: Path) -> None:
  text = f"""# ELF3 AMP V3 motion set

This directory is generated by `scripts/build_elf3_amp_v3_dataset.py` and does
not modify `amp_v2_balanced`.

- Replaced the four V2 crossing side-step clips with two AMASS motions and
  their exact sagittal mirrors.
- Left source trim: frames `[30:105]`; right source trim: `[38:102]`.
- Side clips use 9-frame smoothed root-height correction and at least 5 mm
  clearance against the canonical ELF3 collision geometry; FK/body velocities
  are then recomputed.
- In-place turn is trimmed from 319 to 150 frames (`[60:210]`) while retaining
  lead-in/out transition, then mirrored exactly.
- All four one-direction walking/running arcs receive exact mirrors.
- Recovery data and all other V2 clips are unchanged.

Input SHA-256:

- left: `{_sha256(left)}`
- right: `{_sha256(right)}`
"""
  (output / "README.md").write_text(text, encoding="utf-8")


def build(left: Path, right: Path, output: Path) -> None:
  left = left.resolve()
  right = right.resolve()
  output = output.resolve()
  if not left.is_file() or not right.is_file():
    raise FileNotFoundError("Both converted side-step NPZ inputs are required")
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite existing V3 dataset: {output}")
  staging = output.with_name(output.name + ".building")
  if staging.exists():
    raise FileExistsError(f"Remove stale staging directory first: {staging}")

  shutil.copytree(V2_DIR, staging)
  walk_dir = staging / "WalkandRun"
  for name in OLD_SIDE_FILES:
    (walk_dir / name).unlink()

  _prepare_side(
    left,
    walk_dir / "amass_side_step_left__B22.npz",
    start=30,
    stop=105,
    direction="left",
  )
  _prepare_side(
    right,
    walk_dir / "amass_side_step_right__B23.npz",
    start=38,
    stop=102,
    direction="right",
  )
  mirror_clip(
    walk_dir / "amass_side_step_left__B22.npz",
    walk_dir / "amass_side_step_left__B22__mirror_right.npz",
  )
  mirror_clip(
    walk_dir / "amass_side_step_right__B23.npz",
    walk_dir / "amass_side_step_right__B23__mirror_left.npz",
  )

  turn_source = walk_dir / TURN_SOURCE
  _slice_clip(turn_source, turn_source, 60, 210)
  (walk_dir / TURN_MIRROR).unlink()
  mirror_clip(turn_source, walk_dir / TURN_MIRROR)

  for source_name, mirror_name in ARC_MIRRORS.items():
    mirror_clip(walk_dir / source_name, walk_dir / mirror_name)

  _write_readme(staging, left, right)
  for motion in sorted(staging.rglob("*.npz")):
    validate_clip(motion)
  staging.rename(output)
  motions = list(output.rglob("*.npz"))
  frames = sum(int(_load(path)["joint_pos"].shape[0]) for path in motions)
  print(f"Built {output}: {len(motions)} clips / {frames} frames")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--side-left", type=Path, required=True)
  parser.add_argument("--side-right", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  build(args.side_left, args.side_right, args.output_dir)


if __name__ == "__main__":
  main()
