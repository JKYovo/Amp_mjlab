"""Build ELF3 AMP V3.1 with support-aware side-step and turn posture fixes.

V3.1 is derived from V3 without modifying it.  The two canonical AMASS side
steps are corrected only while a foot is supporting the robot: the ankle pitch
is chosen from the ELF3 collision mesh so the heel and toe sit at comparable
heights.  The in-place turn receives a small, speed-gated hip/ankle posture
correction that moves the COM toward mid-foot while retaining the original
turn timing and approximate foot pitch.  Exact sagittal mirrors are rebuilt
after each canonical clip is corrected.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from scripts.augment_elf3_amp_motions import mirror_clip, validate_clip
from scripts.build_elf3_amp_v3_dataset import (
  _load,
  _metadata,
  _moving_average,
  _replay_elf3,
  _save,
)
from src.assets.robots.elf3.elf3_constants import (
  ELF3_JOINT_NAMES,
  ELF3_PHYSICAL_ROOT,
  ELF3_XML,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_DIR = PROJECT_ROOT / "src/assets/motions/elf3/amp_v3"
DEFAULT_OUTPUT = PROJECT_ROOT / "src/assets/motions/elf3/amp_v3_1"
WALK_DIR = "WalkandRun"

SIDE_PAIRS = (
  (
    "amass_side_step_left__B22.npz",
    "amass_side_step_left__B22__mirror_right.npz",
  ),
  (
    "amass_side_step_right__B23.npz",
    "amass_side_step_right__B23__mirror_left.npz",
  ),
)
TURN_SOURCE = "neutral_idle_turn_360_001__A103.npz"
TURN_MIRROR = "neutral_idle_turn_360_001__A103__mirror_left.npz"

SIDE_MAX_ANKLE_CORRECTION = 0.12
TURN_HIP_CORRECTION = 0.025
TURN_MAX_HEEL_ANKLE_CORRECTION = 0.025
GROUND_CLEARANCE = 0.005
SMOOTHING_WINDOW = 9


@dataclass(frozen=True)
class FootMesh:
  geom_id: int
  vertices: np.ndarray
  forward_fraction: np.ndarray


class Elf3Geometry:
  """Small FK helper used to derive and validate the corrections."""

  def __init__(self) -> None:
    self.model = mujoco.MjModel.from_xml_path(str(ELF3_XML))
    self.data = mujoco.MjData(self.model)
    self.root_body_id = int(self.model.body(ELF3_PHYSICAL_ROOT).id)
    self.joint_qpos_addresses = {
      name: int(self.model.joint(name).qposadr[0]) for name in ELF3_JOINT_NAMES
    }
    self.feet: dict[str, FootMesh] = {}
    for side in ("l", "r"):
      geom = self.model.geom(f"{side}_ankle_x_link_collision_0")
      mesh_id = int(self.model.geom_dataid[geom.id])
      vertex_address = int(self.model.mesh_vertadr[mesh_id])
      vertex_count = int(self.model.mesh_vertnum[mesh_id])
      vertices = self.model.mesh_vert[
        vertex_address : vertex_address + vertex_count
      ].copy()
      # The collision mesh's local +Z axis maps to ELF3's forward (+X) axis.
      forward = vertices[:, 2]
      forward_fraction = (forward - forward.min()) / np.ptp(forward)
      self.feet[side] = FootMesh(
        geom_id=int(geom.id),
        vertices=vertices,
        forward_fraction=forward_fraction,
      )

  def set_frame(
    self,
    arrays: dict[str, np.ndarray],
    frame_index: int,
    *,
    ankle_delta: tuple[str, float] | None = None,
  ) -> None:
    body_names = tuple(str(name) for name in arrays["body_names"].tolist())
    root_index = body_names.index(ELF3_PHYSICAL_ROOT)
    self.data.qpos[:3] = arrays["body_pos_w"][frame_index, root_index]
    self.data.qpos[3:7] = arrays["body_quat_w"][frame_index, root_index]
    self.data.qpos[7:] = arrays["joint_pos"][frame_index]
    if ankle_delta is not None:
      side, delta = ankle_delta
      address = self.joint_qpos_addresses[f"{side}_ankle_y_joint"]
      self.data.qpos[address] += delta
    self.data.qvel.fill(0.0)
    mujoco.mj_forward(self.model, self.data)

  def foot_state(self, side: str) -> tuple[float, float, float]:
    """Return minimum Z, contact location, and heel/toe height difference."""
    foot = self.feet[side]
    rotation = self.data.geom_xmat[foot.geom_id].reshape(3, 3)
    world_vertices = (
      foot.vertices @ rotation.T + self.data.geom_xpos[foot.geom_id]
    )
    height = world_vertices[:, 2]
    near_ground = height <= height.min() + 0.0015
    contact_location = float(foot.forward_fraction[near_ground].mean())
    heel_height = float(
      np.quantile(height[foot.forward_fraction < 0.2], 0.1)
    )
    toe_height = float(
      np.quantile(height[foot.forward_fraction > 0.8], 0.1)
    )
    return float(height.min()), contact_location, heel_height - toe_height

  def com_support_ratio(self) -> float:
    """Return whole-body COM position with heel=0 and toe=1."""
    root_rotation = self.data.xmat[self.root_body_id].reshape(3, 3)
    forward = root_rotation[:, 0].copy()
    forward[2] = 0.0
    forward /= np.linalg.norm(forward)
    support_projection = []
    for foot in self.feet.values():
      rotation = self.data.geom_xmat[foot.geom_id].reshape(3, 3)
      world_vertices = (
        foot.vertices @ rotation.T + self.data.geom_xpos[foot.geom_id]
      )
      support_projection.append(world_vertices @ forward)
    support_projection_array = np.concatenate(support_projection)
    com_projection = float(self.data.subtree_com[self.root_body_id] @ forward)
    return float(
      (com_projection - support_projection_array.min())
      / np.ptp(support_projection_array)
    )


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
  scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
  return scaled * scaled * (3.0 - 2.0 * scaled)


def _side_contact_metrics(
  arrays: dict[str, np.ndarray], geometry: Elf3Geometry
) -> dict[str, float | int]:
  body_names = tuple(str(name) for name in arrays["body_names"].tolist())
  contacts: list[float] = []
  for frame_index in range(len(arrays["joint_pos"])):
    geometry.set_frame(arrays, frame_index)
    foot_states = [geometry.foot_state(side) for side in ("l", "r")]
    lowest = min(state[0] for state in foot_states)
    for side_index, side in enumerate(("l", "r")):
      ankle_index = body_names.index(f"{side}_ankle_x_link")
      horizontal_speed = float(
        np.linalg.norm(
          arrays["body_lin_vel_w"][frame_index, ankle_index, :2]
        )
      )
      if foot_states[side_index][0] - lowest <= 0.012 and horizontal_speed < 0.35:
        contacts.append(foot_states[side_index][1])
  contact_array = np.asarray(contacts)
  return {
    "support_samples": int(len(contact_array)),
    "mean_contact_location_heel_0_toe_1": float(contact_array.mean()),
    "heel_quarter_fraction": float(np.mean(contact_array < 0.25)),
    "toe_quarter_fraction": float(np.mean(contact_array > 0.75)),
  }


def _correct_side_clip(path: Path, geometry: Elf3Geometry) -> None:
  arrays = _load(path)
  before = _side_contact_metrics(arrays, geometry)
  body_names = tuple(str(name) for name in arrays["body_names"].tolist())
  frame_count = len(arrays["joint_pos"])
  foot_heights = np.empty((frame_count, 2), dtype=np.float64)
  foot_speeds = np.empty((frame_count, 2), dtype=np.float64)
  raw_corrections = np.zeros((frame_count, 2), dtype=np.float64)
  candidates = np.linspace(0.0, SIDE_MAX_ANKLE_CORRECTION, 49)

  for frame_index in range(frame_count):
    geometry.set_frame(arrays, frame_index)
    for side_index, side in enumerate(("l", "r")):
      foot_heights[frame_index, side_index] = geometry.foot_state(side)[0]
      ankle_index = body_names.index(f"{side}_ankle_x_link")
      foot_speeds[frame_index, side_index] = np.linalg.norm(
        arrays["body_lin_vel_w"][frame_index, ankle_index, :2]
      )
      best_error = float("inf")
      best_correction = 0.0
      for candidate in candidates:
        geometry.set_frame(
          arrays, frame_index, ankle_delta=(side, float(candidate))
        )
        error = abs(geometry.foot_state(side)[2])
        if error < best_error:
          best_error = error
          best_correction = float(candidate)
      raw_corrections[frame_index, side_index] = best_correction

  relative_height = foot_heights - foot_heights.min(axis=1, keepdims=True)
  height_weight = np.clip((0.020 - relative_height) / 0.015, 0.0, 1.0)
  speed_weight = np.clip((0.55 - foot_speeds) / 0.40, 0.0, 1.0)
  support_weight = height_weight * speed_weight

  applied_corrections: dict[str, np.ndarray] = {}
  for side_index, side in enumerate(("l", "r")):
    correction = _moving_average(
      raw_corrections[:, side_index] * support_weight[:, side_index],
      SMOOTHING_WINDOW,
    )
    joint_index = ELF3_JOINT_NAMES.index(f"{side}_ankle_y_joint")
    arrays["joint_pos"][:, joint_index] += correction.astype(np.float32)
    applied_corrections[side] = correction

  alignment = _replay_elf3(
    arrays,
    ground_clearance=GROUND_CLEARANCE,
    smoothing_window=SMOOTHING_WINDOW,
  )
  after = _side_contact_metrics(arrays, geometry)
  metadata = _metadata(arrays)
  metadata["v3_1_side_support_fix"] = {
    "method": "support-weighted collision-mesh heel/toe leveling",
    "max_candidate_ankle_y_rad": SIDE_MAX_ANKLE_CORRECTION,
    "smoothing_window_frames": SMOOTHING_WINDOW,
    "applied_l_ankle_y_rad": {
      "min": float(applied_corrections["l"].min()),
      "mean": float(applied_corrections["l"].mean()),
      "max": float(applied_corrections["l"].max()),
    },
    "applied_r_ankle_y_rad": {
      "min": float(applied_corrections["r"].min()),
      "mean": float(applied_corrections["r"].mean()),
      "max": float(applied_corrections["r"].max()),
    },
    "contact_before": before,
    "contact_after": after,
    "ground_alignment": alignment,
  }
  arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
  _save(path, arrays)


def _turn_com_metrics(
  arrays: dict[str, np.ndarray], geometry: Elf3Geometry, active: np.ndarray
) -> dict[str, float]:
  ratios = np.empty(len(arrays["joint_pos"]), dtype=np.float64)
  for frame_index in range(len(ratios)):
    geometry.set_frame(arrays, frame_index)
    ratios[frame_index] = geometry.com_support_ratio()
  return {
    "all_frames_mean_heel_0_toe_1": float(ratios.mean()),
    "active_frames_mean_heel_0_toe_1": float(ratios[active].mean()),
    "p05": float(np.quantile(ratios, 0.05)),
    "p50": float(np.quantile(ratios, 0.50)),
    "p95": float(np.quantile(ratios, 0.95)),
  }


def _turn_contact_metrics(
  arrays: dict[str, np.ndarray], geometry: Elf3Geometry, active: np.ndarray
) -> dict[str, dict[str, float | int]]:
  metrics: dict[str, dict[str, float | int]] = {}
  for side_index, side in enumerate(("l", "r")):
    all_contacts: list[float] = []
    active_contacts: list[float] = []
    for frame_index in range(len(arrays["joint_pos"])):
      geometry.set_frame(arrays, frame_index)
      states = [geometry.foot_state(name) for name in ("l", "r")]
      lowest = min(state[0] for state in states)
      state = states[side_index]
      if state[0] - lowest <= 0.012:
        all_contacts.append(state[1])
        if active[frame_index]:
          active_contacts.append(state[1])

    all_array = np.asarray(all_contacts)
    active_array = np.asarray(active_contacts)
    metrics[side] = {
      "support_samples": int(len(all_array)),
      "mean_contact_location_heel_0_toe_1": float(all_array.mean()),
      "heel_quarter_fraction": float(np.mean(all_array < 0.25)),
      "toe_quarter_fraction": float(np.mean(all_array > 0.75)),
      "active_support_samples": int(len(active_array)),
      "active_mean_contact_location_heel_0_toe_1": float(active_array.mean()),
      "active_heel_quarter_fraction": float(np.mean(active_array < 0.25)),
      "active_toe_quarter_fraction": float(np.mean(active_array > 0.75)),
    }
  return metrics


def _correct_turn_clip(path: Path, geometry: Elf3Geometry) -> None:
  arrays = _load(path)
  body_names = tuple(str(name) for name in arrays["body_names"].tolist())
  root_index = body_names.index(ELF3_PHYSICAL_ROOT)
  yaw_speed = np.abs(arrays["body_ang_vel_w"][:, root_index, 2])
  weight = _smoothstep(yaw_speed, 0.12, 0.82)
  weight = _moving_average(weight, SMOOTHING_WINDOW)
  active = weight > 0.5
  com_before = _turn_com_metrics(arrays, geometry, active)
  contact_before = _turn_contact_metrics(arrays, geometry, active)

  hip_correction = (TURN_HIP_CORRECTION * weight).astype(np.float32)
  for side in ("l", "r"):
    hip_index = ELF3_JOINT_NAMES.index(f"{side}_hip_y_joint")
    arrays["joint_pos"][:, hip_index] += hip_correction

  # The source turn intentionally pivots one foot near the toe, but the other
  # foot spends most of the active turn on its heel.  Preserve the toe pivot and
  # correct only a low, supporting foot whose contact is behind 40% of the sole.
  frame_count = len(arrays["joint_pos"])
  foot_heights = np.empty((frame_count, 2), dtype=np.float64)
  contact_locations = np.empty((frame_count, 2), dtype=np.float64)
  for frame_index in range(frame_count):
    geometry.set_frame(arrays, frame_index)
    for side_index, side in enumerate(("l", "r")):
      state = geometry.foot_state(side)
      foot_heights[frame_index, side_index] = state[0]
      contact_locations[frame_index, side_index] = state[1]

  relative_height = foot_heights - foot_heights.min(axis=1, keepdims=True)
  support_weight = np.clip((0.020 - relative_height) / 0.015, 0.0, 1.0)
  heel_weight = np.clip((0.40 - contact_locations) / 0.40, 0.0, 1.0)
  applied_ankle_corrections: dict[str, np.ndarray] = {}
  for side_index, side in enumerate(("l", "r")):
    ankle_correction = _moving_average(
      TURN_MAX_HEEL_ANKLE_CORRECTION
      * support_weight[:, side_index]
      * weight
      * np.sqrt(heel_weight[:, side_index]),
      SMOOTHING_WINDOW,
    )
    ankle_index = ELF3_JOINT_NAMES.index(f"{side}_ankle_y_joint")
    arrays["joint_pos"][:, ankle_index] += ankle_correction.astype(np.float32)
    applied_ankle_corrections[side] = ankle_correction

  alignment = _replay_elf3(
    arrays,
    ground_clearance=GROUND_CLEARANCE,
    smoothing_window=SMOOTHING_WINDOW,
  )
  com_after = _turn_com_metrics(arrays, geometry, active)
  contact_after = _turn_contact_metrics(arrays, geometry, active)
  metadata = _metadata(arrays)
  metadata["v3_1_turn_posture_fix"] = {
    "method": "yaw-speed-gated hip correction plus support-aware heel reduction",
    "peak_hip_correction_rad": TURN_HIP_CORRECTION,
    "max_heel_ankle_correction_rad": TURN_MAX_HEEL_ANKLE_CORRECTION,
    "applied_l_ankle_y_rad": {
      "min": float(applied_ankle_corrections["l"].min()),
      "mean": float(applied_ankle_corrections["l"].mean()),
      "max": float(applied_ankle_corrections["l"].max()),
    },
    "applied_r_ankle_y_rad": {
      "min": float(applied_ankle_corrections["r"].min()),
      "mean": float(applied_ankle_corrections["r"].mean()),
      "max": float(applied_ankle_corrections["r"].max()),
    },
    "active_frame_count": int(active.sum()),
    "weight_first": float(weight[0]),
    "weight_last": float(weight[-1]),
    "com_before": com_before,
    "com_after": com_after,
    "contact_before": contact_before,
    "contact_after": contact_after,
    "ground_alignment": alignment,
  }
  arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
  _save(path, arrays)


def _write_readme(output: Path) -> None:
  text = """# ELF3 AMP V3.1 motion set

This directory is generated from `amp_v3` by
`scripts/build_elf3_amp_v3_1_dataset.py`. The V3 directory is not modified.

- The two canonical AMASS side-step clips use support-aware ankle-pitch
  correction derived from the ELF3 collision mesh. The correction is faded out
  for fast or lifted feet, so swing and crossover timing stay unchanged.
- Their left/right counterparts are regenerated as exact sagittal mirrors.
- The canonical in-place turn uses a maximum 0.025 rad hip-pitch adjustment to
  shift active-turn COM toward mid-foot. A separate maximum 0.025 rad ankle
  adjustment is applied only to a low supporting foot that is loading its heel;
  the original toe-pivot foot and turn speed are retained.
- The mirrored turn is regenerated from the corrected canonical turn.
- Joint/body velocities and all body FK fields are recomputed. Corrected clips
  are aligned to at least 5 mm collision clearance.
- Every other V3 motion is copied byte-for-byte.

The detailed before/after contact and COM measurements are stored in each
corrected NPZ's JSON metadata.
"""
  (output / "README.md").write_text(text, encoding="utf-8")


def build(output: Path, *, overwrite: bool = False) -> None:
  output = output.resolve()
  if not V3_DIR.is_dir():
    raise FileNotFoundError(V3_DIR)
  if output.exists() and not overwrite:
    raise FileExistsError(f"Refusing to overwrite existing V3.1 dataset: {output}")
  staging = output.with_name(output.name + ".building")
  if staging.exists():
    raise FileExistsError(f"Remove stale staging directory first: {staging}")

  shutil.copytree(V3_DIR, staging)
  geometry = Elf3Geometry()
  walk_dir = staging / WALK_DIR
  for source_name, mirror_name in SIDE_PAIRS:
    _correct_side_clip(walk_dir / source_name, geometry)
    (walk_dir / mirror_name).unlink()
    mirror_clip(walk_dir / source_name, walk_dir / mirror_name)

  _correct_turn_clip(walk_dir / TURN_SOURCE, geometry)
  (walk_dir / TURN_MIRROR).unlink()
  mirror_clip(walk_dir / TURN_SOURCE, walk_dir / TURN_MIRROR)

  _write_readme(staging)
  for motion in sorted(staging.rglob("*.npz")):
    validate_clip(motion)
  if output.exists():
    shutil.rmtree(output)
  staging.rename(output)
  motions = list(output.rglob("*.npz"))
  frames = sum(int(_load(path)["joint_pos"].shape[0]) for path in motions)
  print(f"Built {output}: {len(motions)} clips / {frames} frames")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()
  build(args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
  main()
