"""Build V3.1 from immutable V3 using sole-trajectory constrained leg IK.

Only the two canonical side steps, one canonical turn and their exact mirrors
change. The helper validates identical source support masks and rebuilds all
FK/velocity fields. Overwrite retains the previous dataset under artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from scripts.augment_elf3_amp_motions import mirror_clip, validate_clip
from scripts.build_elf3_amp_v3_dataset import (
  _load,
  _metadata,
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


def _correct_clip(path: Path, geometry: Elf3Geometry, *, turning: bool) -> dict:
  from scripts.elf3_contact_ik import correct

  validate_clip(path)
  arrays = _load(path)
  report = correct(arrays, geometry, turning=turning)
  metadata = _metadata(arrays)
  metadata["v3_1_contact_ik"] = report
  arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
  _save(path, arrays)
  return report


def _mirror_and_rebuild(source: Path, output: Path, geometry: Elf3Geometry) -> None:
  from scripts.elf3_contact_ik import qpos_from_arrays, rebuild

  mirror_clip(source, output)
  arrays = _load(output)
  rebuild(arrays, geometry, qpos_from_arrays(arrays), float(arrays['fps'][0]))
  metadata = _metadata(arrays)
  metadata['v3_1_contact_ik'] = {
    'revision': 'contact_ik_1', 'canonical_clip': source.name,
    'method': 'sagittal joint/sole-trajectory mirror, canonical FK/velocities rebuilt',
    'note': 'left/right link COM offsets are not exactly symmetric in ELF3',
  }
  arrays['metadata'] = np.asarray(json.dumps(metadata, sort_keys=True))
  _save(output, arrays)


def _write_readme(output: Path) -> None:
  text = """# ELF3 AMP V3.1 — contact IK revision

Generated from immutable `amp_v3` by
`scripts/build_elf3_amp_v3_1_dataset.py`, using `scripts/elf3_contact_ik.py`.

- Replaces four side-step clips and two in-place-turn clips. Other 18 NPZs
  remain byte-identical to V3.
- Six-joint leg IK preserves the original collision sole-centre XY trajectory
  for **every frame**, while reducing heel/toe pitch in low, slow support.
  Original yaw and lateral roll are retained; large toe-off tilts are preserved.
- Ground clearance is reconstructed per foot. The torso is lowered 15 mm
  relative to the ground-aligned source to avoid straight-knee singularities.
- During the active turn, torso translation advances by at most 12 mm; both
  foot XY trajectories remain constrained. There is no blanket hip-angle edit.
- Joint and sole trajectories are mirrored from corrected canonical clips.
  Both sides' FK and velocities are rebuilt with the actual ELF3 inertial
  offsets, which are not perfectly symmetric. BODY linear velocity at link COM
  therefore need not be an exact signed copy on the other side.
- All FK, joint and body velocities are recomputed. Ankle body velocity and
  collision sole-centre velocity are different measurements.
- The audit uses **the same source-frame mask** before and after correction:
  relative foot height < 18 mm and sole XY speed < 0.20 m/s.
  The old comparisons using a separately selected mask per dataset are invalid.
- See `validation.json` and each corrected clip's `v3_1_contact_ik` metadata
  for trajectory errors, speed, heel fractions, COM and correction magnitudes.

These are kinematic references. The inferred support mask and lowest mesh
vertices do not measure load, centre of pressure, torque feasibility or dynamic
stability. Original small sole translations are retained, not removed.
The COM ratio uses the two-foot geometric envelope, not a force-based support
polygon. Closed-loop policy/real-robot improvement requires separate testing.

The dataset revision does not change training parameters. The accompanying
AMP sampler fix selects clips uniformly per sample, then frames uniformly
within each clip. All 24 clips are eligible, including the four straight
walking clips omitted by the previous 20-batch, restart-at-zero sampler.
"""
  (output / "README.md").write_text(text, encoding="utf-8")


def build(output: Path, *, overwrite: bool = False) -> None:
  output = output.resolve()
  if not V3_DIR.is_dir():
    raise FileNotFoundError(V3_DIR)
  # Disallow replacing an ancestor, the source, or arbitrary existing folders.
  if output == V3_DIR.resolve() or output in V3_DIR.resolve().parents:
    raise ValueError(f"Unsafe dataset destination: {output}")
  if output.exists():
    if not overwrite:
      raise FileExistsError(f"Dataset already exists: {output}")
    readme = output / "README.md"
    if not readme.is_file() or "ELF3 AMP V3.1" not in readme.read_text():
      raise ValueError(f"Refusing to replace an unrecognized dataset: {output}")

  output.parent.mkdir(parents=True, exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent))
  shutil.copytree(V3_DIR, staging, dirs_exist_ok=True)
  geometry = Elf3Geometry()
  walk_dir = staging / WALK_DIR
  reports = {}
  for source_name, mirror_name in SIDE_PAIRS:
    reports[source_name] = _correct_clip(walk_dir / source_name, geometry, turning=False)
    _mirror_and_rebuild(walk_dir / source_name, walk_dir / mirror_name, geometry)
  reports[TURN_SOURCE] = _correct_clip(walk_dir / TURN_SOURCE, geometry, turning=True)
  _mirror_and_rebuild(walk_dir / TURN_SOURCE, walk_dir / TURN_MIRROR, geometry)
  _write_readme(staging)
  for motion in sorted(staging.rglob("*.npz")):
    validate_clip(motion)

  # Test saved float32 data independently, including mirrors, before publication.
  from scripts.validate_elf3_contact_ik import validate_dataset
  reports["saved_data_checks"] = validate_dataset(V3_DIR, staging)
  (staging / "validation.json").write_text(
    json.dumps(reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  backup = None
  if output.exists():
    backup_root = PROJECT_ROOT / "artifacts" / "v3_1_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / (datetime.now().strftime("%Y%m%d_%H%M%S_") + staging.name)
    output.rename(backup)
  try:
    staging.rename(output)
  except OSError:
    if backup is not None:
      backup.rename(output)
    raise
  motions = list(output.rglob("*.npz"))
  frames = sum(int(_load(path)["joint_pos"].shape[0]) for path in motions)
  print(f"Built {output}: {len(motions)} clips / {frames} frames")
  if backup is not None:
    print(f"Previous V3.1 retained at {backup}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--overwrite", action="store_true")
  args = parser.parse_args()
  build(args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
  main()
