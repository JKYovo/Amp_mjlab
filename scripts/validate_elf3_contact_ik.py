"""Independent saved-NPZ checks for the six contact-IK edited ELF3 clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.augment_elf3_amp_motions import (
  _joint_mirror_sign, _mirrored_name, validate_clip,
)
from scripts.build_elf3_amp_v3_dataset import _load
from scripts.convert_elf3_amp_motion import _differentiate_qpos, _minimum_collision_z
from scripts.elf3_contact_ik import capture, qpos_from_arrays


def validate_dataset(source: Path, output: Path) -> dict:
  from scripts.build_elf3_amp_v3_1_dataset import (
    Elf3Geometry, SIDE_PAIRS, TURN_SOURCE, TURN_MIRROR,
  )
  pairs = list(SIDE_PAIRS) + [(TURN_SOURCE, TURN_MIRROR)]
  changed = {name for pair in pairs for name in pair}
  geometry = Elf3Geometry()
  model, data = geometry.model, geometry.data
  results = {}
  files = sorted(source.rglob('*.npz'))
  assert len(files) == len(list(output.rglob('*.npz'))) == 24
  unchanged_count = 0
  for original in files:
    path = output / original.relative_to(source)
    if original.name not in changed:
      assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(path.read_bytes()).digest(), path
      unchanged_count += 1
      continue
    validate_clip(path)
    before, after = _load(original), _load(path)
    assert np.array_equal(before['fps'], after['fps'])
    assert before['joint_pos'].shape == after['joint_pos'].shape
    assert tuple(after['body_names']) == tuple(model.body(i).name for i in range(1, model.nbody))
    source_qpos, qpos = qpos_from_arrays(before), qpos_from_arrays(after)
    fps = float(after['fps'][0])
    qvel = _differentiate_qpos(model, qpos, 1 / fps)
    assert np.max(np.abs(qvel[:, 6:] - after['joint_vel'])) < 3e-5
    # Only leg joints are changed; waist, arms and root orientation retain V3.
    assert np.array_equal(before['joint_pos'][:, :3], after['joint_pos'][:, :3])
    assert np.array_equal(before['joint_pos'][:, 15:], after['joint_pos'][:, 15:])
    # MuJoCo normalizes the free-joint quaternion during FK; allow float32 ULPs.
    assert np.max(np.abs(source_qpos[:, 3:7] - qpos[:, 3:7])) < 2e-7
    max_fk_error = max_velocity_error = 0.
    min_height = float('inf')
    v = np.empty(6)
    for i in range(len(qpos)):
      data.qpos[:] = qpos[i]
      data.qvel[:] = qvel[i]
      mujoco.mj_forward(model, data)
      max_fk_error = max(max_fk_error, float(np.max(np.abs(data.xpos[1:] - after['body_pos_w'][i]))))
      assert np.max(np.abs(data.xquat[1:] - after['body_quat_w'][i])) < 2e-6
      min_height = min(min_height, _minimum_collision_z(model, data))
      for body in range(1, model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, v, 0)
        max_velocity_error = max(max_velocity_error,
          float(np.max(np.abs(v[:3] - after['body_ang_vel_w'][i, body - 1]))),
          float(np.max(np.abs(v[3:] - after['body_lin_vel_w'][i, body - 1]))))
    assert max_fk_error < 2e-6
    assert max_velocity_error < 1e-4, (path, max_velocity_error)
    assert min_height >= .005 - 2e-6
    for name in after['joint_names']:
      joint = model.joint(str(name))
      address = int(joint.qposadr[0])
      assert np.all(qpos[:, address] >= joint.range[0] - 1e-6)
      assert np.all(qpos[:, address] <= joint.range[1] + 1e-6)
    old_p, old_r, old_h, old_c, _ = capture(geometry, source_qpos)
    new_p, new_r, _, new_c, _ = capture(geometry, qpos)
    old_v = np.gradient(old_p, 1 / fps, axis=0)
    new_v = np.gradient(new_p, 1 / fps, axis=0)
    xy_error = float(np.max(np.linalg.norm((old_p - new_p)[:, :, :2], axis=2)))
    assert xy_error < 2e-6
    old_yaw = np.arctan2(old_r[:, :, 1, 0], old_r[:, :, 0, 0])
    new_yaw = np.arctan2(new_r[:, :, 1, 0], new_r[:, :, 0, 0])
    yaw_error = np.arctan2(np.sin(new_yaw - old_yaw), np.cos(new_yaw - old_yaw))
    assert np.max(np.abs(yaw_error)) < 2e-5
    stats = {}
    for k, side in enumerate(('l', 'r')):
      mask = ((old_h[:, k] - old_h.min(axis=1)) < .018) & (np.linalg.norm(old_v[:, k, :2], axis=1) < .20)
      old_speed = np.linalg.norm(old_v[:, k, :2], axis=1)[mask]
      new_speed = np.linalg.norm(new_v[:, k, :2], axis=1)[mask]
      assert new_speed.mean() <= old_speed.mean() + 1e-5
      assert np.max(np.abs(new_speed - old_speed)) < 1e-4
      assert (new_c[mask, k] < .25).mean() <= (old_c[mask, k] < .25).mean() + .01
      stats[side] = {'same_mask_frames': int(mask.sum()),
                     'source_speed_m_s': float(old_speed.mean()),
                     'corrected_speed_m_s': float(new_speed.mean()),
                     'source_heel_fraction': float((old_c[mask, k] < .25).mean()),
                     'corrected_heel_fraction': float((new_c[mask, k] < .25).mean())}
    results[path.name] = {'fk_error_m': max_fk_error,
                          'body_velocity_error': max_velocity_error,
                          'sole_xy_error_m': xy_error,
                          'minimum_collision_z_m': min_height, 'feet': stats}
  for name, mirror in pairs:
    a, b = _load(output / 'WalkandRun' / name), _load(output / 'WalkandRun' / mirror)
    joints, bodies = list(a['joint_names']), list(a['body_names'])
    ji = [joints.index(_mirrored_name(n)) for n in joints]
    bi = [bodies.index(_mirrored_name(n)) for n in bodies]
    signs = np.array([_joint_mirror_sign(n) for n in joints], dtype=np.float32)
    assert np.array_equal(a['joint_pos'][:, ji] * signs, b['joint_pos'])
    assert np.max(np.abs(a['joint_vel'][:, ji] * signs - b['joint_vel'])) < 3e-5
    for field, sign, tolerance in [('body_pos_w', [1, -1, 1], 2e-6),
                                   ('body_quat_w', [1, -1, 1, -1], 2e-6),
                                   ('body_ang_vel_w', [-1, 1, -1], 1e-4)]:
      assert np.max(np.abs(a[field][:, bi] * np.array(sign) - b[field])) < tolerance, (mirror, field)
    # COM linear velocities differ when the asset's paired inertial offsets
    # differ. Compare velocities translated back to each link origin instead.
    origin_vel = []
    for arrays in (a, b):
      velocities = arrays['body_lin_vel_w'].astype(float).copy()
      for i, q in enumerate(qpos_from_arrays(arrays)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        velocities[i] += np.cross(arrays['body_ang_vel_w'][i], data.xpos[1:] - data.xipos[1:])
      origin_vel.append(velocities)
    assert np.max(np.abs(origin_vel[0][:, bi] * [1, -1, 1] - origin_vel[1])) < 1e-4
  assert unchanged_count == 18
  return {'unchanged_npz_count': unchanged_count, 'validated_geometric_mirror_pairs': len(pairs), 'clips': results}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=Path, default=Path('src/assets/motions/elf3/amp_v3'))
  parser.add_argument('--output', type=Path, default=Path('src/assets/motions/elf3/amp_v3_1'))
  args = parser.parse_args()
  print(json.dumps(validate_dataset(args.source, args.output), indent=2))
