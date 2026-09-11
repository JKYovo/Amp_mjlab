"""Contact-aware ELF3 reference editing, with fixed-frame geometric audits.

This is kinematic IK, not a proof of dynamic feasibility or contact forces.
The reference point is the actual collision sole centre, not the ankle axis.
"""

from __future__ import annotations

import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scripts.convert_elf3_amp_motion import _differentiate_qpos, _minimum_collision_z
from src.assets.robots.elf3.elf3_constants import ELF3_PHYSICAL_ROOT


SOLE = np.array([0.03, 0.0, -0.041])


def ramp(x, lo, hi):
  x = np.clip((x - lo) / (hi - lo), 0, 1)
  return x * x * (3 - 2 * x)


def qpos_from_arrays(arrays):
  root = list(arrays['body_names']).index(ELF3_PHYSICAL_ROOT)
  return np.c_[arrays['body_pos_w'][:, root], arrays['body_quat_w'][:, root],
               arrays['joint_pos']].astype(float)


def foot_geometry(geometry):
  model = geometry.model
  feet = []
  for side in ('l', 'r'):
    geom = model.geom(f'{side}_ankle_x_link_collision_0')
    mesh = geometry.feet[side]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, geom.quat)
    vertices = mesh.vertices @ rotation.reshape(3, 3).T + geom.pos
    joint_names = [f'{side}_{part}_joint' for part in
                   ('hip_y', 'hip_x', 'hip_z', 'knee_y', 'ankle_y', 'ankle_x')]
    feet.append((int(model.body(f'{side}_ankle_x_link').id), vertices,
                 [int(model.joint(name).qposadr[0]) for name in joint_names],
                 np.array([model.joint(name).range for name in joint_names])))
  return feet


def capture(geometry, qpos):
  model, data = geometry.model, geometry.data
  feet = foot_geometry(geometry)
  n = len(qpos)
  positions = np.empty((n, 2, 3))
  rotations = np.empty((n, 2, 3, 3))
  heights = np.empty((n, 2))
  contact = np.empty((n, 2))
  com = np.empty(n)
  for i, q in enumerate(qpos):
    data.qpos[:] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for k, (body, vertices, _, _) in enumerate(feet):
      r = data.xmat[body].reshape(3, 3)
      rotations[i, k] = r
      positions[i, k] = data.xpos[body] + r @ SOLE
      world = vertices @ r.T + data.xpos[body]
      z = world[:, 2]
      heights[i, k] = z.min()
      fraction = (vertices[:, 0] - vertices[:, 0].min()) / np.ptp(vertices[:, 0])
      contact[i, k] = fraction[z <= z.min() + .0015].mean()
    com[i] = geometry.com_support_ratio()
  return positions, rotations, heights, contact, com


def rebuild(arrays, geometry, qpos, fps):
  """Match existing NPZ BODY velocity convention; rebuild every FK field."""
  model, data = geometry.model, geometry.data
  qvel = _differentiate_qpos(model, qpos, 1 / fps)
  n = len(qpos)
  bp = np.empty((n, model.nbody - 1, 3), dtype=np.float32)
  bq = np.empty((n, model.nbody - 1, 4), dtype=np.float32)
  bv = np.empty_like(bp)
  bw = np.empty_like(bp)
  velocity = np.empty(6)
  for i in range(n):
    data.qpos[:] = qpos[i]
    data.qvel[:] = qvel[i]
    mujoco.mj_forward(model, data)
    bp[i], bq[i] = data.xpos[1:], data.xquat[1:]
    for body in range(1, model.nbody):
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                              body, velocity, 0)
      bw[i, body - 1], bv[i, body - 1] = velocity[:3], velocity[3:]
  arrays.update(joint_pos=qpos[:, 7:].astype(np.float32),
                joint_vel=qvel[:, 6:].astype(np.float32), body_pos_w=bp,
                body_quat_w=bq, body_lin_vel_w=bv, body_ang_vel_w=bw)


def correct(arrays, geometry, *, turning=False):
  model, data = geometry.model, geometry.data
  feet = foot_geometry(geometry)
  fps = float(np.asarray(arrays['fps']).reshape(-1)[0])
  source = qpos_from_arrays(arrays)
  n = len(source)
  pos, rot, heights, contacts, com = capture(geometry, source)
  source_vel = np.gradient(pos, 1 / fps, axis=0)
  speed = np.linalg.norm(source_vel[:, :, :2], axis=2)
  rel_height = heights - heights.min(axis=1, keepdims=True)
  # Same ORIGINAL mask for both versions: lifting/swinging frames cannot enter
  # the comparison merely because a modified mesh changed which foot is low.
  support = (rel_height < .018) & (speed < .20)
  support_weight = (1 - ramp(rel_height, .007, .035)) * (1 - ramp(speed, .12, .50))
  # Preserve pronounced toe-off pivots; flatten only near-level support.
  tilt = np.arccos(np.clip(rot[:, :, 2, 2], -1, 1))
  support_weight *= 1 - ramp(tilt, .18, .38)
  weight = gaussian_filter1d(support_weight, 1.2, axis=0, mode='nearest')
  target_rot = rot.copy()
  for k in range(2):
    angles = Rotation.from_matrix(rot[:, k]).as_euler('xyz')
    # Correct heel/toe pitch only. Lateral roll is needed for side steps and
    # flattening it as well can drive the ankle-roll joint against its limit.
    angles[:, 1] = 0
    flat = Rotation.from_euler('xyz', angles).as_matrix()
    delta = Rotation.from_matrix(flat @ rot[:, k].transpose(0, 2, 1)).as_rotvec()
    target_rot[:, k] = Rotation.from_rotvec(delta * weight[:, k, None]).as_matrix() @ rot[:, k]

  # Align the original minimum envelope once. Each foot subsequently preserves
  # that envelope, even while its sole is rotated about the tracked sole point.
  shift = gaussian_filter1d(.005 - heights.min(axis=1), 2, mode='nearest')
  shift += max(0., float(np.max(.005 - heights.min(axis=1) - shift)))
  target = pos.copy()
  for k, (_, vertices, _, _) in enumerate(feet):
    lower_offset = np.min(np.einsum('nij,vj->nvi', target_rot[:, k], vertices - SOLE)[:, :, 2], axis=1)
    target[:, k, 2] = heights[:, k] + shift - lower_offset
    target[:, k, 2] = gaussian_filter1d(target[:, k, 2], .7, mode='nearest')

  result = source.copy()
  # A small uniform lowering keeps near-straight source knees away from the
  # extension singularity when a tilted sole is placed flat on the floor.
  result[:, 2] += shift - .015
  if turning:
    root = list(arrays['body_names']).index(ELF3_PHYSICAL_ROOT)
    yaw_speed = np.abs(arrays['body_ang_vel_w'][:, root, 2])
    turn_weight = gaussian_filter1d(ramp(yaw_speed, .12, .82), 2, mode='nearest')
    heading = Rotation.from_quat(source[:, [4, 5, 6, 3]]).as_matrix()[:, :, 0]
    heading[:, 2] = 0
    heading /= np.linalg.norm(heading, axis=1, keepdims=True)
    # Move the torso forward by at most 12 mm while both feet retain their
    # original XY trajectories. The leg IK provides the posture change.
    result[:, :3] += .012 * turn_weight[:, None] * heading
  else:
    turn_weight = np.ones(n)

  max_residual = 0.
  for i in range(n):
    data.qpos[:] = result[i]
    for k, (body, _, addresses, limits) in enumerate(feet):
      def residual(joints):
        data.qpos[addresses] = joints
        mujoco.mj_fwdPosition(model, data)
        r = data.xmat[body].reshape(3, 3)
        position_error = data.xpos[body] + r @ SOLE - target[i, k]
        orientation_error = Rotation.from_matrix(target_rot[i, k] @ r.T).as_rotvec()
        return np.r_[position_error, .20 * orientation_error]
      initial = source[i, addresses]
      solution = least_squares(residual, initial,
                               bounds=(limits[:, 0] + 1e-6, limits[:, 1] - 1e-6),
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=80)
      max_residual = max(max_residual, float(np.max(np.abs(residual(solution.x)))))
      result[i, addresses] = solution.x
    if i % 50 == 0:
      print(f'Contact IK {i + 1}/{n}', flush=True)

  # A constant final vertical offset preserves all computed velocities.
  minimum = float('inf')
  for q in result:
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    minimum = min(minimum, _minimum_collision_z(model, data))
  result[:, 2] += max(0., .005 - minimum)
  after_pos, after_rot, after_heights, after_contacts, after_com = capture(geometry, result)
  after_vel = np.gradient(after_pos, 1 / fps, axis=0)
  xy_error = np.linalg.norm((after_pos - pos)[:, :, :2], axis=2)
  source_qv = _differentiate_qpos(model, source, 1 / fps)
  after_qv = _differentiate_qpos(model, result, 1 / fps)
  metrics = {
    'method': 'sole-centre trajectory constrained 6-DoF leg IK',
    'revision': 'contact_ik_1',
    'velocity_measure': 'finite difference of collision sole centre; identical source mask',
    'support_mask': 'source relative minimum Z < 18mm and sole XY speed < 0.20m/s',
    'ik_max_weighted_residual': max_residual,
    'sole_xy_max_error_m': float(xy_error.max()),
    'max_joint_change_rad': float(np.max(np.abs(result[:, 7:] - source[:, 7:]))),
    'max_added_joint_speed_rad_s': float(np.max(np.abs(after_qv[:, 6:] - source_qv[:, 6:]))),
    'root_lowering_relative_to_ground_aligned_source_m': .015,
    'turn_root_forward_max_m': .012 if turning else 0.,
    'source_joint_speed_max_rad_s': float(np.max(np.abs(source_qv[:, 6:]))),
    'corrected_joint_speed_max_rad_s': float(np.max(np.abs(after_qv[:, 6:]))),
    'added_joint_acceleration_max_rad_s2': float(np.max(np.abs(np.gradient(after_qv[:, 6:] - source_qv[:, 6:], 1 / fps, axis=0)))),
    'minimum_foot_collision_z_m': float(after_heights.min()),
    'com_envelope_ratio_source': float(com[turn_weight > .5].mean()),
    'com_envelope_ratio_corrected': float(after_com[turn_weight > .5].mean()),
    'feet': {},
  }
  for k, side in enumerate(('l', 'r')):
    mask = support[:, k]
    old = np.linalg.norm(source_vel[:, k, :2], axis=1)[mask]
    new = np.linalg.norm(after_vel[:, k, :2], axis=1)[mask]
    metrics['feet'][side] = {
      'support_frames': int(mask.sum()),
      'source_sole_xy_speed_mean_m_s': float(old.mean()),
      'corrected_sole_xy_speed_mean_m_s': float(new.mean()),
      'source_heel_fraction': float((contacts[mask, k] < .25).mean()),
      'corrected_heel_fraction': float((after_contacts[mask, k] < .25).mean()),
      'source_toe_fraction': float((contacts[mask, k] > .75).mean()),
      'corrected_toe_fraction': float((after_contacts[mask, k] > .75).mean()),
    }
  if max_residual > 2e-4 or xy_error.max() > 2e-4:
    raise ValueError(f'Contact IK failed trajectory tolerance: {metrics}')
  if np.max(np.abs(result[:, 7:] - source[:, 7:])) > .35:
    raise ValueError(f'Contact IK exceeds joint correction budget: {metrics}')
  if np.max(np.abs(after_qv[:, 6:] - source_qv[:, 6:])) > 4.:
    raise ValueError(f'Contact IK correction is too abrupt: {metrics}')
  for side in ('l', 'r'):
    foot = metrics['feet'][side]
    if foot['corrected_sole_xy_speed_mean_m_s'] > foot['source_sole_xy_speed_mean_m_s'] + 1e-4:
      raise ValueError(f'Contact IK increased sole translation speed: {metrics}')
  rebuild(arrays, geometry, result, fps)
  return metrics
