# ELF3 AMP V3.1 — contact IK revision

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
