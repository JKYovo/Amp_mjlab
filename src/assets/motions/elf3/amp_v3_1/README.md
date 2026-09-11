# ELF3 AMP V3.1 motion set

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
