# ELF3 AMP V4 — walking-only subset

Built by `scripts/build_elf3_amp_v4_dataset.py` from the repaired contact-IK
`amp_v3_1` dataset. Source unchanged. See `manifest.json` for SHA-256 hashes,
frame counts and the inclusion decision for all 24 source clips.

- Removed 8 jogging references: 4 straight and 4 arc jogging (with mirrors).
- Retained 16 clips byte-for-byte: 4 straight walks, 4 arc walks, 4 side steps,
  2 in-place turns, 1 standing clip and 1 recovery clip. No time/pose edits.
- `WalkandRun/` is a compatibility directory name; no running references remain.
  Reset and AMP both use this filtered dataset.
- AMP sampling is uniform per clip (expected 1/16), then uniform per frame.
  Removing running changes the remaining clips' relative expert weights.
- These are ground-level kinematic references, not terrain-retargeted motions.
  Removing running does not mathematically prohibit an occasional running gait;
  closed-loop training must be evaluated separately.
