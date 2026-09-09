# ELF3 AMP motion data

The 17 locomotion clips in `WalkandRun/` are generated from HoloMotion's
ELF3-native, 50 Hz `elf3_boneseed_filtered_all44134` references with
`scripts/convert_elf3_amp_motion.py`. The converter recomputes all 30 body
poses and world-frame velocities using this repository's canonical ELF3
MuJoCo model. Every output file records its joint/body ordering and identifies
`torso_link` as the physical root.

`Recovery/fallAndGetUp1_subject1.npz` is provisional: it maps the repository's
G1 recovery clip to ELF3 by semantic joint name, reverses the three waist
coordinates because the kinematic chain is rooted in the opposite direction,
clips out-of-range joints, and aligns every frame to the floor using ELF3's
actual collision meshes. Replace this clip when a native ELF3 get-up reference
becomes available.

The robot MJCF and controller parameters originate from the canonical
HoloMotion ELF3 29-DoF training/sim2sim asset. Its source XML SHA-256 is:

```text
e369cccf96a87618e2cc690200daeb325ea47fb3f39dabb91c9cffcce5b3daeb
```
