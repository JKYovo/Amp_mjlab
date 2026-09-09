# ELF3 balanced AMP motion data v2

This directory is an independent successor to `../amp`.  The original dataset
is intentionally left byte-for-byte unchanged.  The original dataset contains
17 WalkandRun clips plus one G1-derived recovery clip.  V2 contains 19
WalkandRun clips plus the same recovery clip, for exactly 20 motions.

Compared with `../amp`, v2 replaces exactly five WalkandRun slots and adds two
new clips:

1. The old asymmetric static clip is replaced, under the same filename, by a
   500-frame (10 s at 50 Hz) repeated ReTWIST ELF3 nominal pose with zero joint
   and body velocities.
2. The old left/right side-walk pair is replaced by the native HoloMotion
   `220713/walk_sideway_right_loop_002__A023` clip and its exact sagittal
   mirror.  Their mean body-frame lateral speeds are -0.656 and +0.656 m/s.
   The native clip's mean whole-body COM offset is 0.009 m behind the two-foot
   geometry center, instead of 0.097-0.117 m for the old pair.
3. A second, lower-speed symmetric side-walk pair is added from native
   HoloMotion `220714/walk_sideway_right_loop_003__A025`.  Its mean body-frame
   lateral speed is -0.364 m/s and its mean COM is 0.012 m ahead of the
   two-foot geometry center.
4. The two old same-direction idle-turn clips are replaced by the native
   HoloMotion `230104/neutral_idle_turn_360_001__A103` clip and its exact
   sagittal mirror.  Their mean body-frame yaw rates are -0.534 and +0.534
   rad/s.  The native clip's mean COM offset is 0.001 m behind the two-foot
geometry center.

The total of 20 motions matches the configured 5 learning epochs x 4 PPO
minibatches.  The current AMP loader cycles one motion per expert minibatch, so
this avoids silently repeating the first motions within every update.

Sagittal mirroring swaps all left/right joints and bodies, applies the axial
joint sign convention, reflects positions and linear velocities across Y=0,
and reflects quaternions and angular velocities with their proper parity.  Run
`scripts/augment_elf3_amp_motions.py` to regenerate the two mirrored clips and
the nominal stand after the two native source clips have been converted.

This dataset should be used for a fresh training run.  Resuming a policy whose
discriminator was trained on `../amp` mixes two expert distributions and does
not reliably remove the learned left/right or rear-COM bias.
