"""Create the walking-only V4 subset, leaving every retained clip unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/assets/motions/elf3/amp_v3_1'
OUTPUT = ROOT / 'src/assets/motions/elf3/amp_v4'
REMOVED = frozenset({
    'arc_jog_left_loop_002__A029.npz',
    'arc_jog_left_loop_002__A029__mirror_right.npz',
    'jog_arc_cw_loop_004__A045.npz',
    'jog_arc_cw_loop_004__A045__mirror_ccw.npz',
    'jog_backward_loop_002__A022.npz',
    'jog_backward_loop_003__A023.npz',
    'jog_forward_loop_003__A021.npz',
    'jog_forward_loop_003__A022.npz',
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be independent directories')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output}')
    files = sorted(source.rglob('*.npz'))
    if len(files) != 24 or {p.name for p in files if p.name in REMOVED} != REMOVED:
        raise ValueError('Expected the reviewed 24-clip V3.1 inventory')
    output.mkdir(parents=True)
    entries = []
    for path in files:
        relative = path.relative_to(source)
        kept = path.name not in REMOVED
        with np.load(path, allow_pickle=False) as data:
            frames = int(data['joint_pos'].shape[0])
            fps = float(np.asarray(data['fps']).reshape(-1)[0])
        digest = sha256(path)
        entries.append({'file': relative.as_posix(), 'kept': kept,
                        'sha256': digest, 'frames': frames, 'fps': fps})
        if kept:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            assert sha256(target) == digest
    report = {'source_dataset': source.name, 'dataset': 'amp_v4',
              'kept_clips': 16, 'removed_clips': 8,
              'kept_frames': sum(e['frames'] for e in entries if e['kept']),
              'entries': entries}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'README.md').write_text('''# ELF3 AMP V4 — walking-only subset

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
''')
    print(f'Built {output}: 16 kept / 8 removed; {report["kept_frames"]} frames')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.source, args.output_dir)
