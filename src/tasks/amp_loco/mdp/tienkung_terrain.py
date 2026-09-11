"""MuJoCo heightfield port of TienKung-Lab's active ELF3 GRAVEL terrain.

Parameters: MelodyAI/TienKung-Lab commit c4e7f0974b0eef90023a014ff21407cfa79bbfec,
legged_lab/terrains/terrain_generator_cfg.py (GRAVEL_TERRAINS_CFG), selected by
legged_lab/envs/elf3/walk_cfg.py. Independent implementation for mjlab.

Match IsaacLab 2.1's endpoint-inclusive grid, border discretization, signed
heights and central 2m spawn maximum. Use a MuJoCo hfield, not PhysX triangles;
random realizations and contact solvers are not bit-identical across engines.
"""
from dataclasses import dataclass
import uuid

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
    SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput,
)


@dataclass(kw_only=True)
class TienKungGravelTerrainCfg(SubTerrainCfg):
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    noise_range: tuple[float, float] = (-0.02, 0.04)
    noise_step: float = 0.02
    border_width: float = 0.25
    strip_cells: int = 2

    def height_grid(self, rng):
        shape = tuple(int(size / self.horizontal_scale) + 1 for size in self.size)
        border = int(self.border_width / self.horizontal_scale) + 1
        inner = tuple(n - 2 * border for n in shape)
        if min(inner) < 1:
            raise ValueError('Gravel tile is too small for its border')
        levels = np.arange(round(self.noise_range[0] / self.vertical_scale),
                           round(self.noise_range[1] / self.vertical_scale) + 1,
                           round(self.noise_step / self.vertical_scale))
        grid = np.zeros(shape, dtype=np.float64)
        # At downsampled_scale == horizontal_scale, upstream spline is sampled
        # at its knots: the resulting quantized heights equal these samples.
        grid[border:-border, border:-border] = rng.choice(levels, size=inner) * self.vertical_scale
        return grid

    def function(self, difficulty, spec, rng):
        del difficulty  # GRAVEL is stationary; there is no height curriculum.
        grid = self.height_grid(rng)
        # MuJoCo-Warp collects at most 50 triangle contacts per geom-hfield
        # pair. A fallen ELF3 torso can exceed this on one large hfield.
        # Split along X into narrow strips, sharing boundary vertices without
        # modifying any height or resolution. Do not patch the engine constant
        # or coarsen the terrain / change the robot's collision geometry.
        geometries = []
        for start in range(0, grid.shape[1] - 1, self.strip_cells):
            stop = min(start + self.strip_cells, grid.shape[1] - 1)
            strip = grid[:, start:stop + 1]
            # MuJoCo normalizes EACH compiled hfield independently. In
            # particular, constant border strips become all zeros. Use each
            # strip's own offset/range so compilation preserves signed heights
            # (a zero-height border must not accidentally become -2 cm).
            low, high = float(strip.min()), float(strip.max())
            amplitude = max(high - low, self.vertical_scale)
            width = (stop - start) * self.horizontal_scale
            field = spec.add_hfield(
                name='tienkung_gravel_' + uuid.uuid4().hex,
                size=[width / 2, self.size[1] / 2, amplitude, 0.1],
                nrow=strip.shape[0], ncol=strip.shape[1],
                userdata=((strip - low) / amplitude).flatten().astype(np.float32).tolist())
            geom = spec.body('terrain').add_geom(
                type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                pos=[start * self.horizontal_scale + width / 2, self.size[1] / 2, low],
                rgba=[0.45, 0.42, 0.36, 1.0], group=0)
            geometries.append(TerrainGeometry(geom=geom, hfield=field))
        x1, x2 = [int((self.size[0] / 2 + d) / self.horizontal_scale) for d in (-1, 1)]
        y1, y2 = [int((self.size[1] / 2 + d) / self.horizontal_scale) for d in (-1, 1)]
        origin = np.array([self.size[0] / 2, self.size[1] / 2, grid[x1:x2, y1:y2].max()])
        return TerrainOutput(origin=origin, geometries=geometries)


def tienkung_gravel_cfg(play=False):
    return TerrainGeneratorCfg(
        curriculum=False, size=(8.0, 8.0), border_width=20.0,
        num_rows=5 if play else 10, num_cols=5 if play else 20,
        sub_terrains={'random_rough': TienKungGravelTerrainCfg(proportion=0.2)},
    )
