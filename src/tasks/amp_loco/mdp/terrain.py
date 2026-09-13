import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


def elf3_v4_rough_terrain_cfg(play: bool = False) -> TerrainGeneratorCfg:
  """Native random roughness, matching the approved V4 terrain preview.

  One heightfield per tile, without the former 20-cm-wide hfield strips.
  The 20-cm sampling grid is coarser than the old 10-cm GRAVEL grid to
  reduce triangle candidates when the torso contacts the ground.
  """
  return TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5 if play else 10,
    num_cols=5 if play else 20,
    sub_terrains={
      "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0,
        noise_range=(0.0, 0.06),
        noise_step=0.02,
        horizontal_scale=0.2,
        downsampled_scale=0.2,
        border_width=0.25,
      ),
    },
  )


RANDOM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.4),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.6,
      noise_range=(0.02, 0.05),
      noise_step=0.02,
      border_width=0.25,
    ),
    # "wave_terrain": terrain_gen.HfWaveTerrainCfg(
    #   proportion=0.1,
    #   amplitude_range=(0.0, 0.2),
    #   num_waves=4,
    #   border_width=0.25,
    # ),
  },
  add_lights=True,
)
