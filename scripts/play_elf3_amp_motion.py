"""Play or record ELF3 AMP NPZ motion files with the canonical MuJoCo asset.

The player deliberately resolves both joints and the physical root by name.
This makes the preview independent of Isaac Lab, mjlab, or MuJoCo array order.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw
import tyro

from src.assets.robots.elf3.elf3_constants import (
  ELF3_JOINT_NAMES,
  ELF3_PHYSICAL_ROOT,
  ELF3_XML,
)


DEFAULT_MOTION = (
  Path(__file__).parents[1]
  / "src/assets/motions/elf3/amp/WalkandRun/walk_forward_loop_002__A022.npz"
)


def _string_tuple(data: np.lib.npyio.NpzFile, key: str) -> tuple[str, ...]:
  if key not in data.files:
    raise ValueError(f"AMP file is missing required field {key!r}")
  return tuple(str(value) for value in np.asarray(data[key]).tolist())


def _resolve_motion_files(input_path: Path, pattern: str) -> list[Path]:
  input_path = input_path.expanduser().resolve()
  if input_path.is_file():
    return [input_path]
  if input_path.is_dir():
    files = sorted(input_path.glob(pattern))
    if files:
      return files
    raise FileNotFoundError(f"No files matching {pattern!r} in {input_path}")
  raise FileNotFoundError(input_path)


def _load_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    joint_names = _string_tuple(data, "joint_names")
    body_names = _string_tuple(data, "body_names")
    if len(set(joint_names)) != len(joint_names):
      raise ValueError(f"{path}: duplicate joint names")
    missing = sorted(set(ELF3_JOINT_NAMES) - set(joint_names))
    if missing:
      raise ValueError(f"{path}: missing ELF3 joints: {missing}")

    root_name = str(np.asarray(data["root_body_name"]).item())
    if root_name != ELF3_PHYSICAL_ROOT:
      raise ValueError(
        f"{path}: root_body_name={root_name!r}, expected {ELF3_PHYSICAL_ROOT!r}"
      )
    if root_name not in body_names:
      raise ValueError(f"{path}: root body {root_name!r} is absent from body_names")

    joint_indices = [joint_names.index(name) for name in ELF3_JOINT_NAMES]
    root_index = body_names.index(root_name)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)[:, joint_indices]
    root_pos = np.asarray(data["body_pos_w"], dtype=np.float64)[:, root_index]
    root_quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, root_index]

  frame_counts = {len(joint_pos), len(root_pos), len(root_quat)}
  if len(frame_counts) != 1 or not frame_counts:
    raise ValueError(f"{path}: inconsistent frame counts")
  if fps <= 0.0 or not np.isfinite(fps):
    raise ValueError(f"{path}: invalid fps {fps}")
  if not all(np.all(np.isfinite(value)) for value in (joint_pos, root_pos, root_quat)):
    raise ValueError(f"{path}: motion contains NaN or infinity")
  return fps, joint_pos, root_pos, root_quat


def _set_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addresses: np.ndarray,
  joint_pos: np.ndarray,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
) -> None:
  data.qpos[:3] = root_pos
  data.qpos[3:7] = root_quat
  data.qpos[joint_qpos_addresses] = joint_pos
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  draw.rectangle((0, 0, min(image.width, 12 + 7 * len(text)), 28), fill=(0, 0, 0))
  draw.text((7, 7), text, fill=(255, 255, 255))
  return np.asarray(image)


def _play_window(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  motions: list[Path],
  joint_qpos_addresses: np.ndarray,
  speed: float,
  loop: bool,
) -> None:
  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.6
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -15.0
    while viewer.is_running():
      for path in motions:
        fps, joint_pos, root_pos, root_quat = _load_motion(path)
        print(f"Playing {path.name}: {len(joint_pos)} frames @ {fps:g} Hz")
        start = time.perf_counter()
        for frame_index in range(len(joint_pos)):
          if not viewer.is_running():
            return
          _set_pose(
            model,
            data,
            joint_qpos_addresses,
            joint_pos[frame_index],
            root_pos[frame_index],
            root_quat[frame_index],
          )
          viewer.cam.lookat[:] = root_pos[frame_index] + (0.0, 0.0, 0.1)
          viewer.sync()
          target = (frame_index + 1) / (fps * speed)
          remaining = target - (time.perf_counter() - start)
          if remaining > 0.0:
            time.sleep(remaining)
      if not loop:
        return


def _record_video(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  motions: list[Path],
  joint_qpos_addresses: np.ndarray,
  output: Path,
  width: int,
  height: int,
  output_fps: float,
  max_seconds_per_clip: float | None,
) -> None:
  output = output.expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.distance = 2.6
  camera.azimuth = 135.0
  camera.elevation = -15.0

  with mujoco.Renderer(model, height=height, width=width) as renderer:
    with imageio.get_writer(
      output,
      fps=output_fps,
      codec="libx264",
      quality=8,
      macro_block_size=None,
    ) as writer:
      for clip_index, path in enumerate(motions, start=1):
        fps, joint_pos, root_pos, root_quat = _load_motion(path)
        duration = len(joint_pos) / fps
        if max_seconds_per_clip is not None:
          duration = min(duration, max_seconds_per_clip)
        output_frames = max(1, int(np.ceil(duration * output_fps)))
        print(
          f"Rendering {clip_index}/{len(motions)} {path.name}: "
          f"{output_frames} output frames"
        )
        for output_index in range(output_frames):
          source_index = min(
            int(np.floor(output_index * fps / output_fps)), len(joint_pos) - 1
          )
          _set_pose(
            model,
            data,
            joint_qpos_addresses,
            joint_pos[source_index],
            root_pos[source_index],
            root_quat[source_index],
          )
          camera.lookat[:] = root_pos[source_index] + (0.0, 0.0, 0.1)
          renderer.update_scene(data, camera=camera)
          frame = renderer.render()
          writer.append_data(_caption(frame, path.stem))
  print(f"Saved {output}")


def main(
  input_path: Path = DEFAULT_MOTION,
  pattern: str = "*.npz",
  mode: Literal["window", "video"] = "window",
  output: Path = Path("artifacts/elf3_amp_preview.mp4"),
  speed: float = 1.0,
  loop: bool = True,
  width: int = 960,
  height: int = 720,
  output_fps: float = 25.0,
  max_seconds_per_clip: float | None = None,
) -> None:
  """Preview one AMP NPZ or every matching NPZ in a directory.

  Args:
    input_path: NPZ file or directory to play.
    pattern: Glob used when input_path is a directory.
    mode: Native interactive window or MP4 recording.
    output: MP4 path in video mode.
    speed: Window playback speed multiplier.
    loop: Repeat after the final clip in window mode.
    width: Recorded video width.
    height: Recorded video height.
    output_fps: Recorded video frame rate.
    max_seconds_per_clip: Optional per-clip truncation for directory previews.
  """
  if speed <= 0.0:
    raise ValueError("speed must be positive")
  if output_fps <= 0.0:
    raise ValueError("output_fps must be positive")

  motions = _resolve_motion_files(input_path, pattern)
  model = mujoco.MjModel.from_xml_path(str(ELF3_XML))
  data = mujoco.MjData(model)
  joint_qpos_addresses = np.asarray(
    [model.joint(name).qposadr[0] for name in ELF3_JOINT_NAMES], dtype=np.int32
  )

  print(f"Asset: {ELF3_XML}")
  print(f"Physical root: {ELF3_PHYSICAL_ROOT}; clips: {len(motions)}")
  if mode == "window":
    _play_window(model, data, motions, joint_qpos_addresses, speed, loop)
  else:
    _record_video(
      model,
      data,
      motions,
      joint_qpos_addresses,
      output,
      width,
      height,
      output_fps,
      max_seconds_per_clip,
    )


if __name__ == "__main__":
  # EGL can be selected by setting MUJOCO_GL=egl before starting this script.
  # The default leaves native window rendering available on desktop systems.
  tyro.cli(main)
