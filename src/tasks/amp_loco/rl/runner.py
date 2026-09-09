import os
import inspect

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner


class _OnnxPolicyWrapper(torch.nn.Module):
  """Thin wrapper that exposes ``act_inference`` as ``forward`` for ONNX export.
  
  Includes the obs normalizer so the exported ONNX model expects raw observations
  and C++ deployment does not need to implement normalization separately.
  """

  def __init__(self, actor_critic, obs_normalizer=None):
    super().__init__()
    self.actor_critic = actor_critic
    self.obs_normalizer = obs_normalizer

  def forward(self, obs):
    if self.obs_normalizer is not None:
      obs = self.obs_normalizer(obs)
    return self.actor_critic.act_inference(obs)


def _onnx_export_kwargs_single_file() -> dict:
  """Build kwargs that request single-file ONNX export across torch versions."""
  try:
    params = inspect.signature(torch.onnx.export).parameters
  except (TypeError, ValueError):
    return {}

  if "external_data" in params:
    return {"external_data": False}
  if "use_external_data_format" in params:
    return {"use_external_data_format": False}
  return {}


def _inline_external_onnx_data(onnx_path: str) -> None:
  """Merge external tensor data back into a single ONNX file if needed."""
  data_path = f"{onnx_path}.data"
  if not os.path.exists(data_path):
    return

  try:
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(model, onnx_path, save_as_external_data=False)
    if os.path.exists(data_path):
      os.remove(data_path)
    print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
  except Exception as exc:
    print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")


class AMPOnPolicyRunner(AmpOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def _restore_step_curricula(self) -> None:
    """Apply curricula derived from the restored global environment step."""
    env = self.env.unwrapped
    manager = env.curriculum_manager
    for term_name in manager.active_terms:
      term_cfg = manager.get_term_cfg(term_name)
      # These curricula are pure functions of common_step_counter.  Do not
      # recompute performance-driven terrain curricula on a newly created env.
      if term_cfg.func.__name__ not in {"commands_vel", "reward_weight"}:
        continue
      state = term_cfg.func(env, slice(None), **term_cfg.params)
      manager._curriculum_state[term_name] = state

    # Environment construction sampled commands before checkpoint loading,
    # using the stage-zero ranges.  Resample once so the first resumed rollout
    # already uses the restored stage instead of waiting 3--8 seconds.
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.command_manager.reset(env_ids)

  def load(self, path: str, load_optimizer: bool = True):
    """Load policy state and restore the environment's curriculum clock."""
    infos = super().load(path, load_optimizer=load_optimizer)
    env_state = infos.get("env_state", {}) if isinstance(infos, dict) else {}
    if "common_step_counter" in env_state:
      common_step_counter = int(env_state["common_step_counter"])
      source = "checkpoint"
    else:
      # Legacy AMP checkpoints did not persist environment state.  A checkpoint
      # named iteration N is written after completing that iteration.
      common_step_counter = (
        int(self.current_learning_iteration) + 1
      ) * int(self.num_steps_per_env)
      source = "iteration fallback"

    env = self.env.unwrapped
    env.common_step_counter = common_step_counter
    if "sim_step_counter" in env_state:
      env._sim_step_counter = int(env_state["sim_step_counter"])
    else:
      env._sim_step_counter = common_step_counter * int(env.cfg.decimation)
    self._restore_step_curricula()
    print(
      "[INFO] Restored curriculum clock: "
      f"common_step_counter={common_step_counter} ({source})"
    )
    return infos

  def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
    """Export the actor network to ONNX using the local ActorCritic model.
    
    The exported model includes the obs normalizer (if empirical_normalization
    is enabled) so that the ONNX model expects raw observations directly.
    """
    policy = self.alg.policy
    # Include normalizer in the ONNX model if empirical normalization is used
    obs_normalizer = None
    if self.empirical_normalization:
      obs_normalizer = self.obs_normalizer
      obs_normalizer.to("cpu")
      obs_normalizer.eval()
    wrapper = _OnnxPolicyWrapper(policy, obs_normalizer)
    wrapper.to("cpu")
    wrapper.eval()
    num_obs = policy.actor[0].in_features
    dummy_input = torch.zeros(1, num_obs)
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      wrapper,
      dummy_input,
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
      **_onnx_export_kwargs_single_file(),
    )
    _inline_external_onnx_data(os.path.join(path, filename))
    # move policy back to training device
    policy.to(self.device)
    if obs_normalizer is not None:
      obs_normalizer.to(self.device)

  def save(self, path: str, infos=None):
    env = self.env.unwrapped
    checkpoint_infos = dict(infos or {})
    checkpoint_infos["env_state"] = {
      "common_step_counter": int(env.common_step_counter),
      "sim_step_counter": int(env._sim_step_counter),
    }
    super().save(path, checkpoint_infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self._export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    _inline_external_onnx_data(onnx_path)
    if self.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
