"""From JKYovo/DWAQ a14f9d5, legged_lab/envs/elf3/feedback.py.

Pure-tensor joint-velocity observation models (no Isaac Sim imports).

The filter changes only a sensor observation. It must not replace the physical
velocity used by an actuator or filter actions/position targets. ``lpf20`` is a
first-order low-pass model, not a fixed 20 ms delay.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch


FEEDBACK_MODEL_VERSION = "joint_velocity_tustin_v1"
DEFAULT_VELOCITY_FEEDBACK = {
    "mode": "ideal",
    "cutoff_hz": 20.0,
    "model_version": FEEDBACK_MODEL_VERSION,
    "latest_critic_contacts": True,
}


def resolve_feedback_settings(settings: Mapping | None) -> dict:
    """Return a validated, serializable copy of the isolated-task configuration."""
    result = dict(DEFAULT_VELOCITY_FEEDBACK)
    if settings is not None:
        if not isinstance(settings, Mapping):
            raise TypeError("velocity_feedback must be a mapping")
        unknown = set(settings) - set(result)
        if unknown:
            raise ValueError(f"Unknown velocity_feedback settings: {sorted(unknown)}")
        result.update(settings)
    if result["mode"] not in ("ideal", "lpf20"):
        raise ValueError("velocity_feedback.mode must be 'ideal' or 'lpf20'")
    if result["model_version"] != FEEDBACK_MODEL_VERSION:
        raise ValueError(f"Unsupported velocity feedback model: {result['model_version']!r}")
    if type(result["latest_critic_contacts"]) is not bool:
        raise TypeError("latest_critic_contacts must be a bool")
    result["cutoff_hz"] = float(result["cutoff_hz"])
    if not math.isfinite(result["cutoff_hz"]) or result["cutoff_hz"] <= 0:
        raise ValueError("cutoff_hz must be finite and positive")
    return result


def tustin_lowpass_coefficients(cutoff_hz: float, physics_dt: float) -> tuple[float, float, float]:
    """Prewarped bilinear coefficients for H(z)=(b0+b1*z^-1)/(1+a1*z^-1).

    Prewarping places the -3 dB / -45 degree point at ``cutoff_hz`` exactly.
    The recurrence is y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1].
    """
    cutoff_hz, physics_dt = float(cutoff_hz), float(physics_dt)
    if not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("physics_dt must be finite and positive")
    if not math.isfinite(cutoff_hz) or not 0 < cutoff_hz < 0.5 / physics_dt:
        raise ValueError("cutoff_hz must be strictly between zero and the physics Nyquist frequency")
    warped = math.tan(math.pi * cutoff_hz * physics_dt)
    b = warped / (1.0 + warped)
    return b, b, (warped - 1.0) / (warped + 1.0)


class JointVelocityFeedback:
    """One batched filter state in native simulator joint order.

    Call ``update`` exactly once after each physics substep. Reading ``value``
    never advances state. Reset input/output to the *current raw velocity*, not
    zero, after the environment has finished writing/resetting its robot state.
    Counters are diagnostic only and never enter policy observations.
    """

    def __init__(
        self,
        initial_velocity: torch.Tensor,
        physics_dt: float,
        mode: str = "ideal",
        cutoff_hz: float = 20.0,
        model_version: str = FEEDBACK_MODEL_VERSION,
    ):
        settings = resolve_feedback_settings({
            "mode": mode, "cutoff_hz": cutoff_hz, "model_version": model_version,
        })
        if initial_velocity.ndim != 2 or not initial_velocity.is_floating_point():
            raise ValueError("initial_velocity must be a floating [environment, joint] tensor")
        self.mode = settings["mode"]
        self.model_version = settings["model_version"]
        self.cutoff_hz = settings["cutoff_hz"]
        self.physics_dt = float(physics_dt)
        self.b0, self.b1, self.a1 = tustin_lowpass_coefficients(self.cutoff_hz, self.physics_dt)
        self._previous_input = initial_velocity.detach().clone()
        self._value = initial_velocity.detach().clone()
        self.update_count = 0
        self.reset_count = 0
        self.steps_since_reset = torch.zeros(initial_velocity.shape[0], dtype=torch.long,
                                             device=initial_velocity.device)

    @property
    def value(self) -> torch.Tensor:
        """Current cached output; consumers must not modify this tensor in-place."""
        return self._value

    def _check_input(self, velocity: torch.Tensor) -> None:
        if velocity.shape != self._value.shape:
            raise ValueError("Joint velocity shape changed after filter initialization")
        if velocity.device != self._value.device or velocity.dtype != self._value.dtype:
            raise ValueError("Joint velocity device/dtype changed after filter initialization")

    @torch.no_grad()
    def update(self, raw_velocity: torch.Tensor) -> torch.Tensor:
        self._check_input(raw_velocity)
        if self.mode == "ideal":
            self._value.copy_(raw_velocity)
        else:
            output = self.b0 * raw_velocity + self.b1 * self._previous_input - self.a1 * self._value
            self._value.copy_(output)
        self._previous_input.copy_(raw_velocity)
        self.steps_since_reset.add_(1)
        self.update_count += 1
        return self._value

    @torch.no_grad()
    def reset(self, raw_velocity: torch.Tensor, env_ids=None) -> None:
        """Reset selected rows from a full current-velocity tensor, without advancing time."""
        self._check_input(raw_velocity)
        if env_ids is None:
            self._previous_input.copy_(raw_velocity)
            self._value.copy_(raw_velocity)
            self.steps_since_reset.zero_()
        else:
            if len(env_ids) == 0:
                return
            self._previous_input[env_ids] = raw_velocity[env_ids]
            self._value[env_ids] = raw_velocity[env_ids]
            self.steps_since_reset[env_ids] = 0
        self.reset_count += 1
