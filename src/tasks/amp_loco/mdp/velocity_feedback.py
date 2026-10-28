"""mjlab integration of JKYovo/DWAQ a14f9d5 joint_velocity_tustin_v1.

Only the actor/critic joint-speed observations are filtered. The standard
JointPositionAction, physical PD feedback, AMP and reward inputs remain raw.
"""
from dataclasses import dataclass

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_feedback_model import JointVelocityFeedback


@dataclass(kw_only=True)
class FeedbackJointPositionActionCfg(JointPositionActionCfg):
    cutoff_hz: float = 20.0

    def build(self, env):
        return FeedbackJointPositionAction(self, env)


class FeedbackJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.feedback = JointVelocityFeedback(
            self._entity.data.joint_vel, env.physics_dt,
            mode='lpf20', cutoff_hz=cfg.cutoff_hz,
        )
        if hasattr(env, 'joint_velocity_feedback'):
            raise ValueError('Only one velocity-feedback model may be installed')
        env.joint_velocity_feedback = self.feedback
        # mjlab calls scene.update exactly once after each integrated physics
        # substep. Hook there, not apply_actions (which runs before integration).
        update_scene = env.scene.update

        def update_with_feedback(dt):
            update_scene(dt=dt)
            self.feedback.update(self._entity.data.joint_vel)

        env.scene.update = update_with_feedback

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # ActionManager.reset follows all reset-state events, so seed the
        # filter from the new state, including partial/automatic resets.
        self.feedback.reset(self._entity.data.joint_vel, env_ids)


def joint_vel_feedback_rel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg('robot')):
    robot = env.scene[asset_cfg.name]
    return (env.joint_velocity_feedback.value[:, asset_cfg.joint_ids]
            - robot.data.default_joint_vel[:, asset_cfg.joint_ids])
