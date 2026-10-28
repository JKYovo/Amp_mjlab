"""GPU integration check for post-physics feedback updates and partial reset."""
import argparse
import torch
import warp as wp
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from src.tasks.amp_loco.mdp.velocity_feedback import joint_vel_feedback_rel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--steps',type=int,default=120)
    args=parser.parse_args()
    cfg=load_env_cfg('BXI-ELF3-AMP-Rough-V4')
    cfg.scene.num_envs=args.num_envs
    cfg.seed=42
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        obs,_=env.reset()
        f=env.joint_velocity_feedback
        n=f.update_count
        assert obs['actor'].shape==(args.num_envs,384),obs['actor'].shape
        for _ in range(args.steps):
            obs,reward,terminated,truncated,info=env.step(torch.zeros((args.num_envs,29),device=env.device))
            for group,value in obs.items(): assert torch.isfinite(value).all(),group
            assert torch.isfinite(reward).all()
            for field in ('qpos','qvel','qacc'):
                assert torch.isfinite(wp.to_torch(getattr(env.sim.wp_data,field))).all(),field
        assert f.update_count-n==args.steps*cfg.decimation,(f.update_count,n)
        before_count=f.update_count
        observed=joint_vel_feedback_rel(env).clone()
        for _ in range(5): torch.testing.assert_close(joint_vel_feedback_rel(env),observed)
        assert before_count==f.update_count
        keep=f.value[1:].clone()
        env.reset(env_ids=torch.tensor([0],device=env.device))
        torch.testing.assert_close(f.value[0],env.scene['robot'].data.joint_vel[0])
        torch.testing.assert_close(f.value[1:],keep)
        print('FEEDBACK_CHECK_OK',dict(num_envs=args.num_envs,steps=args.steps,
              physics_updates=f.update_count-n,cutoff_hz=f.cutoff_hz,
              actor_shape=tuple(obs['actor'].shape),reset_count=f.reset_count),flush=True)
    finally: env.close()


if __name__=='__main__': main()
