"""Verify compiled ELF3 DWAQ parameters and original-AMP/feedback20 scope."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
import yaml
import src.tasks  # noqa: F401
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from src.assets.robots.elf3.elf3_constants import (
    ELF3_DWAQ_CONTRACT as CONTRACT, ELF3_DWAQ_CONTRACT_PATH,
    ELF3_JOINT_NAMES, get_elf3_robot_cfg,
)
from src.tasks.amp_loco.config.elf3.env_cfgs import elf3_amp_rough_v4_loco_env_cfg
from src.tasks.amp_loco.mdp.velocity_feedback import FeedbackJointPositionActionCfg, joint_vel_feedback_rel


def verify_compiled_robot(model, warp_model=None):
    joints={model.joint(i).name.rsplit('/',1)[-1]:i for i in range(model.njnt)}
    actuators={model.actuator(i).name.rsplit('/',1)[-1]:i for i in range(model.nu)}
    rows=[]
    for name in ELF3_JOINT_NAMES:
        j,a=joints[name],actuators[name]
        dof=int(model.jnt_dofadr[j])
        values=dict(armature=float(model.dof_armature[dof]),
                    stiffness=float(model.actuator_gainprm[a,0]),
                    damping=float(-model.actuator_biasprm[a,2]),
                    effort_limit=float(model.actuator_forcerange[a,1]))
        for field,value in values.items():
            np.testing.assert_allclose(value,CONTRACT[field][name],rtol=0.,atol=1.e-8,
                                       err_msg=f'{name}.{field}')
        np.testing.assert_allclose(model.actuator_forcerange[a],[-values['effort_limit'],values['effort_limit']])
        np.testing.assert_allclose(model.jnt_actfrcrange[j],[-values['effort_limit'],values['effort_limit']])
        np.testing.assert_allclose(model.actuator_biasprm[a,1],-values['stiffness'])
        assert model.actuator_forcelimited[a] and model.jnt_actfrclimited[j]
        assert not model.actuator_ctrllimited[a],name
        if warp_model is not None:
            for field,index,expected in (
                ('dof_armature',dof,values['armature']),
                ('actuator_gainprm',a,None),
                ('actuator_biasprm',a,None),
                ('actuator_forcerange',a,None),
                ('jnt_actfrcrange',j,None),
            ):
                data=wp.to_torch(getattr(warp_model,field))
                reference=np.asarray(getattr(model,field)[index])
                # mjwarp model fields have a leading world dimension, possibly one.
                actual=data[:,index] if data.ndim==np.asarray(getattr(model,field)).ndim+1 else data[index]
                target=torch.as_tensor(reference.copy(),device=actual.device,dtype=actual.dtype)
                torch.testing.assert_close(actual,target.expand_as(actual),rtol=0.,atol=1.e-6)
        rows.append(dict(joint=name,**values))
    return dict(joints=rows,verified_joint_count=len(rows),
                contract_sha256=hashlib.sha256(ELF3_DWAQ_CONTRACT_PATH.read_bytes()).hexdigest(),
                verified_gpu_model=warp_model is not None,
                limitation='PhysX 20 rad/s solver cap is recorded, not directly enforced by MuJoCo position actuators')


def verify_original_amp_scope(play=False):
    cfg=load_env_cfg('BXI-ELF3-AMP-Rough-V4',play=play)
    baseline=elf3_amp_rough_v4_loco_env_cfg(play=play)
    assert cfg.rewards==baseline.rewards,'Reward experiment still active'
    for field in ('commands','events','sim','terminations','curriculum','scene'):
        # Scene includes separately constructed sensor/config objects whose
        # Python identity is not a semantic configuration difference.
        assert yaml.dump(asdict(cfg)[field],sort_keys=True)==yaml.dump(asdict(baseline)[field],sort_keys=True),field
    for group in ('actor','critic'):
        assert cfg.observations[group].terms['joint_vel'].func is joint_vel_feedback_rel
        baseline.observations[group].terms['joint_vel'].func=joint_vel_feedback_rel
    assert cfg.observations==baseline.observations
    action=cfg.actions['joint_pos']
    assert type(action) is FeedbackJointPositionActionCfg and action.cutoff_hz==20.
    original=asdict(baseline.actions['joint_pos'])
    actual=asdict(action)
    actual.pop('cutoff_hz')
    assert actual==original
    runner=load_rl_cfg('BXI-ELF3-AMP-Rough-V4')
    assert not runner.resume and runner.algorithm.learning_rate==.001
    assert Path(runner.amp_motion_files).name=='WalkandRun'
    assert cfg.events['init_motion_loader'].params['recovery_dir'] is None
    return dict(rewards=list(cfg.rewards),reward_count=len(cfg.rewards),
                original_rewards_equal=True,non_reward_scope_equal=True,
                cutoff_hz=20.,additional_action_delay=False,additional_sensor_delay=False,
                recovery=False,amp_motion_files=runner.amp_motion_files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda:0'))
    parser.add_argument('--num-envs',type=int,default=32)
    parser.add_argument('--steps',type=int,default=120)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    result=dict(scope_train=verify_original_amp_scope(),scope_play=verify_original_amp_scope(True))
    if args.device=='cpu':
        robot=Entity(get_elf3_robot_cfg())
        result['model']=verify_compiled_robot(robot.spec.compile())
    else:
        cfg=load_env_cfg('BXI-ELF3-AMP-Rough-V4')
        cfg.scene.num_envs=args.num_envs
        cfg.seed=42
        env=ManagerBasedRlEnv(cfg,device=args.device)
        try:
            result['model']=verify_compiled_robot(env.sim.mj_model,env.sim.wp_model)
            obs,_=env.reset()
            feedback=env.joint_velocity_feedback
            start=feedback.update_count
            for _ in range(args.steps):
                obs,reward,_,_,_=env.step(torch.zeros((env.num_envs,29),device=env.device))
                assert all(torch.isfinite(x).all() for x in obs.values())
                assert obs['actor'].shape==(env.num_envs,384)
                assert torch.isfinite(reward).all()
                for field in ('qpos','qvel','qacc'):
                    assert torch.isfinite(wp.to_torch(getattr(env.sim.wp_data,field))).all(),field
            assert feedback.update_count-start==args.steps*cfg.decimation
            keep=feedback.value[1:].clone()
            env.reset(env_ids=torch.tensor([0],device=env.device))
            torch.testing.assert_close(feedback.value[0],env.scene['robot'].data.joint_vel[0])
            torch.testing.assert_close(feedback.value[1:],keep)
            result['integration']=dict(num_envs=env.num_envs,steps=args.steps,
                                       physics_updates=args.steps*cfg.decimation,
                                       partial_reset_checked=True,finite=True)
        finally:
            env.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print('DWAQ_ASSET_CHECK_OK',args.output,flush=True)


if __name__=='__main__':
    main()
