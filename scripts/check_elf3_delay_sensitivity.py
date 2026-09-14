"""Paired CPU-only policy latency sweep; no training, export or logger.

Compile the canonical ELF3 asset using the saved training configuration, then
use standard MuJoCo for isolated rollouts. Avoid importing mjlab/Warp entirely.
This is not a real-robot latency measurement, nor a MuJoCo-Warp collision test.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import time

# Only this diagnostic process loses GPU visibility; the training is untouched.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import mujoco
import numpy as np
import torch
import yaml


class ConfigLoader(yaml.SafeLoader):
    pass


ConfigLoader.add_constructor('tag:yaml.org,2002:python/tuple',
    lambda loader, node: tuple(loader.construct_sequence(node)))
ConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/name:',
    lambda loader, suffix, node: suffix)
ConfigLoader.add_multi_constructor('tag:yaml.org,2002:python/object/apply:',
    lambda loader, suffix, node: loader.construct_sequence(node))


def resolve(patterns, name, default=None):
    if not isinstance(patterns, dict):
        return patterns
    for pattern, value in patterns.items():
        if re.fullmatch(pattern, name):
            return value
    return default


def compile_common_asset(config):
    cfg = yaml.load(Path(config).read_text(), Loader=ConfigLoader)
    robot = cfg['scene']['entities']['robot']
    actor = cfg['observations']['actor']
    assert actor['history_ordering'] == 'time' and actor['history_length'] == 4
    assert list(actor['terms']) == ['base_ang_vel', 'projected_gravity', 'command',
                                  'joint_pos', 'joint_vel', 'actions']
    assert not robot['sort_actuators']
    xml = Path('src/assets/robots/elf3/xmls/elf3.xml').resolve()
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.assets = {f'{spec.meshdir}/{p.name}': p.read_bytes()
                   for p in (xml.parent / spec.meshdir).iterdir() if p.is_file()}
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    spec.delete(spec.geom('floor'))
    collision = robot['collisions'][0]
    for geom in spec.geoms:
        if any(re.fullmatch(p, geom.name) for p in collision['geom_names_expr']):
            for attr, fallback in [('condim', 3), ('priority', 0), ('contype', 1), ('conaffinity', 1)]:
                setattr(geom, attr, resolve(collision[attr], geom.name, fallback))
            for attr in ('friction', 'solref', 'solimp'):
                value = resolve(collision[attr], geom.name)
                if value is not None:
                    getattr(geom, attr)[:len(value)] = value
        elif collision['disable_other_geoms']:
            geom.contype = geom.conaffinity = 0
    spec.worldbody.add_geom(name='terrain', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0., 0., .05], group=0)
    spec.body('torso_link').add_site(name='imu_in_torso', pos=[0., 0., 0.], size=[.01])
    spec.add_sensor(name='imu_ang_vel', type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu_in_torso')
    parameters = robot['articulation']['actuators']
    names = []
    for params in parameters:
        name, = params['target_names_expr']
        names.append(name)
        joint = spec.joint(name)
        actuator = spec.add_actuator(name=name, target=name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = params['stiffness']
        actuator.biasprm[1:3] = [-params['stiffness'], -params['damping']]
        actuator.inheritrange = 0.
        actuator.ctrllimited = False
        actuator.forcelimited = True
        limit = params['effort_limit']
        actuator.forcerange[:] = [-limit, limit]
        actuator.ctrlrange[:] = [joint.range[0] - limit / params['stiffness'],
                                 joint.range[1] + limit / params['stiffness']]
        joint.armature = params['armature']
        joint.frictionloss = params['frictionloss']
    model = spec.compile()
    options = cfg['sim']['mujoco']
    enums = dict(integrator={'implicitfast':mujoco.mjtIntegrator.mjINT_IMPLICITFAST},
                 solver={'newton':mujoco.mjtSolver.mjSOL_NEWTON},
                 jacobian={'auto':mujoco.mjtJacobian.mjJAC_AUTO},
                 cone={'pyramidal':mujoco.mjtCone.mjCONE_PYRAMIDAL})
    for name, value in options.items():
        if name == 'multiccd':
            # Match mjlab MujocoCfg.apply(): False leaves native defaults alone.
            assert not value
        else:
            setattr(model.opt, name, enums[name][value] if name in enums else value)
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    assert [model.joint(i).name for i in joints] == names
    # Literal extraction does not import the robot's mjlab dependencies.
    tree = ast.parse(Path('src/assets/robots/elf3/elf3_constants.py').read_text())
    declared = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'ELF3_JOINT_NAMES' for t in n.targets))
    assert tuple(names) == declared
    default = np.array([resolve(robot['init_state']['joint_pos'], n) for n in names], np.float32)
    scale = np.array([resolve(cfg['actions']['joint_pos']['scale'], n) for n in names], np.float32)
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    ctrl_ids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in joints])
    assert model.nu == 29 and default.shape == scale.shape == (29,)
    for i, params in enumerate(parameters):
        assert model.actuator_gainprm[ctrl_ids[i], 0] == params['stiffness']
        assert model.actuator_biasprm[ctrl_ids[i], 2] == -params['damping']
    data = mujoco.MjData(model)
    data.qpos[:3] = robot['init_state']['pos']
    data.qpos[3:7] = robot['init_state']['rot']
    data.qpos[qadr] = default
    mujoco.mj_forward(model, data)
    minimum = np.inf
    for side in ('l', 'r'):
        geom = model.geom(f'{side}_ankle_x_link_collision_0').id
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        minimum = min(minimum, world[:, 2].min())
    qpos = data.qpos.copy()
    qpos[2] += .003 - minimum
    sensor = model.sensor('imu_ang_vel').id
    mapping = dict(qadr=qadr, vadr=vadr, ctrl_ids=ctrl_ids, default=default, scale=scale,
                   torso=model.body('torso_link').id, gyro=slice(model.sensor_adr[sensor], model.sensor_adr[sensor] + 3),
                   joint_names=names, leg_indices=np.array([i for i, n in enumerate(names)
                       if any(x in n for x in ('hip_', 'knee_', 'ankle_'))]))
    return model, mapping, qpos


def load_actor(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict']
    keys = sorted((k for k in state if k.startswith('actor.') and k.endswith('.weight')),
                   key=lambda k:int(k.split('.')[1]))
    layers = []
    for i, key in enumerate(keys):
        weight = state[key]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
        layer.load_state_dict({'weight':weight, 'bias':state[key[:-6] + 'bias']})
        layers.append(layer)
        if i < len(keys) - 1:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    assert actor[0].in_features == 384 and actor[-1].out_features == 29
    norm = checkpoint['obs_norm_state_dict']
    mean, std = norm['_mean'], norm['_std']
    assert all(torch.isfinite(x).all() for x in (mean, std))
    return actor, lambda obs:(obs - mean) / (std + .01)


def measurement(data, mapping):
    rotation = data.xmat[mapping['torso']].reshape(3, 3)
    return np.concatenate((data.sensordata[mapping['gyro']],
                           -rotation[2],
                           data.qpos[mapping['qadr']] - mapping['default'],
                           data.qvel[mapping['vadr']])).astype(np.float32)


def summarize(records, indices, dt):
    selected = [r for r in records if r['step'] in indices]
    if len(selected) < 20:
        return {'samples': len(selected), 'insufficient_surviving_samples': True}
    target = np.array([r['target'] for r in selected])
    position = np.array([r['leg_position'] for r in selected])
    pitch = np.array([r['pitch'] for r in selected])
    velocity = np.array([r['leg_velocity'] for r in selected])
    angular = np.array([r['angular'] for r in selected])
    window = np.hanning(len(target))[:, None]
    spectrum = np.fft.rfft((target - target.mean(axis=0)) * window, axis=0)
    frequency = np.fft.rfftfreq(len(target), dt)
    spectrum[(frequency < 8.) | (frequency > 25.)] = 0
    high = np.fft.irfft(spectrum, n=len(target), axis=0) / np.sqrt(np.mean(window**2))
    joint_spectrum = np.fft.rfft((position - position.mean(axis=0)) * window, axis=0)
    joint_spectrum[(frequency < 8.) | (frequency > 25.)] = 0
    joint_high = np.fft.irfft(joint_spectrum, n=len(position), axis=0) / np.sqrt(np.mean(window**2))
    return dict(samples=len(selected),
                target_rate_rms_rad_s=float(np.sqrt(np.mean((np.diff(target, axis=0) / dt)**2))),
                target_high_frequency_rms_rad=float(np.sqrt(np.mean(high**2))),
                joint_high_frequency_rms_rad=float(np.sqrt(np.mean(joint_high**2))),
                leg_velocity_rms_rad_s=float(np.sqrt(np.mean(velocity**2))),
                torso_roll_pitch_rate_rms_rad_s=float(np.sqrt(np.mean(angular[:, :2]**2))),
                mean_world_yaw_rate_rad_s=float(angular[:, 2].mean()),
                mean_pitch_deg=float(pitch.mean()), pitch_std_deg=float(pitch.std()))


def run(args):
    if Path(args.output).exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    torch.set_num_threads(2)
    started = time.monotonic()
    model, mapping, nominal = compile_common_asset(args.config)
    dt = model.opt.timestep * 4
    assert abs(dt - .02) < 1e-8
    actors = {label: load_actor(path) for label, path in
              ((getattr(args, 'v31_label', 'v31_81300'), args.v31),
               (getattr(args, 'v4_label', 'v4_old_gravel_22800'), args.v4))}
    if len(actors) != 2:
        raise ValueError('The two checkpoint labels must be distinct')
    # Position-target delays use physics steps (5ms). Observation delays use
    # control samples (20ms); delay only measurements, not command/last action.
    conditions = [(f'action_{ms}ms', ms, 0) for ms in args.action_delays_ms]
    conditions += [(f'observation_{ms}ms', 0, ms) for ms in args.observation_delays_ms if ms]
    if args.combined:
        conditions += [('action_20ms_observation_20ms', 20, 20)]
    for _, action_ms, obs_ms in conditions:
        assert action_ms >= 0 and action_ms % 5 == 0
        assert obs_ms >= 0 and obs_ms % 20 == 0
    worlds = []
    commands = [(0., yaw) for yaw in args.yaw_commands]
    commands += [(vx, 0.) for vx in args.linear_commands]
    for label in actors:
        for name, action_ms, obs_ms in conditions:
            for vx_command, command in commands:
                for seed in args.seeds:
                    data = mujoco.MjData(model)
                    rng = np.random.default_rng(seed)
                    data.qpos[:] = nominal
                    data.qpos[mapping['qadr']] += rng.normal(0, .003, 29)
                    data.qvel[mapping['vadr']] = rng.normal(0, .01, 29)
                    data.qvel[3:6] = rng.normal(0, .015, 3)
                    data.ctrl[mapping['ctrl_ids']] = mapping['default']
                    mujoco.mj_forward(model, data)
                    raw = measurement(data, mapping)
                    single = np.concatenate((raw[:6], np.zeros(3), raw[6:], np.zeros(29))).astype(np.float32)
                    worlds.append(dict(label=label, condition=name, action_ms=action_ms,
                                       obs_ms=obs_ms, command=command, vx_command=vx_command, seed=seed, data=data,
                                       observations=[raw.copy() for _ in range(obs_ms // 20 + 1)],
                                       targets=[mapping['default'].copy() for _ in range(action_ms // 5 + 1)],
                                       history=np.tile(single, (4, 1)), previous=np.zeros(29, np.float32),
                                       failed=False, failure_time_s=None, failure_reason=None, records=[]))
    print('DELAY_TEST_SETUP ' + json.dumps(dict(worlds=len(worlds), physics_dt=model.opt.timestep,
          control_dt=dt, device='cpu', backend='standard MuJoCo', contract_validated=True)), flush=True)
    total_steps = round((args.settle_s + args.move_s + args.stop_s) / dt)
    actor_indices = {label: [i for i, w in enumerate(worlds) if w['label'] == label] for label in actors}
    with torch.inference_mode():
        for step in range(total_steps):
            for w in worlds:
                if w['failed']:
                    continue
                raw = measurement(w['data'], mapping)
                w['observations'].append(raw)
                w['observations'].pop(0)
                delayed = w['observations'][0]
                yaw = w['command'] if args.settle_s <= step * dt < args.settle_s + args.move_s else 0.
                vx = w['vx_command'] if args.settle_s <= step * dt < args.settle_s + args.move_s else 0.
                single = np.concatenate((delayed[:6], [vx, 0., yaw], delayed[6:], w['previous'])).astype(np.float32)
                w['history'] = np.concatenate((w['history'][1:], single[None]), axis=0)
            for label, indices in actor_indices.items():
                actor, normalizer = actors[label]
                batch = torch.from_numpy(np.array([worlds[i]['history'].reshape(-1) for i in indices]))
                actions = actor(normalizer(batch)).numpy()
                for i, action in zip(indices, actions):
                    w = worlds[i]
                    if w['failed']:
                        continue
                    if not np.isfinite(action).all():
                        w.update(failed=True, failure_time_s=step * dt, failure_reason='nonfinite_action')
                        continue
                    w['previous'] = action.copy()
                    proposed = mapping['default'] + mapping['scale'] * action
                    for substep in range(4):
                        w['targets'].append(proposed)
                        w['targets'].pop(0)
                        w['data'].ctrl[mapping['ctrl_ids']] = w['targets'][0]
                        mujoco.mj_step(model, w['data'])
                        if not np.isfinite(w['data'].qpos).all() or not np.isfinite(w['data'].qvel).all():
                            w.update(failed=True, failure_time_s=(step + (substep + 1) / 4) * dt,
                                     failure_reason='nonfinite_physics')
                            break
                    if w['failed']:
                        continue
                    mujoco.mj_forward(model, w['data'])
                    rotation = w['data'].xmat[mapping['torso']].reshape(3, 3)
                    if w['data'].qpos[2] < .62 or rotation[2, 2] < .5:
                        w.update(failed=True, failure_time_s=(step + 1) * dt, failure_reason='fallen')
                        continue
                    pitch = np.degrees(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
                    w['records'].append(dict(step=step, target=proposed[mapping['leg_indices']], pitch=pitch,
                        leg_position=w['data'].qpos[mapping['qadr'][mapping['leg_indices']]].copy(),
                        leg_velocity=w['data'].qvel[mapping['vadr'][mapping['leg_indices']]].copy(),
                        angular=rotation @ w['data'].sensordata[mapping['gyro']]))
            if (step + 1) % 50 == 0:
                print('DELAY_TEST_PROGRESS ' + json.dumps(dict(step=step + 1, total=total_steps,
                      fallen=sum(w['failed'] for w in worlds), wall_s=time.monotonic()-started)), flush=True)
    move = set(range(round(args.settle_s / dt), round((args.settle_s + args.move_s) / dt)))
    stop = set(range(round((args.settle_s + args.move_s) / dt), total_steps))
    results = [dict(policy=w['label'], condition=w['condition'], action_delay_ms=w['action_ms'],
                    observation_delay_ms=w['obs_ms'], yaw_command=w['command'], seed=w['seed'],
                    lin_vel_x_command=w['vx_command'],
                    failed=w['failed'], failure_time_s=w['failure_time_s'], failure_reason=w['failure_reason'],
                    active=summarize(w['records'], move, dt), stopped=summarize(w['records'], stop, dt),
                    active_steady=summarize(w['records'], {s for s in move if s*dt >= args.settle_s + 1.}, dt),
                    stopped_steady=summarize(w['records'], {s for s in stop if s*dt >= args.settle_s + args.move_s + 1.}, dt))
               for w in worlds]
    report = dict(checkpoints={'v31':str(Path(args.v31).resolve()), 'v4':str(Path(args.v4).resolve())},
                  policy_labels={'v31':getattr(args, 'v31_label', 'v31_81300'),
                                 'v4':getattr(args, 'v4_label', 'v4_old_gravel_22800')},
                  backend='standard MuJoCo CPU', contract_validated=True,
                  contract_validation='canonical joint-order literal, saved actor term/history layout and saved PD/action scale',
                  source_config=str(Path(args.config).resolve()),
                  terrain='common nominal plane', randomization=False, noise=False,
                  initial_perturbation='paired 0.003rad joint position / 0.01rad/s joint speed / 0.015rad/s root rotation',
                  pd='shared unchanged mjlab ELF3 builtin position PD', physics_dt=model.opt.timestep,
                  control_dt=dt, settle_s=args.settle_s, move_s=args.move_s, stop_s=args.stop_s,
                  failure_threshold='root z < 0.62m OR torso tilt > 60deg; no automatic reset',
                  jitter_metric='leg target rate / 8-25Hz target band / torso angular rate; gait motion is not automatically jitter',
                  observation_delay='only measured angular velocity/gravity/joint position/speed; command and generated previous action remain current',
                  action_delay='position target only; builtin inner PD feedback remains instantaneous',
                  training_performed=False, wall_s=time.monotonic()-started, results=results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('DELAY_TEST_RESULT ' + str(output.resolve()), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='logs/rsl_rl/elf3_amp_locomotion_v4/2026-09-12_17-22-39_v4_startup_recovery_fresh/params/env.yaml')
    parser.add_argument('--v31', default='logs/rsl_rl/elf3_amp_locomotion_v3_1/2026-09-11_14-44-24_sampling_fix_from79500/model_81300.pt')
    parser.add_argument('--v4', default='logs/rsl_rl/elf3_amp_locomotion_v4/2026-09-12_17-22-39_v4_startup_recovery_fresh/model_22800.pt')
    parser.add_argument('--v31-label', default='v31_81300')
    parser.add_argument('--v4-label', default='v4_old_gravel_22800')
    parser.add_argument('--action-delays-ms', nargs='+', type=int, default=[0, 5, 10, 20, 40])
    parser.add_argument('--observation-delays-ms', nargs='+', type=int, default=[20, 40])
    parser.add_argument('--yaw-commands', nargs='+', type=float, default=[0., 1., -1.])
    parser.add_argument('--linear-commands', nargs='+', type=float, default=[.6, -.4])
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    parser.add_argument('--settle-s', type=float, default=2.)
    parser.add_argument('--move-s', type=float, default=6.)
    parser.add_argument('--stop-s', type=float, default=4.)
    parser.add_argument('--combined', action='store_true')
    parser.add_argument('--output', default='logs/diagnostics/elf3_delay_sensitivity_20260914.json')
    run(parser.parse_args())
