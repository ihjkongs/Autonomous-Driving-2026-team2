"""DAgger-style labels on learner-visited states, never used at evaluation."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import get_context
import json
import pickle
from pathlib import Path
import random
import numpy as np
from utils.env import F1Wrapper, normalize_speed, unnormalize_speed
from utils.policy_evaluation import initialize_policy, policy_action
from utils.rule_based import PurePursuitPlanner, nearest_point_on_trajectory


def collect_episode(job):
    args, name, seed, beta, max_examples = job
    config = deepcopy(args)
    config.seed, config.reset_config = seed, {'type': 'cl_random_static'}
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed+377)
    env = F1Wrapper(config, [name])
    examples, labels = [], []
    record = dict(map=name, seed=seed, beta=beta, teacher_steps=0, outcome='timeout')
    try:
        _, _ = env.reset(options={'map': name})
        poses = env.unwrapped.sim.agent_poses.copy()
        lateral = rng.uniform(-.25, .25)
        yaw = rng.uniform(-.10, .10)
        poses[0, :2] += lateral*np.array([-np.sin(poses[0, 2]), np.cos(poses[0, 2])])
        poses[0, 2] += yaw
        observation, info = env.reset(options={'map': name, 'poses': poses})
        record.update(initial_pose=poses[0].tolist(), lateral_perturbation_m=float(lateral),
                      yaw_perturbation_rad=float(yaw),
                      response=env.unwrapped.sim.agents[0].response_parameters)
        parameters = config.expert_parameters
        planner = PurePursuitPlanner(env.unwrapped.track, config.length_f+config.length_r,
                                    parameters['reference'])
        seen = 0
        for step in range(1, config.max_episode_steps+1):
            raw = info['obs_dict']
            position = np.array([raw['poses_x'][0], raw['poses_y'][0]])
            _, distance, _, _ = nearest_point_on_trajectory(position, planner.waypoints[:, :2])
            if distance > 1.25:
                record['outcome'] = 'off_route_limit'
                break
            try:
                speed, steer = planner.plan(raw, parameters)
            except RuntimeError:
                record['outcome'] = 'oracle_lost_path'
                break
            # A recovery oracle slows while correcting appreciable cross-track error.
            if distance > .15:
                speed = min(speed, 2.5)
            if distance > .5:
                speed = min(speed, 1.5)
            label = np.clip([steer/config.max_steer,
                             normalize_speed(speed, config.min_speed, config.max_speed)], -1., 1.)
            student = policy_action(observation)
            hard = abs(label[0]-student[0]) > .03 or abs(label[1]-student[1]) > .15
            if hard or step % 6 == 0:
                seen += 1
                slot = seen-1 if seen <= max_examples else rng.randint(seen)
                if slot < max_examples:
                    pair = np.asarray(observation, dtype=np.float32).copy()
                    if slot == len(examples):
                        examples.append(pair)
                        labels.append(np.asarray(label, dtype=np.float32))
                    else:
                        examples[slot] = pair
                        labels[slot] = np.asarray(label, dtype=np.float32)
            teacher = step <= 40 or distance > .35 or rng.rand() < beta
            applied = label if teacher else student
            record['teacher_steps'] += int(teacher)
            physical = np.array([applied[0]*config.max_steer,
                                unnormalize_speed(applied[1], config.min_speed, config.max_speed)])
            raw, _, terminated, truncated, info = env.unwrapped.step(physical)
            observation = env.getObs(raw)
            info['obs_dict'] = raw
            if terminated or truncated:
                record['outcome'] = ('collision' if env.unwrapped.collisions[0] else
                    'checkpoint' if bool(np.asarray(info['checkpoint_done']).all()) else 'timeout')
                break
        record.update(steps=step, examples=len(examples), sampled_states=seen)
        return np.asarray(examples, dtype=np.float32), np.asarray(labels, dtype=np.float32), record
    finally:
        env.close()


def collect_recovery(args, actor_path, scale_path, maps, forbidden_maps, seeds, output,
                     beta=.15, workers=4, max_examples=2000):
    if set(maps) & set(forbidden_maps):
        raise ValueError('Held-out maps must never enter recovery collection')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(args, name, seed, beta, max_examples) for name in maps for seed in seeds]
    observations, actions, records = [], [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
            initializer=initialize_policy, initargs=(str(actor_path), str(scale_path))) as pool:
        for x, y, record in pool.map(collect_episode, jobs):
            if len(x):
                observations.append(x)
                actions.append(y)
            records.append(record)
            print('[Recovery]', record['map'], record['seed'], record['outcome'], record['examples'], flush=True)
    if not observations:
        raise RuntimeError('No valid recovery labels were collected')
    data = dict(observations=np.concatenate(observations), actions=np.concatenate(actions),
        metadata=dict(maps=list(maps), forbidden_maps=list(forbidden_maps), episodes=records,
                      label_source='Recovery oracle on learner-visited states; mixed control during COLLECTION only.',
                      evaluation_uses_teacher=False))
    with (output/'corrections.pkl').open('wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (output/'metadata.json').write_text(json.dumps(data['metadata'], indent=2)+'\n')
    return data
