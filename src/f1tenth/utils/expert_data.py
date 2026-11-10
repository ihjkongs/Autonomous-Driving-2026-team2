"""Collect whole, collision-free laps; reject stale or unverified BC data."""
from copy import deepcopy
import hashlib
import json
import pickle
from pathlib import Path
import random

import numpy as np

from utils.env import F1Wrapper, normalize_speed
from utils.expert_config import ROOT, load_expert_config
from utils.rule_based import PurePursuitPlanner

VEHICLE_KEYS = ('timestep', 'min_speed', 'max_speed', 'max_steer', 'length_f',
                'length_r', 'mu', 'C_Sf', 'C_Sr', 'h', 'm', 'I', 's_min', 's_max',
                'sv_min', 'sv_max', 'v_switch', 'a_max', 'v_min', 'v_max',
                'width', 'length', 'lidar_pos_offset', 'model', 'integrator',
                'control_input', 'num_agents', 'obs_dim', 'action_dim')


def signature(args):
    sources = ['utils/rule_based.py', 'utils/expert_data.py', 'utils/env.py',
               'utils/expert_config.py', 'configs/expert.yaml',
               'installation/f1tenth_gym/envs/integrator.py',
               'installation/f1tenth_gym/envs/dynamic_models.py']
    payload = dict(parameters=load_expert_config(),
                   vehicle={key: args[key] for key in VEHICLE_KEYS},
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in sources})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_verified(path, args, maps):
    with Path(path).open('rb') as stream:
        data = pickle.load(stream)
    meta = data.get('metadata', {})
    if meta.get('signature') != signature(args):
        raise ValueError(f'{path}: stale or unverified expert data. Collect a new dataset with collect_expert.py.')
    episodes = meta.get('episodes', [])
    if (set(meta.get('maps', [])) != set(maps) or
            set(episode.get('map') for episode in episodes) != set(maps) or not episodes or any(
            episode.get('outcome') != 'complete' for episode in episodes)):
        raise ValueError('Expert dataset must contain verified complete laps for every requested track.')
    observations, actions = data['observations'], data['actions']
    if (observations.ndim != 2 or observations.shape[1] != args.obs_dim or
            actions.shape != (len(observations), args.action_dim) or
            not np.isfinite(observations).all() or not np.isfinite(actions).all() or
            np.any(np.abs(actions) > 1.0)):
        raise ValueError('Invalid expert observations/actions')
    return data


def collect_verified(args, maps, path, total_steps=100000, max_steps=20000):
    path = Path(path)
    if total_steps <= 0 or max_steps <= 0 or not maps:
        raise ValueError('Positive step limits and at least one track are required.')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}; choose a new output path.')
    config = deepcopy(args)
    config.max_episode_steps = max_steps
    config.reset_config = {'type': 'cl_random_static'}
    np.random.seed(config.seed)
    random.seed(config.seed)
    parameters = load_expert_config()
    env = F1Wrapper(config, maps)
    observations, actions, episodes = [], [], []
    total, cycle = 0, 0
    try:
        # Finish each full round of tracks: no track is silently omitted.
        while total < total_steps or not episodes:
            for name in maps:
                observation, info = env.reset(options={'map': name})
                track = env.unwrapped.track
                planner = PurePursuitPlanner(track, config.length_f + config.length_r,
                                             parameters['reference'])
                ep_observations, ep_actions = [], []
                initial_pose = env.unwrapped.sim.agent_poses[0].tolist()
                progress = 0.0
                for step in range(1, max_steps + 1):
                    speed, steer = planner.plan(info['obs_dict'], parameters)
                    action = np.clip([steer / config.max_steer,
                                      normalize_speed(speed, config.min_speed, config.max_speed)], -1., 1.)
                    if not np.isfinite(action).all() or not np.isfinite(observation).all():
                        raise ValueError('Non-finite expert observation or action')
                    ep_observations.append(np.asarray(observation, dtype=np.float32))
                    ep_actions.append(np.asarray(action, dtype=np.float32))
                    observation, _, terminated, truncated, info = env.step(action)
                    progress += float(env.delta_s)
                    if terminated or truncated:
                        break
                complete = (not bool(env.collision) and
                            bool(np.asarray(info['checkpoint_done']).all()) and
                            progress >= .9 * float(track.centerline.spline.s[-1]))
                if not complete:
                    raise RuntimeError(f'{name}: expert failed at {step} steps; dataset was NOT saved.')
                start = total
                observations.append(np.asarray(ep_observations, dtype=np.float32))
                actions.append(np.asarray(ep_actions, dtype=np.float32))
                total += step
                episodes.append(dict(map=name, cycle=cycle, start=start, stop=total,
                                     outcome='complete', steps=step, initial_pose=initial_pose,
                                     elapsed_s=float(env.unwrapped.current_time),
                                     progress_fraction=progress / float(track.centerline.spline.s[-1])))
                print(f'[Expert collection] {name}: complete, {step} steps, total={total}', flush=True)
            cycle += 1
    finally:
        env.close()
    data = dict(observations=np.concatenate(observations), actions=np.concatenate(actions),
                metadata=dict(signature=signature(args), parameters=parameters,
                              maps=list(maps), seed=config.seed, episodes=episodes,
                              reset_mode='cl_random_static', collection_max_steps=max_steps,
                              acceptance='Whole lap, no collision, checkpoint and >=90% forward progress'))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    path.with_suffix('.metadata.json').write_text(json.dumps(data['metadata'], indent=2) + '\n')
    return data
