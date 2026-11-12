"""Pure LiDAR policy rollouts. Pose/map are used ONLY to score progress."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import random
import subprocess
import traceback

import numpy as np
import torch

from utils.env import F1Wrapper, unnormalize_speed
from utils.rule_based import nearest_point_on_trajectory

_ACTOR = None
_MEAN = None
_SCALE = None
_PHYSICAL_CONTRACT = None


def initialize_policy(actor_path, scale_path):
    global _ACTOR, _MEAN, _SCALE, _PHYSICAL_CONTRACT
    torch.set_num_threads(1)
    _PHYSICAL_CONTRACT = None
    if str(actor_path).endswith('.ts'):
        extra = {'metadata.json': ''}
        _ACTOR = torch.jit.load(str(actor_path), map_location='cpu', _extra_files=extra).eval()
        _PHYSICAL_CONTRACT = json.loads(extra['metadata.json'])
        if _PHYSICAL_CONTRACT['format'] != 'f1tenth-physical-lidar-v1':
            raise ValueError('Unsupported standalone policy format')
        _MEAN, _SCALE = 0., 1.
        return
    _ACTOR = torch.load(actor_path, map_location='cpu').eval()
    with open(scale_path, 'rb') as stream:
        _MEAN, variance, _ = pickle.load(stream)
    _SCALE = np.sqrt(variance + 1e-8)


def policy_action(scan):
    # No position, track identity, waypoints, velocity or expert command enters.
    state = np.asarray((np.asarray(scan) - _MEAN) / _SCALE, dtype=np.float32)
    with torch.inference_mode():
        if _PHYSICAL_CONTRACT is not None:
            physical = _ACTOR(torch.from_numpy(state).unsqueeze(0)).numpy().reshape(-1)
            contract = _PHYSICAL_CONTRACT
            normalized = np.array([physical[0]/contract['max_steer'],
                2.*(physical[1]-contract['min_speed'])/(contract['max_speed']-contract['min_speed'])-1.])
            return np.clip(normalized, -1., 1.)
        return np.clip(_ACTOR(torch.from_numpy(state)).loc.numpy(), -1., 1.)


def episode(job):
    args, name, mode, seed, video_path = job
    config = deepcopy(args)
    required_laps = int(config.get('evaluation_laps', 1))
    if required_laps < 1:
        raise ValueError('evaluation_laps must be positive')
    config.seed, config.reset_config = seed, {'type': mode}
    random.seed(seed); np.random.seed(seed)
    result = dict(map=name, reset=mode, seed=seed, outcome='error', steps=0,
                  required_laps=required_laps)
    env, writer, frames = None, None, 0
    try:
        env = F1Wrapper(config, [name], render_mode='rgb_array' if video_path else None)
        observation, info = env.reset(options={'map': name})
        result['initial_pose'] = env.unwrapped.sim.agent_poses[0].tolist()
        result['longitudinal_response'] = env.unwrapped.sim.agents[0].response_parameters
        max_actual_speed, fast_steps = 0., 0
        center = np.column_stack([env.unwrapped.track.centerline.xs,
                                  env.unwrapped.track.centerline.ys]).astype(np.float64)
        center = np.vstack([center, center[0]])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
        length = float(arc[-1])
        def progress_at(raw):
            _, _, fraction, index = nearest_point_on_trajectory(
                np.array([raw['poses_x'][0], raw['poses_y'][0]], dtype=np.float64), center)
            return float(arc[index] + fraction * (arc[index + 1] - arc[index]))
        previous, progress = progress_at(info['obs_dict']), 0.
        for step in range(1, config.max_episode_steps + 1):
            action = policy_action(observation)
            if action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError('Invalid learned action')
            physical = np.array([action[0]*config.max_steer,
                                 unnormalize_speed(action[1], config.min_speed, config.max_speed)])
            raw, _, terminated, truncated, info = env.unwrapped.step(physical)
            observation = env.getObs(raw)
            actual_speed = float(env.unwrapped.sim.agents[0].state[3])
            max_actual_speed = max(max_actual_speed, actual_speed)
            fast_steps += int(actual_speed >= 5.5)
            if not np.isfinite(observation).all() or not np.isfinite(env.unwrapped.sim.agents[0].state).all():
                raise ValueError('Non-finite scan or physics state')
            current = progress_at(raw)
            delta = (current - previous + length/2.) % length - length/2.
            progress += delta; previous = current
            result['steps'] = step
            if video_path and (step % 8 == 1 or terminated or truncated):
                import imageio_ffmpeg
                frame = np.ascontiguousarray(env.render())
                if writer is None:
                    path = Path(video_path);path.parent.mkdir(parents=True, exist_ok=True)
                    h,w = frame.shape[:2]
                    writer = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner',
                        '-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
                        '-r',str(1/(8*config.timestep)),'-i','pipe:0','-an','-c:v','libx264',
                        '-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
                writer.stdin.write(frame.tobytes());frames += 1
            collision = bool(env.unwrapped.collisions[0])
            # Keep state and scan history across finish crossings for repeated laps.
            # The simulator's normal checkpoint termination is for one lap only.
            reached_laps = required_laps > 1 and progress >= required_laps*length-.5
            if collision or truncated or reached_laps or (terminated and required_laps == 1):
                checkpoint = bool(np.asarray(info['checkpoint_done']).all())
                result['outcome'] = ('collision' if collision else 'complete'
                    if reached_laps or (required_laps == 1 and checkpoint and progress >= .9*length)
                    else 'early_checkpoint' if checkpoint and required_laps == 1 else 'timeout')
                break
        else:
            result['outcome'] = 'timeout'
        result.update(elapsed_s=float(env.unwrapped.current_time), forward_progress_m=progress,
                      track_length_m=length, progress_fraction=progress/(required_laps*length),
                      completed_laps=max(0, int((progress+.5)/length)),
                      max_actual_speed_mps=max_actual_speed,
                      fraction_at_or_above_5_5mps=fast_steps/max(1, result['steps']))
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        if writer:
            writer.stdin.close()
            if writer.wait() != 0:
                result.update(outcome='error',error='Video encoding failed')
            result.update(video=str(video_path),video_frames=frames)
        if env is not None: env.close()
    return result


def summarize(records):
    completed = [r for r in records if r['outcome']=='complete']
    return dict(episodes=len(records), completed=len(completed),
                completion_rate=len(completed)/max(1,len(records)),
                mean_episode_s=float(np.mean([r['elapsed_s'] for r in completed])) if completed else None,
                mean_lap_s=float(np.mean([r['elapsed_s']/r.get('required_laps', 1) for r in completed])) if completed else None,
                mean_progress=float(np.mean([min(1.,max(0.,r.get('progress_fraction',0.))) for r in records])),
                counts={outcome:sum(r['outcome']==outcome for r in records)
                        for outcome in ['complete','collision','timeout','early_checkpoint','error']})


def selection_key(summary):
    """More completions always beat lap time; progress breaks all-failure ties."""
    return (summary['completion_rate'],
            -summary['mean_lap_s'] if summary['mean_lap_s'] is not None else 0.,
            summary['mean_progress'])


def evaluate_policy(args, actor_path, scale_path, maps, seeds, modes,
                    output, workers=6, video_seed=None):
    output = Path(output);output.mkdir(parents=True, exist_ok=False)
    jobs=[]
    for name in maps:
        for mode in modes:
            for seed in seeds:
                video = str(output/'video'/(name+'.mp4')) if seed==video_seed and mode=='cl_grid_static' else None
                jobs.append((args,name,mode,seed,video))
    records=[]
    with (output/'episodes.jsonl').open('w') as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                initializer=initialize_policy,initargs=(str(actor_path),str(scale_path))) as pool:
            for result in pool.map(episode,jobs):
                records.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
                print(f"[Policy {len(records)}/{len(jobs)}] {result['map']} {result['reset']} "
                      f"seed={result['seed']} {result['outcome']} steps={result['steps']}",flush=True)
    summary=summarize(records)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary, records
