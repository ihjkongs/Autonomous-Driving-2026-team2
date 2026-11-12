"""Evaluate the current Pure Pursuit controller without collecting BC training data."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import traceback
import subprocess
from concurrent.futures import ProcessPoolExecutor

from easydict import EasyDict
import numpy as np
from ruamel.yaml import YAML

from utils.expert_config import MAPS, load_expert_config
from utils.experiment import apply_experiment

PARAM = load_expert_config()
from utils.env import F1Wrapper, normalize_speed, unnormalize_speed
from utils.rule_based import PurePursuitPlanner, nearest_point_on_trajectory


ROOT = Path(__file__).resolve().parent
RESET_MODES = ('cl_grid_static', 'cl_random_static')


def evaluate(args, map_name, mode, seed):
    random.seed(seed)
    np.random.seed(seed)
    config = EasyDict(args)
    config.seed = seed
    config.reset_config = {'type': mode}
    result = dict(map=map_name, reset=mode, seed=seed, outcome='error', steps=0)
    env = None
    writer = None
    frames = 0
    trajectory = []
    try:
        record_video = (map_name in config.get('video_maps', []) and
                        seed == config.video_seed and mode == config.video_reset)
        env = F1Wrapper(args=config, maps=[map_name], render_mode='rgb_array' if record_video else None)
        observation, info = env.reset(options={'map': map_name})
        track = env.unwrapped.track
        parameters = config.get('expert_parameters', PARAM)
        planner = PurePursuitPlanner(track, config.length_f + config.length_r, parameters.get('reference', 'raceline'))
        raw = info['obs_dict']
        # Keep the native reset pose; do not reset again using rounded observations.
        result['initial_pose'] = env.unwrapped.sim.agent_poses[0].tolist()
        result['longitudinal_response'] = env.unwrapped.sim.agents[0].response_parameters
        result['initial_expert_distance_m'] = float(nearest_point_on_trajectory(
            np.array([raw['poses_x'][0], raw['poses_y'][0]]), planner.waypoints[:, :2]
        )[1])
        center = np.column_stack([track.centerline.xs, track.centerline.ys]).astype(np.float64)
        center = np.vstack([center, center[0]])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
        def center_s(raw):
            _, _, fraction, index = nearest_point_on_trajectory(
                np.array([raw['poses_x'][0], raw['poses_y'][0]], dtype=np.float64), center)
            return float(arc[index] + fraction * (arc[index + 1] - arc[index]))
        native = config.get('step_mode', 'wrapper') == 'native'
        track_length = float(arc[-1]) if native else float(track.centerline.spline.s[-1])
        result['track_length_m'] = track_length
        previous_s = center_s(raw) if native else float(env.position_frenet[0])
        progress = 0.0
        saturated = 0
        max_yaw_rate, max_slip = 0.0, 0.0
        max_actual_speed, max_requested_speed, fast_steps = 0.0, 0.0, 0
        result['first_commands'] = []
        result['at_collection_horizon'] = None
        for step in range(1, config.max_episode_steps + 1):
            speed, steer = planner.plan(info['obs_dict'], parameters)
            action = np.clip([
                steer / config.max_steer,
                normalize_speed(speed, config.min_speed, config.max_speed),
            ], -1.0, 1.0)
            if not np.isfinite(action).all():
                raise ValueError('Non-finite expert action')
            saturated += int(abs(steer) >= config.max_steer)
            if native:
                # Identical physics and action conversion to F1Wrapper.step;
                # omit its unused reward and costly spline minimization.
                physical = np.array([action[0] * config.max_steer,
                                     unnormalize_speed(action[1], config.min_speed, config.max_speed)])
                raw, _, terminated, truncated, info = env.unwrapped.step(physical)
                info['obs_dict'] = raw
                observation = env.getObs(raw)
            else:
                observation, _, terminated, truncated, info = env.step(action)
            collision = bool(env.unwrapped.collisions[0])
            state = env.unwrapped.sim.agents[0].state
            if not np.isfinite(state).all():
                raise ValueError('Non-finite vehicle physics state')
            max_yaw_rate = max(max_yaw_rate, abs(float(state[5])))
            max_slip = max(max_slip, abs(float(state[6])))
            max_actual_speed = max(max_actual_speed, float(state[3]))
            max_requested_speed = max(max_requested_speed, float(speed))
            fast_steps += int(state[3] >= 5.5)
            if step % 8 == 1 or terminated or truncated:
                raw = info['obs_dict']
                trajectory.append([step, float(raw['poses_x'][0]), float(raw['poses_y'][0]),
                                   float(raw['poses_theta'][0]), float(speed), float(steer), float(state[3])])
                if record_video:
                    import imageio_ffmpeg
                    frame = np.ascontiguousarray(env.render())
                    if writer is None:
                        path = Path(config.output_dir) / 'video' / (map_name + '.mp4')
                        path.parent.mkdir(exist_ok=True)
                        height, width = frame.shape[:2]
                        writer = subprocess.Popen([
                            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                            '-r', str(1. / (8 * config.timestep)), '-i', 'pipe:0', '-an',
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path),
                        ], stdin=subprocess.PIPE)
                        result['video'] = str(path)
                    writer.stdin.write(frame.tobytes())
                    frames += 1
            result['steps'] = step
            if not np.isfinite(observation).all():
                raise ValueError('Non-finite LiDAR observation')
            current_s = center_s(info['obs_dict']) if native else float(env.position_frenet[0])
            delta = current_s - previous_s
            if delta < -track_length / 2:
                delta += track_length
            elif delta > track_length / 2:
                delta -= track_length
            progress += delta
            previous_s = current_s
            if step <= 10:
                result['first_commands'].append(dict(
                    step=step, requested_speed_mps=float(speed),
                    requested_steer_rad=float(steer), action=action.tolist(),
                ))
            if step == config.collection_horizon:
                result['at_collection_horizon'] = dict(
                    collision=collision, progress_fraction=progress / track_length,
                    checkpoint=bool(np.asarray(info['checkpoint_done']).all()),
                )
            if terminated or truncated:
                checkpoint = bool(np.asarray(info['checkpoint_done']).all())
                if collision:
                    result['outcome'] = 'collision'
                elif checkpoint and progress >= 0.9 * track_length:
                    result['outcome'] = 'complete'
                elif checkpoint:
                    result['outcome'] = 'early_checkpoint'
                else:
                    result['outcome'] = 'timeout'
                break
        else:
            result['outcome'] = 'timeout'
        result.update(
            elapsed_s=float(env.unwrapped.current_time),
            forward_progress_m=progress, progress_fraction=progress / track_length,
            steering_saturation_fraction=saturated / max(result['steps'], 1),
            legacy_collector_accepts_prefix=min(result['steps'], config.collection_horizon) > 200,
            max_yaw_rate_radps=max_yaw_rate, max_slip_angle_rad=max_slip,
            max_actual_speed_mps=max_actual_speed, max_requested_speed_mps=max_requested_speed,
            fraction_at_or_above_5_5mps=fast_steps / max(result['steps'], 1),
        )
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                result.update(outcome='error', error='Video encoding failed')
            result['video_frames'] = frames
        if config.get('output_dir') and trajectory:
            directory = Path(config.output_dir) / 'trajectories'
            directory.mkdir(exist_ok=True)
            np.savez_compressed(str(directory / f'{map_name}_{mode}_{seed}.npz'),
                                samples=np.asarray(trajectory))
        if env is not None:
            env.close()
    return result


def summarize(records, maps):
    rows = []
    for name in maps:
        row = {'map': name}
        for mode in RESET_MODES:
            attempts = [r for r in records if r['map'] == name and r['reset'] == mode]
            label = 'grid' if mode == 'cl_grid_static' else 'random'
            row[label + '_attempts'] = len(attempts)
            for outcome in ('complete', 'collision', 'timeout', 'error', 'early_checkpoint'):
                row[label + '_' + outcome] = sum(r['outcome'] == outcome for r in attempts)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', nargs='+', choices=MAPS, default=MAPS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 124])
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--reference', choices=['centerline', 'raceline'])
    parser.add_argument('--lookahead', type=float)
    parser.add_argument('--speed-cap', type=float)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--step-mode', choices=['wrapper', 'native'], default='wrapper')
    parser.add_argument('--video-maps', nargs='*', choices=MAPS, default=[])
    parser.add_argument('--video-seed', type=int, default=124)
    parser.add_argument('--video-reset', choices=RESET_MODES, default='cl_grid_static')
    cli = parser.parse_args()
    if cli.max_steps <= 0 or not cli.seeds or cli.workers <= 0:
        parser.error('A positive step limit and at least one seed are required.')
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d-%H%M%S')
    output = cli.output or ROOT / 'results/expert_validation' / stamp
    output.mkdir(parents=True, exist_ok=False)
    args = EasyDict()
    for name in ('f1tenth', 'dynamic'):
        path = ROOT / 'configs/task' / (name + '.yaml')
        with path.open() as f:
            args.update(YAML(typ='safe').load(f))
        shutil.copy2(path, output / path.name)
    args.collection_horizon = args.max_episode_steps
    args.max_episode_steps = cli.max_steps
    args.step_mode = cli.step_mode
    args.output_dir = str(output.resolve())
    args.video_maps, args.video_seed, args.video_reset = cli.video_maps, cli.video_seed, cli.video_reset
    args.expert_parameters = dict(PARAM)
    apply_experiment(args, cli.experiment)
    if cli.reference:
        args.expert_parameters['reference'] = cli.reference
    if cli.lookahead is not None:
        args.expert_parameters['tlad'] = cli.lookahead
    if cli.speed_cap is not None:
        args.expert_parameters['speed_cap'] = cli.speed_cap
    sources = [ROOT / 'utils/rule_based.py', ROOT / 'utils/env.py', ROOT / 'main_il.py',
               ROOT / 'utils/expert_config.py', ROOT / 'configs/expert.yaml',
               ROOT / 'installation/f1tenth_gym/envs/integrator.py',
               ROOT / 'installation/f1tenth_gym/envs/dynamic_models.py',
               ROOT / 'utils/expert_data.py', ROOT / 'collect_expert.py', Path(__file__).resolve()]
    sources += [ROOT / name for name in ['utils/speed_profile.py', 'utils/experiment.py',
                'installation/f1tenth_gym/envs/longitudinal.py',
                'installation/f1tenth_gym/envs/base_classes.py',
                'installation/f1tenth_gym/envs/laser_models.py',
                'installation/f1tenth_gym/envs/f110_env.py']]
    for source in sources:
        target = output / 'sources' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    metadata = dict(
        maps=cli.maps, seeds=cli.seeds, reset_modes=RESET_MODES,
        parameters=args.expert_parameters, vehicle_config=dict(args),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        completion_rule='No collision, simulator checkpoint, and >=90% forward centerline progress.',
        collection_rule='Balanced complete laps only, no collision, checkpoint and >=90% forward progress; stale data rejected.',
    )
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil.copy2(__file__, output / 'validate_expert.py')
    records = []
    start = time.monotonic()
    total = len(cli.maps) * len(RESET_MODES) * len(cli.seeds)
    print('Output:', output, flush=True)
    with (output / 'episodes.jsonl').open('w') as stream:
        jobs = [(args, name, mode, seed) for name in cli.maps for mode in RESET_MODES for seed in cli.seeds]
        with ProcessPoolExecutor(max_workers=cli.workers) as pool:
            for record in pool.map(evaluate_job, jobs):
                records.append(record)
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                print(f"[{len(records)}/{total}] {record['map']} {record['reset']} seed={record['seed']}: "
                      f"{record['outcome']} steps={record['steps']}", flush=True)
    rows = summarize(records, cli.maps)
    with (output / 'by_track.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        episodes=len(records), elapsed_wall_s=time.monotonic() - start,
        counts={mode: {outcome: sum(r['reset'] == mode and r['outcome'] == outcome for r in records)
                       for outcome in ('complete', 'collision', 'timeout', 'error', 'early_checkpoint')}
                for mode in RESET_MODES},
        tracks_with_all_attempts_complete=[name for name in cli.maps
            if all(r['outcome'] == 'complete' for r in records if r['map'] == name)],
        by_track=rows,
    )
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['counts']), flush=True)
    if any(record['outcome'] != 'complete' for record in records):
        raise SystemExit(1)


def evaluate_job(job):
    return evaluate(*job)


if __name__ == '__main__':
    main()
