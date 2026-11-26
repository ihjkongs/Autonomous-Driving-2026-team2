"""Offline expert evaluation on imported train/val maps, including continuous laps.

Uses simulator pose only for expert control and scoring. Does not train a policy,
collect imitation labels, change deployment settings or command a real vehicle.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

from easydict import EasyDict
import numpy as np
from ruamel.yaml import YAML

from utils.env import F1Wrapper
from utils.experiment import apply_experiment
from utils.expert_config import ROOT
from utils.rule_based import PurePursuitPlanner, nearest_point_on_trajectory


def evaluate(job):
    config, name, mode, seed, candidate, laps = job
    config = deepcopy(config)
    config.seed = seed
    config.reset_config = {'type': mode}
    random.seed(seed)
    np.random.seed(seed)
    result = dict(map=name, reset=mode, seed=seed, candidate=candidate,
                  required_laps=laps, outcome='error', steps=0)
    env = None
    try:
        env = F1Wrapper(config, [name])
        _, info = env.reset(options={'map': name})
        native = env.unwrapped
        track = native.track
        params = config.expert_parameters
        result['reference_kind'] = 'external_racing_line' if params.get('raceline_directory') else params.get('reference','centerline')
        if params.get('raceline_directory'):
            directory = Path(params['raceline_directory'])
            if not directory.is_absolute():
                directory = ROOT/directory
            result['racing_line_sha256'] = hashlib.sha256((directory/(name+'.npz')).read_bytes()).hexdigest()
        planner = PurePursuitPlanner(track, config.length_f + config.length_r,
                                     params.get('reference', 'centerline'))
        xy = np.column_stack([track.centerline.xs, track.centerline.ys]).astype(float)
        xy = np.vstack([xy, xy[0]])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        length = float(arc[-1])

        def position(raw):
            point = np.array([raw['poses_x'][0], raw['poses_y'][0]])
            _, distance, fraction, index = nearest_point_on_trajectory(point, xy)
            return float(arc[index] + fraction * (arc[index+1]-arc[index])), float(distance)

        raw = info['obs_dict']
        previous, _ = position(raw)
        progress = speed_sum = requested_sum = max_speed = max_requested = 0.
        max_error = max_steer = 0.
        fast_steps = 0
        crossing_times = []
        result['longitudinal_response'] = native.sim.agents[0].response_parameters
        for step in range(1, config.max_episode_steps + 1):
            speed, steer = planner.plan(raw, params)
            if not np.isfinite([speed, steer]).all():
                raise ValueError('Non-finite expert command')
            command = np.array([np.clip(steer, -config.max_steer, config.max_steer),
                                np.clip(speed, config.min_speed, config.max_speed)])
            raw, _, _, truncated, info = native.step(command)
            state = native.sim.agents[0].state
            if not np.isfinite(state).all() or not np.isfinite(raw['scans']).all():
                raise ValueError('Non-finite state or scan')
            actual = float(state[3])
            speed_sum += actual
            requested_sum += float(command[1])
            max_speed = max(max_speed, actual)
            max_requested = max(max_requested, float(command[1]))
            max_steer = max(max_steer, abs(float(steer)))
            fast_steps += int(actual >= 5.5)
            current, distance = position(raw)
            max_error = max(max_error, distance)
            progress += (current - previous + length/2) % length - length/2
            previous = current
            # Count full progress from the sampled start, preserving state across laps.
            while progress >= (len(crossing_times)+1)*length and len(crossing_times) < laps:
                crossing_times.append(float(native.current_time))
            result['steps'] = step
            if native.collisions[0]:
                result['outcome'] = 'collision'
                break
            if len(crossing_times) >= laps:
                result['outcome'] = 'complete'
                break
            if truncated:
                result['outcome'] = 'timeout'
                break
        else:
            result['outcome'] = 'timeout'
        result.update(elapsed_s=float(native.current_time), track_length_m=length,
                      forward_progress_m=progress, progress_fraction=progress/(laps*length),
                      completed_laps=len(crossing_times), crossing_times_s=crossing_times,
                      mean_actual_speed_mps=speed_sum/step,
                      max_actual_speed_mps=max_speed,
                      mean_requested_speed_mps=requested_sum/step,
                      max_requested_speed_mps=max_requested,
                      max_centerline_error_m=max_error, max_unclipped_steering_rad=max_steer,
                      fraction_at_or_above_5_5mps=fast_steps/step)
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    return result


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['candidate']].append(row)
    scores = []
    for name, rows in groups.items():
        complete = [r for r in rows if r['outcome'] == 'complete']
        scores.append(dict(candidate=name, episodes=len(rows),
            counts=dict(Counter(r['outcome'] for r in rows)),
            completed=len(complete),
            mean_lap_s=(sum(r['elapsed_s'] for r in complete) /
                        sum(r['required_laps'] for r in complete) if complete else None),
            mean_actual_speed_mps=(sum(r['mean_actual_speed_mps']*r['steps'] for r in complete)/
                                   sum(r['steps'] for r in complete) if complete else None),
            max_actual_speed_mps=max((r.get('max_actual_speed_mps', 0.) for r in rows), default=0.)))
    return sorted(scores, key=lambda s: (s['episodes']-s['completed'], s['mean_lap_s'] or float('inf')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'map_datasets/hj_maps_trainval_261124/manifest.json')
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='train')
    parser.add_argument('--maps', nargs='+')
    parser.add_argument('--limit-per-group', type=int)
    parser.add_argument('--experiment', type=Path, default=ROOT/'configs/experiment_6mps_randomized.yaml')
    parser.add_argument('--candidates', type=Path, help='YAML: candidate name -> expert parameter overrides')
    parser.add_argument('--seeds', type=int, nargs='+', default=[41])
    parser.add_argument('--modes', nargs='+', choices=['cl_grid_static','cl_random_static'], default=['cl_grid_static'])
    parser.add_argument('--laps', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=12000)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    if min(cli.laps, cli.max_steps, cli.workers) < 1:
        parser.error('laps, max-steps and workers must be positive')
    manifest = json.loads(cli.manifest.read_text())
    maps = [m for m in manifest['maps'] if cli.split == 'all' or m['split'] == cli.split]
    if cli.maps:
        known = {m['name'] for m in maps}
        if set(cli.maps)-known:
            parser.error('Requested maps are outside the selected split')
        maps = [m for m in maps if m['name'] in cli.maps]
    if cli.limit_per_group:
        chosen = []
        for group in sorted({m['group'] for m in maps}):
            subset = sorted([m for m in maps if m['group']==group], key=lambda m:m['name'])
            indexes = np.linspace(0, len(subset)-1, min(cli.limit_per_group,len(subset)), dtype=int)
            chosen.extend(subset[i] for i in indexes)
        maps = chosen
    config = EasyDict()
    for name in ['f1tenth', 'dynamic']:
        config.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    apply_experiment(config, cli.experiment)
    config.max_episode_steps = cli.max_steps
    candidates = YAML(typ='safe').load(cli.candidates.read_text()) if cli.candidates else {'baseline': {}}
    jobs = []
    for candidate, overrides in candidates.items():
        args = deepcopy(config)
        args.expert_parameters.update(overrides or {})
        for entry in maps:
            for mode in cli.modes:
                for seed in cli.seeds:
                    jobs.append((args, entry['name'], mode, seed, candidate, cli.laps))
    cli.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(split=cli.split, maps=maps, candidates=candidates,
                    config=dict(config), laps=cli.laps, seeds=cli.seeds, modes=cli.modes,
                    completion_rule='Full forward centerline length per lap, no collision; continuous vehicle state.',
                    manifest_sha256=hashlib.sha256(cli.manifest.read_bytes()).hexdigest(),
                    sources_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__).resolve(), ROOT/'utils/rule_based.py', ROOT/'utils/speed_profile.py', ROOT/'utils/racing_line.py',
                                  ROOT/'installation/f1tenth_gym/envs/longitudinal.py']})
    (cli.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    records = []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=cli.workers) as pool, (cli.output/'episodes.jsonl').open('w') as stream:
        futures = [pool.submit(evaluate, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            if len(records)%40 == 0 or record['outcome'] != 'complete':
                print(f"{len(records)}/{len(jobs)} {record['candidate']} {record['map']}: {record['outcome']}", flush=True)
    report = dict(elapsed_wall_s=time.monotonic()-start, candidates=summarize(records))
    (cli.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
