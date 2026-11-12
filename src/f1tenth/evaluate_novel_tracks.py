"""Qualify procedural test tracks with the expert, then test a frozen policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
from copy import deepcopy
from utils.policy_evaluation import evaluate_policy
from validate_expert import evaluate_job
from create_novel_tracks import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--expert-only', action='store_true')
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    with (cli.run/'backup/args.pkl').open('rb') as stream:
        args = pickle.load(stream)
    args.max_episode_steps = args.expert_max_steps
    args.collection_horizon = 1000
    args.step_mode = 'native'
    args.output_dir = str(cli.output.resolve())
    args.video_maps, args.video_seed, args.video_reset = [], 3911, 'cl_grid_static'
    maps, seeds, modes = [case[0] for case in CASES], [3911, 7717], ['cl_grid_static', 'cl_random_static']
    jobs = [(deepcopy(args), name, mode, seed) for name in maps for mode in modes for seed in seeds]
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        records = list(pool.map(evaluate_job, jobs))
    (cli.output/'expert_episodes.json').write_text(json.dumps(records, indent=2)+'\n')
    if any(r['outcome'] != 'complete' for r in records):
        raise RuntimeError('Expert cannot qualify all procedural test cases; inspect before interpreting policy score')
    if not cli.expert_only:
        summary, _ = evaluate_policy(args, cli.run/'sim2real/actor.pt',
            cli.run/'sim2real/agent_obs_scale.pkl', maps, seeds, modes,
            cli.output/'policy', workers=cli.workers)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
