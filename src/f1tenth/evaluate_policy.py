"""Evaluate a saved LiDAR neural network without expert interventions."""
import argparse
from pathlib import Path
import pickle
from utils.expert_config import MAPS
from utils.policy_evaluation import evaluate_policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[1,124])
    parser.add_argument('--maps',nargs='+',choices=MAPS,default=MAPS)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--video-seed',type=int)
    cli=parser.parse_args()
    with (cli.run/'backup/args.pkl').open('rb') as stream:args=pickle.load(stream)
    args.max_episode_steps=args.expert_max_steps
    summary,_=evaluate_policy(args,cli.run/'sim2real/actor.pt',cli.run/'sim2real/agent_obs_scale.pkl',
        cli.maps,cli.seeds,['cl_grid_static','cl_random_static'],cli.output,cli.workers,cli.video_seed)
    print(summary,flush=True)
    if summary['counts']['error']:raise SystemExit(1)


if __name__=='__main__':main()
