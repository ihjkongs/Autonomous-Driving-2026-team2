"""Collect balanced, verified expert laps without training a neural network."""
import argparse
from pathlib import Path
from easydict import EasyDict
from ruamel.yaml import YAML
from utils.expert_config import ROOT, MAPS
from utils.expert_data import collect_verified
from utils.experiment import apply_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'expert_data_verified.pkl')
    parser.add_argument('--total-steps', type=int, default=100000)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--step-mode', choices=['wrapper', 'native'], default='wrapper')
    parser.add_argument('--maps', nargs='+', choices=MAPS, default=MAPS)
    cli = parser.parse_args()
    if cli.total_steps <= 0 or cli.max_steps <= 0:
        parser.error('Step limits must be positive.')
    config = EasyDict()
    for name in ('f1tenth', 'dynamic'):
        with (ROOT / 'configs/task' / (name + '.yaml')).open() as stream:
            config.update(YAML(typ='safe').load(stream))
    config.seed = cli.seed
    config.collection_step_mode = cli.step_mode
    apply_experiment(config, cli.experiment)
    data = collect_verified(config, cli.maps, cli.output, cli.total_steps, cli.max_steps)
    print(f"Saved {len(data['actions'])} verified examples to {cli.output}")


if __name__ == '__main__':
    main()
