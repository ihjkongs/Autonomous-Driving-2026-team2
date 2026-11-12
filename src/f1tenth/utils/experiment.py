"""Opt-in simulator experiments; deployed configuration is never rewritten."""
from pathlib import Path
from ruamel.yaml import YAML
from utils.expert_config import ROOT, load_expert_config


def apply_experiment(args, path):
    if not path:
        return args
    path = Path(path).resolve()
    config = YAML(typ='safe').load(path.read_text())
    unknown = set(config) - {'description', 'assumptions', 'task', 'dynamics', 'expert'}
    if unknown:
        raise ValueError('Unknown experiment sections: ' + str(unknown))
    args.update(config.get('task', {}))
    args.update(config.get('dynamics', {}))
    args.expert_parameters = dict(load_expert_config(), **config.get('expert', {}))
    args.experiment_config = config
    args.experiment_path = str(path)
    args.lidar_frames = int(args.get('lidar_frames', 1))
    if args.lidar_frames < 1:
        raise ValueError('lidar_frames must be positive')
    args.obs_dim = 1080 * args.lidar_frames
    return args
