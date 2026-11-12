"""Continue a completed BC run only with identical data and normalization."""
import hashlib
import json
import pickle
from pathlib import Path
from shutil import copyfile
import numpy as np
import torch


def restore_training(agent, args, dataset_signature):
    source = Path(args.resume_from).resolve()
    summary = json.loads((source / 'training_summary.json').read_text())
    if summary['dataset_signature'] != dataset_signature:
        raise ValueError('Resume requires the identical verified expert dataset signature')
    with (source / 'sim2real/agent_obs_scale.pkl').open('rb') as stream:
        mean, variance, count = pickle.load(stream)
    if (not np.array_equal(mean, agent.obs_rms.mean) or
            not np.array_equal(variance, agent.obs_rms.var) or count != agent.obs_rms.count):
        raise ValueError('Resume normalization does not match the current dataset')
    checkpoint_path = source / 'checkpoint/model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.optimizer.load_state_dict(checkpoint['optim'])
    backup = Path(args.backup_dir)
    copyfile(checkpoint_path, backup / 'initial_model.pt')
    copyfile(source / 'sim2real/agent_obs_scale.pkl', backup / 'initial_obs_scale.pkl')
    return dict(run=str(source), source_best_update=summary['best_update'],
                selection=summary['selection'],
                checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
