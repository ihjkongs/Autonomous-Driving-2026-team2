"""Fresh LiDAR CNN + mirrored BC + learner-state aggregation, with held-out tracks."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import time
import numpy as np
import torch
from easydict import EasyDict
from ruamel.yaml import YAML
from algorithm.imitation.lidar_policy import TemporalLidarPolicy, PhysicalLidarPolicy
from utils.expert_config import ROOT, MAPS
from utils.experiment import apply_experiment
from utils.expert_data import load_verified
from utils.policy_evaluation import evaluate_policy, selection_key
from utils.recovery_collection import collect_recovery

HELDOUT = ['Spa', 'Sakhir', 'Sepang', 'MoscowRaceway', 'Spielberg']


def split_examples(data, training_maps, heldout_maps):
    if set(training_maps) & set(heldout_maps):
        raise ValueError('Training and held-out tracks overlap')
    if set(training_maps) | set(heldout_maps) != set(data['metadata']['maps']):
        raise ValueError('Split must cover the dataset without unknown tracks')
    indices = np.concatenate([np.arange(ep['start'], ep['stop'])
        for ep in data['metadata']['episodes'] if ep['map'] in training_maps])
    if not len(indices):
        raise ValueError('Empty training split')
    return data['observations'][indices].copy(), data['actions'][indices].copy(), indices


class Sampler:
    def __init__(self, observations, actions, seed):
        self.x, self.y = observations, actions
        weights = 1. + 2.*np.abs(actions[:, 0]) + 2.*(actions[:, 1] < -.2)
        self.cdf = np.cumsum(weights, dtype=np.float64)
        self.rng = np.random.RandomState(seed)

    def batch(self, size):
        index = np.searchsorted(self.cdf, self.rng.rand(size)*self.cdf[-1])
        return self.x[index], self.y[index]


def mirror_batch(x, y, mask):
    x, y = x.clone(), y.clone()
    frames = x.reshape(len(x), 4, 1080)
    frames[mask] = frames[mask].flip(-1)
    y[mask, 0] *= -1.
    return x, y


def train_block(actor, base, corrections, updates, batch_size, device, lr, seed, stream, stage):
    actor.train()
    base_sampler = Sampler(*base, seed)
    correction_sampler = Sampler(*corrections, seed+1) if corrections is not None else None
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, updates, eta_min=lr*.15)
    start, losses = time.monotonic(), []
    for update in range(1, updates+1):
        count = batch_size//2 if correction_sampler else batch_size
        x, y = base_sampler.batch(count)
        if correction_sampler:
            cx, cy = correction_sampler.batch(batch_size-count)
            x, y = np.concatenate([x, cx]), np.concatenate([y, cy])
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        xt, yt = mirror_batch(xt, yt, torch.rand(len(xt), device=device) < .5)
        xt = (xt + .005*torch.randn_like(xt)).clamp(0., 30.)
        prediction = actor.action_mean(xt)
        error = (prediction-yt).square()
        loss = (6.*error[:, 0]+error[:, 1]).mean()/2.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        schedule.step()
        losses.append(float(loss))
        if update % 250 == 0 or update == updates:
            record = dict(stage=stage, update=update, mean_weighted_loss=float(np.mean(losses[-250:])),
                          lr=schedule.get_last_lr()[0], elapsed_s=time.monotonic()-start)
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            print('[Train recovery]', json.dumps(record), flush=True)
    actor.eval()
    return dict(updates=updates, mean_last_250_loss=float(np.mean(losses[-250:])), elapsed_s=time.monotonic()-start)


def export_actor(actor, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    cpu = deepcopy(actor).cpu().eval()
    torch.save(cpu, directory/'actor.pt')
    # No statistics from training OR held-out tracks: preprocessing lives in actor.
    with (directory/'agent_obs_scale.pkl').open('wb') as stream:
        pickle.dump([np.zeros(args.obs_dim, np.float32), np.ones(args.obs_dim, np.float32), 0.], stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'expert_data_6mps_randomized_strict_20261112.pkl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, default=ROOT/'configs/experiment_6mps_randomized.yaml')
    parser.add_argument('--heldout', nargs='+', choices=MAPS, default=HELDOUT)
    parser.add_argument('--initial-updates', type=int, default=8000)
    parser.add_argument('--round-updates', type=int, default=4000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=41)
    cli = parser.parse_args()
    if min(cli.initial_updates, cli.round_updates, cli.batch_size, cli.workers) <= 0 or cli.rounds < 0:
        parser.error('Invalid training limits')
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'backup').mkdir()
    args = EasyDict(seed=cli.seed, expert_max_steps=20000)
    for name in ('f1tenth', 'dynamic'):
        args.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    apply_experiment(args, cli.experiment)
    args.max_episode_steps = args.expert_max_steps
    if args.obs_dim != 4320:
        raise ValueError('This experiment requires four chronological LiDAR frames')
    training_maps = [name for name in MAPS if name not in cli.heldout]
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_verified(cli.dataset, args, MAPS)
    x, y, indices = split_examples(data, training_maps, cli.heldout)
    dataset_metadata = data['metadata']
    del data
    np.save(output/'backup/base_indices.npy', indices)
    manifest = dict(training_maps=training_maps, heldout_maps=cli.heldout, base_examples=len(x),
        fresh_weights=True, uses_previous_23_track_model=False, track_statistics_used=False,
        checkpoint_selection_maps=training_maps, checkpoint_selection_seed=401,
        final_seeds=[613, 22817, 99173], dataset_signature=dataset_metadata['signature'],
        method='Fresh temporal CNN, reflection augmentation, recovery-oracle DAgger; teacher only during collection.',
        device=str(device), config={k: str(v) if isinstance(v, Path) else v for k, v in vars(cli).items()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with (output/'backup/args.pkl').open('wb') as stream:
        pickle.dump(args, stream)
    source_names = ['train_recovery.py', 'algorithm/imitation/lidar_policy.py',
                    'utils/recovery_collection.py', 'utils/policy_evaluation.py', 'utils/env.py',
                    'utils/rule_based.py', 'utils/speed_profile.py', 'utils/expert_data.py',
                    'installation/f1tenth_gym/envs/base_classes.py',
                    'installation/f1tenth_gym/envs/laser_models.py',
                    'installation/f1tenth_gym/envs/longitudinal.py']
    for name in source_names:
        target = output/'backup/sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    actor = TemporalLidarPolicy().to(device)
    best_key, best_stage, best_summary = None, None, None
    corrections, stages = None, []
    with (output/'training_metrics.jsonl').open('w') as stream:
        for stage in range(cli.rounds+1):
            if stage:
                actor.load_state_dict(torch.load(output/'best_weights.pt', map_location=device))
                beta = [.25, .15, .05][min(stage-1, 2)]
                correction = collect_recovery(args, output/'sim2real/actor.pt',
                    output/'sim2real/agent_obs_scale.pkl', training_maps, cli.heldout,
                    [1000+stage*37, 1001+stage*37], output/'recovery'/str(stage),
                    beta=beta, workers=cli.workers)
                new = (correction['observations'], correction['actions'])
                corrections = new if corrections is None else (
                    np.concatenate([corrections[0], new[0]]), np.concatenate([corrections[1], new[1]]))
                del correction, new
            training = train_block(actor, (x, y), corrections,
                cli.initial_updates if stage == 0 else cli.round_updates,
                cli.batch_size, device, 3e-4 if stage == 0 else 1e-4,
                cli.seed+stage, stream, stage)
            directory = output/'stages'/str(stage)
            export_actor(actor, directory, args)
            summary, _ = evaluate_policy(args, directory/'actor.pt', directory/'agent_obs_scale.pkl',
                training_maps, [401], ['cl_grid_static', 'cl_random_static'],
                directory/'development', workers=cli.workers)
            if summary['counts']['error']:
                raise RuntimeError('Development evaluation failed')
            key = selection_key(summary)
            if best_key is None or key > best_key:
                best_key, best_stage, best_summary = key, stage, summary
                torch.save(actor.state_dict(), output/'best_weights.pt')
                export_actor(actor, output/'sim2real', args)
            record = dict(stage=stage, training=training, development=summary,
                          correction_examples=0 if corrections is None else len(corrections[0]), best_stage=best_stage)
            stages.append(record)
            (output/'stages.json').write_text(json.dumps(stages, indent=2)+'\n')
            print('[Recovery selection]', json.dumps(record), flush=True)
    # Final suite is opened once, AFTER all architecture/weight selection.
    summary, records = evaluate_policy(args, output/'sim2real/actor.pt',
        output/'sim2real/agent_obs_scale.pkl', MAPS, manifest['final_seeds'],
        ['cl_grid_static', 'cl_random_static'], output/'evaluation_final', workers=cli.workers)
    def subset(names):
        selected = [r for r in records if r['map'] in names]
        from utils.policy_evaluation import summarize
        return summarize(selected)
    result = dict(best_stage=best_stage, selection=best_summary, final=summary,
        training_tracks_final=subset(training_maps), heldout_tracks_final=subset(cli.heldout),
        all_evaluation_episodes_passed=summary['completed'] == summary['episodes'],
        teacher_used_during_evaluation=False, physical_vehicle_validated=False,
        model_sha256=hashlib.sha256((output/'sim2real/actor.pt').read_bytes()).hexdigest())
    from export_driving_policy import export
    export(output, output/'sim2real/driving_policy.ts')
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('[Recovery final]', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
