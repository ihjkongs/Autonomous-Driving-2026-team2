"""Train BC over every balanced demonstration batch with episode holdout."""

import argparse
import json
import pickle
import random
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from easydict import EasyDict
from ruamel.yaml import YAML

from algorithm.common.models import Actor
from utils.vectorize import RunningMeanStd


def load_config(device):
    config = EasyDict()
    yaml = YAML()
    for path in ("configs/task/f1tenth.yaml", "configs/task/dynamic.yaml",
                 "configs/algorithm/imitation.yaml"):
        with open(path, encoding="utf-8") as source:
            config.update(yaml.load(source))
    config.device = device
    return config


def mse_on_indices(actor, observations, actions, indices, obs_rms, batch_size,
                   device):
    actor.eval()
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            states = torch.as_tensor(
                obs_rms.normalize(observations[chosen]), dtype=torch.float32,
                device=device,
            )
            labels = torch.as_tensor(actions[chosen], device=device)
            total_loss += torch.nn.functional.mse_loss(
                actor(states).loc, labels, reduction="sum"
            ).item()
    return total_loss / (len(indices) * actions.shape[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="bc_centerline_v1")
    parser.add_argument("--algo_idx", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "gpu" and
                          torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_path = Path("results") / args.name / "expert_data.npz"
    with np.load(data_path) as data:
        observations = data["observations"]
        actions = data["actions"]
        is_validation = data["is_validation"]
        map_indices = data["map_indices"]
        episode_indices = data["episode_indices"]
    train_indices = np.flatnonzero(~is_validation)
    val_indices = np.flatnonzero(is_validation)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("training and validation sets must both be nonempty")
    if set(episode_indices[train_indices]) & set(episode_indices[val_indices]):
        raise ValueError("an episode appears in both training and validation")
    if set(map_indices[train_indices]) != set(map_indices[val_indices]):
        raise ValueError("training and validation map sets differ")

    config = load_config(device)
    if observations.shape[1] != config.obs_dim or actions.shape[1] != config.action_dim:
        raise ValueError("demonstration dimensions do not match task configuration")
    obs_rms = RunningMeanStd("agent_obs", config.obs_dim)
    obs_rms.update(observations[train_indices])
    actor = Actor(config).to(device)
    actor.initialize()
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)

    result_dir = Path("results") / args.name / str(args.algo_idx)
    checkpoint_dir = result_dir / "checkpoint"
    sim2real_dir = result_dir / "sim2real"
    backup_dir = result_dir / "backup"
    for directory in (checkpoint_dir, sim2real_dir, backup_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for source in ("configs/task/f1tenth.yaml", "configs/task/dynamic.yaml",
                   "configs/algorithm/imitation.yaml"):
        copyfile(source, backup_dir / Path(source).name)
    with open(backup_dir / "args.pkl", "wb") as destination:
        pickle.dump(config, destination)
    obs_rms.save(str(sim2real_dir))
    RunningMeanStd("agent_reward", 1).save(str(sim2real_dir))

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        actor.train()
        shuffled = np.random.permutation(train_indices)
        total_loss = 0.0
        for start in range(0, len(shuffled), args.batch_size):
            chosen = shuffled[start:start + args.batch_size]
            states = torch.as_tensor(
                obs_rms.normalize(observations[chosen]), dtype=torch.float32,
                device=device,
            )
            labels = torch.as_tensor(actions[chosen], device=device)
            loss = torch.nn.functional.mse_loss(actor(states).loc, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * len(chosen)
        train_mse = total_loss / len(train_indices)
        val_mse = mse_on_indices(
            actor, observations, actions, val_indices, obs_rms, args.batch_size,
            device,
        )
        torch.save(actor, checkpoint_dir / f"actor_epoch_{epoch}.pt")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(actor, sim2real_dir / "actor.pt")
            best_epoch = epoch
        record = {"epoch": epoch, "train_mse": train_mse,
                  "val_mse": val_mse, "best_val_epoch": best_epoch}
        history.append(record)
        print(record, flush=True)

    with open(result_dir / "training_history.json", "w", encoding="utf-8") as dest:
        json.dump({"name": args.name, "algo_idx": args.algo_idx,
                   "train_samples": len(train_indices),
                   "val_samples": len(val_indices), "best_epoch": best_epoch,
                   "history": history}, dest, indent=2)
    print(f"Saved best validation actor (epoch {best_epoch}) to {sim2real_dir}")


if __name__ == "__main__":
    main()
