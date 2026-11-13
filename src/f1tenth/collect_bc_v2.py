"""Collect balanced centerline Pure Pursuit demonstrations with map metadata."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from easydict import EasyDict
from ruamel.yaml import YAML

from leaderboard import MAPS
from utils.env import F1Wrapper, unnormalize_speed
from utils.rule_based import PurePursuitPlanner
from utils.speed_profile import curvature_speed_profile


EXPERT_PARAMS = {
    "mass": 3.463388126201571,
    "lf": 0.15597534362552312,
    "tlad": 0.82461887897713965,
    "vgain": 0.35,
}
EPS = 1e-8


def make_env(seed, reset_type, max_steps, max_speed=None):
    args = EasyDict(seed=seed)
    yaml = YAML()
    for path in ("configs/task/f1tenth.yaml", "configs/task/dynamic.yaml"):
        with open(path, encoding="utf-8") as source:
            args.update(yaml.load(source))
    args.reset_config = {"type": reset_type}
    args.max_episode_steps = max_steps
    if max_speed is not None:
        args.max_speed = max_speed
    return F1Wrapper(args, maps=MAPS)


def collect_segment(env, map_name, seed, segment_steps, params, drop_collision_tail,
                    speed_profile, lateral_accel_limit, accel_limit, brake_limit):
    observation, info = env.reset(seed=seed, options={"map": map_name})
    track = env._env.unwrapped.track
    planner = PurePursuitPlanner(
        track=SimpleNamespace(raceline=track.centerline),
        wb=0.17145 + 0.15875,
    )
    if speed_profile == "curvature":
        profile = curvature_speed_profile(
            track.centerline, env.min_speed, env.max_speed, lateral_accel_limit,
            accel_limit, brake_limit,
        )
        line_points = np.column_stack((track.centerline.xs, track.centerline.ys))
        nearest = np.argmin(
            np.sum((planner.waypoints[:, None, :2] - line_points[None, :, :]) ** 2,
                   axis=2),
            axis=1,
        )
        # The planner closes the path and may drop duplicate waypoints.
        planner.waypoints[:, 2] = profile[nearest] / params["vgain"]
    start = [float(info["obs_dict"][key][0])
             for key in ("poses_x", "poses_y", "poses_theta")]
    observations, actions = [], []
    requested_speeds, applied_speeds = [], []
    outcome = "segment"

    for _ in range(segment_steps):
        speed, steer = planner.plan(info["obs_dict"], params)
        speed_action = 2 * (speed - env.min_speed) / (
            env.max_speed - env.min_speed
        ) - 1
        action = np.clip(
            [steer / (env.max_steer + EPS), speed_action], -1.0, 1.0
        )
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action.astype(np.float32))
        requested_speeds.append(float(speed))
        applied_speeds.append(
            float(unnormalize_speed(action[1], env.min_speed, env.max_speed))
        )
        observation, _, terminated, truncated, info = env.step(action)
        completed = bool(np.asarray(info["checkpoint_done"])[0])
        collision = bool(np.asarray(info["obs_dict"]["collisions"])[0])
        if terminated or truncated:
            if completed:
                outcome = "complete"
            elif collision:
                outcome = "collision"
            elif truncated:
                outcome = "timeout"
            else:
                outcome = "other"
            break

    if outcome == "collision":
        keep = max(0, len(observations) - drop_collision_tail)
        observations = observations[:keep]
        actions = actions[:keep]

    metadata = {
        "map": map_name,
        "seed": seed,
        "start_pose": start,
        "outcome": outcome,
        "steps_kept": len(observations),
        "requested_speed_mean": float(np.mean(requested_speeds)),
        "requested_speed_min": float(np.min(requested_speeds)),
        "requested_speed_max": float(np.max(requested_speeds)),
        "requested_speed_std": float(np.std(requested_speeds)),
        "applied_speed_mean": float(np.mean(applied_speeds)),
        "speed_profile": speed_profile,
    }
    return observations, actions, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="bc_centerline_v1")
    parser.add_argument("--maps", nargs="+", default=MAPS, choices=MAPS)
    parser.add_argument("--train-segments", type=int, default=4)
    parser.add_argument("--val-segments", type=int, default=1)
    parser.add_argument("--segment-steps", type=int, default=1000)
    parser.add_argument("--min-safe-steps", type=int, default=200)
    parser.add_argument("--drop-collision-tail", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--reset-type", default="cl_random_static")
    parser.add_argument("--vgain", type=float, default=EXPERT_PARAMS["vgain"])
    parser.add_argument("--tlad", type=float, default=EXPERT_PARAMS["tlad"])
    parser.add_argument("--speed-profile", choices=("constant", "curvature"),
                        default="constant")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="override task max_speed; curvature profile defaults to 8 m/s")
    parser.add_argument("--lateral-accel-limit", type=float, default=2.0)
    parser.add_argument("--accel-limit", type=float, default=1.5)
    parser.add_argument("--brake-limit", type=float, default=2.0)
    args = parser.parse_args()
    if (args.train_segments < 1 or args.val_segments < 1 or
            args.segment_steps < args.min_safe_steps or args.max_attempts < 1 or
            args.vgain <= 0 or args.tlad <= 0 or
            args.lateral_accel_limit <= 0 or args.accel_limit <= 0 or
            args.brake_limit <= 0 or
            (args.max_speed is not None and args.max_speed <= 0)):
        parser.error("invalid segment count, step count, or expert parameter")

    params = dict(EXPERT_PARAMS, vgain=args.vgain, tlad=args.tlad)
    max_speed = args.max_speed
    if max_speed is None and args.speed_profile == "curvature":
        max_speed = 8.0
    env = make_env(args.seed_base, args.reset_type, 10000, max_speed=max_speed)
    observations, actions, map_indices, episode_indices, splits = [], [], [], [], []
    records = []
    try:
        for map_index, map_name in enumerate(args.maps):
            for split_name, required in (("train", args.train_segments),
                                         ("val", args.val_segments)):
                accepted = 0
                for attempt in range(args.max_attempts):
                    if accepted >= required:
                        break
                    seed = (args.seed_base + map_index * 1000 + attempt +
                            (500000 if split_name == "val" else 0))
                    obs, act, record = collect_segment(
                        env, map_name, seed, args.segment_steps, params,
                        args.drop_collision_tail, args.speed_profile,
                        args.lateral_accel_limit, args.accel_limit,
                        args.brake_limit,
                    )
                    record["split"] = split_name
                    record["accepted"] = len(obs) >= args.min_safe_steps
                    if record["accepted"]:
                        episode_index = len(records)
                        observations.extend(obs)
                        actions.extend(act)
                        map_indices.extend([map_index] * len(obs))
                        episode_indices.extend([episode_index] * len(obs))
                        splits.extend([split_name == "val"] * len(obs))
                        accepted += 1
                    records.append(record)
                print(f"{map_name} {split_name}: {accepted}/{required} segments",
                      flush=True)
                if accepted < required:
                    raise RuntimeError(
                        f"{map_name} {split_name}: only {accepted}/{required} safe segments"
                    )
    finally:
        env.close()

    output_dir = Path("results") / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "expert_data.npz"
    np.savez(
        data_path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        map_indices=np.asarray(map_indices, dtype=np.int16),
        episode_indices=np.asarray(episode_indices, dtype=np.int32),
        is_validation=np.asarray(splits, dtype=bool),
    )
    metadata = {"name": args.name, "maps": args.maps, "params": params,
                "reset_type": args.reset_type, "segment_steps": args.segment_steps,
                "speed_profile": args.speed_profile,
                "min_speed": env.min_speed, "max_speed": env.max_speed,
                "lateral_accel_limit": args.lateral_accel_limit,
                "accel_limit": args.accel_limit,
                "brake_limit": args.brake_limit,
                "train_segments": args.train_segments,
                "val_segments": args.val_segments, "records": records}
    with open(output_dir / "expert_metadata.json", "w", encoding="utf-8") as dest:
        json.dump(metadata, dest, indent=2)
    print(f"Saved {len(observations)} steps to {data_path}", flush=True)


if __name__ == "__main__":
    main()
