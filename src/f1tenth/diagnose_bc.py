"""Compare a saved BC actor with its Pure Pursuit expert on fixed map starts."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from easydict import EasyDict
from ruamel.yaml import YAML

from leaderboard import MAPS
from utils.env import F1Wrapper, unnormalize_speed
from utils.rule_based import PurePursuitPlanner
from utils.speed_profile import curvature_speed_profile
from utils.vectorize import RunningMeanStd


EXPERT_PARAMS = {
    "mass": 3.463388126201571,
    "lf": 0.15597534362552312,
    "tlad": 0.82461887897713965,
    "vgain": 0.5,
}
EPS = 1e-8


def load_args(reset_type, max_steps, seed, max_speed=None):
    args = EasyDict()
    yaml = YAML()
    for path in ("configs/task/f1tenth.yaml", "configs/task/dynamic.yaml"):
        with open(path, encoding="utf-8") as source:
            args.update(yaml.load(source))
    args.reset_config = {"type": reset_type}
    args.max_episode_steps = max_steps
    args.seed = seed
    if max_speed is not None:
        args.max_speed = max_speed
    return args


def run_episode(env, policy, actor, obs_rms, device, map_name, seed, speed_label,
                expert_params, expert_line, speed_profile, lateral_accel_limit,
                accel_limit, brake_limit):
    observation, info = env.reset(seed=seed, options={"map": map_name})
    planner = None
    if policy == "expert":
        track = env._env.unwrapped.track
        if expert_line == "centerline":
            track = SimpleNamespace(raceline=track.centerline)
        planner = PurePursuitPlanner(
            track=track, wb=0.17145 + 0.15875
        )
        if speed_profile == "curvature":
            line = env._env.unwrapped.track.centerline if expert_line == "centerline" else env._env.unwrapped.track.raceline
            profile = curvature_speed_profile(
                line, env.min_speed, env.max_speed, lateral_accel_limit,
                accel_limit, brake_limit,
            )
            line_points = np.column_stack((line.xs, line.ys))
            nearest = np.argmin(
                np.sum((planner.waypoints[:, None, :2] - line_points[None, :, :]) ** 2,
                       axis=2),
                axis=1,
            )
            planner.waypoints[:, 2] = profile[nearest] / expert_params["vgain"]

    requested_speeds = []
    applied_speeds = []
    steps = 0
    while steps < env._env.unwrapped.max_episode_steps:
        if policy == "actor":
            state = torch.as_tensor(
                obs_rms.normalize(observation), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                action = actor(state).loc.cpu().numpy()
        else:
            speed, steer = planner.plan(info["obs_dict"], expert_params)
            if speed_label == "original":
                speed_action = speed / env.max_speed
            else:
                speed_action = (
                    2 * (speed - env.min_speed) / (env.max_speed - env.min_speed) - 1
                )
            action = np.clip(
                [steer / (env.max_steer + EPS), speed_action], -1.0, 1.0
            )
            requested_speeds.append(float(speed))
            applied_speeds.append(
                float(unnormalize_speed(action[1], env.min_speed, env.max_speed))
            )

        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    completed = bool(np.asarray(info["checkpoint_done"])[0])
    collision = bool(np.asarray(info["obs_dict"]["collisions"])[0])
    if completed:
        outcome = "complete"
    elif collision:
        outcome = "collision"
    elif truncated or steps >= env._env.unwrapped.max_episode_steps:
        outcome = "timeout"
    else:
        outcome = "other"

    result = {
        "map": map_name,
        "seed": seed,
        "outcome": outcome,
        "steps": steps,
        "sim_seconds": float(env._env.unwrapped.current_time),
        "lap_seconds": float(env._env.unwrapped.lap_times[0]) if completed else None,
    }
    if policy == "expert":
        result["requested_speed_mean"] = float(np.mean(requested_speeds))
        result["applied_speed_mean"] = float(np.mean(applied_speeds))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("actor", "expert"), required=True)
    parser.add_argument("--name", default="bc")
    parser.add_argument("--algo_idx", type=int, default=1)
    parser.add_argument("--actor-path", default=None,
                        help="full Actor checkpoint; default is sim2real/actor.pt")
    parser.add_argument("--tag", default=None, help="suffix for the JSON report filename")
    parser.add_argument("--maps", nargs="+", default=MAPS, choices=MAPS)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=124)
    parser.add_argument("--reset-type", default="cl_grid_static")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--speed-label", choices=("original", "inverse"), default="original")
    parser.add_argument("--expert-line", choices=("raceline", "centerline"),
                        default="raceline")
    parser.add_argument("--vgain", type=float, default=EXPERT_PARAMS["vgain"])
    parser.add_argument("--tlad", type=float, default=EXPERT_PARAMS["tlad"])
    parser.add_argument("--speed-profile", choices=("constant", "curvature"),
                        default="constant")
    parser.add_argument("--max-speed", type=float, default=None,
                        help="override task max_speed; curvature profile defaults to 8 m/s")
    parser.add_argument("--lateral-accel-limit", type=float, default=2.0)
    parser.add_argument("--accel-limit", type=float, default=1.5)
    parser.add_argument("--brake-limit", type=float, default=2.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if (args.trials < 1 or args.max_steps < 1 or args.vgain <= 0 or
            args.tlad <= 0 or args.lateral_accel_limit <= 0 or
            args.accel_limit <= 0 or args.brake_limit <= 0 or
            (args.max_speed is not None and args.max_speed <= 0)):
        parser.error("--trials, --max-steps, --vgain and --tlad must be positive")

    max_speed = args.max_speed
    if max_speed is None and args.speed_profile == "curvature":
        max_speed = 8.0
    task_args = load_args(args.reset_type, args.max_steps, args.seed_base,
                          max_speed=max_speed)
    env = F1Wrapper(task_args, maps=MAPS)
    actor = obs_rms = None
    if args.policy == "actor":
        model_dir = Path("results") / args.name / str(args.algo_idx) / "sim2real"
        obs_rms = RunningMeanStd("agent_obs", env.obs_dim)
        obs_rms.load(str(model_dir))
        actor_path = args.actor_path or str(model_dir / "actor.pt")
        actor = torch.load(actor_path, map_location=args.device).eval()

    episodes = []
    expert_params = dict(EXPERT_PARAMS, vgain=args.vgain, tlad=args.tlad)
    try:
        for map_name in args.maps:
            for trial in range(args.trials):
                seed = args.seed_base + trial
                result = run_episode(
                    env, args.policy, actor, obs_rms, args.device,
                    map_name, seed, args.speed_label, expert_params,
                    args.expert_line, args.speed_profile,
                    args.lateral_accel_limit, args.accel_limit, args.brake_limit,
                )
                episodes.append(result)
            per_map = episodes[-args.trials:]
            completed = sum(item["outcome"] == "complete" for item in per_map)
            print(f"{map_name}: {completed}/{args.trials} complete; "
                  f"outcomes={[item['outcome'] for item in per_map]}", flush=True)
    finally:
        env.close()

    totals = {outcome: sum(item["outcome"] == outcome for item in episodes)
              for outcome in ("complete", "collision", "timeout", "other")}
    report = {"policy": args.policy, "speed_label": args.speed_label,
              "expert_line": args.expert_line,
              "speed_profile": args.speed_profile,
              "max_speed": env.max_speed,
              "actor_path": actor_path if args.policy == "actor" else None,
              "reset_type": args.reset_type, "seed_base": args.seed_base,
              "trials": args.trials, "max_steps": args.max_steps,
              "expert_params": expert_params if args.policy == "expert" else None,
              "totals": totals, "episodes": episodes}
    output_dir = Path("results") / args.name / str(args.algo_idx) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = ""
    if args.policy == "expert":
        settings = (f"_{args.expert_line}_vgain{args.vgain:g}_"
                    f"tlad{args.tlad:g}")
        if args.speed_profile == "curvature":
            settings += (f"_curvature_a{args.lateral_accel_limit:g}_"
                         f"acc{args.accel_limit:g}_brake{args.brake_limit:g}")
    if args.tag:
        settings += f"_{args.tag}"
    output_path = output_dir / (
        f"{args.policy}_{args.speed_label}_{args.reset_type}_{args.seed_base}_"
        f"{args.trials}{settings}.json"
    )
    with open(output_path, "w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2)
    print(f"Totals: {totals}; report: {output_path}", flush=True)


if __name__ == "__main__":
    main()
