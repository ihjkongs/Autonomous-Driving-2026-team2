"""Compare a student's commands with expert labels on its own visited states."""
import argparse
import json
import pickle
from pathlib import Path
import random
import numpy as np
from utils.env import F1Wrapper, unnormalize_speed
from utils.rule_based import PurePursuitPlanner, nearest_point_on_trajectory
from utils.expert_config import load_expert_config, MAPS
from utils.policy_evaluation import initialize_policy, policy_action


def diagnose(args, name, seed):
    args.seed, args.reset_config = seed, {'type': 'cl_random_static'}
    np.random.seed(seed)
    random.seed(seed)
    env = F1Wrapper(args, [name])
    records = []
    try:
        observation, info = env.reset(options={'map': name})
        planner = PurePursuitPlanner(env.unwrapped.track, args.length_f+args.length_r,
                                    args.get('expert_parameters', {}).get('reference', 'centerline'))
        parameters = args.get('expert_parameters', load_expert_config())
        front_mask = np.abs(np.linspace(-2.35, 2.35, 1080)) <= np.deg2rad(15)
        steer_errors, excess_speeds = [], []
        outcome = 'timeout'
        for step in range(1, args.max_episode_steps+1):
            raw = info['obs_dict']
            action = policy_action(observation)
            requested_speed = float(unnormalize_speed(action[1], args.min_speed, args.max_speed))
            requested_steer = float(action[0]*args.max_steer)
            expert_speed, expert_steer = planner.plan(raw, parameters)
            actual_speed = float(env.unwrapped.sim.agents[0].state[3])
            _, lateral_distance, _, _ = nearest_point_on_trajectory(
                np.array([raw['poses_x'][0], raw['poses_y'][0]]), planner.waypoints[:, :2])
            error = abs(requested_steer-float(np.clip(expert_steer, -args.max_steer, args.max_steer)))
            steer_errors.append(error)
            excess_speeds.append(max(0., requested_speed-expert_speed))
            if step <= 20 or step % 8 == 1:
                records.append(dict(time_s=float(env.unwrapped.current_time),
                    actual_speed_mps=actual_speed, requested_speed_mps=requested_speed,
                    expert_speed_mps=float(expert_speed), requested_steer_rad=requested_steer,
                    expert_steer_rad=float(expert_steer), distance_from_reference_m=float(lateral_distance),
                    front_median_m=float(np.median(raw['scans'][0][front_mask]))))
            raw, _, terminated, truncated, info = env.unwrapped.step(np.array([requested_steer, requested_speed]))
            observation = env.getObs(raw)
            info['obs_dict'] = raw
            if terminated or truncated:
                outcome = ('collision' if env.unwrapped.collisions[0] else
                           'checkpoint' if bool(np.asarray(info['checkpoint_done']).all()) else 'timeout')
                break
        return dict(map=name, seed=seed, outcome=outcome, steps=step,
            elapsed_s=float(env.unwrapped.current_time),
            mean_abs_steering_label_error_rad=float(np.mean(steer_errors)),
            mean_positive_speed_label_error_mps=float(np.mean(excess_speeds)),
            final_distance_from_reference_m=float(lateral_distance), samples=records)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--maps', nargs='+', choices=MAPS, default=['IMS', 'Shanghai', 'Montreal'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[53])
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    with (cli.run/'backup/args.pkl').open('rb') as stream:
        args = pickle.load(stream)
    args.max_episode_steps = args.expert_max_steps
    initialize_policy(cli.run/'sim2real/actor.pt', cli.run/'sim2real/agent_obs_scale.pkl')
    results = []
    for name in cli.maps:
        for seed in cli.seeds:
            result = diagnose(args, name, seed)
            results.append(result)
            print(name, seed, result['outcome'], result['mean_abs_steering_label_error_rad'], flush=True)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(dict(run=str(cli.run), episodes=results,
        note='Student alone controls the car. Expert labels are diagnostic comparisons on visited states; checkpoint is not an independently measured full lap.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
