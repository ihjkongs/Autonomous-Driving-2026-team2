# backup
from shutil import copyfile

from copy import deepcopy
import argparse
import os
import time
import random
import json
from pathlib import Path
import numpy as np
import torch
import wandb
from ruamel.yaml import YAML
from easydict import EasyDict

import gymnasium as gym

from utils.env import F1Wrapper
from utils.logger import Logger
from utils.video import create_video
from utils.color import cprint

from algorithm import algo_dict

# CUDA kernels run asynchronously during normal training.
EPS = 1e-8

from utils.expert_config import MAPS, load_expert_config
from utils.expert_data import load_verified, collect_verified
from utils.policy_evaluation import evaluate_policy, selection_key

EVAL_MAPS = ['Sochi', 'IMS', 'Catalunya', 'Zandvoort', 'Silverstone', 'SaoPaulo',       # easy
             'YasMarina', 'MoscowRaceway', 'Shanghai', 'MexicoCity', 'Montreal']        # hard

PARAM = load_expert_config()

def getParser():
    parser = argparse.ArgumentParser(description='F1tenth.')
    # mode
    parser.add_argument('--test',  action='store_true', help='for test.')
    # algorithm
    parser.add_argument('--name', type=str, default=None, help='algo name.')
    parser.add_argument('--algo_idx', type=int, default=1, help='algo index.')
    parser.add_argument('--model_num', type=int, default=None, help='for checkpoint num.')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--comment', type=str, default=None, help='wandb comment saved in run name.')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    parser.add_argument('--log_freq', type=int, default=int(1e4), help='# of time steps for logging.')
    parser.add_argument('--save_freq', type=int, default=int(5e5), help='# of time steps for save.')
    # train
    parser.add_argument('--epochs', type=int, default=5000, help='Number of minibatch gradient updates, not full dataset passes.')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Updates between fixed-suite policy evaluations.')
    parser.add_argument('--eval_workers', type=int, default=6)
    parser.add_argument('--expert_data', default='expert_data_verified.pkl', help='Verified expert dataset path.')
    parser.add_argument('--expert_max_steps', type=int, default=20000, help='Per-lap collection timeout.')
    parser.add_argument('--total_steps', type=int, default=int(1e5), help='total interaction steps.')
    # for eval
    parser.add_argument('--eval_num', type=int, default=5, help='# of evaluation epochs.')
    parser.add_argument('--eval_freq', type=int, default=int(5e4), help='# of time steps for eval.')
    parser.add_argument('--save_video', action='store_true', help='save video.')
    parser.add_argument('--live_video', action='store_true', help='live video.')
    return parser

def eval(args, env, agent, name, eval_num=1):
    cprint('[Evaluation]\t{:^15}\tLap Time'.format('Map'), color='cyan')

    steps, scores, frames = [], [], []
    for _ in range(eval_num):
        step, score, done, _frames = 0, 0, False, []
        observation, info = env.reset()
        while not done and step < env._env.max_episode_steps:
            step += 1

            with torch.no_grad():
                clipped_action_tensor = agent.getAction(observation, deterministic=True)
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                observation, reward, terminate, truncate, info = env.step(clipped_action)
            score += reward
            done = terminate or truncate
            _frames.append(env.render())

        if bool(np.asarray(info['checkpoint_done']).all()) and not env.collision:
            print('[Success]   \t{:^15}\t{:>5s}'.format(env._env.map, f'{env._env.lap_times[0]:.2f}'))
        else:
            print('[Fail]      \t{:^15}\t{:>5s}'.format(env._env.map, f'{env._env.lap_times[0]:.2f}'))

        steps.append(step)
        scores.append(score)
        frames.append(_frames)

    max_score_idx = np.argmax(scores)
    step_mean, score_mean = np.mean(steps), np.mean(scores)
    cprint(f'[Eval]-{name} \t steps: {step_mean:.2f} \t score: {score_mean:.2f}', color='cyan')
    frames = frames[max_score_idx]
    if args.save_video:
        create_video(frames, output_name=f'{args.save_dir}/video/{name}')
    return step_mean, score_mean

def train(args):
    # Keep each experiment isolated, including normalization statistics.
    if (Path(args.save_dir) / 'checkpoint/model.pt').exists():
        raise FileExistsError('Choose a new --algo_idx; an existing model will not be overwritten.')
    if args.epochs <= 0 or args.eval_interval <= 0 or args.eval_workers <= 0:
        raise ValueError('Training and evaluation limits must be positive.')
    for source in ['configs/task/f1tenth.yaml', 'configs/task/dynamic.yaml',
                   'configs/expert.yaml', 'configs/algorithm/imitation.yaml']:
        copyfile(source, str(Path(args.backup_dir) / Path(source).name))
    for source in ['main_il.py', 'algorithm/imitation/agent.py',
                   'algorithm/common/models.py', 'utils/policy_evaluation.py']:
        target = Path(args.backup_dir) / 'sources' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, str(target))
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.action_bound_min, args.action_bound_max = -np.ones(2), np.ones(2)
    args.train_frequency = args.batch_size
    agent = algo_dict['imitation'](args)
    Path(agent.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(agent.sim2real_dir).mkdir(parents=True, exist_ok=True)
    agent.actor.initialize()
    loggers = {key: Logger(args.log_dir, key) for key in
               ('actor_loss', 'score', 'ep_len', 'completion_rate', 'lap_time')}
    if args.wandb:
        wandb.init(project='[F1tenth] Baselines', config=dict(args),
                   name=f'{args.name}/{args.algo_idx}')
    if Path(args.expert_data).is_file():
        data = load_verified(args.expert_data, args, MAPS)
    else:
        data = collect_verified(args, MAPS, args.expert_data, args.total_steps, args.expert_max_steps)
    metadata = data['metadata']
    (Path(args.backup_dir) / 'dataset_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    agent.step(data['observations'], data['actions'])
    count = len(data['actions'])
    del data
    eval_args = deepcopy(args); eval_args.max_episode_steps = args.expert_max_steps
    best_key, best_update, best_summary = None, None, None
    losses = []
    start = time.monotonic()
    with (Path(args.save_dir) / 'training_metrics.jsonl').open('w') as stream:
        for update in range(1, args.epochs+1):
            metrics = agent.train(); losses.append(metrics['actor_loss'])
            loggers['actor_loss'].write(update, metrics['actor_loss'])
            if update % 100 == 0:
                row = dict(update=update, mean_loss_last_100=float(np.mean(losses[-100:])),
                           elapsed_s=time.monotonic()-start)
                stream.write(json.dumps(row)+'\n');stream.flush()
                print('[Train]', json.dumps(row), flush=True)
                if args.wandb:
                    wandb.log({'train/actor_loss': row['mean_loss_last_100']}, step=update)
            if update % args.eval_interval != 0 and update != args.epochs:
                continue
            # Candidate uses its own weights; all candidates see identical starts.
            candidate = Path(args.save_dir) / 'candidate_actor.pt'
            torch.save(agent.actor, str(candidate))
            agent.obs_rms.save(args.save_dir)
            summary, records = evaluate_policy(eval_args, candidate,
                Path(args.save_dir) / 'agent_obs_scale.pkl', MAPS, [17, 19],
                ['cl_grid_static', 'cl_random_static'],
                Path(args.save_dir) / 'selection' / str(update), workers=args.eval_workers)
            if summary['counts']['error']:
                raise RuntimeError('Policy evaluation infrastructure failed; inspect episode errors.')
            rank = selection_key(summary)
            if best_key is None or rank > best_key:
                best_key, best_update, best_summary = rank, update, summary
                agent.save(log=False)
            agent.save(model_num=update, log=False)
            row = dict(update=update, evaluation=summary, selected_update=best_update,
                       selected_rank=list(best_key))
            stream.write(json.dumps(row)+'\n');stream.flush()
            print('[Selection]', json.dumps(row), flush=True)
            loggers['score'].write(update, summary['completion_rate'])
            loggers['completion_rate'].write(update, summary['completion_rate'])
            loggers['ep_len'].write(update, float(np.mean([r['steps'] for r in records])))
            loggers['lap_time'].write(update, summary['mean_lap_s'])
            for logger in loggers.values():
                logger.save()
            if args.wandb:
                wandb.log({'eval/completion_rate': summary['completion_rate'],
                           'eval/mean_lap_s': summary['mean_lap_s']}, step=update)
    candidate.unlink(missing_ok=True)
    result = dict(updates=args.epochs, examples=count,
                  equivalent_dataset_passes=args.epochs*args.batch_size/count,
                  first_100_mean_loss=float(np.mean(losses[:100])),
                  last_100_mean_loss=float(np.mean(losses[-100:])),
                  best_update=best_update, selection=best_summary,
                  selection_rule='Completion rate, then lower completed-lap time, then progress.',
                  selection_seeds=[17, 19], selection_modes=['cl_grid_static', 'cl_random_static'],
                  input='Single LiDAR scan only (1080 ranges).',
                  device=str(args.device), elapsed_s=time.monotonic()-start,
                  dataset_signature=metadata['signature'])
    (Path(args.save_dir)/'training_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[Training finished]',json.dumps(result),flush=True)
    if args.wandb:
        wandb.finish()


def test(args):
    # for random seed
    np.random.seed(args.seed)
    random.seed(args.seed)    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define Environment
    render_mode = None
    if args.live_video:
        render_mode = "human_fast"
    if args.save_video:
        render_mode = "rgb_array"
    args.max_episode_steps = 10000
    test_env = F1Wrapper(args=args, maps=args.map if args.map else EVAL_MAPS, render_mode=render_mode)

    # set args value for env
    args.action_bound_min = test_env.action_space.low
    args.action_bound_max = test_env.action_space.high

    # define agent
    args.train_frequency = args.batch_size      # for code compatibility
    agent = algo_dict['imitation'](args)
    agent.load(args.model_num)

    # test
    eval(args, test_env, agent, name='test', eval_num=args.eval_num)


if __name__ == "__main__":
    # base configurations
    parser = getParser()
    args = parser.parse_args()
    if args.name == None:
        args.name = 'imitation'
    args = EasyDict(vars(args))

    # load configurations & merge them
    with open('configs/task/f1tenth.yaml', 'r') as f:
        task_args = EasyDict(YAML().load(f))
    with open('configs/task/dynamic.yaml', 'r') as f:
        dynamic_args = EasyDict(YAML().load(f))
    with open(f'configs/algorithm/imitation.yaml', 'r') as f:
        agent_args = EasyDict(YAML().load(f))
    args.update(task_args)
    args.update(dynamic_args)
    args.update(agent_args)
    args.save_dir = f"results/{args.name}/{args.algo_idx}"

    # ==== processing args ==== #
    # directory
    args.log_dir = f'{args.save_dir}/logs'
    args.video_dir = f'{args.save_dir}/video'
    args.backup_dir = f'{args.save_dir}/backup'
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    os.makedirs(args.backup_dir, exist_ok=True)
    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_idx}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device('cuda:0')
        cprint('[torch] cuda is used.')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.')
    args.device = device

    if args.test:
        test(args)
    else:
        train(args)
