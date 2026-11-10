from functools import partial
from importlib import import_module

from algorithm.imitation import Agent as Imitation


def _create_agent(module_name, args):
    # Load unfinished RL exercises only when that algorithm is selected.
    return import_module(module_name).Agent(args)

algo_dict = {
    'imitation':Imitation,
    'ppo': partial(_create_agent, 'algorithm.ppo'),
    'sac': partial(_create_agent, 'algorithm.sac'),
    }
