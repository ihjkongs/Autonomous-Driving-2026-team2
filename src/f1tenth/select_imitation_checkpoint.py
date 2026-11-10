"""Compare saved BC checkpoints on broader fixed starts, without retraining."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
import torch
from utils.expert_config import MAPS
from utils.policy_evaluation import evaluate_policy, selection_key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--updates',nargs='+',type=int,default=[3000,4000,5000])
    parser.add_argument('--seeds',nargs='+',type=int,default=[17,19])
    parser.add_argument('--workers',type=int,default=6)
    cli=parser.parse_args();run=cli.run;output=run/'robust_selection'
    output.mkdir(exist_ok=False)
    shutil.copy2(__file__,output/'select_imitation_checkpoint.py')
    with (run/'backup/args.pkl').open('rb') as stream:args=pickle.load(stream)
    args.max_episode_steps=args.expert_max_steps
    actor=torch.load(str(run/'sim2real/actor.pt'),map_location='cpu').eval()
    key,best_update,best_summary=None,None,None
    summaries=[]
    for update in cli.updates:
        checkpoint=torch.load(str(run/'checkpoint'/f'model_{update}.pt'),map_location='cpu')
        actor.load_state_dict(checkpoint['actor'])
        candidate=output/'candidate_actor.pt';torch.save(actor,str(candidate))
        summary,_=evaluate_policy(args,candidate,run/'sim2real/agent_obs_scale.pkl',MAPS,
            cli.seeds,['cl_grid_static','cl_random_static'],output/str(update),cli.workers)
        if summary['counts']['error']:raise RuntimeError('Evaluation error; selected model unchanged.')
        summaries.append(dict(update=update,summary=summary))
        rank=selection_key(summary)
        if key is None or rank>key:
            key,best_update,best_summary=rank,update,summary
            torch.save(actor,str(output/'selected_actor.pt'))
        print('[Broader selection]',update,summary,flush=True)
    (output/'candidate_actor.pt').unlink()
    initial=run/'initial_selection';initial.mkdir(exist_ok=False)
    shutil.copytree(run/'sim2real',initial/'sim2real')
    shutil.copy2(run/'checkpoint/model.pt',initial/'model.pt')
    shutil.copy2(run/'training_summary.json',initial/'training_summary.json')
    shutil.copy2(output/'selected_actor.pt',run/'sim2real/actor.pt')
    shutil.copy2(run/'checkpoint'/f'model_{best_update}.pt',run/'checkpoint/model.pt')
    result=dict(candidates=summaries,best_update=best_update,selection=best_summary,
                seeds=cli.seeds,modes=['cl_grid_static','cl_random_static'])
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    summary=json.loads((run/'training_summary.json').read_text())
    summary.update(initial_best_update=summary['best_update'],initial_selection=summary['selection'],
                   best_update=best_update,selection=best_summary,
                   selection_seeds=cli.seeds,selection_modes=result['modes'],
                   post_training_selection=True)
    (run/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('[Selected]',best_update,best_summary,flush=True)


if __name__=='__main__':main()
