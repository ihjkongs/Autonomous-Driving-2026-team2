"""Fresh small-map racing-line BC/DAgger; validation maps never label training."""
import argparse
from collections import defaultdict
from datetime import datetime
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
from algorithm.imitation.lidar_policy import TemporalLidarPolicy
from utils.expert_config import ROOT
from utils.experiment import apply_experiment
from utils.small_map_learning import collect, select_maps, DiskSampler
from utils.policy_evaluation import evaluate_policy, selection_key, summarize
from train_recovery import export_actor, mirror_batch


def configuration(experiment=None):
    args=EasyDict(seed=41,expert_max_steps=12000)
    for name in ('f1tenth','dynamic'):
        args.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    apply_experiment(args,experiment or ROOT/'configs/experiment_small_maps_raceline_6mps.yaml')
    args.max_episode_steps=12000
    args.full_progress_completion=True
    args.evaluation_laps=2
    return args


def train(actor,samplers,updates,batch_size,device,lr,stage,stream):
    actor.train()
    optimizer=torch.optim.AdamW(actor.parameters(),lr=lr,weight_decay=1e-5)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,updates,eta_min=lr*.15)
    start=time.monotonic();losses=[]
    for step in range(1,updates+1):
        if len(samplers)==1:x,y=samplers[0].batch(batch_size)
        else:
            bx,by=samplers[0].batch(batch_size//2)
            sampler=samplers[1+(step%(len(samplers)-1))]
            cx,cy=sampler.batch(batch_size-len(bx))
            x,y=np.concatenate([bx,cx]),np.concatenate([by,cy])
        xt=torch.from_numpy(x).to(device);yt=torch.from_numpy(y).to(device)
        xt,yt=mirror_batch(xt,yt,torch.rand(len(xt),device=device)<.5)
        xt=(xt+.005*torch.randn_like(xt)).clamp(0.,30.)
        error=(actor.action_mean(xt)-yt).square()
        loss=(6*error[:,0]+error[:,1]).mean()/2
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step();schedule.step();losses.append(float(loss))
        if step%250==0 or step==updates:
            row=dict(stage=stage,update=step,loss=float(np.mean(losses[-250:])),elapsed_s=time.monotonic()-start)
            stream.write(json.dumps(row)+'\n');stream.flush();print('[Train]',json.dumps(row),flush=True)
    actor.eval()
    return dict(updates=updates,loss=float(np.mean(losses[-250:])),elapsed_s=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment',type=Path,
        default=ROOT/'configs/experiment_small_maps_raceline_6mps.yaml')
    p.add_argument('--mode',choices=['collect','train','all'],default='all')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--initial-updates',type=int,default=20000)
    p.add_argument('--round-updates',type=int,default=8000)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=512)
    cli=p.parse_args()
    if min(cli.workers,cli.initial_updates,cli.round_updates,cli.batch_size)<=0 or cli.rounds<0:
        p.error('Training limits must be positive and rounds non-negative')
    output=cli.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if cli.mode!='collect' and (output/'training_metrics.jsonl').exists():
        raise FileExistsError('Refusing to overwrite an existing training run')
    experiment=cli.experiment.resolve()
    experiment_source=str(experiment.relative_to(ROOT))
    args=configuration(experiment)
    manifest=json.loads((ROOT/'map_datasets/hj_maps_trainval_261124/manifest.json').read_text())
    training=select_maps(manifest,'train');heldout=select_maps(manifest,'val')
    groups=defaultdict(list)
    for m in manifest['maps']:
        if m['split']=='train':groups[m['group']].append(m['name'])
    development=[]
    for names in groups.values():development.extend(names[::max(1,len(names)//16)])
    if cli.mode in ('collect','all'):
        if (output/'data/base/metadata.json').exists():raise FileExistsError('Use --mode train to reuse collected data')
        collect(args,training,heldout,output/'data/base',workers=cli.workers,laps=2)
    if cli.mode=='collect':return
    torch.set_num_threads(1);random.seed(41);np.random.seed(41);torch.manual_seed(41)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backup=output/'backup';backup.mkdir(exist_ok=True)
    with (backup/'args.pkl').open('wb') as f:pickle.dump(args,f)
    source_files=['train_small_map_policy.py','train_recovery.py','export_driving_policy.py',
                 'utils/small_map_learning.py','utils/policy_evaluation.py','utils/env.py',
                 'utils/rule_based.py','utils/racing_line.py','utils/speed_profile.py','utils/expert_data.py',
                 'algorithm/imitation/lidar_policy.py','configs/experiment_small_maps_raceline_6mps.yaml',
                 'configs/expert.yaml','configs/task/f1tenth.yaml','configs/task/dynamic.yaml',
                 'installation/f1tenth_gym/envs/base_classes.py',
                 'installation/f1tenth_gym/envs/f110_env.py',
                 'installation/f1tenth_gym/envs/longitudinal.py',
                 'installation/f1tenth_gym/envs/laser_models.py']
    if experiment_source not in source_files:source_files.append(experiment_source)
    for file in source_files:
        dest=backup/'sources'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/file,dest)
    record=dict(training_maps=training,heldout_maps=heldout,selection_maps=development,
        fresh_initialization=True,previous_model_used=False,input='4 chronological raw LiDAR scans only',
        rays_per_frame=1080,spatial_pooling=4,features_per_frame=270,teacher_speed_cap_mps=6,
        collection_response_randomized=True,teacher_at_inference=False,device=str(device),seed=41,
        config={**vars(cli),'output':str(output),'experiment':str(experiment)},
        effective_experiment=experiment_source,
        started_at=datetime.now().astimezone().isoformat(),
        torch_version=torch.__version__,numpy_version=np.__version__,
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        source_sha256={file:hashlib.sha256((ROOT/file).read_bytes()).hexdigest() for file in source_files})
    (output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    samplers=[DiskSampler(output/'data/base',41,training,heldout)]
    from utils.expert_data import signature
    if samplers[0].metadata['signature'] != signature(args):
        raise ValueError('Expert dataset configuration/source signature changed')
    if set(samplers[0].metadata['maps']) != set(training) or any(
            ep['outcome']!='complete' for ep in samplers[0].metadata['episodes']):
        raise ValueError('Base dataset must cover every training map with complete expert laps')
    actor=TemporalLidarPolicy().to(device)
    best_key=None;stages=[];best_stage=None
    with (output/'training_metrics.jsonl').open('w') as stream:
        for stage in range(cli.rounds+1):
            if stage:
                actor.load_state_dict(torch.load(output/'best_weights.pt',map_location=device))
                failed=stages[-1].get('failed_maps',[])
                recovery_maps=list(dict.fromkeys(development+failed))
                collect(args,recovery_maps,heldout,output/f'data/recovery{stage}',workers=cli.workers,
                    recovery_actor=output/'sim2real/actor.pt',recovery_scale=output/'sim2real/agent_obs_scale.pkl',
                    beta=[.25,.15,.05][min(stage-1,2)],laps=2,seeds=(1000+stage*37,1001+stage*37))
                samplers.append(DiskSampler(output/f'data/recovery{stage}',41+stage,training,heldout))
            stats=train(actor,samplers,cli.initial_updates if stage==0 else cli.round_updates,
                cli.batch_size,device,3e-4 if stage==0 else 1e-4,stage,stream)
            directory=output/f'stages/{stage}';export_actor(actor,directory,args)
            summary,rows=evaluate_policy(args,directory/'actor.pt',directory/'agent_obs_scale.pkl',
                development,[401],['cl_grid_static','cl_random_static'],directory/'development',workers=cli.workers)
            if summary['counts']['error']:raise RuntimeError('Evaluation errors; inspect development episodes')
            key=selection_key(summary)
            if best_key is None or key>best_key:
                best_key,best_stage=key,stage
                torch.save(actor.state_dict(),output/'best_weights.pt');export_actor(actor,output/'sim2real',args)
            stages.append(dict(stage=stage,training=stats,development=summary,best_stage=best_stage,
                failed_maps=sorted({r['map'] for r in rows if r['outcome']!='complete'})))
            (output/'stages.json').write_text(json.dumps(stages,indent=2)+'\n')
            print('[Selection]',json.dumps(stages[-1]),flush=True)
    from export_driving_policy import export
    contract=export(output,output/'sim2real/driving_policy.ts')
    args.evaluation_laps=3
    final,rows=evaluate_policy(args,output/'sim2real/driving_policy.ts',None,
        training+heldout,[613,22817],['cl_grid_static','cl_random_static'],output/'evaluation_final',workers=cli.workers)
    result=dict(best_stage=best_stage,final=final,
        training=summarize([r for r in rows if r['map'] in training]),
        heldout=summarize([r for r in rows if r['map'] in heldout]),
        artifact=str(output/'sim2real/driving_policy.ts'),contract=contract,
        sha256=hashlib.sha256((output/'sim2real/driving_policy.ts').read_bytes()).hexdigest(),
        physical_vehicle_validated=False,teacher_during_evaluation=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[Final]',json.dumps(result),flush=True)

if __name__=='__main__':main()
