"""Offline one-factor and interaction diagnostics for expert speed bottlenecks."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime
import hashlib,json
from multiprocessing import get_context
from pathlib import Path
import time
import numpy as np
from compare_expert_speed_caps import rebuild_profile,aggregate
from train_small_map_policy import configuration
from utils.expert_config import ROOT
from validate_small_map_expert import evaluate


# Hypothetical sensitivity tests; none of these are physical tuning advice.
CASES={
 'baseline':dict(),
 'corner_lat6':dict(expert={'lateral_acceleration':6.}),
 'plan_accel4':dict(expert={'planning_acceleration':4.}),
 'plan_brake4':dict(expert={'braking_deceleration':4.}),
 'preview0':dict(expert={'reaction_time':0.}),
 'command_accel4':dict(response={'command_acceleration':4.}),
 'command_brake6':dict(response={'command_deceleration':6.}),
 'vehicle_accel_x2':dict(response={'acceleration':4.},ranges={'acceleration':[3.,5.]}),
 'vehicle_brake_x2':dict(response={'deceleration':6.},ranges={'deceleration':[4.,7.]}),
 'delay0':dict(response={'delay':0.},ranges={'delay':[0.,0.]}),
 'tau005':dict(response={'time_constant':.05},ranges={'time_constant':[.05,.05]}),
 'delay0_tau005':dict(response={'delay':0.,'time_constant':.05},ranges={'delay':[0.,0.],'time_constant':[.05,.05]}),
 'accel_stack_x2':dict(expert={'planning_acceleration':4.},response={'command_acceleration':4.,'acceleration':4.},ranges={'acceleration':[3.,5.]}),
 'brake_stack_x2':dict(expert={'braking_deceleration':4.},response={'command_deceleration':6.,'deceleration':6.},ranges={'deceleration':[4.,7.]}),
}
PROFILE_KEYS=('lateral_acceleration','planning_acceleration','braking_deceleration','speed_cap','curvature_window','vgain')


def make_case(baseline,label):
    case=deepcopy(baseline);changes=CASES[label]
    case.expert_parameters.update(changes.get('expert',{}))
    case.longitudinal_response.update(changes.get('response',{}))
    # Retain range insertion order and the number of RNG draws, including fixed ranges.
    case.longitudinal_response['ranges'].update(changes.get('ranges',{}))
    return case


def choose_maps(manifest,train_per_group=12,val_per_group=4):
    selected=[]
    for group in sorted({m['group'] for m in manifest['maps']}):
        entries=sorted([m for m in manifest['maps'] if m['group']==group],key=lambda m:m['name'])
        count=train_per_group if entries[0]['split']=='train' else val_per_group
        indexes=np.linspace(0,len(entries)-1,min(count,len(entries)),dtype=int)
        selected.extend(entries[int(i)] for i in indexes)
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    cli=parser.parse_args();output=cli.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if cli.workers<1:parser.error('Workers must be positive')
    base=configuration();source=ROOT/base.expert_parameters['raceline_directory']
    manifest=json.loads((ROOT/'map_datasets/hj_maps_trainval_261124/manifest.json').read_text())
    selected=choose_maps(manifest);configs={name:make_case(base,name) for name in CASES}
    dirs={};signatures={};profile_audit=[]
    for label,config in configs.items():
        work=config.expert_parameters;key=tuple(float(work[k]) for k in PROFILE_KEYS)
        if key in dirs:
            config.expert_parameters['raceline_directory']=str(dirs[key]);continue
        if label=='baseline':directory=source
        else:directory=output/'profiles'/label;directory.mkdir(parents=True)
        dirs[key]=directory;signatures[label]=dict(zip(PROFILE_KEYS,key))
        for entry in manifest['maps']:
            name=entry['name'];path=source/(name+'.npz')
            with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
            speed=rebuild_profile(data['points'],work)
            if label=='baseline':
                if np.max(np.abs(speed-data['speed']))>1e-10:raise ValueError('Base profile regeneration changed')
            else:
                data['speed']=speed;np.savez_compressed(directory/(name+'.npz'),**data)
            if label=='baseline':
                curve_only=rebuild_profile(data['points'],dict(work,planning_acceleration=1e6,braking_deceleration=1e6))
                ds=np.linalg.norm(np.roll(data['points'],-1,axis=0)-data['points'],axis=1)
                profile_audit.append(dict(map=name,split=entry['split'],group=entry['group'],length_m=float(ds.sum()),
                    fraction_local_corner_limit_below6=float(ds[curve_only<6-1e-8].sum()/ds.sum()),
                    fraction_plan_below_local_corner_limit=float(ds[speed<curve_only-1e-8].sum()/ds.sum()),
                    minimum_local_corner_speed_mps=float(curve_only.min()),
                    original_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        config.expert_parameters['raceline_directory']=str(directory)
    metadata=dict(started_at=datetime.now().astimezone().isoformat(),
        purpose='Sensitivity and interaction analysis, fixed paths, hypothetical offline perturbations, no parameter recommendations.',
        selected_maps=selected,all_map_profile_audit=profile_audit,cases=CASES,
        baseline_config=dict(base),effective_configs={k:dict(v) for k,v in configs.items()},
        seeds=[124,2026],modes=['cl_grid_static','cl_random_static'],laps=3,
        profile_signatures=signatures,profile_directories={str(k):str(v) for k,v in dirs.items()},
        matched_random_response_draws=True,model_training=False,real_vehicle_test=False,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['diagnose_expert_speed_limits.py','compare_expert_speed_caps.py','validate_small_map_expert.py','utils/speed_profile.py','utils/rule_based.py','installation/f1tenth_gym/envs/longitudinal.py']})
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    jobs=[]
    for m in selected:
        for mode in metadata['modes']:
            for seed in metadata['seeds']:
                for label,config in configs.items():jobs.append((config,m['name'],mode,seed,label,3))
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=cli.workers,mp_context=get_context('spawn')) as pool,(output/'episodes.jsonl').open('w') as stream:
        for future in as_completed([pool.submit(evaluate,j) for j in jobs]):
            row=future.result();rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if len(rows)%100==0 or row['outcome']!='complete':
                print(f"[Diagnosis {len(rows)}/{len(jobs)}] {row['candidate']} {row['map']} {row['outcome']}",flush=True)
    indexed={(r['map'],r['reset'],r['seed'],r['candidate']):r for r in rows}
    summaries={}
    for label in CASES:
        group=[r for r in rows if r['candidate']==label];pairs=[]
        changed=set(CASES[label].get('response',{}))|set(CASES[label].get('ranges',{}))
        for r in group:
            old=indexed[(r['map'],r['reset'],r['seed'],'baseline')]
            for key,value in old['longitudinal_response'].items():
                if key not in changed:
                    assert value==r['longitudinal_response'][key],(label,key,value,r['longitudinal_response'][key])
            if r['outcome']==old['outcome']=='complete':pairs.append((old,r))
        a=aggregate([x for x,y in pairs]);b=aggregate([y for x,y in pairs])
        summaries[label]=dict(all_trials=aggregate(group),paired_complete_cases=len(pairs),
            paired_baseline=a,paired_case=b,
            lap_reduction_percent=(1-b['mean_lap_s_completed_only']/a['mean_lap_s_completed_only'])*100 if pairs else None,
            actual_speed_gain_percent=(b['time_weighted_actual_speed_mps']/a['time_weighted_actual_speed_mps']-1)*100 if pairs else None)
    summary=dict(finished_at=datetime.now().astimezone().isoformat(),elapsed_s=time.monotonic()-start,
        episodes=len(rows),cases=summaries,failed_episodes=[r for r in rows if r['outcome']!='complete'],
        profile_audit=dict(maps=len(profile_audit),
            distance_weighted_fraction_local_corner_limited=sum(r['fraction_local_corner_limit_below6']*r['length_m'] for r in profile_audit)/sum(r['length_m'] for r in profile_audit),
            distance_weighted_fraction_further_limited_by_accel_braking_plan=sum(r['fraction_plan_below_local_corner_limit']*r['length_m'] for r in profile_audit)/sum(r['length_m'] for r in profile_audit)),
        interpretation='Different perturbation magnitudes and interacting constraints; percentages are not additive causal shares. Crashes must be considered before lap-time improvements.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if any(r['outcome']=='error' for r in rows):raise SystemExit('Evaluation infrastructure failure')

if __name__=='__main__':main()
