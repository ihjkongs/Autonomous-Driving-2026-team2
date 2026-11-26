"""Paired offline expert cap study with fixed racing-line geometry and dynamics."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time
import numpy as np
from utils.expert_config import ROOT
from utils.racing_line import geometry
from utils.speed_profile import speed_profile
from train_small_map_policy import configuration
from validate_small_map_expert import evaluate


def rebuild_profile(points,work):
    """Same profile construction used by racing_line.optimize; no path fitting."""
    _,kappa=geometry(points)
    lateral=np.sqrt(work['lateral_acceleration']/np.maximum(np.abs(kappa),1e-6))
    return speed_profile(points,np.minimum(8.,lateral),work)[1]


def prepare_profiles(source,destination,names,work,cap):
    source,destination=Path(source),Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    records=[]
    for name in names:
        original=source/(name+'.npz')
        with np.load(original,allow_pickle=False) as f:data={key:f[key].copy() for key in f.files}
        baseline=rebuild_profile(data['points'],work)
        error=float(np.max(np.abs(baseline-data['speed'])))
        if error>1e-10:raise ValueError(f'Baseline regeneration changed {name}: {error}')
        updated=dict(work,speed_cap=float(cap))
        speed=rebuild_profile(data['points'],updated)
        old_profile=speed_profile(data['points'],data['speed'],work)[1]
        literal_profile=speed_profile(data['points'],data['speed'],updated)[1]
        if not np.array_equal(old_profile,literal_profile):
            raise ValueError('Changing only cap should not change a route already bounded below both caps')
        old_speed=data['speed'].copy();data['speed']=speed
        np.savez_compressed(destination/(name+'.npz'),**data)
        with np.load(destination/(name+'.npz'),allow_pickle=False) as check:
            assert np.array_equal(check['points'],data['points'])
        ds=np.linalg.norm(np.roll(data['points'],-1,axis=0)-data['points'],axis=1)
        records.append(dict(map=name,baseline_profile_max_error=error,
            original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            updated_sha256=hashlib.sha256((destination/(name+'.npz')).read_bytes()).hexdigest(),
            original_max_speed_mps=float(old_speed.max()),updated_max_speed_mps=float(speed.max()),
            max_speed_increase_mps=float(np.max(speed-old_speed)),
            path_fraction_speed_increased=float(ds[speed>old_speed+1e-8].sum()/ds.sum()),
            path_fraction_planned_above_6mps=float(ds[speed>6.+1e-8].sum()/ds.sum()),
            literal_cap_change_planner_profile_identical=True,geometry_identical=True))
    return records


def aggregate(rows):
    complete=[r for r in rows if r['outcome']=='complete'];steps=sum(r['steps'] for r in complete)
    return dict(episodes=len(rows),completed=len(complete),counts=dict(Counter(r['outcome'] for r in rows)),
        completed_laps=sum(r.get('completed_laps',0) for r in rows),
        mean_lap_s_completed_only=float(np.mean([r['elapsed_s']/r['required_laps'] for r in complete])) if complete else None,
        time_weighted_actual_speed_mps=sum(r['mean_actual_speed_mps']*r['steps'] for r in complete)/steps if steps else None,
        time_weighted_requested_speed_mps=sum(r['mean_requested_speed_mps']*r['steps'] for r in complete)/steps if steps else None,
        max_actual_speed_mps=max((r.get('max_actual_speed_mps',0) for r in rows),default=0),
        max_requested_speed_mps=max((r.get('max_requested_speed_mps',0) for r in rows),default=0),
        episodes_requested_above_6mps=sum(r.get('max_requested_speed_mps',0)>6.+1e-8 for r in rows),
        episodes_actual_above_6mps=sum(r.get('max_actual_speed_mps',0)>6.+1e-8 for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--laps',type=int,default=3)
    parser.add_argument('--seeds',nargs='+',type=int,default=[124,2026])
    cli=parser.parse_args()
    if min(cli.workers,cli.laps)<1:parser.error('Workers/laps must be positive')
    output=cli.output.resolve();output.mkdir(parents=True,exist_ok=False)
    args=configuration()
    manifest_path=ROOT/'map_datasets/hj_maps_trainval_261124/manifest.json'
    manifest=json.loads(manifest_path.read_text());names=[r['name'] for r in manifest['maps']]
    source=ROOT/args.expert_parameters['raceline_directory']
    started=datetime.now().astimezone().isoformat()
    profiles=prepare_profiles(source,output/'racelines_cap8',names,args.expert_parameters,8.)
    (output/'profile_audit.json').write_text(json.dumps(profiles,indent=2)+'\n')
    candidates={}
    for cap in (6,8):
        conf=deepcopy(args);conf.max_episode_steps=12000
        conf.max_speed=float(cap);conf.expert_parameters['speed_cap']=float(cap)
        conf.expert_parameters['raceline_directory']=str(source if cap==6 else output/'racelines_cap8')
        candidates[f'cap{cap}']=conf
    # Exactly the cap, wrapper command ceiling and cached speed-profile path differ.
    left,right=deepcopy(candidates['cap6']),deepcopy(candidates['cap8'])
    for c in [left,right]:
        c.pop('max_speed');c.expert_parameters.pop('speed_cap');c.expert_parameters.pop('raceline_directory')
    assert left==right
    metadata=dict(started_at=started,maps=manifest['maps'],laps=cli.laps,seeds=cli.seeds,
        modes=['cl_grid_static','cl_random_static'],workers=cli.workers,
        candidates={k:dict(v) for k,v in candidates.items()},
        changed=['expert speed_cap','command upper bound max_speed','precomputed route-speed envelope'],
        unchanged=['racing-line points','steering/lookahead','lateral acceleration','braking/planning acceleration','response distributions','physical v_max=8'],
        geometry_reoptimized=False,neural_network_trained=False,real_vehicle_commands_sent=False,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in [
            'compare_expert_speed_caps.py','validate_small_map_expert.py','utils/rule_based.py',
            'utils/racing_line.py','utils/speed_profile.py','configs/experiment_small_maps_raceline_6mps.yaml',
            'installation/f1tenth_gym/envs/longitudinal.py']})
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    jobs=[]
    for name in names:
        for mode in metadata['modes']:
            for seed in cli.seeds:
                for label,config in candidates.items():jobs.append((config,name,mode,seed,label,cli.laps))
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=cli.workers,mp_context=get_context('spawn')) as pool,(output/'episodes.jsonl').open('w') as stream:
        futures=[pool.submit(evaluate,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();rows.append(result)
            stream.write(json.dumps(result)+'\n');stream.flush()
            if len(rows)%80==0 or result['outcome']!='complete':
                print(f"[Cap comparison {len(rows)}/{len(jobs)}] {result['candidate']} {result['map']} {result['outcome']}",flush=True)
    def paired(records):
        index={(r['map'],r['reset'],r['seed'],r['candidate']):r for r in records}
        pairs=[]
        for key,r in index.items():
            if key[3]!='cap6':continue
            other=index.get(key[:3]+('cap8',))
            assert other is not None
            assert r['longitudinal_response']==other['longitudinal_response']
            if r['outcome']=='complete' and other['outcome']=='complete':pairs.append((r,other))
        a=aggregate([r[0] for r in pairs]);b=aggregate([r[1] for r in pairs])
        return dict(paired_completed_trials=len(pairs),baseline6=a,candidate8=b,
            mean_lap_reduction_percent=(1-b['mean_lap_s_completed_only']/a['mean_lap_s_completed_only'])*100 if pairs else None,
            mean_speed_gain_percent=(b['time_weighted_actual_speed_mps']/a['time_weighted_actual_speed_mps']-1)*100 if pairs else None,
            mean_trial_lap_delta_s=float(np.mean([(a['elapsed_s']-b['elapsed_s'])/a['required_laps'] for a,b in pairs])) if pairs else None)
    summary=dict(started_at=started,finished_at=datetime.now().astimezone().isoformat(),elapsed_wall_s=time.monotonic()-start,
        candidates={label:aggregate([r for r in rows if r['candidate']==label]) for label in candidates},
        paired=paired(rows),by_split={},
        profiles=dict(maps=len(profiles),all_geometry_identical=all(r['geometry_identical'] for r in profiles),
          max_baseline_regeneration_error=max(r['baseline_profile_max_error'] for r in profiles),
          maps_profile_speed_increased=sum(r['max_speed_increase_mps']>1e-8 for r in profiles),
          maps_planned_above_6=sum(r['updated_max_speed_mps']>6.+1e-8 for r in profiles),
          maximum_planned_speed_cap8_mps=max(r['updated_max_speed_mps'] for r in profiles),
          literal_cap_only_all_planner_profiles_unchanged=all(r['literal_cap_change_planner_profile_identical'] for r in profiles)))
    for split in ['train','val']:
        names_split={m['name'] for m in manifest['maps'] if m['split']==split}
        summary['by_split'][split]=paired([r for r in rows if r['map'] in names_split])
    failures=[r for r in rows if r['outcome']!='complete']
    summary['failures']=failures
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if any(r['outcome']=='error' for r in rows):raise SystemExit('Expert evaluation infrastructure error')

if __name__=='__main__':main()
