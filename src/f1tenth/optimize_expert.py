"""Search bounded simulator expert settings, then verify the fastest safe candidate.

No real-vehicle commands, BC training, dataset overwrite or active config mutation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

from easydict import EasyDict
from ruamel.yaml import YAML
from utils.expert_config import MAPS, ROOT, load_expert_config
from validate_expert import evaluate


def run_job(job):
    args, name, mode, seed, candidate = job
    result = evaluate(args, name, mode, seed)
    result['candidate'] = candidate
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--baseline-config', type=Path)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=False)
    base = EasyDict()
    for name in ('f1tenth', 'dynamic'):
        base.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    base.seed, base.collection_horizon, base.max_episode_steps = 17, 1000, 20000
    base.step_mode = 'native'
    baseline_path = cli.baseline_config or ROOT / 'configs/expert_baseline.yaml'
    baseline = (YAML(typ='safe').load(baseline_path.read_text())
                if baseline_path.exists() else load_expert_config())
    parameters = {'baseline': baseline}
    for distance in (.8, 1., 1.2):
        for lateral in (1.5, 2.5, 3.5):
            params = dict(baseline, tlad=distance, max_lookahead=distance,
                          speed_cap=min(4., base.max_speed), lateral_acceleration=lateral)
            parameters[f'L{distance:g}_A{lateral:g}'] = params
    sources = ['optimize_expert.py', 'validate_expert.py', 'utils/rule_based.py',
               'utils/env.py', 'installation/f1tenth_gym/envs/integrator.py',
               'installation/f1tenth_gym/envs/dynamic_models.py']
    (cli.output/'metadata.json').write_text(json.dumps(dict(
        parameters=parameters, vehicle=dict(base),
        selection='All trials must complete; rank by mean lap time. Finite grid search, not a global optimum.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    for name in sources:
        target=cli.output/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=cli.workers) as pool:
        def run_stage(stage, candidates, maps, seeds):
            records=[];jobs=[]
            for candidate in candidates:
                for name in maps:
                    for mode in ('cl_grid_static','cl_random_static'):
                        for seed in seeds:
                            args=deepcopy(base);args.expert_parameters=parameters[candidate]
                            jobs.append((args,name,mode,seed,candidate))
            with (cli.output/(stage+'.jsonl')).open('w') as stream:
                for record in pool.map(run_job,jobs):
                    records.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
                    if len(records)%12==0 or record['outcome']!='complete':
                        print(f"{stage} {len(records)}/{len(jobs)} {record['candidate']} {record['map']}: {record['outcome']}",flush=True)
            return records
        smoke=run_stage('screen',parameters,['Sochi','Spa','Nuerburgring','Shanghai','Oschersleben','IMS'],[17])
        scores=[]
        for candidate in parameters:
            rs=[r for r in smoke if r['candidate']==candidate]
            scores.append(dict(candidate=candidate,complete=sum(r['outcome']=='complete' for r in rs),
                               trials=len(rs),mean_s=sum(r.get('elapsed_s',500.) for r in rs)/len(rs)))
        (cli.output/'screen_summary.json').write_text(json.dumps(scores,indent=2)+'\n')
        eligible=sorted((s for s in scores if s['complete']==s['trials'] and s['candidate']!='baseline'),key=lambda s:s['mean_s'])
        winner=None
        for index,score in enumerate(eligible):
            candidate=score['candidate']
            all_tracks=run_stage(f'all_tracks_{candidate}',[candidate],MAPS,[17])
            if not all(r['outcome']=='complete' for r in all_tracks):
                continue
            final=run_stage(f'repeated_{candidate}',[candidate],MAPS,[1,124,7,42,2026,2027,31415])
            if all(r['outcome']=='complete' for r in final):
                baseline_records=[json.loads(l) for l in (ROOT/'results/expert_validation/final_23_tracks/episodes.jsonl').read_text().splitlines()]
                before={(r['map'],r['reset'],r['seed']):r for r in baseline_records}
                comparisons=[]
                for name in MAPS:
                    rs=[r for r in final if r['map']==name]
                    previous=[before[(r['map'],r['reset'],r['seed'])]['elapsed_s'] for r in rs]
                    new=sum(r['elapsed_s'] for r in rs)/len(rs);old=sum(previous)/len(previous)
                    comparisons.append(dict(map=name,complete=len(rs),trials=len(rs),before_mean_s=old,
                                            after_mean_s=new,reduction_percent=100*(1-new/old)))
                old=sum(r['before_mean_s'] for r in comparisons);new=sum(r['after_mean_s'] for r in comparisons)
                winner=dict(candidate=candidate,parameters=parameters[candidate],episodes=len(final),
                            complete=len(final),before_track_mean_s=old/len(MAPS),
                            after_track_mean_s=new/len(MAPS),reduction_percent=100*(1-new/old),
                            by_track=comparisons,elapsed_wall_s=time.monotonic()-start)
                break
        if winner is None:
            raise RuntimeError('No candidate maintained all-repeat completion; baseline unchanged.')
    (cli.output/'winner.json').write_text(json.dumps(winner,indent=2)+'\n')
    with (cli.output/'expert_fast.yaml').open('w') as stream:
        YAML().dump(winner['parameters'],stream)
    print(json.dumps(winner),flush=True)


if __name__=='__main__':
    main()
