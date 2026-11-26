"""Build offline racing-line files; simulator rollouts must verify their quality."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from easydict import EasyDict
from ruamel.yaml import YAML

from utils.expert_config import ROOT
from utils.experiment import apply_experiment
from utils.racing_line import optimize


def run(job):
    entry,config,output,margin,step,iterations = job
    name = entry['name']
    try:
        directory = ROOT/'installation/maps'/name
        arrays,report = optimize(directory,name,config.expert_parameters,config,margin,step,iterations)
        destination = Path(output)/(name+'.npz')
        np.savez_compressed(destination,**arrays)
        report.update(outcome='generated',split=entry['split'],group=entry['group'],
                      file_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      map_source_sha256=entry['files'])
        return report
    except Exception:
        return dict(map=name,outcome='error',split=entry['split'],error=traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'map_datasets/hj_maps_trainval_261124/manifest.json')
    parser.add_argument('--split',choices=['train','val','all'],default='train')
    parser.add_argument('--maps',nargs='+')
    parser.add_argument('--limit-per-group',type=int)
    parser.add_argument('--experiment',type=Path,default=ROOT/'configs/experiment_small_maps_6mps.yaml')
    parser.add_argument('--margin',type=float,default=.12)
    parser.add_argument('--sample-step',type=float,default=.25)
    parser.add_argument('--iterations',type=int,default=90)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--output',type=Path,required=True)
    cli = parser.parse_args()
    if min(cli.margin,cli.sample_step,cli.iterations,cli.workers)<=0:
        parser.error('Positive margin, sample-step, iterations and workers required')
    entries=[m for m in json.loads(cli.manifest.read_text())['maps'] if cli.split=='all' or m['split']==cli.split]
    if cli.maps:
        if set(cli.maps)-{m['name'] for m in entries}:
            parser.error('Requested maps are outside the selected split')
        entries=[m for m in entries if m['name'] in cli.maps]
    if cli.limit_per_group:
        chosen=[]
        for group in sorted({m['group'] for m in entries}):
            subset=sorted([m for m in entries if m['group']==group],key=lambda m:m['name'])
            chosen.extend(subset[i] for i in np.linspace(0,len(subset)-1,min(cli.limit_per_group,len(subset)),dtype=int))
        entries=chosen
    config=EasyDict()
    for name in ['f1tenth','dynamic']:
        config.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    apply_experiment(config,cli.experiment)
    cli.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),ROOT/'utils/racing_line.py',ROOT/'utils/speed_profile.py']
    metadata=dict(description=__doc__,entries=entries,config=dict(config),margin_m=cli.margin,
                  step_m=cli.sample_step,iterations=cli.iterations,
                  method='Bounded local minimum-curvature optimization followed by a speed-envelope lap-time objective; clearance-checked blends ranked by estimated time. Not a global optimum.',
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (cli.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    start=time.monotonic();records=[]
    with ProcessPoolExecutor(max_workers=cli.workers) as pool,(cli.output/'generation.jsonl').open('w') as stream:
        futures=[pool.submit(run,(e,config,str(cli.output),cli.margin,cli.sample_step,cli.iterations)) for e in entries]
        for future in as_completed(futures):
            row=future.result();records.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
            print(f"{len(records)}/{len(entries)} {row['map']}: {row['outcome']} "
                  f"{row.get('center_estimated_lap_s',0):.2f} -> {row.get('estimated_lap_s',0):.2f}",flush=True)
    summary=dict(maps=len(records),generated=sum(r['outcome']=='generated' for r in records),
                 noncenter=sum(r.get('max_abs_offset_m',0)>.02 for r in records),elapsed_s=time.monotonic()-start)
    (cli.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if summary['generated']!=summary['maps']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
