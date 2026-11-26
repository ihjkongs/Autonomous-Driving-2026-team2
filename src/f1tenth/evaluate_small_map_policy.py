"""Evaluate the frozen deployment artifact on all imported small maps."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
from utils.expert_config import ROOT
from utils.small_map_learning import select_maps
from utils.policy_evaluation import evaluate_policy,summarize


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=Path,required=True)
 p.add_argument('--output',type=Path)
 p.add_argument('--workers',type=int,default=12)
 p.add_argument('--laps',type=int,default=3)
 p.add_argument('--seeds',nargs='+',type=int,default=[613,22817])
 cli=p.parse_args()
 if min(cli.workers,cli.laps)<1:p.error('Workers and laps must be positive')
 run=cli.run.resolve();artifact=run/'sim2real/driving_policy.ts'
 with (run/'backup/args.pkl').open('rb') as f:args=pickle.load(f)
 args.evaluation_laps=cli.laps;args.full_progress_completion=True;args.max_episode_steps=12000
 manifest=json.loads((ROOT/'map_datasets/hj_maps_trainval_261124/manifest.json').read_text())
 training=select_maps(manifest,'train');heldout=select_maps(manifest,'val')
 if set(training)&set(heldout):raise ValueError('Train/val overlap')
 sha=hashlib.sha256(artifact.read_bytes()).hexdigest()
 protocol=dict(started_at=datetime.now().astimezone().isoformat(),artifact=str(artifact),sha256=sha,
  workers=cli.workers,required_laps=cli.laps,seeds=cli.seeds,maps=training+heldout,
  modes=['cl_grid_static','cl_random_static'],full_progress_completion=True,
  teacher_during_evaluation=False,model_selection_finished=True)
 (run/'final_evaluation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 summary,rows=evaluate_policy(args,artifact,None,training+heldout,cli.seeds,
  protocol['modes'],cli.output or run/'evaluation_final',workers=cli.workers)
 stages=json.loads((run/'stages.json').read_text())
 verification=json.loads((run/'export_verification.json').read_text())
 result=dict(best_stage=stages[-1]['best_stage'],final=summary,
  training=summarize([r for r in rows if r['map'] in training]),
  heldout=summarize([r for r in rows if r['map'] in heldout]),
  artifact=str(artifact),contract=verification['metadata'],sha256=sha,
  physical_vehicle_validated=False,teacher_during_evaluation=False,
  evaluated_deployment_artifact_directly=True,finished_at=datetime.now().astimezone().isoformat())
 assert sha==hashlib.sha256(artifact.read_bytes()).hexdigest(),'Artifact changed during evaluation'
 (run/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('[Final]',json.dumps(result),flush=True)
 if summary['counts']['error']:raise SystemExit('Evaluation infrastructure errors')

if __name__=='__main__':main()
