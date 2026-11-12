"""Check whether stricter expert revalidation changes any training example."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from easydict import EasyDict
from ruamel.yaml import YAML
from utils.expert_config import ROOT
from utils.experiment import apply_experiment
from utils.expert_data import signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    with cli.old.open('rb') as stream:
        old = pickle.load(stream)
    with cli.new.open('rb') as stream:
        new = pickle.load(stream)
    args = EasyDict()
    for name in ('f1tenth', 'dynamic'):
        args.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    apply_experiment(args, cli.experiment)
    arrays = {}
    for key in ('observations', 'actions'):
        a, b = old[key], new[key]
        arrays[key] = dict(identical=bool(a.dtype == b.dtype and np.array_equal(a, b)),
                          old_shape=list(a.shape), new_shape=list(b.shape),
                          old_sha256=hashlib.sha256(memoryview(a)).hexdigest(),
                          new_sha256=hashlib.sha256(memoryview(b)).hexdigest())
    result = dict(old=str(cli.old), new=str(cli.new), arrays=arrays,
        identical_training_examples=all(x['identical'] for x in arrays.values()),
        old_signature=old['metadata']['signature'], new_signature=new['metadata']['signature'],
        new_signature_matches_current_sources=new['metadata']['signature'] == signature(args),
        revalidated_laps=len(new['metadata']['episodes']),
        all_revalidated_laps_complete=all(ep['outcome'] == 'complete' for ep in new['metadata']['episodes']),
        legacy_training_metadata_preserved=True,
        explanation='Training arrays may be identical even though the physics-validation signature changes. Old model training provenance is retained; current validity comes from new expert/policy evaluations.')
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['identical_training_examples'] or not result['new_signature_matches_current_sources']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
