from copy import deepcopy
import numpy as np
from diagnose_expert_speed_limits import make_case,CASES,choose_maps
from train_small_map_policy import configuration


def sample(config,seed):
    response=deepcopy(config.longitudinal_response);ranges=response.pop('ranges');rng=np.random.default_rng(seed)
    for key,bounds in ranges.items():response[key]=rng.uniform(*bounds)
    return response


def test_changing_a_response_range_preserves_other_random_samples_and_source():
    base=configuration();before=deepcopy(base)
    for label,changes in CASES.items():
        case=make_case(base,label)
        assert list(case.longitudinal_response['ranges'])==list(base.longitudinal_response['ranges'])
        a,b=sample(base,1937),sample(case,1937)
        changed=set(changes.get('response',{}))|set(changes.get('ranges',{}))
        for key,value in a.items():
            if key not in changed:assert b[key]==value,(label,key)
    assert base==before


def test_stratified_map_selection_is_fixed_unique_and_covers_groups():
    manifest={'maps':[dict(name=f'{group}{i:03}',group=group,split=split) for group,split,n in
        [('a','train',240),('b','train',120),('c','train',60),('d','train',60),('e','val',30),('f','val',20),('g','val',50)] for i in range(n)]}
    selected=choose_maps(manifest)
    assert len(selected)==60 and len({m['name'] for m in selected})==60
    assert sum(m['split']=='train' for m in selected)==48
    assert sum(m['split']=='val' for m in selected)==12
