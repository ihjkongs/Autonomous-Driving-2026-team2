import numpy as np
import pytest
import torch
from algorithm.imitation.lidar_policy import TemporalLidarPolicy, PhysicalLidarPolicy
from train_recovery import split_examples, mirror_batch
from utils.recovery_collection import collect_recovery


def test_heldout_rows_never_enter_training_split():
    data = dict(observations=np.array([[1.], [2.], [999.], [1000.]]),
                actions=np.zeros((4, 2)), metadata=dict(maps=['train', 'test'], episodes=[
                    dict(map='train', start=0, stop=2), dict(map='test', start=2, stop=4)]))
    x, _, indices = split_examples(data, ['train'], ['test'])
    assert np.array_equal(indices, [0, 1]) and x.max() == 2.
    with pytest.raises(ValueError, match='overlap'):
        split_examples(data, ['train', 'test'], ['test'])


def test_recovery_collection_rejects_heldout_maps_before_any_execution(tmp_path):
    with pytest.raises(ValueError, match='Held-out'):
        collect_recovery(None, None, None, ['Sochi'], ['Sochi'], [1], tmp_path/'no_write')
    assert not (tmp_path/'no_write').exists()


def test_reflection_changes_steering_only_and_preserves_original_arrays():
    x = torch.arange(2*4320).float().reshape(2, 4320)
    y = torch.tensor([[.2, .7], [-.3, .8]])
    a, b = mirror_batch(x, y, torch.tensor([True, False]))
    assert torch.equal(a[0].reshape(4, 1080), x[0].reshape(4, 1080).flip(-1))
    assert torch.equal(a[1], x[1])
    assert torch.allclose(b, torch.tensor([[-.2, .7], [-.3, .8]]))
    assert y[0, 0] > 0


def test_policy_vector_batch_and_portable_physical_export_agree(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = TemporalLidarPolicy().eval()
    scans = torch.rand(2, 4320)*30.
    with torch.no_grad():
        action = model(scans).loc
        single = model(scans[0]).loc
        assert torch.allclose(action[0], single, atol=1e-6)
        assert action.shape == (2, 2) and torch.isfinite(action).all() and action.abs().max() <= 1.
        wrapper = PhysicalLidarPolicy(model)
        expected = wrapper(scans)
        traced = torch.jit.trace(wrapper, scans[:1])
        traced.save(str(tmp_path/'policy.ts'))
        actual = torch.jit.load(str(tmp_path/'policy.ts'))(scans)
        assert torch.allclose(actual, expected, atol=1e-6)
        assert actual[:, 0].abs().max() <= .48
        assert actual[:, 1].min() >= 1. and actual[:, 1].max() <= 6.


def test_single_file_evaluator_matches_original_normalized_policy(tmp_path):
    import json
    from utils.policy_evaluation import initialize_policy, policy_action
    from train_recovery import export_actor
    from easydict import EasyDict
    torch.set_num_threads(1)
    actor = TemporalLidarPolicy().eval()
    traced = torch.jit.trace(PhysicalLidarPolicy(actor), torch.zeros(1, 4320))
    contract = dict(format='f1tenth-physical-lidar-v1', max_steer=.48, min_speed=1., max_speed=6.)
    path = tmp_path/'policy.ts'
    torch.jit.save(traced, str(path), _extra_files={'metadata.json': json.dumps(contract)})
    scan = np.random.default_rng(9).uniform(0, 30, 4320).astype(np.float32)
    initialize_policy(path, None)
    exported_action = policy_action(scan)
    with torch.no_grad():
        expected = actor(torch.from_numpy(scan)).loc.numpy()
    assert np.allclose(exported_action, expected, atol=1e-6)
    # A subsequent legacy load must clear the physical-output contract.
    export_actor(actor, tmp_path/'legacy', EasyDict(obs_dim=4320))
    initialize_policy(tmp_path/'legacy/actor.pt', tmp_path/'legacy/agent_obs_scale.pkl')
    assert np.allclose(policy_action(scan), expected, atol=1e-6)


def test_evaluation_does_not_query_the_expert(tmp_path, monkeypatch):
    from easydict import EasyDict
    from ruamel.yaml import YAML
    from utils.expert_config import ROOT
    from utils.experiment import apply_experiment
    from utils.rule_based import PurePursuitPlanner
    from utils.policy_evaluation import initialize_policy, episode
    from train_recovery import export_actor
    def forbidden(*args, **kwargs):
        raise AssertionError('Expert queried during evaluation')
    monkeypatch.setattr(PurePursuitPlanner, 'plan', forbidden)
    args = EasyDict(seed=9)
    for name in ('f1tenth', 'dynamic'):
        args.update(YAML(typ='safe').load((ROOT/'configs/task'/(name+'.yaml')).read_text()))
    apply_experiment(args, ROOT/'configs/experiment_6mps_randomized.yaml')
    args.max_episode_steps = 30
    export_actor(TemporalLidarPolicy().eval(), tmp_path/'actor', args)
    initialize_policy(tmp_path/'actor/actor.pt', tmp_path/'actor/agent_obs_scale.pkl')
    result = episode((args, 'IMS', 'cl_grid_static', 9, None))
    assert result['outcome'] != 'error', result.get('error')
