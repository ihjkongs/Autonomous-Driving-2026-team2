"""Physical response, braking anticipation and temporal observation regressions."""
from collections import deque
from types import SimpleNamespace
import numpy as np
import pytest
from f1tenth_gym.envs.longitudinal import LongitudinalResponse
from utils.speed_profile import speed_profile, preview_speed
from utils.env import F1Wrapper


def response(**overrides):
    config = dict(acceleration=2., deceleration=3., time_constant=.25,
                  delay=.1, command_acceleration=2., command_deceleration=3.)
    config.update(overrides)
    return LongitudinalResponse(config, .025)


def test_velocity_remains_continuous_on_abrupt_braking():
    model = response()
    velocity = 0.
    for _ in range(200):
        velocity += .025 * model.update(6., velocity)
    before = velocity
    acceleration = model.update(1., velocity)
    assert -3. <= acceleration <= 2.
    assert before + .025*acceleration > 5.9
    for _ in range(20):
        acceleration = model.update(1., velocity)
        velocity += .025 * acceleration
        assert -3. <= acceleration <= 2.
    assert velocity > 4.


def test_delay_and_reset_do_not_leak_previous_episode():
    model = response()
    assert all(model.update(6., 0.) == 0 for _ in range(5))
    assert model.update(6., 0.) > 0
    model.reset()
    assert model.command == 0.
    assert all(model.update(6., 0.) == 0 for _ in range(5))


def test_closed_profile_respects_braking_and_acceleration():
    # Straight approach to tight corners, including braking across path seam.
    points = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.],
                       [4., 1.], [4., 2.], [3., 2.], [2., 2.], [1., 2.], [0., 2.], [0., 1.]])
    work = dict(vgain=1., speed_cap=6., lateral_acceleration=3., min_speed=1.,
                braking_deceleration=2., planning_acceleration=1.5, curvature_window=.75)
    arc, speed, _ = speed_profile(points, np.full(len(points), 8.), work)
    ds = np.diff(arc)
    assert np.all(speed**2 <= np.roll(speed, -1)**2 + 4*ds + 1e-9)
    assert np.all(np.roll(speed, -1)**2 <= speed**2 + 3*ds + 1e-9)
    assert speed[2] < 6.  # Anticipates the corner while still on straight.
    assert preview_speed(arc, speed, 11, 2.) <= min(speed[11], speed[0], speed[1])


def test_history_is_oldest_first_and_reset_pads_current_scan():
    wrapper = SimpleNamespace(lidar_frames=4, scan_history=deque(maxlen=4))
    first = F1Wrapper.getObs(wrapper, {'scans': [np.ones(1080)]}, reset=True)
    assert first.shape == (4320,) and np.all(first == 1.)
    second = F1Wrapper.getObs(wrapper, {'scans': [np.full(1080, 2.)]})
    assert np.array_equal(second.reshape(4, 1080)[:, 0], [1., 1., 1., 2.])
    fresh = F1Wrapper.getObs(wrapper, {'scans': [np.full(1080, 3.)]}, reset=True)
    assert np.all(fresh == 3.)


def test_profile_changes_invalidate_expert_data_signature():
    from easydict import EasyDict
    from ruamel.yaml import YAML
    from utils.expert_config import ROOT
    from utils.experiment import apply_experiment
    from utils.expert_data import signature
    args = EasyDict()
    for name in ('f1tenth', 'dynamic'):
        args.update(YAML(typ='safe').load((ROOT / 'configs/task' / (name + '.yaml')).read_text()))
    old = signature(args)
    apply_experiment(args, ROOT / 'configs/experiment_6mps.yaml')
    assert args.max_speed == 6. and args.obs_dim == 4320
    assert signature(args) != old
    previous = signature(args)
    args.longitudinal_response.delay = .2
    assert signature(args) != previous


def test_native_collection_matches_wrapper_including_temporal_history(tmp_path):
    from easydict import EasyDict
    from ruamel.yaml import YAML
    from utils.expert_config import ROOT
    from utils.experiment import apply_experiment
    from utils.expert_data import collect_verified, load_verified
    args = EasyDict(seed=73)
    for name in ('f1tenth', 'dynamic'):
        args.update(YAML(typ='safe').load((ROOT / 'configs/task' / (name + '.yaml')).read_text()))
    apply_experiment(args, ROOT / 'configs/experiment_6mps_randomized.yaml')
    datasets = []
    for mode in ('wrapper', 'native'):
        args.collection_step_mode = mode
        path = tmp_path / (mode + '.pkl')
        data = collect_verified(args, ['IMS'], path, total_steps=1, max_steps=10000)
        load_verified(path, args, ['IMS'])
        datasets.append(data)
    assert np.array_equal(datasets[0]['actions'], datasets[1]['actions'])
    assert np.array_equal(datasets[0]['observations'], datasets[1]['observations'])
    assert datasets[0]['metadata']['episodes'][0]['longitudinal_response'] == datasets[1]['metadata']['episodes'][0]['longitudinal_response']


@pytest.mark.parametrize('key', ['acceleration', 'deceleration', 'time_constant'])
def test_invalid_response_is_rejected(key):
    with pytest.raises(ValueError):
        response(**{key: 0.})


def test_resume_restores_weights_and_rejects_changed_normalization(tmp_path):
    import json
    import pickle
    import torch
    from easydict import EasyDict
    from utils.training_resume import restore_training
    source = tmp_path / 'source'
    (source / 'sim2real').mkdir(parents=True)
    (source / 'checkpoint').mkdir()
    (source / 'training_summary.json').write_text(json.dumps(dict(
        dataset_signature='verified', best_update=5000, selection={'completed': 92})))
    actor = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(actor.parameters())
    torch.save(dict(actor=actor.state_dict(), optim=optimizer.state_dict()), source/'checkpoint/model.pt')
    rms = SimpleNamespace(mean=np.zeros(2), var=np.ones(2), count=12)
    with (source/'sim2real/agent_obs_scale.pkl').open('wb') as stream:
        pickle.dump([rms.mean, rms.var, rms.count], stream)
    other = torch.nn.Linear(2, 2)
    agent = SimpleNamespace(actor=other, optimizer=torch.optim.Adam(other.parameters()), obs_rms=rms)
    (tmp_path/'backup').mkdir()
    args = EasyDict(resume_from=source, backup_dir=tmp_path/'backup', device='cpu')
    with pytest.raises(ValueError, match='signature'):
        restore_training(agent, args, 'different')
    rms.count = 13
    with pytest.raises(ValueError, match='normalization'):
        restore_training(agent, args, 'verified')
    rms.count = 12
    restored = restore_training(agent, args, 'verified')
    assert restored['source_best_update'] == 5000
    assert torch.equal(other.weight, actor.weight)
    assert (tmp_path/'backup/initial_model.pt').exists()


def test_collision_detects_body_overlap_even_at_rest_and_ignores_receding_wall():
    from f1tenth_gym.envs.laser_models import check_ttc_jit
    angles = np.array([0.])
    body = np.array([.13])
    assert check_ttc_jit(np.array([.10]), 3., angles, np.ones(1), body, .025)
    assert check_ttc_jit(np.array([.10]), 0., angles, np.ones(1), body, .025)
    assert not check_ttc_jit(np.array([.20]), -3., angles, np.ones(1), body, .025)
    assert check_ttc_jit(np.array([.20]), 3., angles, np.ones(1), body, .025)


def test_collision_geometry_uses_body_length_and_is_per_vehicle():
    from f1tenth_gym.envs.base_classes import RaceCar
    a = SimpleNamespace(params=dict(length=.57, width=.29), lidar_pos_offset=.155,
                        scan_angles=np.array([0., np.pi/2., np.pi]))
    b = SimpleNamespace(params=dict(length=1., width=.4), lidar_pos_offset=.1,
                        scan_angles=a.scan_angles)
    RaceCar._update_body_distances(a)
    before = a.side_distances.copy()
    RaceCar._update_body_distances(b)
    assert np.allclose(a.side_distances, [.13, .145, .44])
    assert np.allclose(b.side_distances, [.4, .2, .6])
    assert np.array_equal(a.side_distances, before)
