"""Regression tests for expert target geometry and failure handling."""
from types import SimpleNamespace
import numpy as np
import pytest

from utils.rule_based import PurePursuitPlanner, first_point_on_trajectory_intersecting_circle


def track(points):
    line = SimpleNamespace(xs=np.array(points)[:, 0], ys=np.array(points)[:, 1],
                           vxs=np.full(len(points), 4.0))
    return SimpleNamespace(centerline=line, raceline=line)


def test_circle_intersection_is_translation_invariant():
    # World-coordinate cancellation used to lose targets far from the origin.
    path = np.array([[-2., 0.], [2., 0.]], dtype=np.float32)
    for offset in (np.array([0., 0.]), np.array([-128., -48.]), np.array([10000., 10000.])):
        point = offset + [0., .25]
        target, _, _ = first_point_on_trajectory_intersecting_circle(point, 1., path + offset, .5)
        assert target is not None
        assert np.linalg.norm(target - point) == pytest.approx(1., abs=1e-8)
        assert target[0] > point[0]


def test_target_is_intersection_not_previous_waypoint():
    planner = PurePursuitPlanner(track([[0., 0.], [4., 0.], [4., 4.], [0., 4.]]), .35)
    target, _ = planner._get_current_waypoint(planner.waypoints, 1., np.array([.5, .2]), 0.)
    assert np.linalg.norm(target[:2] - [.5, .2]) == pytest.approx(1., abs=1e-6)
    assert target[0] > .5


def test_reacquisition_has_speed_and_forward_target():
    planner = PurePursuitPlanner(track([[0., 0.], [4., 0.], [4., 4.], [0., 4.]]), .35)
    target, _ = planner._get_current_waypoint(planner.waypoints, .5, np.array([1., -.8]), 0.)
    assert target.shape == (3,)
    assert target[0] == pytest.approx(1.5)
    assert target[2] == 4.


def test_closing_segment_is_available():
    planner = PurePursuitPlanner(track([[0., 0.], [4., 0.], [4., 4.], [0., 4.]]), .35)
    assert np.array_equal(planner.waypoints[0], planner.waypoints[-1])
    target, _ = planner._get_current_waypoint(planner.waypoints, 1., np.array([0., .2]), -np.pi / 2)
    assert np.linalg.norm(target[:2] - [0., .2]) == pytest.approx(1., abs=1e-6)
    assert target[0] > 0.


def test_lost_path_does_not_produce_expert_label():
    planner = PurePursuitPlanner(track([[0., 0.], [4., 0.], [4., 4.], [0., 4.]]), .35)
    obs = dict(poses_x=[100.], poses_y=[100.], poses_theta=[0.], linear_vels_x=[0.])
    with pytest.raises(RuntimeError, match='lost'):
        planner.plan(obs, dict(tlad=1., vgain=.5))


def test_nearest_projection_matches_vector_reference():
    from utils.rule_based import nearest_point_on_trajectory
    rng = np.random.RandomState(81)
    points = rng.normal(size=(100, 2)).cumsum(axis=0)
    for point in rng.normal(size=(20, 2)):
        delta = np.diff(points, axis=0)
        fraction = np.clip(((point - points[:-1]) * delta).sum(axis=1) / (delta**2).sum(axis=1), 0., 1.)
        projections = points[:-1] + fraction[:, None] * delta
        distances = np.linalg.norm(projections - point, axis=1)
        target, distance, t, index = nearest_point_on_trajectory(point, points)
        expected = distances.argmin()
        assert index == expected
        assert distance == pytest.approx(distances[expected])
        assert t == pytest.approx(fraction[expected])
        assert np.allclose(target, projections[expected])


def test_unverified_old_data_is_rejected(tmp_path):
    import pickle
    from easydict import EasyDict
    from ruamel.yaml import YAML
    from utils.expert_config import ROOT
    from utils.expert_data import load_verified
    args = EasyDict()
    for name in ('f1tenth', 'dynamic'):
        args.update(YAML(typ='safe').load((ROOT / 'configs/task' / (name + '.yaml')).read_text()))
    path = tmp_path / 'old.pkl'
    path.write_bytes(pickle.dumps(dict(observations=np.zeros((2, 1080)), actions=np.zeros((2, 2)))))
    with pytest.raises(ValueError, match='unverified'):
        load_verified(path, args, ['Sochi'])


def test_low_speed_rk4_matches_fine_step_reference():
    from f1tenth_gym.envs.integrator import RK4Integrator
    from f1tenth_gym.envs.dynamic_models import vehicle_dynamics_st
    from ruamel.yaml import YAML
    from utils.expert_config import ROOT
    params = YAML(typ='safe').load((ROOT / 'configs/task/dynamic.yaml').read_text())
    params['lf'], params['lr'] = params['length_f'], params['length_r']
    integrator = RK4Integrator()
    initial = np.array([0., 0., .2, 1., 0., .2, .03])
    actual, reference = initial.copy(), initial.copy()
    for _ in range(40):
        actual = integrator.integrate(vehicle_dynamics_st, actual, np.zeros(2), .025, params)
    for _ in range(1000):
        reference = integrator.integrate(vehicle_dynamics_st, reference, np.zeros(2), .001, params)
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual - reference)) < 1e-5
    assert abs(actual[5]) < 1.
