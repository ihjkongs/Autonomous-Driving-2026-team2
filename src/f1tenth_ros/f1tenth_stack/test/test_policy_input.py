"""Tests for the simulator-to-real observation and action contract."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'policy_input', Path(__file__).parents[1] / 'f1tenth_stack/policy_input.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_matches_simulator_angle_grid():
    expected = np.linspace(1, 20, 1080).astype(np.float32)
    obs = policy.scan_to_observation(expected, -2.35, 4.7 / 1079, .02, 30)
    np.testing.assert_allclose(obs, expected)


def test_resamples_actual_angles_and_reversed_scan():
    angles = np.linspace(-2.355, 2.355, 1081)
    values = 5 + angles
    expected = 5 + np.linspace(-2.35, 2.35, 1080)
    for rays, start, increment in (
        (values, angles[0], angles[1] - angles[0]),
        (values[::-1], angles[-1], angles[-2] - angles[-1]),
    ):
        obs = policy.scan_to_observation(rays, start, increment, .02, 30)
        np.testing.assert_allclose(obs, expected, atol=1e-6)


def test_no_return_is_training_maximum():
    obs = policy.scan_to_observation([np.inf] * 1080, -2.35, 4.7 / 1079, .02, 30)
    assert np.all(obs == 30)


@pytest.mark.parametrize('bad', [np.nan, -np.inf, 0.0])
def test_invalid_returns_rejected(bad):
    values = np.full(1080, 5.0)
    values[200] = bad
    with pytest.raises(ValueError):
        policy.scan_to_observation(values, -2.35, 4.7 / 1079, .02, 30)


def test_missing_coverage_rejected():
    with pytest.raises(ValueError):
        policy.scan_to_observation([5] * 1080, -2, 4 / 1079, .02, 30)


def test_short_range_sensor_rejected():
    with pytest.raises(ValueError):
        policy.scan_to_observation([5] * 1080, -2.35, 4.7 / 1079, .02, 10)


def test_normalization_matches_training_and_rejects_bad_statistics():
    obs = np.arange(1080, dtype=np.float32)
    mean, var = np.ones(1080), np.full(1080, 4.0)
    np.testing.assert_allclose(policy.normalize_observation(obs, mean, var),
                               (obs - mean) / np.sqrt(var + 1e-8), rtol=1e-6)
    var[10] = -1
    with pytest.raises(ValueError):
        policy.normalize_observation(obs, mean, var)


def test_actions_use_training_limits():
    assert policy.decode_action([-1, -1], .48, 1, 4) == (-.48, 1)
    assert policy.decode_action([1, 1], .48, 1, 4) == (.48, 4)
    assert policy.decode_action([0, 0], .48, 1, 4) == (0, 2.5)
    with pytest.raises(ValueError):
        policy.decode_action([np.nan, 0], .48, 1, 4)
