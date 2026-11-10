"""Command slew and restart behavior, independent of ROS and model inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'speed_limiter', Path(__file__).parents[1] / 'f1tenth_stack/speed_limiter.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SpeedLimiter = module.SpeedLimiter


def test_start_acceleration_and_sudden_braking_request():
    limiter = SpeedLimiter(2.0, 3.0)
    times = np.arange(0, 3, .025)
    speeds = [limiter.update(4.0, float(t)) for t in times]
    assert speeds[0] == 0
    assert speeds[-1] == 4
    np.testing.assert_array_less(np.diff(speeds), np.full(len(speeds)-1, .05+1e-10))
    assert limiter.update(1.0, 3.0) == pytest.approx(3.925)
    assert limiter.update(4.0, 3.025) == pytest.approx(3.975)


def test_irregular_callback_intervals_and_target_reaching():
    limiter = SpeedLimiter(2.0, 3.0)
    limiter.update(4, 0)
    assert limiter.update(4, .013) == pytest.approx(.026)
    assert limiter.update(4, .048) == pytest.approx(.096)
    assert limiter.update(.10, .073) == pytest.approx(.10)
    assert limiter.update(.09, .098) == pytest.approx(.09)


def test_long_gap_does_not_allow_large_jump():
    limiter = SpeedLimiter(2.0, 3.0, max_dt=.1)
    limiter.update(4, 0)
    assert limiter.update(4, 10) == pytest.approx(.2)


def test_reset_immediately_discards_old_speed_and_restarts_from_zero():
    limiter = SpeedLimiter(2.0, 3.0)
    for t in np.arange(0, 3, .025):
        limiter.update(4, float(t))
    assert limiter.speed == 4
    limiter.reset()
    assert limiter.speed == 0
    assert limiter.update(4, 5) == 0
    assert limiter.update(4, 5.025) == pytest.approx(.05)


def test_zero_target_is_normal_deceleration_unless_reset_is_called():
    limiter = SpeedLimiter(2.0, 3.0)
    limiter.update(4, 0)
    limiter.update(4, .1)
    assert limiter.update(0, .125) == pytest.approx(.125)
    limiter.reset()
    assert limiter.speed == 0


@pytest.mark.parametrize('target, now', [(np.nan, 1), (np.inf, 1), (-1, 1), (1, np.nan)])
def test_invalid_inputs_do_not_poison_state(target, now):
    limiter = SpeedLimiter(2.0, 3.0)
    limiter.update(4, 0)
    limiter.update(4, .025)
    with pytest.raises(ValueError):
        limiter.update(target, now)
    assert limiter.speed == pytest.approx(.05)
    assert limiter.last_time == .025


def test_backwards_time_rejected_and_duplicate_timestamp_has_no_step():
    limiter = SpeedLimiter(2.0, 3.0)
    limiter.update(4, 1)
    assert limiter.update(4, 1) == 0
    with pytest.raises(ValueError):
        limiter.update(4, .9)


@pytest.mark.parametrize('acc, dec', [(0, 3), (2, -1), (np.nan, 3), (2, np.inf)])
def test_invalid_limits_rejected(acc, dec):
    with pytest.raises(ValueError):
        SpeedLimiter(acc, dec)
