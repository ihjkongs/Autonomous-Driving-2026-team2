"""Continuous-lap evaluation must not accept one-lap termination or a crash."""
from types import SimpleNamespace

from easydict import EasyDict
import numpy as np
import pytest

import validate_small_map_expert as runner


class SquareEnvironment:
    def __init__(self, collision_step=None):
        self.unwrapped = self
        self.track = SimpleNamespace(centerline=SimpleNamespace(
            xs=np.array([0., 1., 1., 0.]), ys=np.array([0., 0., 1., 1.])))
        agent = SimpleNamespace(state=np.array([0., 0., 0., 1.]), response_parameters={})
        self.sim = SimpleNamespace(agents=[agent])
        self.collisions = [False]
        self.current_time = 0.
        self.steps = 0
        self.collision_step = collision_step
        self.closed = False

    def raw(self):
        xy = [(0.,0.), (1.,0.), (1.,1.), (0.,1.)][self.steps % 4]
        return dict(poses_x=[xy[0]], poses_y=[xy[1]], scans=np.array([1.]))

    def reset(self, options):
        return None, {'obs_dict': self.raw()}

    def step(self, command):
        self.steps += 1
        self.current_time += .025
        self.collisions = [self.steps == self.collision_step]
        return self.raw(), 0., self.steps >= 4, False, {}

    def close(self):
        self.closed = True


@pytest.mark.parametrize('collision_step,expected', [(None,'complete'), (12,'collision')])
def test_three_laps_preserve_state_and_reject_finish_collision(monkeypatch, collision_step, expected):
    env = SquareEnvironment(collision_step)
    monkeypatch.setattr(runner, 'F1Wrapper', lambda *args: env)
    monkeypatch.setattr(runner, 'PurePursuitPlanner', lambda *args:
                        SimpleNamespace(plan=lambda raw, params: (1., 0.)))
    config = EasyDict(length_f=.2, length_r=.15, expert_parameters={},
                     max_episode_steps=20, min_speed=1., max_speed=6., max_steer=.48)
    row = runner.evaluate((config, 'square', 'cl_grid_static', 41, 'test', 3))
    assert row['outcome'] == expected
    assert row['steps'] == 12
    assert row['forward_progress_m'] == pytest.approx(12.)
    assert row['mean_actual_speed_mps'] == 1.
    assert env.closed


def test_selection_prefers_completion_and_normalizes_lap_count():
    base = dict(required_laps=3, steps=120, mean_actual_speed_mps=2., max_actual_speed_mps=3.)
    records = [dict(base, candidate='safe', outcome='complete', elapsed_s=30.),
               dict(base, candidate='fast_crash', outcome='collision', elapsed_s=1.)]
    scores = runner.summarize(records)
    assert scores[0]['candidate'] == 'safe'
    assert scores[0]['mean_lap_s'] == 10.
