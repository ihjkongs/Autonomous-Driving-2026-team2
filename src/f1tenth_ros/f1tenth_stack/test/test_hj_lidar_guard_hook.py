"""A9 guard wired into the speed agent core: SHA checks, parity, stop/reset behaviour."""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).parents[1]
sys.path.insert(0, str(PKG))
from f1tenth_stack import hj_speed_policy_runtime as rt  # noqa: E402

SRC = Path(__file__).parents[3]
GUARD = PKG / 'f1tenth_stack/hj_lidar_following_guard_a9.py'
PARAMS = PKG / 'config/hj_lidar_guard_a9_params.yaml'
DATA = SRC.parent / '.local/hj/disposable/261123/lidar_guard_dev/data_guarded_A9'
GRID = dict(angle_min=-2.35, angle_increment=4.7/1079, range_min=0., range_max=30.)


def hook():
    return rt.load_guard_hook(GUARD, rt.A9_GUARD_SHA256, PARAMS, rt.A9_PARAMS_SHA256)


class Fixed:
    def predict(self, obs):
        return .1, 8.


def make_core(guard, speed_limit=8.):
    return rt.SpeedAgentCore(rt.ScanSpeedAssembler(), Fixed(), speed_limit, 100., 100., .2, guard)


def test_sha_checks(tmp_path):
    assert rt.file_sha256(GUARD) == rt.A9_GUARD_SHA256
    assert rt.file_sha256(PARAMS) == rt.A9_PARAMS_SHA256
    bad = tmp_path/'g.py'
    shutil.copy(GUARD, bad)
    bad.write_bytes(bad.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='SHA256'):
        rt.load_guard_hook(bad, rt.A9_GUARD_SHA256, PARAMS, rt.A9_PARAMS_SHA256)
    with pytest.raises(ValueError, match='SHA256'):
        rt.load_guard_hook(GUARD, rt.A9_GUARD_SHA256, PARAMS, '0'*64)
    assert 'algorithm' not in sys.modules or sys.modules['algorithm'].__dict__.get('__file__')


def test_shim_equals_repo_safe_following_speed():
    sys.path.insert(0, str(SRC/'f1tenth'))
    try:
        from utils.traffic_expert import safe_following_speed as ref
    except Exception:
        pytest.skip('simulator repo not importable')
    finally:
        sys.path.pop(0)
    from f1tenth_stack.hj_guard_shims import safe_following_speed as ours
    rng = np.random.default_rng(0)
    for gap, v in rng.uniform(0, 12, (2000, 2)):
        assert ours(gap, 0., 1.5, .8, 1.) == ref(gap, 0., braking=1.5, reaction=.8, standstill_gap=1.)


def feed(core, obs, k, t=None):
    """Push observation row k through the full core path (scan frame + odom)."""
    t = 10+k*.025 if t is None else t
    core.assembler.on_odom(float(obs[-1]), t, t)
    return core.step(obs[3240:4320], GRID['angle_min'], GRID['angle_increment'], 0., 30., t, t)


def car_obs(gap, speed=3.):
    scan = np.full(1080, 20.)
    centre = 540
    scan[centre-25:centre+26] = gap
    obs = np.concatenate([np.tile(scan, 4), [speed]]).astype(np.float32)
    return obs


def test_guard_on_off():
    obs = car_obs(2.0)
    off = make_core(rt.no_guard)
    on = make_core(hook())
    for k in range(3):
        o_off, o_on = feed(off, obs, k), feed(on, obs, k)
    assert o_off[2] == 8.
    assert o_on[2] < 3. and o_on[2] == on.guard.last[0] and o_on[1] == 8.   # policy speed untouched


def test_clear_road_not_limited():
    core = make_core(hook())
    out = feed(core, np.concatenate([np.full(4320, 20.), [3.]]).astype(np.float32), 0)
    assert out[2] == 8. and core.guard.last[0] == np.inf


def test_nonfinite_input_stops_guard_direct():
    h = hook()
    bad = car_obs(5.)
    bad[100] = np.nan
    assert h(8., dict(obs=bad, steer=0.)) == 0.
    assert h(8., dict(obs=car_obs(5.), steer=float('nan'))) == 0.
    assert h(8., dict(obs=car_obs(5.)[:-1], steer=0.)) == 0.


def test_nonfinite_scan_or_odom_through_core_rejects_and_resets_memory():
    h = hook()
    core = make_core(h)
    obs = car_obs(2.0)
    for k in range(12):
        feed(core, obs, k)
    assert h.memory['run'] >= 10 or h.memory['gap'] is not None
    bad = obs.copy()
    bad[3300] = np.nan
    with pytest.raises(rt.Rejected):
        feed(core, bad, 12)
    assert h.memory == dict(run=0, gap=None, age_s=0., travel=0.)
    for k in range(13, 25):
        feed(core, obs, k)
    core.assembler.on_odom(float('nan'), 10.6, 10.6)
    with pytest.raises(rt.Rejected):
        core.step(obs[3240:4320], GRID['angle_min'], GRID['angle_increment'], 0., 30., 10.6, 10.6)
    assert h.memory['run'] == 0 and h.memory['gap'] is None


def test_history_restart_after_gap_resets_memory():
    h = hook()
    core = make_core(h)
    obs = car_obs(2.0)
    for k in range(12):
        feed(core, obs, k)
    assert h.memory['run'] > 1
    feed(core, obs, 99, t=10+12*.025+1.0)       # gap > scan_timeout -> history restarts
    assert h.memory['run'] == 1                 # reset then one new confirmation step


def test_explicit_stop_resets_memory():
    h = hook()
    core = make_core(h)
    obs = car_obs(2.0)
    for k in range(12):
        feed(core, obs, k)
    core.stop()
    assert h.memory == dict(run=0, gap=None, age_s=0., travel=0.) and h.last is None


# ---- parity with simulator guard logs ----
EPISODES = ['single_261120_HJCompetition_train_012_grid3x10_corners20',
            'single_261120_HJCompetition_train_203_grid2x9_corners4',
            'two_261120_HJCompetition_train_203_grid2x9_corners4',
            'two_261120_HJCompetition_train_076_grid2x9_corners6']


@pytest.mark.skipif(not DATA.is_dir(), reason='guard dev data absent')
@pytest.mark.parametrize('name', EPISODES)
def test_parity_with_recorded_guarded_policy_limits(name):
    rec = np.load(DATA/(name+'.npz'))['guard_limit']
    obs = np.fromfile(DATA/(name+'.obs.f32'), dtype=np.float32).reshape(-1, 4321)
    assert len(obs) == len(rec)
    h = hook()
    core = make_core(h, speed_limit=1e3)
    got = np.empty(len(rec))
    for k in range(len(obs)):
        out = feed(core, obs[k], k)
        got[k] = h.last[0]
        want = min(8., rec[k])
        assert out[2] == want
    assert np.array_equal(got, rec)          # includes inf == inf, bit identical
    assert np.isfinite(rec).any() or name.startswith('single')
