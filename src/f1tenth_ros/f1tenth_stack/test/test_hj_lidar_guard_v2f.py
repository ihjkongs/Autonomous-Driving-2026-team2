"""V2f guard wired into the speed agent core: SHA pins, variant resolution, parity, 71 config."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).parents[1]
sys.path.insert(0, str(PKG))
from f1tenth_stack import hj_speed_policy_runtime as rt  # noqa: E402

SRC = Path(__file__).parents[3]
GUARD = PKG / 'f1tenth_stack/hj_lidar_following_guard_v2f.py'
PARAMS = PKG / 'config/hj_lidar_guard_v2f_params.yaml'
ORIG = SRC / 'f1tenth/hj_lidar_following_guard_v2.py'
FROZEN = (SRC / 'f1tenth/results/hj/261123/68_bc_compact_dagger_from67_4000_seed261123/'
          'evaluation/lidar_following_guard/frozen_V2f/guarded_V2f_params.json')
LOCAL = SRC.parent / '.local/hj/disposable/261123/lidar_guard_dev'
DATA = LOCAL / 'data_guarded_A9'            # stored observations
REC = LOCAL / 'v2/parity_V2f'               # V2f GuardedPolicy (system 71 actor.pt, trace) limits
GRID = dict(angle_min=-2.35, angle_increment=4.7/1079, range_min=0., range_max=30.)


def hook():
    return rt.load_guard_hook(GUARD, rt.V2F_GUARD_SHA256, PARAMS, rt.V2F_PARAMS_SHA256)


class Fixed:
    def predict(self, obs):
        return .1, 8.


def make_core(guard, speed_limit=8.):
    return rt.SpeedAgentCore(rt.ScanSpeedAssembler(), Fixed(), speed_limit, 100., 100., .2, guard)


def feed(core, obs, k, t=None):
    t = 10+k*.025 if t is None else t
    core.assembler.on_odom(float(obs[-1]), t, t)
    return core.step(obs[3240:4320], GRID['angle_min'], GRID['angle_increment'], 0., 30., t, t)


def car_obs(gap, speed=3.):
    scan = np.full(1080, 20.)
    scan[540-25:540+26] = gap
    return np.concatenate([np.tile(scan, 4), [speed]]).astype(np.float32)


def test_shas_and_byte_identity():
    assert rt.file_sha256(GUARD) == rt.V2F_GUARD_SHA256 == \
        '234aa1b8b982ebef6276c6d41b52353ea8955ffe4cfb26ba91fbe4d464fe4cf2'
    assert rt.file_sha256(PARAMS) == rt.V2F_PARAMS_SHA256 == \
        'a44923a9164e7250e8a07c6715065c245c4a4b8170ffeb183f7afc1321a4acb4'
    if ORIG.is_file():
        assert ORIG.read_bytes() == GUARD.read_bytes()
    if FROZEN.is_file():
        assert FROZEN.read_bytes() == PARAMS.read_bytes()


def test_sha_mismatch_blocks_load(tmp_path):
    bad = tmp_path/'g.py'
    shutil.copy(GUARD, bad)
    bad.write_bytes(bad.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='SHA256'):
        rt.load_guard_hook(bad, rt.V2F_GUARD_SHA256, PARAMS, rt.V2F_PARAMS_SHA256)
    with pytest.raises(ValueError, match='SHA256'):
        rt.load_guard_hook(GUARD, rt.V2F_GUARD_SHA256, PARAMS, rt.A9_PARAMS_SHA256)
    with pytest.raises(ValueError, match='SHA256'):
        rt.load_guard_hook(GUARD, rt.A9_GUARD_SHA256, PARAMS, rt.V2F_PARAMS_SHA256)


def test_params_are_v2_param_set():
    params = json.loads(PARAMS.read_text())
    mod = rt.load_guard_module(GUARD, rt.V2F_GUARD_SHA256)
    assert set(params) == set(mod.DEFAULT_PARAMS)      # nothing missing/extra vs the guard
    assert params['hold_floor_mode'] == 'gap_clamp' and params['held_slow_floor'] is True


def test_variant_resolution():
    cfg = lambda: '/share/config'  # noqa: E731
    f, s, p, ps = rt.resolve_guard_paths('v2f', '', '', '', '', cfg)
    assert Path(f).name == 'hj_lidar_following_guard_v2f.py' and s == rt.V2F_GUARD_SHA256
    assert p == '/share/config/hj_lidar_guard_v2f_params.yaml' and ps == rt.V2F_PARAMS_SHA256
    f, s, p, ps = rt.resolve_guard_paths('a9', '', '', '', '', cfg)
    assert Path(f).name == 'hj_lidar_following_guard_a9.py' and s == rt.A9_GUARD_SHA256
    assert Path(p).name == 'hj_lidar_guard_a9_params.yaml' and ps == rt.A9_PARAMS_SHA256
    assert rt.resolve_guard_paths('custom', 'a', 'b', 'c', 'd', cfg) == ('a', 'b', 'c', 'd')
    with pytest.raises(ValueError):
        rt.resolve_guard_paths('custom', 'a', 'b', '', 'd', cfg)
    with pytest.raises(ValueError):
        rt.resolve_guard_paths('v3', '', '', '', '', cfg)
    # the packaged files really exist and match their pins
    for variant in ('a9', 'v2f'):
        f, s, p, ps = rt.resolve_guard_paths(variant, '', '', '', '', lambda: PKG/'config')
        assert rt.file_sha256(f) == s and rt.file_sha256(p) == ps


def test_71_config_pins_v2f_and_default_config_stays_off():
    yaml = pytest.importorskip('yaml')
    c71 = yaml.safe_load((PKG/'config/hj_speed_agent_71.yaml').read_text())['hj_speed_agent_node']['ros__parameters']
    assert c71['guard_enabled'] is True and c71['guard_variant'] == 'v2f'
    assert c71['guard_sha256'] == rt.V2F_GUARD_SHA256 and c71['guard_params_sha256'] == rt.V2F_PARAMS_SHA256
    assert c71['dry_run'] is True and c71['speed_limit'] == 1.0
    base = yaml.safe_load((PKG/'config/hj_speed_agent.yaml').read_text())['hj_speed_agent_node']['ros__parameters']
    assert base['guard_enabled'] is False and base['dry_run'] is True
    assert set(c71) == set(base)                       # same parameter set, no stray keys


def test_guard_on_off_and_clear_road():
    obs = car_obs(2.0)
    off, on = make_core(rt.no_guard), make_core(hook())
    for k in range(3):
        o_off, o_on = feed(off, obs, k), feed(on, obs, k)
    assert o_off[2] == 8. and o_on[2] < 3. and o_on[2] == on.guard.last[0] and o_on[1] == 8.
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


FRESH = dict(run=0, gap=None, age_s=0., travel=0., leader_speed=0., slow=False)


def test_memory_reset_on_bad_scan_gap_and_stop():
    h = hook()
    core = make_core(h)
    obs = car_obs(2.0)
    for k in range(12):
        feed(core, obs, k)
    assert h.memory['run'] >= 10 and h.memory['gap'] is not None
    bad = obs.copy()
    bad[3300] = np.nan
    with pytest.raises(rt.Rejected):
        feed(core, bad, 12)
    assert h.memory == FRESH
    for k in range(13, 25):
        feed(core, obs, k)
    feed(core, obs, 99, t=10+25*.025+1.0)      # gap > scan_timeout -> history restarts
    assert h.memory['run'] == 1
    core.stop()
    assert h.memory == FRESH and h.last is None


def test_held_mode_floor_after_occlusion():
    """Leader seen at 1.2 m then occluded: V2f holds (limit finite) and never raises speed."""
    h = hook()
    core = make_core(h)
    seen, clear = car_obs(1.2, speed=3.), np.concatenate([np.full(4320, 20.), [3.]]).astype(np.float32)
    for k in range(12):
        feed(core, seen, k)
    out = feed(core, clear, 12)
    assert np.isfinite(h.last[0]) and out[2] <= 8.


EPISODES = ['single_261120_HJCompetition_train_012_grid3x10_corners20',
            'single_261120_HJCompetition_train_203_grid2x9_corners4',
            'two_261120_HJCompetition_train_203_grid2x9_corners4',
            'two_261120_HJCompetition_train_076_grid2x9_corners6',
            'two_261120_HJCompetition_train_010_grid3x10_corners26']


@pytest.mark.skipif(not (DATA.is_dir() and REC.is_dir()), reason='guard dev data absent')
@pytest.mark.parametrize('name', EPISODES)
def test_parity_with_recorded_v2f_guardedpolicy_limits(name):
    """Node path (assembler -> core -> hook) vs V2f GuardedPolicy trace on the same stored obs: bit identical."""
    rec = np.load(REC/(name+'.npz'))
    limit, mode = rec['guard_limit'], rec['mode']
    obs = np.fromfile(DATA/(name+'.obs.f32'), dtype=np.float32).reshape(-1, 4321)
    assert len(obs) == len(limit)
    h = hook()
    core = make_core(h, speed_limit=1e3)
    got = np.empty(len(limit))
    for k in range(len(obs)):
        out = feed(core, obs[k], k)
        got[k] = h.last[0]
        assert out[2] == min(8., limit[k])
    assert np.array_equal(got, limit)          # inf == inf included
    assert (mode > 0).any() == np.isfinite(limit).any()
