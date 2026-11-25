"""Scan+odom-speed runtime: assembly parity with simulator data, stop-on-bad-odom, hook."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

PKG = Path(__file__).parents[1]
sys.path.insert(0, str(PKG))
from f1tenth_stack import hj_speed_policy_runtime as rt  # noqa: E402

SRC = Path(__file__).parents[3]
B68 = SRC / 'f1tenth/results/hj/261123/68_bc_compact_dagger_from67_4000_seed261123'
TS = B68 / 'sim2real/driving_policy.ts'
FAR3 = B68 / 'data/collection/far3.npz'
ANGLES = np.linspace(-2.35, 2.35, 1080)
GRID = dict(angle_min=-2.35, angle_increment=4.7/1079, range_min=0., range_max=30.)


class FakePolicy:
    def __init__(self, speed=3.):
        self.speed, self.inputs = speed, []

    def predict(self, obs):
        self.inputs.append(obs)
        return .1, self.speed


def core(speed_limit=10., guard=rt.no_guard, **kw):
    asm = rt.ScanSpeedAssembler(**kw)
    return rt.SpeedAgentCore(asm, FakePolicy(), speed_limit, 100., 100., .2, guard), asm


def step(c, t, ranges=None):
    ranges = np.full(1080, 5.) if ranges is None else ranges
    return c.step(ranges, GRID['angle_min'], GRID['angle_increment'], 0., 30., t, t)


def test_no_odom_rejects():
    c, _ = core()
    with pytest.raises(rt.Rejected, match='no odom'):
        step(c, 1.)


@pytest.mark.parametrize('vx', [float('nan'), float('inf'), -float('inf'), -.5, 9.])
def test_bad_odom_value_rejects(vx):
    c, a = core()
    a.on_odom(vx, 1., 1.)
    with pytest.raises(rt.Rejected):
        step(c, 1.)


def test_odom_skew_and_stale_receipt_reject():
    c, a = core(odom_max_skew=.1, odom_timeout=.25)
    a.on_odom(2., 1., 1.)
    with pytest.raises(rt.Rejected, match='skew'):
        step(c, 1.2)
    a.on_odom(2., 1.3, 1.)
    with pytest.raises(rt.Rejected, match='stale'):
        step(c, 1.3)  # mono 1.3 > 1.0+.25


def test_odom_nonfinite_time_rejects():
    c, a = core()
    a.on_odom(2., float('nan'), 1.)
    with pytest.raises(rt.Rejected):
        step(c, 1.)


def test_sign_and_deadband():
    c, a = core()
    a.on_odom(-.05, 1., 1.)           # small negative noise -> abs
    step(c, 1.)
    assert c.policy.inputs[-1][-1] == pytest.approx(.05)
    a.on_odom(0., 1.025, 1.025)
    step(c, 1.025)
    assert c.policy.inputs[-1][-1] == 0.
    a.on_odom(-.3, 1.05, 1.05)        # reverse beyond tolerance -> stop
    with pytest.raises(rt.Rejected, match='negative'):
        step(c, 1.05)


def test_history_order_and_gap_reset():
    c, a = core()
    for i in range(5):
        t = 1+i*.025
        a.on_odom(1., t, t)
        step(c, t, np.full(1080, float(i+1)))
    obs = c.policy.inputs[-1]
    assert obs.shape == (4321,)
    assert [obs[k*1080] for k in range(4)] == [2., 3., 4., 5.] and obs[-1] == 1.
    first = c.policy.inputs[0]
    assert all(first[k*1080] == 1. for k in range(4))   # repeat first scan
    t = 1.2+.5                                           # gap > scan_timeout
    a.on_odom(1., t, t)
    step(c, t, np.full(1080, 9.))
    assert all(c.policy.inputs[-1][k*1080] == 9. for k in range(4))


def test_bad_scan_resets_and_rejects():
    c, a = core()
    a.on_odom(1., 1., 1.)
    step(c, 1.)
    bad = np.full(1080, 5.)
    bad[3] = np.nan
    a.on_odom(1., 1.025, 1.025)
    with pytest.raises(rt.Rejected, match='scan'):
        step(c, 1.025, bad)
    assert not a.history


def test_speed_limit_slew_and_guard_hook():
    seen = []

    def guard(target, ctx):
        seen.append((target, ctx['policy_speed']))
        return target/2
    c, a = core(speed_limit=1., guard=guard)
    c.limiter.max_acceleration = 100.
    for i in range(3):
        t = 1+i*.025
        a.on_odom(1., t, t)
        out = step(c, t)
    assert out[1] == 3. and out[2] == .5 and seen[0] == (1., 3.)   # limit, then hook
    assert out[3] <= out[2]

    def bad_guard(target, ctx):
        return target+1.
    c2, a2 = core(speed_limit=1., guard=bad_guard)
    a2.on_odom(1., 1., 1.)
    with pytest.raises(ValueError, match='only reduce'):
        step(c2, 1.)


def test_odom_fresh_for_watchdog():
    _, a = core(odom_timeout=.25)
    assert not a.odom_fresh(1.)
    a.on_odom(1., 1., 1.)
    assert a.odom_fresh(1.2) and not a.odom_fresh(1.3)


def test_decode_matches_policy_contract_formula():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1.5, 1.5, (1000, 2))
    got = rt.decode_physical(a, 0., 8., .48, .1)
    c = np.clip(a, -1, 1)
    sp = (c[:, 1]+1)*.5*8.
    want = np.stack([c[:, 0]*.48, np.where(sp <= .1, 0., sp)], -1)
    assert np.array_equal(got, want)


# ---- simulator-data parity (skipped when the local result files are absent) ----
needs_data = pytest.mark.skipif(not (FAR3.is_file() and TS.is_file()), reason='68 artifacts absent')


def episodes(obs, ep, ts, count):
    out = []
    for e in np.unique(ep)[:count]:
        rows = np.flatnonzero(ep == e)
        assert (np.diff(ts[rows]) == 1).all() and (np.diff(rows) == 1).all()
        out.append(rows)
    return out


def replay(rows, obs, grid):
    """Feed delivered scan frames as LaserScans; return assembled 4321 rows."""
    asm = rt.ScanSpeedAssembler()
    got = []
    for i, r in enumerate(rows):
        t = 10+i*.025
        asm.on_odom(float(obs[r, -1]), t, t)
        got.append(asm.assemble(grid['ranges'](obs[r, 3240:4320]), grid['angle_min'], grid['angle_increment'],
                                0., 30., t, t))
    return np.asarray(got)


@needs_data
def test_assembly_bit_identical_to_saved_observations():
    z = np.load(FAR3, mmap_mode='r')
    ep, ts = np.asarray(z['episode_indices']), np.asarray(z['timestep_indices'])
    grid = dict(GRID, ranges=lambda f: f)
    for rows in episodes(None, ep, ts, 3):
        rows = rows[:600]
        obs = np.asarray(z['observations'][rows[0]:rows[-1]+1])
        got = replay(np.arange(len(rows)), obs, grid)
        diff = np.abs(got-obs).max()
        assert diff == 0., diff


@needs_data
def test_ts_output_same_for_assembled_and_saved_input():
    import torch
    z = np.load(FAR3, mmap_mode='r')
    ep, ts = np.asarray(z['episode_indices']), np.asarray(z['timestep_indices'])
    rows = episodes(None, ep, ts, 1)[0][:200]
    obs = np.asarray(z['observations'][rows[0]:rows[-1]+1])
    got = replay(np.arange(len(rows)), obs, dict(GRID, ranges=lambda f: f))
    pol = rt.SpeedPolicy(TS)
    ref = torch.jit.load(str(TS), map_location='cpu').eval()
    for g, o in zip(got, obs):
        with torch.inference_mode():
            want = ref(torch.from_numpy(o[None])).numpy()[0]
        assert np.array_equal(np.array(pol.predict(g)), want)


# ---- speed_source: vesc_core (pure functions) ----
def test_erpm_proportional_roundtrip_is_identity():
    for gain in (2500., 5300.):
        for v in (0.5, 1., 2.5, 4., 6.):
            assert rt.erpm_to_speed(gain*v, gain, 0., 0.) == pytest.approx(v, abs=1e-12)
    # Command ERPM = gain*v + offset (offset -325) is recovered exactly only when the
    # core offset equals that intercept, i.e. only under the assumption that the VESC follows
    # the commanded ERPM; otherwise the default (0) leaves offset/gain = -0.13 m/s.
    assert rt.erpm_to_speed(2500.*2.-325., 2500., -325., 0.) == pytest.approx(2.)
    assert rt.erpm_to_speed(2500.*2.-325., 2500., 0., 0.) == pytest.approx(2.-.13)


def test_erpm_zero_and_deadband():
    assert rt.erpm_to_speed(0., 2500.) == 0.                 # standstill -> 0 (default offset 0)
    assert rt.erpm_to_speed(0., 2500., 0., 0.) == 0.
    assert rt.erpm_to_speed(100., 2500., 0., .05) == 0.      # 0.04 m/s inside deadband
    assert rt.erpm_to_speed(150., 2500., 0., .05) == pytest.approx(.06)


@pytest.mark.parametrize('erpm', [float('nan'), float('inf'), -float('inf')])
def test_erpm_nonfinite_rejects(erpm):
    with pytest.raises(rt.Rejected):
        rt.erpm_to_speed(erpm, 2500.)


@pytest.mark.parametrize('gain', [0., -1., float('nan')])
def test_erpm_bad_gain_raises(gain):
    with pytest.raises(ValueError):
        rt.erpm_to_speed(1000., gain, 0.)


def test_vesc_core_speed_through_assembler_negative_and_timeout():
    c, a = core()
    a.on_odom(rt.erpm_to_speed(2500.*2., 2500.), 1., 1.)
    step(c, 1.)
    assert c.policy.inputs[-1][-1] == pytest.approx(2.)
    a.on_odom(rt.erpm_to_speed(-2500., 2500.), 1.025, 1.025)   # reverse -> stop
    with pytest.raises(rt.Rejected, match='negative'):
        step(c, 1.025)
    a.on_odom(1., 1.05, 1.05)                                  # stale core -> stop
    with pytest.raises(rt.Rejected, match='stale'):
        step(c, 1.4)
    assert rt.SPEED_SOURCES == ('odom', 'vesc_core')
