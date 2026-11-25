"""hj_bag_replay with fake bag data built from a simulator recording (no ROS, no rosbags)."""
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).parents[1]
sys.path.insert(0, str(PKG))
from f1tenth_stack import hj_bag_replay as br  # noqa: E402

SRC = Path(__file__).parents[3]
B68 = SRC / 'f1tenth/results/hj/261123/68_bc_compact_dagger_from67_4000_seed261123'
TS = B68 / 'sim2real/driving_policy.ts'
FAR3 = B68 / 'data/collection/far3.npz'
needs_data = pytest.mark.skipif(not (FAR3.is_file() and TS.is_file()), reason='68 artifacts absent')


def fake_bag(rows=150, episode_pos=0):
    """Simulator rows -> LaserScan/Odometry-like arrays. Returns (data, true_speed)."""
    z = np.load(FAR3, mmap_mode='r')
    ep = np.asarray(z['episode_indices'])
    idx = np.flatnonzero(ep == np.unique(ep)[episode_pos])[:rows]
    obs = np.asarray(z['observations'][idx[0]:idx[-1] + 1])
    t = 10 + np.arange(len(idx)) * .025
    data = dict(scan_t=t, ranges=obs[:, 3240:4320].astype(np.float64), angle_min=-2.35,
                angle_increment=4.7 / 1079, range_min=0., range_max=30.,
                odom_t=t - .005, odom_vx=obs[:, -1].astype(np.float64))
    return data, obs[:, -1].astype(np.float64)


@needs_data
def test_replay_matches_recorded_and_detects_difference(tmp_path):
    data, _ = fake_bag()
    core = br.build_core(str(TS), speed_limit=4., guard_variant='v2f')
    res = br.replay(data, core)
    assert not res['rejected'] and np.isfinite(res['command']).all()
    assert (res['target'] <= np.minimum(res['policy_speed'], 4.) + 1e-9).all()
    # a bag recorded from the same path compares with zero error
    for name, key in (('prev', 'command'), ('pred', 'policy_speed'), ('drive', 'command')):
        data[name + '_t'] = res['t'].copy()
        data[name + '_steer'] = res['steer'].copy()
        data[name + '_speed'] = res[key].copy()
    s = br.compare(res, data)
    assert s['prev']['n'] == len(res['t']) and s['prev']['speed_max'] == 0.
    assert s['pred']['steer_max'] == 0. and s['drive']['speed_mae'] == 0.
    # a different recorded command is reported
    data['prev_speed'] = data['prev_speed'] + .5
    assert br.compare(res, data)['prev']['speed_mean_diff'] == pytest.approx(-.5)
    # the same data through a fresh core gives identical output (deterministic)
    res2 = br.replay(data, br.build_core(str(TS), speed_limit=4., guard_variant='v2f'))
    assert np.array_equal(res['command'], res2['command'])
    # csv round trip, npz round trip and plot
    br.save_csv_dir(data, tmp_path / 'csv')
    back = br.load_csv_dir(tmp_path / 'csv')
    assert np.allclose(back['ranges'], data['ranges']) and np.allclose(back['prev_speed'], data['prev_speed'])
    np.savez(tmp_path / 'b.npz', **data)
    assert br.load_npz(tmp_path / 'b.npz')['angle_min'] == -2.35
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    br.make_plot(res, data, {'x': np.full(len(res['t']), 1.)}, tmp_path / 'p.png')
    assert (tmp_path / 'p.png').stat().st_size > 1000


@needs_data
def test_cli_runs_on_npz(tmp_path):
    data, _ = fake_bag(rows=40)
    np.savez(tmp_path / 'b.npz', **data)
    assert br.main(['--npz', str(tmp_path / 'b.npz'), '--policy', str(TS), '--out',
                    str(tmp_path / 'o'), '--speed-limit', '2']) == 0
    assert (tmp_path / 'o/summary.json').is_file() and (tmp_path / 'o/replay.csv').is_file()


def test_load_bag_without_rosbags_is_clear_error(tmp_path):
    try:
        import rosbags  # noqa: F401
        pytest.skip('rosbags present')
    except ImportError:
        with pytest.raises(RuntimeError, match='rosbags'):
            br.load_bag(tmp_path)


def test_icp_recovers_translation_on_synthetic_room():
    rng = np.random.default_rng(0)
    # L-shaped, asymmetric room so ICP is observable in both axes
    ang = np.linspace(-2.35, 2.35, 1080)

    def scan(x):
        d = np.stack([np.cos(ang), np.sin(ang)], 1)
        r = np.full(len(ang), 30.)
        for n, off in ((np.array([1., 0.]), 12. - x), (np.array([-1., 0.]), 3. + x),
                       (np.array([0., 1.]), 2.), (np.array([0., -1.]), 1.5)):
            den = d @ n
            with np.errstate(divide='ignore'):
                tt = np.where(den > 1e-6, off / den, 30.)
            r = np.minimum(r, tt)
        return r + rng.normal(0, .003, len(r))
    a = br.scan_xy(scan(0.), -2.35, 4.7 / 1079, 0., 30.)
    b = br.scan_xy(scan(.3), -2.35, 4.7 / 1079, 0., 30.)
    # init 0 sticks in the along-wall local minimum (documented limit); start near the truth
    tx, ty, yaw, rms = br.icp_translation(a, b, init=(0.25, 0., 0.))
    assert tx == pytest.approx(.3, abs=.03) and abs(ty) < .03 and rms < .05


def test_front_wall_speed_on_constant_approach():
    n, v = 60, 2.0
    t = np.arange(n) * .025
    ranges = np.full((n, 1080), 10.)
    ranges -= (v * t)[:, None]
    data = dict(scan_t=t, ranges=ranges, angle_min=-2.35, angle_increment=4.7 / 1079)
    est = br.estimate_speed_front(data)
    assert np.nanmedian(est) == pytest.approx(v, abs=.01)


@needs_data
def test_lidar_icp_error_on_simulator_recording():
    """Quantify the LiDAR ruler against the simulator's true measured speed (median only).

    ICP is not trusted in featureless straights; the median ratio is what the procedure uses.
    """
    data, truth = fake_bag(rows=120, episode_pos=2)
    est, _ = br.estimate_speed_icp(data)
    s = br.speed_ratio_summary(est, truth)
    print('ICP vs true speed (fake data):', s)
    assert s['n'] > 50
    assert 0.75 < s['median_ratio'] < 1.25
