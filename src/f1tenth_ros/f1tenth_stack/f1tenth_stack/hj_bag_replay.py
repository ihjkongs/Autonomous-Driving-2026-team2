#!/usr/bin/env python3
"""Offline replay of a recorded drive through the system-71 path (no ROS needed).

Input  : rosbag2 directory (needs the pure-python ``rosbags`` package; untested here) or a
         converted ``.npz`` / csv directory (see ``load_npz`` / ``load_csv_dir``).
Replay : /scan + /odom -> ScanSpeedAssembler -> TorchScript policy -> speed_limit -> guard
         -> SpeedLimiter, exactly ``hj_speed_policy_runtime.SpeedAgentCore``.
Compare: recomputed steer / policy speed / guard-limited target / command against what the
         bag recorded (/agent/prediction, /agent/command_preview, /drive).
Extra  : LiDAR-only forward-speed estimate (front-wall range rate and 2D ICP, numpy only)
         compared with the /odom speed. This is a rough ruler, NOT ground truth: see
         ``estimate_speed_icp`` for the limits.

npz keys: scan_t(N) ranges(N,M) angle_min angle_increment range_min range_max
          odom_t(K) odom_vx(K)
          optional pred_t pred_steer pred_speed | prev_t prev_steer prev_speed |
                   drive_t drive_steer drive_speed
csv dir : scan.csv (t, r0..rM-1) + scan_meta.json (angle_min, angle_increment, range_min,
          range_max), odom.csv (t,vx), optional prediction.csv / preview.csv / drive.csv
          (t,steer,speed).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TOPICS = dict(scan='/scan', odom='/odom', pred='/agent/prediction',
              prev='/agent/command_preview', drive='/drive')


# ----------------------------------------------------------------------------- loaders
def _stamp(msg, fallback):
    h = getattr(msg, 'header', None)
    if h is not None and (h.stamp.sec or h.stamp.nanosec):
        return h.stamp.sec + h.stamp.nanosec * 1e-9
    return fallback * 1e-9


def load_bag(path, topics=None):
    """Read a rosbag2 (sqlite3/mcap) with ``rosbags``. NOT tested on a real bag (확인 필요)."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError('rosbags package missing (pip install rosbags); or convert the bag '
                           'to npz/csv and use --npz/--csv-dir') from exc
    topics = dict(TOPICS, **(topics or {}))
    out = {k: [] for k in ('scan', 'odom', 'pred', 'prev', 'drive')}
    meta = None
    with AnyReader([Path(path)]) as reader:
        conns = [c for c in reader.connections if c.topic in topics.values()]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = _stamp(msg, ts)
            if conn.topic == topics['scan']:
                meta = (msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
                out['scan'].append((t, np.asarray(msg.ranges, dtype=np.float64)))
            elif conn.topic == topics['odom']:
                out['odom'].append((t, msg.twist.twist.linear.x))
            else:
                key = [k for k in ('pred', 'prev', 'drive') if topics[k] == conn.topic][0]
                out[key].append((t, msg.drive.steering_angle, msg.drive.speed))
    if not out['scan'] or not out['odom'] or meta is None:
        raise RuntimeError('bag has no %s or %s messages' % (topics['scan'], topics['odom']))
    out['scan'].sort(key=lambda r: r[0])
    out['odom'].sort(key=lambda r: r[0])
    data = dict(scan_t=np.array([r[0] for r in out['scan']]),
                ranges=np.stack([r[1] for r in out['scan']]),
                angle_min=meta[0], angle_increment=meta[1], range_min=meta[2],
                range_max=meta[3],
                odom_t=np.array([r[0] for r in out['odom']]),
                odom_vx=np.array([r[1] for r in out['odom']]))
    for key, name in (('pred', 'pred'), ('prev', 'prev'), ('drive', 'drive')):
        if out[key]:
            rows = sorted(out[key])
            data[name + '_t'] = np.array([r[0] for r in rows])
            data[name + '_steer'] = np.array([r[1] for r in rows])
            data[name + '_speed'] = np.array([r[2] for r in rows])
    return data


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files}
    for k in ('angle_min', 'angle_increment', 'range_min', 'range_max'):
        data[k] = float(data[k])
    return data


def load_csv_dir(path):
    path = Path(path)
    meta = json.loads((path / 'scan_meta.json').read_text())
    scan = np.loadtxt(path / 'scan.csv', delimiter=',', ndmin=2)
    odom = np.loadtxt(path / 'odom.csv', delimiter=',', ndmin=2)
    data = dict(scan_t=scan[:, 0], ranges=scan[:, 1:], odom_t=odom[:, 0], odom_vx=odom[:, 1],
                **{k: float(meta[k]) for k in
                   ('angle_min', 'angle_increment', 'range_min', 'range_max')})
    for name, fn in (('pred', 'prediction.csv'), ('prev', 'preview.csv'), ('drive', 'drive.csv')):
        if (path / fn).is_file():
            a = np.loadtxt(path / fn, delimiter=',', ndmin=2)
            data[name + '_t'], data[name + '_steer'], data[name + '_speed'] = a[:, 0], a[:, 1], a[:, 2]
    return data


def save_csv_dir(data, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savetxt(path / 'scan.csv', np.column_stack([data['scan_t'], data['ranges']]),
               delimiter=',', fmt='%.9g')
    np.savetxt(path / 'odom.csv', np.column_stack([data['odom_t'], data['odom_vx']]),
               delimiter=',', fmt='%.9g')
    (path / 'scan_meta.json').write_text(json.dumps(
        {k: float(data[k]) for k in ('angle_min', 'angle_increment', 'range_min', 'range_max')}))
    for name, fn in (('pred', 'prediction.csv'), ('prev', 'preview.csv'), ('drive', 'drive.csv')):
        if name + '_t' in data:
            np.savetxt(path / fn, np.column_stack(
                [data[name + '_t'], data[name + '_steer'], data[name + '_speed']]),
                delimiter=',', fmt='%.9g')


# ----------------------------------------------------------------------------- replay
def build_core(policy_file, speed_limit=1.0, guard_variant='v2f', max_acceleration=2.0,
               max_deceleration=3.0, scan_timeout=0.2, **assembler_kw):
    """Same construction as hj_speed_agent_node (guard_variant '' = no guard)."""
    from f1tenth_stack import hj_speed_policy_runtime as rt
    guard = rt.no_guard
    if guard_variant:
        cfg = Path(__file__).resolve().parents[1] / 'config'
        gf, gs, pf, ps = rt.resolve_guard_paths(guard_variant, '', '', '', '', lambda: cfg)
        guard = rt.load_guard_hook(gf, gs, pf, ps)
    assembler = rt.ScanSpeedAssembler(4, scan_timeout, **assembler_kw)
    return rt.SpeedAgentCore(assembler, rt.SpeedPolicy(policy_file, 'cpu'), speed_limit,
                             max_acceleration, max_deceleration, scan_timeout, guard=guard)


def replay(data, core):
    """Feed scans/odom in time order. Returns dict of per-scan arrays (NaN where rejected).

    Mirrors the node: each odom is delivered before the first scan with a later stamp; the
    stamp time base is used as the monotonic clock (receipt time is not in a bag).
    """
    st, ot = np.asarray(data['scan_t']), np.asarray(data['odom_t'])
    n = len(st)
    out = {k: np.full(n, np.nan) for k in ('steer', 'policy_speed', 'target', 'command',
                                          'guard_limit', 'odom_v')}
    rejected = []
    j = 0
    guard = core.guard
    for i in range(n):
        while j < len(ot) and ot[j] <= st[i]:
            core.assembler.on_odom(float(data['odom_vx'][j]), float(ot[j]), float(ot[j]))
            j += 1
        try:
            if hasattr(guard, 'last'):
                guard.last = None
            steer, speed, target, command = core.step(
                data['ranges'][i], data['angle_min'], data['angle_increment'],
                data['range_min'], data['range_max'], float(st[i]), float(st[i]))
        except Exception as exc:  # Rejected etc: node would command stop
            rejected.append((i, str(exc)))
            continue
        out['steer'][i], out['policy_speed'][i] = steer, speed
        out['target'][i], out['command'][i] = target, command
        out['odom_v'][i] = core.assembler.odom[0]
        last = getattr(guard, 'last', None)
        if last is not None:
            out['guard_limit'][i] = float(last[0])
    out['t'] = st
    out['rejected'] = rejected
    return out


def _nearest(t_src, t_ref, tol):
    idx = np.clip(np.searchsorted(t_ref, t_src), 1, len(t_ref) - 1)
    left_closer = np.abs(t_src - t_ref[idx - 1]) < np.abs(t_src - t_ref[idx])
    idx = np.where(left_closer, idx - 1, idx)
    ok = np.abs(t_ref[idx] - t_src) <= tol
    return idx, ok


def compare(res, data, tol=0.02):
    """Per recorded stream: recomputed vs recorded error statistics."""
    summary = {}
    pairs = (('pred', 'steer', 'policy_speed'), ('prev', 'steer', 'command'),
             ('drive', 'steer', 'command'))
    for name, ks, kv in pairs:
        if name + '_t' not in data:
            continue
        idx, ok = _nearest(res['t'], np.asarray(data[name + '_t']), tol)
        ok &= np.isfinite(res[ks])
        if not ok.any():
            summary[name] = dict(n=0)
            continue
        ds = res[ks][ok] - data[name + '_steer'][idx[ok]]
        dv = res[kv][ok] - data[name + '_speed'][idx[ok]]
        summary[name] = dict(
            n=int(ok.sum()), steer_mae=float(np.abs(ds).mean()), steer_max=float(np.abs(ds).max()),
            speed_mae=float(np.abs(dv).mean()), speed_max=float(np.abs(dv).max()),
            speed_mean_diff=float(dv.mean()))
    fin = np.isfinite(res['guard_limit'])
    summary['guard'] = dict(
        n_scans=int(len(res['t'])), n_rejected=len(res['rejected']),
        guard_active_frac=float(np.mean(res['target'][np.isfinite(res['target'])] <
                                        res['policy_speed'][np.isfinite(res['target'])] - 1e-9))
        if np.isfinite(res['target']).any() else None,
        guard_limit_min=float(np.nanmin(res['guard_limit'])) if fin.any() else None)
    return summary


# ----------------------------------------------------------------------------- LiDAR ruler
def scan_xy(ranges, angle_min, inc, range_min, range_max, step=6, margin=0.3):
    r = np.asarray(ranges, dtype=np.float64)
    ang = angle_min + inc * np.arange(len(r))
    keep = np.isfinite(r) & (r > max(range_min, 0.1)) & (r < range_max - margin)
    keep[np.arange(len(r)) % step != 0] = False
    return np.column_stack([r[keep] * np.cos(ang[keep]), r[keep] * np.sin(ang[keep])])


def icp_translation(pa, pb, init=(0., 0., 0.), iters=12, trim=0.8):
    """Rigid 2D point-to-point ICP aligning pb (later scan) onto pa (earlier scan).

    Returns (tx, ty, yaw, rms). (tx, ty) is where the later sensor origin sits in the
    earlier sensor frame, i.e. the vehicle displacement (x forward).
    """
    tx, ty, yaw = init
    if len(pa) < 10 or len(pb) < 10:
        return np.nan, np.nan, np.nan, np.nan
    rms = np.nan
    for _ in range(iters):
        c, s = np.cos(yaw), np.sin(yaw)
        q = pb @ np.array([[c, s], [-s, c]]) + np.array([tx, ty])
        d2 = ((q[:, None, :] - pa[None, :, :]) ** 2).sum(-1)
        nn = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(q)), nn])
        keep = dist <= np.quantile(dist, trim)
        if keep.sum() < 8:
            return np.nan, np.nan, np.nan, np.nan
        src, dst = pb[keep], pa[nn[keep]]
        mu_s, mu_d = src.mean(0), dst.mean(0)
        h = (src - mu_s).T @ (dst - mu_d)
        u, _, vt = np.linalg.svd(h)
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0:
            vt[-1] *= -1
            rot = vt.T @ u.T
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        t = mu_d - rot @ mu_s
        tx, ty = float(t[0]), float(t[1])
        rms = float(np.sqrt((dist[keep] ** 2).mean()))
    return tx, ty, yaw, rms


def estimate_speed_icp(data, lag=4, max_rms=0.15):
    """Forward speed from ICP between scan i-lag and scan i: v = tx / dt.

    LIMITS: point-to-point ICP is unobservable along featureless straight walls (corridors):
    the along-track translation then collapses toward the initial guess (here the previous
    estimate), so speed is biased low/stale on long straights, which is exactly where a
    LiDAR ruler is most wanted. Ranges of the estimate are only trustworthy where rms is
    small and both side walls AND a front/back structure exist. Use as a rough cross-check
    of /odom (mean ratio, trend), not as ground truth. Output NaN where rms > max_rms.
    """
    st = np.asarray(data['scan_t'])
    n = len(st)
    v = np.full(n, np.nan)
    rms = np.full(n, np.nan)
    pts = [scan_xy(data['ranges'][i], data['angle_min'], data['angle_increment'],
                   data['range_min'], data['range_max']) for i in range(n)]
    prev_v = 0.
    for i in range(lag, n):
        dt = st[i] - st[i - lag]
        if dt <= 0 or dt > 0.5:
            continue
        # multi-start along x (+ previous estimate); lowest rms wins, ties go to the prior
        best = None
        for g in (0., prev_v * dt, 0.2, 0.4, 0.6):
            res = icp_translation(pts[i - lag], pts[i], init=(g, 0., 0.))
            if np.isfinite(res[3]):
                key = res[3] + 0.02 * abs(res[0] - prev_v * dt)
                if best is None or key < best[0]:
                    best = (key, res)
        if best is None:
            continue
        tx, ty, yaw, r = best[1]
        if np.isfinite(tx) and r <= max_rms:
            v[i], rms[i] = tx / dt, r
            prev_v = v[i]
    return v, rms


def estimate_speed_front(data, half_width=4, window_s=0.4, min_range=0.8, max_jump=0.5):
    """Front-wall method: median range of the beams around 0 rad, v = -d(range)/dt from a
    sliding least-squares line fit over window_s. Valid only when driving straight at a wall
    (range decreasing smoothly); NaN where the fit residual is large or range < min_range."""
    st = np.asarray(data['scan_t'])
    mid = int(round((0. - data['angle_min']) / data['angle_increment']))
    sec = np.asarray(data['ranges'])[:, mid - half_width:mid + half_width + 1]
    with np.errstate(invalid='ignore'):
        d = np.where(np.isfinite(sec), sec, np.nan)
        d = np.nanmedian(d, axis=1)
    v = np.full(len(st), np.nan)
    for i in range(len(st)):
        sel = np.flatnonzero((st > st[i] - window_s) & (st <= st[i]))
        if len(sel) < 5:
            continue
        x, y = st[sel] - st[i], d[sel]
        ok = np.isfinite(y) & (y > min_range)
        if ok.sum() < 5 or np.abs(np.diff(y[ok])).max() > max_jump:
            continue
        a, b = np.polyfit(x[ok], y[ok], 1)
        resid = np.abs(y[ok] - (a * x[ok] + b)).max()
        if resid < 0.03:
            v[i] = -a
    return v


def speed_ratio_summary(est, odom_v, min_speed=0.5):
    ok = np.isfinite(est) & np.isfinite(odom_v) & (odom_v > min_speed)
    if not ok.any():
        return dict(n=0)
    ratio = est[ok] / odom_v[ok]
    return dict(n=int(ok.sum()), mean_ratio_est_over_odom=float(np.mean(ratio)),
                median_ratio=float(np.median(ratio)), p10_ratio=float(np.quantile(ratio, .1)),
                p90_ratio=float(np.quantile(ratio, .9)),
                mean_abs_err_mps=float(np.mean(np.abs(est[ok] - odom_v[ok]))),
                gain_fit_est_per_odom=float(np.sum(est[ok] * odom_v[ok]) / np.sum(odom_v[ok] ** 2)))


# ----------------------------------------------------------------------------- plot / CLI
def make_plot(res, data, speed_est, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = res['t'] - res['t'][0]
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(t, res['steer'], label='replay steer')
    if 'prev_t' in data:
        ax[0].plot(np.asarray(data['prev_t']) - res['t'][0], data['prev_steer'], '--',
                   label='recorded preview')
    elif 'drive_t' in data:
        ax[0].plot(np.asarray(data['drive_t']) - res['t'][0], data['drive_steer'], '--',
                   label='recorded drive')
    ax[0].set_ylabel('steer [rad]')
    ax[1].plot(t, res['policy_speed'], label='policy speed')
    ax[1].plot(t, res['target'], label='after speed_limit+guard')
    ax[1].plot(t, res['command'], label='replay command')
    if 'prev_t' in data:
        ax[1].plot(np.asarray(data['prev_t']) - res['t'][0], data['prev_speed'], '--',
                   label='recorded preview')
    if 'drive_t' in data:
        ax[1].plot(np.asarray(data['drive_t']) - res['t'][0], data['drive_speed'], ':',
                   label='recorded drive')
    ax[1].set_ylabel('speed [m/s]')
    ax[2].plot(t, res['guard_limit'], label='guard limit')
    ax[2].plot(t, res['policy_speed'], label='policy speed', alpha=.6)
    ax[2].set_ylabel('guard [m/s]')
    ax[3].plot(t, res['odom_v'], label='/odom v')
    for name, est in speed_est.items():
        ax[3].plot(t, est, label=name, alpha=.8)
    ax[3].set_ylabel('measured speed [m/s]')
    ax[3].set_xlabel('time [s]')
    for a in ax:
        a.grid(alpha=.3)
        a.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--bag')
    src.add_argument('--npz')
    src.add_argument('--csv-dir')
    p.add_argument('--policy', required=True, help='driving_policy.ts of system 71 (BC68)')
    p.add_argument('--speed-limit', type=float, default=1.0)
    p.add_argument('--guard', default='v2f', help="'v2f', 'a9' or '' for no guard")
    p.add_argument('--out', default='bag_replay_out')
    p.add_argument('--lidar-speed', action='store_true', help='also run ICP/front-wall ruler')
    p.add_argument('--export-npz', help='with --bag: save converted data and continue')
    a = p.parse_args(argv)
    data = (load_bag(a.bag) if a.bag else load_npz(a.npz) if a.npz else load_csv_dir(a.csv_dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.export_npz:
        np.savez_compressed(a.export_npz, **data)
    core = build_core(a.policy, a.speed_limit, a.guard)
    res = replay(data, core)
    summary = compare(res, data)
    est = {}
    if a.lidar_speed:
        est['LiDAR ICP'], _ = estimate_speed_icp(data)
        est['LiDAR front wall'] = estimate_speed_front(data)
        summary['lidar_vs_odom'] = {k: speed_ratio_summary(v, res['odom_v']) for k, v in est.items()}
    summary['rejected_first5'] = res['rejected'][:5]
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savetxt(out / 'replay.csv', np.column_stack(
        [res['t'], res['steer'], res['policy_speed'], res['target'], res['command'],
         res['guard_limit'], res['odom_v']] + list(est.values())), delimiter=',',
        header='t,steer,policy_speed,target,command,guard_limit,odom_v,' + ','.join(
            k.replace(' ', '_') for k in est), comments='')
    try:
        make_plot(res, data, est, out / 'replay.png')
    except ImportError:
        print('matplotlib missing: no plot')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
