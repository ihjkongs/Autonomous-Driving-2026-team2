"""LiDAR + measured-speed following guard v2 for the 68 actor (HJ68-LIDAR-following-guard-v2; A9 file hj_lidar_following_guard.py is frozen/untouched).
v2 changes ONLY (preregistration_v2.json): (1) held/memory mode keeps a floor and, with held_leader_speed, lets the held gap open at the last
estimated leader speed (no assumed stopped leader); (2) candidates bounded by invalid or max-range beams (faulty/30 m) are not leaders.
GuardedPolicy v2 can record per-step guard state in an internal buffer (trace, default off).

Original A9 docstring:

guard_speed(obs, policy_steer, params, memory=None) -> (speed_limit_mps, debug)
  obs: delivered 4321 vector = 4 chronological 1080-beam scans (newest last) + measured ego speed.
  Only obs and the policy steering are read; no pose/map/opponent information.
  Detection: newest scan is split into range-continuous segments; a segment that looks like a car rear
  (narrow, shallow, isolated by a range jump on at least one side, forward +-bearing window) is a leader candidate.
  Leader speed is assumed 0 (conservative) and safe_following_speed(braking1.5, reaction.8, standstill1.0) is applied.
  memory (optional dict, per episode): confirms a candidate over `confirm_steps` consecutive steps and then holds the last
  confirmed gap (minus ego travel) for `hold_s` seconds when the leader is occluded (corners). memory=None -> stateless.
  The guard never raises speed (GuardedPolicy takes min). Non-finite input -> speed limit 0 (stop).
"""
import numpy as np
import torch
from algorithm.imitation.highspeed_lidar import HighSpeedLidarPolicy
from utils.traffic_expert import safe_following_speed

BEAMS = 1080
FOV = 4.7
THETA = -2.35 + np.arange(BEAMS) * FOV / (BEAMS - 1)
COS, SIN = np.cos(THETA), np.sin(THETA)

DEFAULT_PARAMS = dict(
    # safe_following_speed (oracle values)
    braking=1.5, reaction=.8, standstill_gap=1.,
    sensor_to_bumper_m=.13,          # ego front bumper is 0.13 m ahead of the lidar (lidar_pos_offset .155, half length .285)
    leader_speed_assumed_mps=0.,     # no leader-speed estimate in v001 (conservative)
    latency_margin_m=0.,
    # segment (car-rear) candidate rules, newest frame only
    break_abs_m=.25, break_rel=.03,  # range jump between neighbours that separates segments
    range_max_m=12., bearing_max_deg=80.,
    near_range_m=1.5, near_bearing_max_deg=40.,   # close segments (<near_range_m) must be roughly ahead (side-wall stubs)
    width_min_m=.2, width_max_m=.7, beams_min=5, depth_var_max_m=.65, jump_min_m=.3,
    # memory (stateful mode)
    # motion test (4-frame range rate of the candidate vs ego speed): static-looking candidates are walls/pillars
    creep_floor_mps=.8,   # visible candidates may not command below this while ego < motion_min_speed (motion test impossible; anti live-lock)
    motion_test=True, static_rho=.7, motion_min_speed_mps=1.0, motion_span_frames=3, frame_dt_s=.025,
    dt_s=.025, confirm_steps=10, hold_s=1.5, hold_gap_max_m=6.,
    stateless=False,
    # ---- v2 additions ----
    hold_floor_mps=.8,                 # held mode floor (A9: held limit could reach 0)
    hold_floor_mode='gap_clamp',       # 'gap_clamp' (held gap decay stops at the gap whose safe speed is the floor), 'speed' (limit floor), 'off'
    held_leader_speed=False,           # held gap opens at the last estimated leader speed (release when gap opens)
    exclude_invalid_neighbors=True,    # segment next to a <=0.05 m / >=29.9 m / non-finite beam is not a leader candidate
    invalid_low_m=.14, invalid_high_m=29.9,   # .14 m: a return closer than the half body width at the lidar is physically impossible (stuck-near fault)
    held_slow_floor=True,              # a track confirmed while the motion test is impossible (ego<motion_min_speed) keeps creep_floor in held mode
)


def _candidate(scan, p):
    """(nearest car-like segment (range_m, bearing_rad) or None, number of qualifying segments)."""
    n = len(scan)
    d = np.abs(np.diff(scan))
    thr = p['break_abs_m'] + p['break_rel'] * np.minimum(scan[:-1], scan[1:])
    br = np.flatnonzero(d > thr)
    i0 = np.r_[0, br + 1]
    i1 = np.r_[br, n - 1]
    best = None
    count = 0
    bmax = np.deg2rad(p['bearing_max_deg'])
    for a, b in zip(i0, i1):
        if b - a + 1 < p['beams_min']:
            continue
        seg = scan[a:b + 1]
        rm = float(seg.min())
        if rm > p['range_max_m'] or rm >= 29.9:
            continue
        j = a + int(np.argmin(seg))
        if abs(THETA[j]) > bmax or (rm < p['near_range_m'] and abs(THETA[j]) > np.deg2rad(p['near_bearing_max_deg'])):
            continue
        rM = float(seg.max())
        if rM - rm > p['depth_var_max_m']:
            continue
        width = float(np.hypot(seg[-1] * COS[b] - seg[0] * COS[a], seg[-1] * SIN[b] - seg[0] * SIN[a]))
        if not p['width_min_m'] <= width <= p['width_max_m']:
            continue
        if p['exclude_invalid_neighbors'] and (
                (a > 0 and not (p['invalid_low_m'] < scan[a - 1] < p['invalid_high_m'])) or
                (b < n - 1 and not (p['invalid_low_m'] < scan[b + 1] < p['invalid_high_m']))):
            continue
        jl = scan[a - 1] - rM if a > 0 else -9.
        jr = scan[b + 1] - rM if b < n - 1 else -9.
        if max(jl, jr) <= p['jump_min_m']:
            continue
        count += 1
        if best is None or rm < best[0]:
            best = (rm, float(THETA[j]))
    return best, count


def _closing(obs, cand, speed, p):
    """Range-rate closing speed of the candidate over motion_span_frames (None if not measurable)."""
    if speed < p['motion_min_speed_mps']:
        return None
    k = int(p['motion_span_frames'])
    old = np.clip(obs[(3 - k) * 1080:(4 - k) * 1080].astype(np.float64), 0., 30.)
    dt = k * p['frame_dt_s']
    lo = int(np.searchsorted(THETA, cand[1] - np.deg2rad(8.))); hi = int(np.searchsorted(THETA, cand[1] + np.deg2rad(8.)))
    seg = old[lo:hi]
    if seg.size == 0:
        return None
    window = seg[(seg > cand[0] - .3) & (seg < cand[0] + speed * dt + .8)]
    if window.size == 0:
        return None
    return (float(window.min()) - cand[0]) / dt


def _is_static(obs, scan, cand, speed, p):
    """True if the candidate closes at ~ego speed (rho>=static_rho) -> static scene, not a moving car."""
    closing = _closing(obs, cand, speed, p)
    return closing is not None and closing >= p['static_rho'] * speed * np.cos(cand[1])


def _limit_for_gap(gap, p):
    return float(safe_following_speed(max(0., gap - p['latency_margin_m']), p['leader_speed_assumed_mps'],
                                      braking=p['braking'], reaction=p['reaction'], standstill_gap=p['standstill_gap']))


def _gap_for_speed(speed, p):
    """Inverse of safe_following_speed for the assumed leader speed: gap whose safe speed equals `speed`."""
    b, r, g, v0 = p['braking'], p['reaction'], p['standstill_gap'], p['leader_speed_assumed_mps']
    usable = ((speed + b * r) ** 2 - (b * r) ** 2 - v0 ** 2) / (2. * b)
    return g + max(0., usable) + p['latency_margin_m']


def new_memory():
    return dict(run=0, gap=None, age_s=0., travel=0., leader_speed=0., slow=False)


def guard_speed(obs, policy_steer, params=None, memory=None):
    p = DEFAULT_PARAMS if params is None else params
    obs = np.asarray(obs)
    if obs.shape != (4321,) or not np.isfinite(obs).all() or not np.isfinite(policy_steer):
        return 0., dict(valid=False, reason='nonfinite_or_bad_shape', active=True)
    scan = np.clip(obs[3240:4320].astype(np.float64), 0., 30.)
    speed = max(0., float(obs[4320]))
    cand, ncand = _candidate(scan, p)
    leader_est = 0.
    if cand is not None and p['motion_test']:
        closing = _closing(obs, cand, speed, p)
        if closing is not None:
            if closing >= p['static_rho'] * speed * np.cos(cand[1]):
                cand = None
            else:
                leader_est = float(np.clip(speed * np.cos(cand[1]) - closing, 0., 8.))
    limit, source, gap = np.inf, None, None
    if cand is not None:
        gap = cand[0] - p['sensor_to_bumper_m']
        limit, source = _limit_for_gap(gap, p), 'visible'
        if speed < p['motion_min_speed_mps']:
            limit = max(limit, p['creep_floor_mps'])
    if memory is not None and not p['stateless']:
        if cand is not None:
            memory['run'] += 1
            if memory['run'] >= p['confirm_steps'] and gap <= p['hold_gap_max_m']:
                memory.update(gap=gap, age_s=0., travel=0., leader_speed=leader_est if p['held_leader_speed'] else 0.,
                              slow=speed < p['motion_min_speed_mps'])
        else:
            memory['run'] = 0
            if memory['gap'] is not None:
                memory['age_s'] += p['dt_s']
                memory['travel'] += (speed - memory['leader_speed']) * p['dt_s']
                if memory['age_s'] <= p['hold_s']:
                    held_gap = max(0., memory['gap'] - memory['travel'])
                    if p['hold_floor_mode'] == 'gap_clamp':
                        # decay by ego travel may not push the held gap below the gap whose safe speed equals hold_floor_mps,
                        # but a leader last seen closer than that keeps its own (smaller) gap -> protection of close leaders unchanged
                        held_gap = max(held_gap, min(memory['gap'], _gap_for_speed(p['hold_floor_mps'], p)))
                    held = _limit_for_gap(held_gap, p)
                    if p['held_slow_floor'] and memory['slow']:
                        held = max(held, p['creep_floor_mps'])
                    if p['hold_floor_mode'] == 'speed':
                        held = max(held, p['hold_floor_mps'])
                    if held < limit:
                        limit, source = held, 'held'
                else:
                    memory['gap'] = None
    return limit, dict(valid=True, source=source, gap_m=gap, candidate=cand, n_candidates=ncand, active=bool(np.isfinite(limit)))


class GuardedPolicy(HighSpeedLidarPolicy):
    """Wraps the 68 actor; only the normalized speed output may be lowered. Subclasses the actor class only so that
    unchanged evaluators' isinstance/contract checks pass; all state/attributes are delegated to the wrapped actor."""

    def __init__(self, actor, params=None):
        torch.nn.Module.__init__(self)
        self.actor = actor
        self.guard_params = dict(DEFAULT_PARAMS if params is None else params)
        self.guard_memory = new_memory()
        self.guard_log = None
        self.trace = None      # optional list of per-step records [limit, unlimited_norm_speed, mode(0 none,1 visible,2 held), gap_m, n_candidates, speed_mps]

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'actor':
                raise
            return getattr(self.actor, name)

    def reset_episode(self):
        self.guard_memory = new_memory()
        if hasattr(self.actor, 'reset_episode'):
            self.actor.reset_episode()

    def action_mean(self, raw):
        mean = self.actor.action_mean(raw)
        if raw.dim() != 1:                       # batched calls: stateless, per row
            rows = [self._limit_row(raw[i].numpy(), mean[i], None) for i in range(raw.shape[0])]
            return torch.stack(rows)
        return self._limit_row(raw.detach().numpy(), mean, self.guard_memory)

    def _limit_row(self, obs, mean, memory):
        a = self.actor
        limit, debug = guard_speed(obs, float(mean[0]) * .48, self.guard_params, memory)
        out = mean.clone()
        lo, hi = float(getattr(a, 'min_speed', 0.)), float(a.max_speed)
        norm = 2. * (min(max(limit, lo), hi) - lo) / (hi - lo) - 1. if np.isfinite(limit) else 1.
        out[1] = min(float(out[1]), norm)        # never raises
        if self.guard_log is not None:
            self.guard_log.append((float(limit), float(mean[1]), debug.get('source')))
        if self.trace is not None:
            self.trace.append([float(limit), float(mean[1]), {None: 0, 'visible': 1, 'held': 2}[debug.get('source')],
                               float('nan') if debug.get('gap_m') is None else float(debug['gap_m']), float(debug.get('n_candidates', -1)), float(obs[4320])])
        return out

    def forward(self, raw):
        mean = self.action_mean(raw)
        return torch.distributions.Normal(mean, torch.full_like(mean, .1))
