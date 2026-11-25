"""LiDAR + measured-speed following guard for the 68 actor (HJ68-LIDAR-following-guard-v001).

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
    near_range_m=0., near_bearing_max_deg=40.,   # close segments (<near_range_m) must be roughly ahead (side-wall stubs)
    width_min_m=.2, width_max_m=.7, beams_min=5, depth_var_max_m=.65, jump_min_m=.3,
    # memory (stateful mode)
    # motion test (4-frame range rate of the candidate vs ego speed): static-looking candidates are walls/pillars
    creep_floor_mps=0.,   # visible candidates may not command below this while ego < motion_min_speed (motion test impossible; anti live-lock)
    motion_test=False, static_rho=.85, motion_min_speed_mps=1.0, motion_span_frames=3, frame_dt_s=.025,
    dt_s=.025, confirm_steps=10, hold_s=.8, hold_gap_max_m=6.,
    stateless=False,
)


def _candidate(scan, p):
    """Nearest car-like segment (range_m, bearing_rad) or None."""
    n = len(scan)
    d = np.abs(np.diff(scan))
    thr = p['break_abs_m'] + p['break_rel'] * np.minimum(scan[:-1], scan[1:])
    br = np.flatnonzero(d > thr)
    i0 = np.r_[0, br + 1]
    i1 = np.r_[br, n - 1]
    best = None
    bmax = np.deg2rad(p['bearing_max_deg'])
    for a, b in zip(i0, i1):
        if b - a + 1 < p['beams_min']:
            continue
        seg = scan[a:b + 1]
        rm = float(seg.min())
        if rm > p['range_max_m'] or rm >= 29.9 or (best is not None and rm >= best[0]):
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
        jl = scan[a - 1] - rM if a > 0 else -9.
        jr = scan[b + 1] - rM if b < n - 1 else -9.
        if max(jl, jr) <= p['jump_min_m']:
            continue
        best = (rm, float(THETA[j]))
    return best


def _is_static(obs, scan, cand, speed, p):
    """True if the candidate closes at ~ego speed (rho>=static_rho) -> static scene, not a moving car."""
    if speed < p['motion_min_speed_mps']:
        return False
    k = int(p['motion_span_frames'])
    old = np.clip(obs[(3 - k) * 1080:(4 - k) * 1080].astype(np.float64), 0., 30.)
    dt = k * p['frame_dt_s']
    lo = int(np.searchsorted(THETA, cand[1] - np.deg2rad(8.))); hi = int(np.searchsorted(THETA, cand[1] + np.deg2rad(8.)))
    seg = old[lo:hi]
    if seg.size == 0:
        return False
    window = seg[(seg > cand[0] - .3) & (seg < cand[0] + speed * dt + .8)]
    if window.size == 0:
        return False
    closing = (float(window.min()) - cand[0]) / dt
    return closing >= p['static_rho'] * speed * np.cos(cand[1])


def _limit_for_gap(gap, p):
    return float(safe_following_speed(max(0., gap - p['latency_margin_m']), p['leader_speed_assumed_mps'],
                                      braking=p['braking'], reaction=p['reaction'], standstill_gap=p['standstill_gap']))


def new_memory():
    return dict(run=0, gap=None, age_s=0., travel=0.)


def guard_speed(obs, policy_steer, params=None, memory=None):
    p = DEFAULT_PARAMS if params is None else params
    obs = np.asarray(obs)
    if obs.shape != (4321,) or not np.isfinite(obs).all() or not np.isfinite(policy_steer):
        return 0., dict(valid=False, reason='nonfinite_or_bad_shape', active=True)
    scan = np.clip(obs[3240:4320].astype(np.float64), 0., 30.)
    speed = max(0., float(obs[4320]))
    cand = _candidate(scan, p)
    if cand is not None and p['motion_test'] and _is_static(obs, scan, cand, speed, p):
        cand = None
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
                memory.update(gap=gap, age_s=0., travel=0.)
        else:
            memory['run'] = 0
            if memory['gap'] is not None:
                memory['age_s'] += p['dt_s']
                memory['travel'] += speed * p['dt_s']
                if memory['age_s'] <= p['hold_s']:
                    held = _limit_for_gap(max(0., memory['gap'] - memory['travel']), p)
                    if held < limit:
                        limit, source = held, 'held'
                else:
                    memory['gap'] = None
    return limit, dict(valid=True, source=source, gap_m=gap, candidate=cand, active=bool(np.isfinite(limit)))


class GuardedPolicy(HighSpeedLidarPolicy):
    """Wraps the 68 actor; only the normalized speed output may be lowered. Subclasses the actor class only so that
    unchanged evaluators' isinstance/contract checks pass; all state/attributes are delegated to the wrapped actor."""

    def __init__(self, actor, params=None):
        torch.nn.Module.__init__(self)
        self.actor = actor
        self.guard_params = dict(DEFAULT_PARAMS if params is None else params)
        self.guard_memory = new_memory()
        self.guard_log = None

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
        return out

    def forward(self, raw):
        mean = self.action_mean(raw)
        return torch.distributions.Normal(mean, torch.full_like(mean, .1))
