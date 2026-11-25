"""ROS-independent logic for the scan+odom-speed (4321 input) policy.

Input = 4 chronological resampled 1080-ray scans (oldest->newest) + current
measured forward speed in m/s (last scalar), as in the simulator observation.
Any missing/stale/non-finite input raises ``Rejected``; the caller must stop.
"""
from collections import deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from f1tenth_stack.policy_input import scan_to_observation
from f1tenth_stack.speed_limiter import SpeedLimiter

RAYS = 1080


class Rejected(ValueError):
    """Input unusable this cycle; command zero speed."""


def decode_physical(action, min_speed, max_speed, max_steer, stop_deadband):
    """Same formula as utils.policy_contract.PolicyContract.decode (native -> physical)."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape[-1:] != (2,) or not np.isfinite(action).all():
        raise ValueError('Finite steering/speed pair required')
    action = np.clip(action, -1., 1.)
    speed = (action[..., 1]+1.)*.5*(max_speed-min_speed)+min_speed
    if stop_deadband:
        speed = np.where(speed <= stop_deadband, 0., speed)
    return np.stack([action[..., 0]*max_steer, speed], axis=-1)


SPEED_SOURCES = ('odom', 'vesc_core')


def erpm_to_speed(erpm, gain, offset=0., deadband=0.05):
    """Measured motor ERPM -> speed: v = (erpm - offset)/gain, offset default 0 (ERPM 0 = standstill).

    ``offset`` is the measured-ERPM intercept (core_speed_offset_erpm), NOT the command offset
    speed_to_erpm_offset of ackermann_to_vesc; only a nonzero regression intercept justifies it.
    vesc_to_odom uses (erpm + command_offset)/gain, which reads -0.13 at rest for offset -325.
    |v| < deadband -> 0 (vesc_to_odom uses 0.05). Raises Rejected on non-finite input.
    The signed result still goes through odom_speed() for the reverse/max checks.
    """
    if not (math.isfinite(gain) and gain > 0 and math.isfinite(offset) and
            math.isfinite(deadband) and deadband >= 0):
        raise ValueError('gain must be finite and positive; offset/deadband finite')
    if not math.isfinite(erpm):
        raise Rejected('vesc erpm not finite')
    v = (erpm - offset) / gain
    return 0. if abs(v) < deadband else v


def odom_speed(vx, reverse_tolerance, max_speed_input):
    """Measured speed fed to the policy.

    The simulator feeds hypot(vx, vy) >= 0. vesc_to_odom gives signed
    linear.x and already zeroes |v| < 0.05. Small negatives (|v|<=tol) are
    noise -> abs(); a larger negative value means reverse or a wrong sign
    convention -> reject (stop). Values above max_speed_input -> reject.
    """
    if not math.isfinite(vx):
        raise Rejected('odom speed not finite')
    if vx < -reverse_tolerance:
        raise Rejected('odom speed %.3f is negative beyond tolerance (reverse/sign?)' % vx)
    speed = abs(vx)
    if speed > max_speed_input:
        raise Rejected('odom speed %.3f above plausible maximum %.3f' % (speed, max_speed_input))
    return speed


class ScanSpeedAssembler:
    """Keeps 4-frame scan history and the latest odometry; builds the 4321 vector."""

    def __init__(self, frames=4, scan_timeout=.2, odom_max_skew=.1, odom_timeout=.25,
                 reverse_tolerance=.1, max_speed_input=8., laser_yaw_offset=0.):
        values = (scan_timeout, odom_max_skew, odom_timeout, reverse_tolerance, max_speed_input)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('timeouts/tolerances must be finite and positive')
        self.frames, self.scan_timeout = int(frames), scan_timeout
        self.odom_max_skew, self.odom_timeout = odom_max_skew, odom_timeout
        self.reverse_tolerance, self.max_speed_input = reverse_tolerance, max_speed_input
        self.laser_yaw_offset = laser_yaw_offset
        self.history = deque(maxlen=self.frames)
        self.last_mono = None
        self.odom = None  # (vx, odom_time, odom_mono)
        self.fresh_history = True  # last assembled vector started a new history

    def reset(self):
        self.history.clear()
        self.last_mono = None

    def on_odom(self, vx, odom_time, mono):
        """odom_time shares the time base of scan_time passed to assemble()."""
        self.odom = (float(vx), float(odom_time), float(mono))

    def odom_fresh(self, mono):
        return self.odom is not None and 0. <= mono-self.odom[2] <= self.odom_timeout

    def assemble(self, ranges, angle_min, angle_increment, range_min, range_max,
                 scan_time, mono):
        if not math.isfinite(scan_time) or not math.isfinite(mono):
            raise Rejected('non-finite time')
        if self.odom is None:
            raise Rejected('no odom received')
        vx, odom_time, odom_mono = self.odom
        if not math.isfinite(odom_time):
            raise Rejected('odom time not finite')
        if mono-odom_mono > self.odom_timeout or mono < odom_mono:
            raise Rejected('odom stale (%.3fs since receipt)' % (mono-odom_mono))
        if abs(scan_time-odom_time) > self.odom_max_skew:
            raise Rejected('scan-odom stamp skew %.3fs > %.3fs' % (scan_time-odom_time, self.odom_max_skew))
        speed = odom_speed(vx, self.reverse_tolerance, self.max_speed_input)
        try:
            scan = scan_to_observation(ranges, angle_min, angle_increment, range_min, range_max,
                                       self.laser_yaw_offset)
        except ValueError as exc:
            self.reset()
            raise Rejected('scan invalid: %s' % exc)
        if scan.shape != (RAYS,) or not np.isfinite(scan).all():
            self.reset()
            raise Rejected('scan invalid')
        # Same history rules as StandalonePolicy.predict.
        if self.last_mono is not None and (mono <= self.last_mono or mono-self.last_mono > self.scan_timeout):
            self.reset()
        self.fresh_history = not self.history
        if not self.history:
            for _ in range(self.frames):
                self.history.append(scan.copy())
        else:
            self.history.append(scan.copy())
        self.last_mono = mono
        return np.concatenate(list(self.history)+[np.array([speed], dtype=np.float32)]).astype(np.float32)


class SpeedPolicy:
    """TorchScript scan+speed policy (physical output [steer rad, speed m/s])."""

    def __init__(self, path, device='cpu'):
        import torch
        self.torch = torch
        extra = {'metadata.json': ''}
        self.module = torch.jit.load(str(Path(path).expanduser()), map_location=device,
                                     _extra_files=extra).eval()
        self.metadata = md = json.loads(extra['metadata.json'])
        if (md.get('format') != 'f1tenth-physical-lidar-v1' or md.get('lidar_frames') != 4 or
                md.get('rays_per_frame') != RAYS or md.get('obs_dim') != 4*RAYS+1 or
                md.get('speed_input') is not True or md.get('fov_rad') != 4.7 or
                md.get('max_range_m') != 30.):
            raise ValueError('Not a 4-frame scan+speed (4321) policy contract')
        limits = [md[k] for k in ('max_steer', 'min_speed', 'max_speed')]
        if not np.isfinite(limits).all() or limits[0] <= 0 or limits[1] < 0 or limits[2] < limits[1]:
            raise ValueError('Invalid output limits')
        self.device = device
        for _ in range(3):  # finish TorchScript warm-up before time-limited callbacks
            self.predict(np.zeros(4*RAYS+1, dtype=np.float32))

    def predict(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (4*RAYS+1,) or not np.isfinite(obs).all():
            raise Rejected('invalid policy input')
        with self.torch.inference_mode():
            out = self.module(self.torch.as_tensor(obs, device=self.device).unsqueeze(0))
            action = out.cpu().numpy().reshape(-1)
        md = self.metadata
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Policy returned invalid actions')
        if (abs(action[0]) > md['max_steer']+1e-5 or action[1] < md['min_speed']-1e-5 or
                action[1] > md['max_speed']+1e-5):
            raise ValueError('Policy action violates embedded limits')
        return float(action[0]), float(action[1])


def no_guard(target_speed, context):
    """Hook for a future LiDAR-following guard: returns the (possibly reduced)
    target speed. Called after the policy speed and speed_limit, before the
    speed limiter. Default: no-op."""
    return target_speed


A9_GUARD_SHA256 = '731d2786768a297d625fc5f075789b2ea79b894c4e62d1751bfbf1e0424dc7cf'
A9_PARAMS_SHA256 = '3767b3b780c38d4653dc7a3a8b890e75b9b8a98e5c71052e2367dd56116c8051'
V2F_GUARD_SHA256 = '234aa1b8b982ebef6276c6d41b52353ea8955ffe4cfb26ba91fbe4d464fe4cf2'
V2F_PARAMS_SHA256 = 'a44923a9164e7250e8a07c6715065c245c4a4b8170ffeb183f7afc1321a4acb4'


# Selectable guards. file/params are relative to the package dir / the installed config dir.
GUARD_VARIANTS = {
    'a9': dict(file='hj_lidar_following_guard_a9.py', sha256=A9_GUARD_SHA256,
               params='hj_lidar_guard_a9_params.yaml', params_sha256=A9_PARAMS_SHA256),
    'v2f': dict(file='hj_lidar_following_guard_v2f.py', sha256=V2F_GUARD_SHA256,
                params='hj_lidar_guard_v2f_params.yaml', params_sha256=V2F_PARAMS_SHA256),
}


def resolve_guard_paths(variant, guard_file, guard_sha256, params_file, params_sha256, config_dir):
    """(guard_file, guard_sha256, params_file, params_sha256); '' entries take the variant's.

    variant 'custom' has no defaults: all four must be given. config_dir is a callable
    returning the installed config directory (only called when needed).
    """
    if variant == 'custom':
        if not all((guard_file, guard_sha256, params_file, params_sha256)):
            raise ValueError("guard_variant 'custom' needs guard_file, guard_sha256, "
                             "guard_params_file and guard_params_sha256")
        return guard_file, guard_sha256, params_file, params_sha256
    if variant not in GUARD_VARIANTS:
        raise ValueError('guard_variant must be one of %s or custom, got %r' % (
            sorted(GUARD_VARIANTS), variant))
    v = GUARD_VARIANTS[variant]
    return (guard_file or str(Path(__file__).with_name(v['file'])), guard_sha256 or v['sha256'],
            params_file or str(Path(config_dir()) / v['params']), params_sha256 or v['params_sha256'])


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_guard_module(path, expected_sha256):
    """Load a guard file (byte-identical copy of the simulator file) after SHA check.

    The file imports algorithm.imitation.highspeed_lidar and utils.traffic_expert from
    the simulator repo; those names are served by hj_guard_shims during the import only.
    """
    path = Path(path)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError('guard file SHA256 mismatch: %s != %s' % (actual, expected_sha256))
    from f1tenth_stack import hj_guard_shims
    import types
    names = ('algorithm', 'algorithm.imitation', 'algorithm.imitation.highspeed_lidar',
             'utils', 'utils.traffic_expert')
    saved = {n: sys.modules.get(n) for n in names}
    fakes = {n: types.ModuleType(n) for n in names}
    fakes['algorithm.imitation.highspeed_lidar'].HighSpeedLidarPolicy = hj_guard_shims.HighSpeedLidarPolicy
    fakes['utils.traffic_expert'].safe_following_speed = hj_guard_shims.safe_following_speed
    try:
        sys.modules.update(fakes)
        spec = importlib.util.spec_from_file_location('hj_lidar_guard_loaded_' + actual[:8], str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for n, old in saved.items():
            if old is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = old
    return module


class GuardHook:
    """Adapter: guard_speed(obs, steer, params, memory) as a reduce-only speed hook.

    The guard counts steps (dt_s per call), not wall time; see README for the
    consequences when the real scan period differs from 0.025 s.
    """

    def __init__(self, module, params):
        self.module, self.params = module, dict(params)
        self.memory = module.new_memory()
        self.last = None

    def reset(self):
        self.memory = self.module.new_memory()
        self.last = None

    def __call__(self, target_speed, context):
        limit, debug = self.module.guard_speed(context['obs'], context['steer'],
                                               self.params, self.memory)
        self.last = (limit, debug)
        if not (limit == limit):  # NaN
            return 0.
        return min(target_speed, float(limit))


def load_guard_hook(guard_file, guard_sha256, params_file, params_sha256):
    actual = file_sha256(params_file)
    if actual != params_sha256:
        raise ValueError('guard params SHA256 mismatch: %s != %s' % (actual, params_sha256))
    params = json.loads(Path(params_file).read_text())
    return GuardHook(load_guard_module(guard_file, guard_sha256), params)


class SpeedAgentCore:
    """One control cycle: assemble -> policy -> speed_limit -> guard hook -> slew limiter."""

    def __init__(self, assembler, policy, speed_limit=1., max_acceleration=2.,
                 max_deceleration=3., scan_timeout=.2, guard=no_guard):
        if not (math.isfinite(speed_limit) and speed_limit > 0):
            raise ValueError('speed_limit must be finite and positive')
        self.assembler, self.policy, self.speed_limit, self.guard = assembler, policy, speed_limit, guard
        self.limiter = SpeedLimiter(max_acceleration, max_deceleration, max_dt=min(.1, scan_timeout))

    def stop(self):
        self.limiter.reset()
        self.assembler.reset()
        reset = getattr(self.guard, 'reset', None)
        if reset is not None:
            reset()  # guard episode memory must not survive a stop/input loss

    def step(self, ranges, angle_min, angle_increment, range_min, range_max, scan_time, mono):
        """Returns (steer, policy_speed, target_speed, command_speed); raises on bad input."""
        try:
            return self._step(ranges, angle_min, angle_increment, range_min, range_max,
                              scan_time, mono)
        except Exception:
            self.stop()
            raise

    def _step(self, ranges, angle_min, angle_increment, range_min, range_max, scan_time, mono):
        obs = self.assembler.assemble(ranges, angle_min, angle_increment, range_min,
                                      range_max, scan_time, mono)
        if self.assembler.fresh_history and hasattr(self.guard, 'reset'):
            self.guard.reset()  # history restarted (gap/first scan) -> new episode memory
        steer, speed = self.policy.predict(obs)
        target = min(speed, self.speed_limit)
        target = self.guard(target, dict(obs=obs, steer=steer, policy_speed=speed, mono=mono))
        if not math.isfinite(target) or target < 0 or target > min(speed, self.speed_limit)+1e-9:
            raise ValueError('guard hook may only reduce the target speed')
        command = self.limiter.update(target, mono)
        return steer, speed, target, command
