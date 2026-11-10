"""ROS-independent scan conversion matching the training simulator."""
import numpy as np

OBS_DIM = 1080
FOV = 4.7
MAX_RANGE = 30.0


def scan_to_observation(ranges, angle_min, angle_increment, range_min,
                        range_max, yaw_offset=0.0):
    """Resample laser-frame rays at vehicle-frame training angles, in metres.

    yaw_offset is the laser's mounting yaw relative to the vehicle (radians).
    Reject missing angular coverage and unknown/invalid returns rather than
    manufacturing a valid-looking scan. +inf denotes no return within range.
    """
    values = np.asarray(ranges, dtype=np.float64)
    metadata = [angle_min, angle_increment, range_min, range_max, yaw_offset]
    if not np.isfinite(metadata).all() or angle_increment == 0 or range_max <= range_min:
        raise ValueError('Invalid LaserScan metadata')
    if values.ndim != 1 or len(values) < 2:
        raise ValueError('LaserScan needs at least two rays')
    if np.isnan(values).any() or np.isneginf(values).any():
        raise ValueError('LaserScan contains NaN or negative infinity')
    if (values < range_min).any():
        raise ValueError('LaserScan contains a return below range_min')
    # A shorter sensor range cannot reproduce the training sensor horizon.
    if range_max < MAX_RANGE:
        raise ValueError('Sensor range_max is below the training range of 30 m')
    angles = angle_min + np.arange(len(values)) * angle_increment
    values = np.clip(values, 0.0, MAX_RANGE)
    if angle_increment < 0:
        angles, values = angles[::-1], values[::-1]
    targets = np.linspace(-FOV / 2, FOV / 2, OBS_DIM) - yaw_offset
    if targets[0] < angles[0] - 1e-6 or targets[-1] > angles[-1] + 1e-6:
        raise ValueError('LaserScan does not cover the training angles [-2.35, 2.35]')
    return np.interp(targets, angles, values).astype(np.float32)


def normalize_observation(obs, mean, var):
    obs, mean, var = (np.asarray(v) for v in (obs, mean, var))
    if any(v.shape != (OBS_DIM,) for v in (obs, mean, var)):
        raise ValueError('Observation and normalization must each have 1080 entries')
    if not all(np.isfinite(v).all() for v in (obs, mean, var)) or (var < 0).any():
        raise ValueError('Invalid observation or normalization statistics')
    return ((obs - mean) / np.sqrt(var + 1e-8)).astype(np.float32)


def decode_action(action, max_steer, min_speed, max_speed):
    action = np.asarray(action).reshape(-1)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError('Policy must return two finite actions')
    steering, speed = np.clip(action, -1.0, 1.0)
    return (float(steering * max_steer),
            float((max_speed - min_speed) * speed / 2 + (max_speed + min_speed) / 2))
