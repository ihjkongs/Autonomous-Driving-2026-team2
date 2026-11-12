"""Closed-path speed envelope with bounded acceleration and advance braking."""
import numpy as np


def speed_profile(points, route_speed, work):
    points = np.asarray(points, dtype=np.float64)
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(ds <= 1e-8):
        raise ValueError('Speed profile needs a non-degenerate closed path')
    arc = np.r_[0., np.cumsum(ds)]
    length = arc[-1]
    window = float(work.get('curvature_window', .75))
    brake = float(work['braking_deceleration'])
    acceleration = float(work.get('planning_acceleration', brake))
    if min(window, brake, acceleration) <= 0:
        raise ValueError('Speed planning distances and limits must be positive')
    def sample(s):
        return np.column_stack([np.interp(s % length, arc, np.r_[points[:, j], points[0, j]])
                                for j in (0, 1)])
    before, after = sample(arc[:-1] - window), sample(arc[:-1] + window)
    incoming, outgoing = points - before, after - points
    angle = np.arctan2(incoming[:, 0]*outgoing[:, 1] - incoming[:, 1]*outgoing[:, 0],
                       np.sum(incoming*outgoing, axis=1))
    curvature = np.abs(angle) / window
    speed = np.minimum(np.asarray(route_speed)*work['vgain'], work['speed_cap'])
    speed = np.minimum(speed, np.sqrt(work['lateral_acceleration'] / np.maximum(curvature, 1e-6)))
    speed = np.maximum(work.get('min_speed', 1.), speed)
    # Two complete periodic sweeps propagate any corner across the start seam.
    for _ in range(2):
        for i in range(len(speed)-1, -1, -1):
            speed[i] = min(speed[i], np.sqrt(speed[(i+1) % len(speed)]**2 + 2*brake*ds[i]))
        for i in range(len(speed)):
            j = (i+1) % len(speed)
            speed[j] = min(speed[j], np.sqrt(speed[i]**2 + 2*acceleration*ds[i]))
    return arc, speed, curvature


def preview_speed(arc, speed, index, distance):
    """Conservative minimum over the complete latency horizon, including seam."""
    if not np.isfinite(distance) or distance < 0:
        raise ValueError('Preview distance must be finite and nonnegative')
    if distance >= arc[-1]:
        return float(np.min(speed))
    stop = (arc[index] + distance) % arc[-1]
    offsets = (arc[:-1] - arc[index]) % arc[-1]
    end = float(np.interp(stop, arc, np.r_[speed, speed[0]]))
    return min(float(np.min(speed[offsets <= distance])), end)
