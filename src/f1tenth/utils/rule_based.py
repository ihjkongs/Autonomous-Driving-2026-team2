'''
    Pure Pursuit Controller for Imitation Learning
'''

from typing import Tuple
import numpy as np
from numba import njit
from utils.speed_profile import speed_profile, preview_speed

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    best_distance_squared = np.inf
    best_i, best_t = 0, 0.0
    best_x, best_y = 0.0, 0.0
    for i in range(trajectory.shape[0] - 1):
        dx = trajectory[i + 1, 0] - trajectory[i, 0]
        dy = trajectory[i + 1, 1] - trajectory[i, 1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-16:
            continue
        t = ((point[0] - trajectory[i, 0]) * dx + (point[1] - trajectory[i, 1]) * dy) / length_squared
        t = min(1.0, max(0.0, t))
        x, y = trajectory[i, 0] + t * dx, trajectory[i, 1] + t * dy
        distance_squared = (point[0] - x)**2 + (point[1] - y)**2
        if distance_squared < best_distance_squared:
            best_distance_squared = distance_squared
            best_i, best_t, best_x, best_y = i, t, x, y
    return np.array([best_x, best_y]), np.sqrt(best_distance_squared), best_t, best_i



@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(
    point, radius, trajectory, t=0.0, wrap=False
):
    """First forward circle intersection, computed in local float64 coordinates.

    Subtract the circle center BEFORE taking dot products. Expanding squared
    world coordinates in float32 loses the small (metre-scale) differences.
    The closed path supplied by the planner includes its first point at the end.
    """
    n = trajectory.shape[0] - 1
    start_i = min(int(t), n - 1)
    start_t = t - start_i
    count = n + 1 if wrap else n - start_i
    for offset in range(count):
        i = (start_i + offset) % n
        sx = float(trajectory[i, 0]) - float(point[0])
        sy = float(trajectory[i, 1]) - float(point[1])
        vx = float(trajectory[i + 1, 0]) - float(trajectory[i, 0])
        vy = float(trajectory[i + 1, 1]) - float(trajectory[i, 1])
        a = vx * vx + vy * vy
        if a < 1e-16:
            continue
        b = 2.0 * (vx * sx + vy * sy)
        c = sx * sx + sy * sy - radius * radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            continue
        root = np.sqrt(discriminant)
        for u in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            lower = start_t if offset == 0 else 0.0
            upper = start_t if offset == n else 1.0
            if lower <= u <= upper:
                p = np.empty(2, dtype=np.float64)
                p[0] = float(trajectory[i, 0]) + u * vx
                p[1] = float(trajectory[i, 1]) + u * vy
                return p, i, u
    return None, None, None

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32),
        lookahead_point[0:2] - position,
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.0
    radius = 1 / (2.0 * waypoint_y / lookahead_distance**2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle

class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, track, wb, reference='raceline'):
        self.wheelbase = wb
        if reference not in ('raceline', 'centerline'):
            raise ValueError('reference must be raceline or centerline')
        line = getattr(track, reference)
        self.waypoints = np.stack(
            [line.xs, line.ys, line.vxs]
        ).T
        # Explicitly include the closing segment, without zero-length segments.
        keep = np.r_[True, np.linalg.norm(np.diff(self.waypoints[:, :2], axis=0), axis=1) > 1e-8]
        self.waypoints = self.waypoints[keep]
        if np.linalg.norm(self.waypoints[-1, :2] - self.waypoints[0, :2]) > 1e-8:
            self.waypoints = np.vstack([self.waypoints, self.waypoints[0]])
        self.max_reacquire = 20.0

        self.drawn_waypoints = []
        self.lookahead_point = None
        self.current_index = None
        self._nearest_index = None
        self._speed_profile = None
        self._speed_profile_parameters = None

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        # NOTE: specify type or numba complains
        self.waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        ).astype(np.float32)

    def render_lookahead_point(self, e):
        """
        Callback to render the lookahead point.
        """
        if self.lookahead_point is not None:
            points = self.lookahead_point[:2][None]  # shape (1, 2)
            e.render_points(points, color=(0, 0, 128), size=2)

    def render_local_plan(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """
        if self.current_index is not None:
            points = self.waypoints[self.current_index : self.current_index + 10, :2]
            e.render_lines(points, color=(0, 128, 0), size=1)

    def _get_current_waypoint(
        self, waypoints, lookahead_distance, position, theta
    ) -> Tuple[np.ndarray, int]:
        """
        Returns the current waypoint to follow given the current pose.

        Args:
            waypoints: The waypoints to follow (Nx3 array)
            lookahead_distance: The lookahead distance [m]
            position: The current position (2D array)
            theta: The current heading [rad]

        Returns:
            waypoint: The current waypoint to follow (x, y, speed)
            i: The index of the current waypoint
        """
        wpts = waypoints[:, :2]
        lookahead_distance = np.float32(lookahead_distance)
        if self._nearest_index is None:
            nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        else:
            # Adjacent legs of a hairpin can be closer than the leg being
            # followed. Preserve route order instead of jumping backwards.
            n = len(wpts) - 1
            indices = (self._nearest_index + np.arange(min(22, n + 1))) % n
            nearest_point, nearest_dist, t, local_i = nearest_point_on_trajectory(position, wpts[indices])
            i = (self._nearest_index + local_i) % n
        self._nearest_index = i
        if nearest_dist < lookahead_distance:
            t1 = np.float32(i + t)
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(
                position, lookahead_distance, wpts, t1, wrap=True
            )
            if i2 is None:
                return None, None
            current_waypoint = np.empty((3,), dtype=np.float32)
            # x, y
            current_waypoint[0:2] = lookahead_point
            # speed
            current_waypoint[2] = waypoints[i, -1]
            return current_waypoint, i
        elif nearest_dist < self.max_reacquire:
            # NOTE: specify type or numba complains
            # Reacquisition still needs an (x, y, speed) waypoint.
            # Move forward along the path instead of aiming sideways at the
            # closest vertex when starting away from the reference line.
            j = i
            distance = 0.0
            start = nearest_point
            while distance < lookahead_distance:
                end = wpts[j + 1]
                segment = np.linalg.norm(end - start)
                if distance + segment >= lookahead_distance:
                    fraction = (lookahead_distance - distance) / max(segment, 1e-8)
                    target = start + fraction * (end - start)
                    return np.array([target[0], target[1], waypoints[i, -1]]), i
                distance += segment
                j = (j + 1) % (len(wpts) - 1)
                start = wpts[j]
            return None, None
        else:
            return None, None

    def plan(self, obs_dict, work):
        """
        gives actuation given observation
        info['obs_dict']["poses_x"][0],
        info['obs_dict']["poses_y"][0],
        info['obs_dict']["poses_theta"][0],
        work["tlad"],
        work["vgain"],
        )
        """
        pose_x, pose_y, pose_theta = obs_dict['poses_x'][0], obs_dict['poses_y'][0], obs_dict['poses_theta'][0]
        lookahead_distance, vgain = work['tlad'], work['vgain']
        velocity = abs(float(obs_dict['linear_vels_x'][0]))
        lookahead_distance += work.get('lookahead_time', 0.0) * velocity
        lookahead_distance = min(lookahead_distance, work.get('max_lookahead', lookahead_distance))
        position = np.array([pose_x, pose_y])
        lookahead_point, i = self._get_current_waypoint(
            self.waypoints, lookahead_distance, position, pose_theta
        )

        if lookahead_point is None:
            raise RuntimeError('Expert lost the reference path; refusing to label straight-ahead driving as expert data.')

        # for rendering
        self.lookahead_point = lookahead_point
        self.current_index = i

        # actuation
        speed, steering_angle = get_actuation(
            pose_theta,
            self.lookahead_point,
            position,
            float(np.linalg.norm(self.lookahead_point[:2] - position)),
            self.wheelbase,
        )
        speed = vgain * speed

        if 'speed_cap' in work:
            curvature = abs(np.tan(steering_angle) / self.wheelbase)
            curve_speed = np.sqrt(work.get('lateral_acceleration', 1.5) / max(curvature, 1e-6))
            speed = min(speed, work['speed_cap'], curve_speed)
            speed = max(work.get('min_speed', 1.0), speed)

        if work.get('anticipatory_braking', False):
            if self._speed_profile_parameters != dict(work):
                self._speed_profile = speed_profile(self.waypoints[:-1, :2],
                                                    self.waypoints[:-1, 2], work)
                self._speed_profile_parameters = dict(work)
            arc, planned, _ = self._speed_profile
            speed = min(speed, preview_speed(arc, planned, self._nearest_index,
                        velocity * work.get('reaction_time', 0.0)))

        return speed, steering_angle
