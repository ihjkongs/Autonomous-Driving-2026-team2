"""Limit forward speed-command slew; this does not estimate vehicle speed."""
import math


class SpeedLimiter:
    def __init__(self, max_acceleration, max_deceleration, max_dt=0.1):
        values = (max_acceleration, max_deceleration, max_dt)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Speed limiter rates and max_dt must be finite and positive')
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_dt = max_dt
        self.reset()

    def reset(self):
        """Forget the old command after an immediate stop or input loss."""
        self.speed = 0.0
        self.last_time = None

    def update(self, target, now):
        if not math.isfinite(target) or target < 0 or not math.isfinite(now):
            raise ValueError('Target must be finite and nonnegative; time must be finite')
        if self.last_time is None:
            self.last_time = now
            return self.speed
        dt = now - self.last_time
        if dt < 0:
            raise ValueError('Speed limiter requires monotonic time')
        # A delayed callback must not turn a long gap into a large speed step.
        dt = min(dt, self.max_dt)
        change = target - self.speed
        change = min(self.max_acceleration * dt,
                     max(-self.max_deceleration * dt, change))
        self.speed = max(0.0, self.speed + change)
        self.last_time = now
        return self.speed
