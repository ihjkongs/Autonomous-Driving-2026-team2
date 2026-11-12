"""Bounded delayed speed response, not an electrical VESC or tire-slip model."""
from collections import deque
import math


class LongitudinalResponse:
    def __init__(self, config, timestep):
        self.dt = float(timestep)
        self.acceleration = float(config['acceleration'])
        self.deceleration = float(config['deceleration'])
        self.tau = float(config['time_constant'])
        self.delay = float(config['delay'])
        self.command_acceleration = float(config['command_acceleration'])
        self.command_deceleration = float(config['command_deceleration'])
        positive = (self.dt, self.acceleration, self.deceleration, self.tau,
                    self.command_acceleration, self.command_deceleration)
        if not all(math.isfinite(x) and x > 0 for x in positive) or not math.isfinite(self.delay) or self.delay < 0:
            raise ValueError('Invalid longitudinal response parameters')
        self.delay_steps = int(math.ceil(self.delay / self.dt - 1e-12))
        self.reset()

    def reset(self):
        self.command = 0.0
        self.first = True
        self.queue = deque([0.0] * self.delay_steps)

    def update(self, target, velocity):
        if not math.isfinite(target) or target < 0 or not math.isfinite(velocity):
            raise ValueError('Response model requires a finite forward speed target')
        # Match deployment's initial zero command; integrate at fixed simulator dt.
        if self.first:
            self.first = False
        else:
            delta = max(-self.command_deceleration * self.dt,
                        min(self.command_acceleration * self.dt, target - self.command))
            self.command += delta
        self.queue.append(self.command)
        delayed = self.queue.popleft()
        requested = (delayed - velocity) / self.tau
        return max(-self.deceleration, min(self.acceleration, requested))
