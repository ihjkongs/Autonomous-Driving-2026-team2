"""Track-independent temporal LiDAR encoder with preprocessing in the model."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalLidarPolicy(nn.Module):
    def __init__(self, frames=4):
        super().__init__()
        self.frames = frames
        self.obs_dim = frames * 1080
        self.encoder = nn.Sequential(
            nn.Conv1d(2*frames-1, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.SiLU(),
            nn.Flatten(), nn.Linear(64*34, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2))

    def action_mean(self, raw):
        single = raw.dim() == 1
        scans = raw.reshape(-1, self.frames, 1080).clamp(0., 30.)
        scans = F.avg_pool1d(scans, 4)
        distance = torch.log1p(scans) / math.log(31.)
        motion = ((distance[:, 1:] - distance[:, :-1])*10.).clamp(-1., 1.)
        mean = torch.tanh(self.encoder(torch.cat([distance, motion], dim=1)))
        return mean[0] if single else mean

    def forward(self, raw):
        mean = self.action_mean(raw)
        return torch.distributions.Normal(mean, torch.full_like(mean, .1))


class PhysicalLidarPolicy(nn.Module):
    """Export raw chronological scans -> steering radians and target m/s."""
    def __init__(self, actor, max_steer=.48, min_speed=1., max_speed=6.):
        super().__init__()
        self.actor = actor
        self.max_steer = max_steer
        self.min_speed, self.max_speed = min_speed, max_speed

    def forward(self, scans):
        normalized = self.actor.action_mean(scans)
        steer = normalized[..., 0] * self.max_steer
        speed = self.min_speed + (normalized[..., 1]+1.)*.5*(self.max_speed-self.min_speed)
        return torch.stack([steer, speed], dim=-1)
