"""Stand-ins for the two repo imports of hj_lidar_following_guard.py, so the guard file
can run byte-identical on a machine without src/f1tenth.

safe_following_speed is copied verbatim from src/f1tenth/utils/traffic_expert.py
(sha256 of that file 50f2f8a9f53c5a9d08a70983fc0f0cf56fad0f2e6fb23252d21ad2ae847514d6);
equality with the original is checked in test_hj_lidar_guard_hook.py.
"""
import numpy as np
import torch


def safe_following_speed(bumper_gap, leader_speed, braking=2., reaction=.6, standstill_gap=.8):
    """Equal-braking stopping envelope; parameters must be qualified in simulation."""
    if min(braking, reaction, standstill_gap) <= 0 or not np.isfinite([bumper_gap, leader_speed]).all():
        raise ValueError('Finite gap/speed and positive safety parameters required')
    usable = max(0., bumper_gap-standstill_gap)
    if usable <= 0:
        return 0.
    return max(0., -braking*reaction+np.sqrt((braking*reaction)**2+max(0., leader_speed)**2+2*braking*usable))


class HighSpeedLidarPolicy(torch.nn.Module):
    """Only a base class for the (unused on the vehicle) GuardedPolicy wrapper."""
