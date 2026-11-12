"""One-file TorchScript policy runner; no map, waypoints or expert required."""
from collections import deque
import json
from pathlib import Path
import numpy as np
import torch


class StandalonePolicy:
    def __init__(self, path, device='cpu', scan_timeout=.2):
        self.device = torch.device(device)
        extra = {'metadata.json': ''}
        self.module = torch.jit.load(str(Path(path).expanduser()), map_location=self.device, _extra_files=extra).eval()
        if not extra['metadata.json']:
            raise ValueError('Standalone policy has no embedded metadata')
        self.metadata = json.loads(extra['metadata.json'])
        self.frames = int(self.metadata['lidar_frames'])
        if (self.metadata.get('format') != 'f1tenth-physical-lidar-v1' or
                self.metadata['rays_per_frame'] != 1080 or self.frames < 1 or
                self.metadata['obs_dim'] != self.frames*1080 or
                self.metadata['fov_rad'] != 4.7 or self.metadata['max_range_m'] != 30.):
            raise ValueError('Incompatible standalone policy input contract')
        limits = [self.metadata[k] for k in ('max_steer', 'min_speed', 'max_speed')]
        if not np.isfinite(limits).all() or limits[0] <= 0 or limits[1] < 0 or limits[2] < limits[1]:
            raise ValueError('Invalid standalone policy output limits')
        self.scan_timeout = float(scan_timeout)
        if not np.isfinite(self.scan_timeout) or self.scan_timeout <= 0:
            raise ValueError('Invalid scan timeout')
        self.history = deque(maxlen=self.frames)
        self.last_time = None
        # TorchScript may optimize its execution plan on the second invocation.
        # Finish that startup work before accepting a time-limited scan callback.
        for _ in range(3):
            self._infer(np.zeros(self.frames*1080, dtype=np.float32))

    def reset(self):
        self.history.clear()
        self.last_time = None

    def _infer(self, observation):
        with torch.inference_mode():
            tensor = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            action = self.module(tensor).cpu().numpy().reshape(-1)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('Standalone policy returned invalid actions')
        if (abs(action[0]) > self.metadata['max_steer']+1e-5 or
                action[1] < self.metadata['min_speed']-1e-5 or action[1] > self.metadata['max_speed']+1e-5):
            raise ValueError('Standalone policy action violates embedded limits')
        return float(action[0]), float(action[1])

    def predict(self, scan, now):
        scan = np.asarray(scan, dtype=np.float32)
        if scan.shape != (1080,) or not np.isfinite(scan).all() or not np.isfinite(now):
            self.reset()
            raise ValueError('Invalid policy scan or time')
        if self.last_time is not None and (now <= self.last_time or now-self.last_time > self.scan_timeout):
            self.reset()
        if not self.history:
            for _ in range(self.frames):
                self.history.append(scan.copy())
        else:
            self.history.append(scan.copy())
        self.last_time = now
        return self._infer(np.concatenate(list(self.history)))
