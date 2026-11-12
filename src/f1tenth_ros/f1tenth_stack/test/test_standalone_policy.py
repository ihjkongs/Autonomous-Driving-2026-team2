"""One-file loading, chronological history and stale-scan recovery without ROS."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    'standalone_policy', Path(__file__).parents[1]/'f1tenth_stack/standalone_policy.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EchoPhysical(torch.nn.Module):
    def forward(self, x):
        return torch.stack([(x[:, -1]-x[:, 0])*.001, 1.+x.mean(-1)*.01], dim=-1)


def artifact(path, **overrides):
    metadata = dict(format='f1tenth-physical-lidar-v1', lidar_frames=4, obs_dim=4320,
                    rays_per_frame=1080, fov_rad=4.7, max_range_m=30.,
                    max_steer=.48, min_speed=1., max_speed=6.)
    metadata.update(overrides)
    model = torch.jit.trace(EchoPhysical(), torch.zeros(1, 4320))
    torch.jit.save(model, str(path), _extra_files={'metadata.json': json.dumps(metadata)})
    return path


@pytest.mark.parametrize('device', ['cpu']+(['cuda'] if torch.cuda.is_available() else []))
def test_raw_scan_history_and_long_gap_reset(tmp_path, device):
    runner = module.StandalonePolicy(artifact(tmp_path/'policy.ts'), device=device)
    assert runner.predict(np.full(1080, 5.), 1.) == pytest.approx((0., 1.05), abs=1e-6)
    assert runner.predict(np.full(1080, 6.), 1.025) == pytest.approx((.001, 1.0525), abs=1e-6)
    assert runner.predict(np.full(1080, 10.), 2.) == pytest.approx((0., 1.1), abs=1e-6)
    runner.reset()
    assert runner.predict(np.full(1080, 15.), 3.) == pytest.approx((0., 1.15), abs=1e-6)


def test_invalid_scan_discards_temporal_state(tmp_path):
    runner = module.StandalonePolicy(artifact(tmp_path/'policy.ts'))
    runner.predict(np.full(1080, 5.), 1.)
    with pytest.raises(ValueError):
        runner.predict(np.full(1080, np.nan), 1.025)
    assert not runner.history
    assert runner.predict(np.full(1080, 7.), 1.05)[0] == 0.


def test_missing_or_incompatible_contract_is_rejected(tmp_path):
    plain = torch.jit.trace(EchoPhysical(), torch.zeros(1, 4320))
    plain.save(str(tmp_path/'missing.ts'))
    with pytest.raises(ValueError, match='metadata'):
        module.StandalonePolicy(tmp_path/'missing.ts')
    with pytest.raises(ValueError, match='contract'):
        module.StandalonePolicy(artifact(tmp_path/'wrong.ts', obs_dim=1080))
