"""Pack the selected neural policy and its physical I/O contract into one file."""
import argparse
import json
from pathlib import Path
import pickle
import torch
from algorithm.imitation.lidar_policy import PhysicalLidarPolicy


def export(run, output):
    with (run/'backup/args.pkl').open('rb') as stream:
        args = pickle.load(stream)
    actor = torch.load(run/'sim2real/actor.pt', map_location='cpu').eval()
    metadata = dict(format='f1tenth-physical-lidar-v1', lidar_frames=args.lidar_frames,
        obs_dim=args.obs_dim, rays_per_frame=1080, fov_rad=4.7, max_range_m=30.,
        timestep_s=args.timestep, max_steer=args.max_steer,
        min_speed=args.min_speed, max_speed=args.max_speed,
        frame_order='oldest_to_newest', initial_history='repeat_first_scan',
        input='raw_lidar_metres', output=['steering_angle_rad', 'target_speed_mps'],
        runtime_expert=False, runtime_track_configuration=False)
    wrapper = PhysicalLidarPolicy(actor, args.max_steer, args.min_speed, args.max_speed)
    traced = torch.jit.trace(wrapper, torch.zeros(1, args.obs_dim))
    torch.jit.save(traced, str(output), _extra_files={'metadata.json': json.dumps(metadata)})
    extra = {'metadata.json': ''}
    loaded = torch.jit.load(str(output), _extra_files=extra)
    scans = torch.rand(5, args.obs_dim)*30.
    with torch.no_grad():
        if not torch.allclose(loaded(scans), wrapper(scans), atol=1e-6):
            raise RuntimeError('Export changed model outputs')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    cli = parser.parse_args()
    output = cli.output or cli.run/'sim2real/driving_policy.ts'
    print(json.dumps(dict(path=str(output), metadata=export(cli.run, output)), indent=2))


if __name__ == '__main__':
    main()
