#!/usr/bin/env python3
"""LiDAR policy runner. Default mode publishes predictions only, never drive."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import torch
import yaml
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_stack.policy_input import (
    OBS_DIM, scan_to_observation, normalize_observation, decode_action,
)
from f1tenth_stack.speed_limiter import SpeedLimiter
from f1tenth_stack.standalone_policy import StandalonePolicy


def experiment_run(sim_path, name, index):
    parent, number = sim_path / 'results' / name, str(index)
    candidates = ([parent / number] if (parent / number).is_dir() else [])
    candidates += sorted(p for p in parent.glob(number + '_*') if p.is_dir())
    if len(candidates) != 1:
        raise ValueError('Expected one model directory for experiment %s; found %s' % (number, candidates))
    return candidates[0]


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        defaults = {
            'project_root': str(Path.home() / 'Autonomous-Driving-2026-team2'),
            'name': 'imitation', 'algo_idx': 2, 'scan_topic': '/scan',
            'drive_topic': '/drive', 'prediction_topic': '/agent/prediction',
            'command_preview_topic': '/agent/command_preview',
            'dry_run': True, 'device': 'auto', 'laser_yaw_offset': 0.0,
            'speed_limit': 1.0, 'scan_timeout': 0.2,
            'max_acceleration': 2.0, 'max_deceleration': 3.0,
            'policy_file': '',
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.params = {key: self.get_parameter(key).value for key in defaults}
        self.dry_run = self.params['dry_run']
        if not all(np.isfinite(self.params[k]) and self.params[k] > 0
                   for k in ('speed_limit', 'scan_timeout')):
            raise ValueError('speed_limit and scan_timeout must be finite and positive')
        topics = [self.params[k] for k in (
            'drive_topic', 'prediction_topic', 'command_preview_topic')]
        if len({t.strip('/') for t in topics}) != 3:
            raise ValueError('Drive and diagnostic topics must be distinct')
        self.speed_limiter = SpeedLimiter(
            self.params['max_acceleration'], self.params['max_deceleration'],
            max_dt=min(0.1, self.params['scan_timeout']))
        sim_path = Path(self.params['project_root']).expanduser().resolve() / 'src/f1tenth'
        if not self.params['policy_file'] and not sim_path.is_dir():
            raise ValueError('Set project_root to the repository containing src/f1tenth')
        sys.path.insert(0, str(sim_path))
        run = None
        if not self.params['policy_file']:
            run = experiment_run(sim_path, self.params['name'], self.params['algo_idx'])
            standalone = run / 'sim2real/driving_policy.ts'
            if standalone.is_file():
                self.params['policy_file'] = str(standalone)
        device = self.params['device']
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.policy_runtime = None
        if self.params['policy_file']:
            self.policy_runtime = StandalonePolicy(self.params['policy_file'], self.device,
                                                   self.params['scan_timeout'])
            self.task = self.policy_runtime.metadata
            self.obs_dim = self.task['obs_dim']
        else:
            with (run / 'backup/f1tenth.yaml').open() as stream:
                self.task = yaml.safe_load(stream)
            self.obs_dim = self.task['obs_dim']
            if self.task['obs_dim'] != OBS_DIM or self.task['action_dim'] != 2:
                raise ValueError('Legacy runner requires 1080 inputs; use policy_file for standalone models')
            with (run / 'sim2real/agent_obs_scale.pkl').open('rb') as stream:
                self.mean, self.var, count = pickle.load(stream)
            normalize_observation(np.zeros(OBS_DIM), self.mean, self.var)
            if not np.isfinite(count) or count <= 0:
                raise ValueError('Normalization statistics have no valid samples')
            self.actor = torch.load(str(run / 'sim2real/actor.pt'), map_location='cpu')
            self.actor.to(self.device).eval()
            with torch.no_grad():
                self.actor(torch.zeros((1, OBS_DIM), device=self.device)).loc
        # No publisher to the vehicle command topic exists in dry-run mode.
        self.pred_pub = self.create_publisher(
            AckermannDriveStamped, self.params['prediction_topic'], 1)
        self.preview_pub = self.create_publisher(
            AckermannDriveStamped, self.params['command_preview_topic'], 1)
        self.drive_pub = None
        if not self.dry_run:
            self.drive_pub = self.create_publisher(
                AckermannDriveStamped, self.params['drive_topic'], 1)
        self.last_valid_scan = None
        self.last_log = 0.0
        self.last_error_log = 0.0
        self.scan_count = 0
        self.create_subscription(
            LaserScan, self.params['scan_topic'], self.scan_callback, qos_profile_sensor_data)
        self.create_timer(0.05, self.watchdog)
        self.get_logger().info(
            'Model loaded on %s; dry_run=%s; waiting for %s. Predictions: %s; '
            'speed_limit=%.2f m/s, acceleration=%.2f, deceleration=%.2f m/s^2' % (
                self.device, self.dry_run, self.params['scan_topic'],
                self.params['prediction_topic'], self.params['speed_limit'],
                self.params['max_acceleration'], self.params['max_deceleration']))

    def stop(self):
        # Stops bypass ordinary deceleration limiting, including in dry-run.
        self.speed_limiter.reset()
        if self.policy_runtime is not None:
            self.policy_runtime.reset()
        self.last_valid_scan = None
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.preview_pub.publish(msg)
        if self.drive_pub is not None:
            self.drive_pub.publish(msg)

    def watchdog(self):
        if (self.last_valid_scan is None or
                time.monotonic() - self.last_valid_scan > self.params['scan_timeout']):
            self.stop()

    def scan_callback(self, scan):
        try:
            if scan.header.stamp.sec or scan.header.stamp.nanosec:
                age = (self.get_clock().now().nanoseconds -
                       (scan.header.stamp.sec * 10**9 + scan.header.stamp.nanosec)) / 1e9
                if age > self.params['scan_timeout'] or age < -0.05:
                    raise ValueError('Stale scan or inconsistent ROS clock')
            obs = self.getObs(scan)
            started = time.monotonic()
            if (self.last_valid_scan is not None and
                    started-self.last_valid_scan > self.params['scan_timeout']):
                self.stop()
            if self.policy_runtime is not None:
                steer, speed = self.policy_runtime.predict(obs, started)
            else:
                norm = normalize_observation(obs, self.mean, self.var)
                with torch.no_grad():
                    x = torch.as_tensor(norm, device=self.device).unsqueeze(0)
                    action = self.actor(x).loc.cpu().numpy().reshape(-1)
                steer, speed = decode_action(
                    action, self.task['max_steer'], self.task['min_speed'], self.task['max_speed'])
            if time.monotonic() - started > self.params['scan_timeout']:
                raise ValueError('Policy inference exceeded scan_timeout')
            now = time.monotonic()
            target_speed = min(speed, self.params['speed_limit'])
            command_speed = self.speed_limiter.update(target_speed, now)
            self.last_valid_scan = now
            self.scan_count += 1
            msg = AckermannDriveStamped()
            msg.header = scan.header
            msg.drive.steering_angle = steer
            msg.drive.speed = speed
            self.pred_pub.publish(msg)
            command = AckermannDriveStamped()
            command.header = scan.header
            command.drive.steering_angle = steer
            command.drive.speed = command_speed
            self.preview_pub.publish(command)
            if self.drive_pub is not None:
                self.drive_pub.publish(command)
            if time.monotonic() - self.last_log >= 1.0:
                self.last_log = time.monotonic()
                self.get_logger().info(
                    'scan=%d rays=%d frame=%s steer=%.3f rad speed=%.3f m/s '
                    'target_speed=%.3f command_speed=%.3f m/s dry_run=%s' % (
                        self.scan_count, len(scan.ranges), scan.header.frame_id,
                        steer, speed, target_speed, command_speed, self.dry_run))
        except Exception as exc:
            self.last_valid_scan = None
            self.stop()
            if time.monotonic() - self.last_error_log >= 1.0:
                self.last_error_log = time.monotonic()
                self.get_logger().error('Scan rejected: %s' % exc)

    def getObs(self, scan):
        # Resample one scan; standalone runtime manages the model's history.
        return scan_to_observation(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max, self.params['laser_yaw_offset'])


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AgentNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
