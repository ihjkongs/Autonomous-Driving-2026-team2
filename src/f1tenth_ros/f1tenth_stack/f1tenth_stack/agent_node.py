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


class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        defaults = {
            'project_root': str(Path.home() / 'Autonomous-Driving-2026-team2'),
            'name': 'imitation', 'algo_idx': 2, 'scan_topic': '/scan',
            'drive_topic': '/drive', 'prediction_topic': '/agent/prediction',
            'dry_run': True, 'device': 'auto', 'laser_yaw_offset': 0.0,
            'speed_limit': 1.0, 'scan_timeout': 0.2,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.params = {key: self.get_parameter(key).value for key in defaults}
        self.dry_run = self.params['dry_run']
        if self.params['speed_limit'] <= 0 or self.params['scan_timeout'] <= 0:
            raise ValueError('speed_limit and scan_timeout must be positive')
        sim_path = Path(self.params['project_root']).expanduser().resolve() / 'src/f1tenth'
        if not sim_path.is_dir():
            raise ValueError('Set project_root to the repository containing src/f1tenth')
        sys.path.insert(0, str(sim_path))
        run = sim_path / 'results' / self.params['name'] / str(self.params['algo_idx'])
        with (run / 'backup/f1tenth.yaml').open() as stream:
            self.task = yaml.safe_load(stream)
        if self.task['obs_dim'] != OBS_DIM or self.task['action_dim'] != 2:
            raise ValueError('This runner requires a 1080-input, 2-output policy')
        with (run / 'sim2real/agent_obs_scale.pkl').open('rb') as stream:
            self.mean, self.var, count = pickle.load(stream)
        normalize_observation(np.zeros(OBS_DIM), self.mean, self.var)
        if not np.isfinite(count) or count <= 0:
            raise ValueError('Normalization statistics have no valid samples')
        device = self.params['device']
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.actor = torch.load(str(run / 'sim2real/actor.pt'), map_location='cpu')
        self.actor.to(self.device).eval()
        with torch.no_grad():
            self.actor(torch.zeros((1, OBS_DIM), device=self.device)).loc
        # No publisher to the vehicle command topic exists in dry-run mode.
        self.pred_pub = self.create_publisher(
            AckermannDriveStamped, self.params['prediction_topic'], 1)
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
            'Model loaded on %s; dry_run=%s; waiting for %s. Predictions: %s' % (
                self.device, self.dry_run, self.params['scan_topic'],
                self.params['prediction_topic']))

    def stop(self):
        if self.drive_pub is not None:
            msg = AckermannDriveStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
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
            obs = scan_to_observation(
                scan.ranges, scan.angle_min, scan.angle_increment,
                scan.range_min, scan.range_max, self.params['laser_yaw_offset'])
            norm = normalize_observation(obs, self.mean, self.var)
            started = time.monotonic()
            with torch.no_grad():
                x = torch.as_tensor(norm, device=self.device).unsqueeze(0)
                action = self.actor(x).loc.cpu().numpy().reshape(-1)
            steer, speed = decode_action(
                action, self.task['max_steer'], self.task['min_speed'], self.task['max_speed'])
            if time.monotonic() - started > self.params['scan_timeout']:
                raise ValueError('Policy inference exceeded scan_timeout')
            self.last_valid_scan = time.monotonic()
            self.scan_count += 1
            msg = AckermannDriveStamped()
            msg.header = scan.header
            msg.drive.steering_angle = steer
            msg.drive.speed = speed
            self.pred_pub.publish(msg)
            if self.drive_pub is not None:
                msg.drive.speed = min(speed, self.params['speed_limit'])
                self.drive_pub.publish(msg)
            if time.monotonic() - self.last_log >= 1.0:
                self.last_log = time.monotonic()
                self.get_logger().info(
                    'scan=%d rays=%d frame=%s steer=%.3f rad speed=%.3f m/s dry_run=%s' % (
                        self.scan_count, len(scan.ranges), scan.header.frame_id,
                        steer, speed, self.dry_run))
        except Exception as exc:
            self.last_valid_scan = None
            self.stop()
            if time.monotonic() - self.last_error_log >= 1.0:
                self.last_error_log = time.monotonic()
                self.get_logger().error('Scan rejected: %s' % exc)


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
