#!/usr/bin/env python3
"""Scan+odom-speed policy runner (4321 input). Default mode never drives (dry_run)."""
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import torch
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from f1tenth_stack.hj_speed_policy_runtime import (
    Rejected, SPEED_SOURCES, ScanSpeedAssembler, SpeedAgentCore, SpeedPolicy, erpm_to_speed,
    load_guard_hook, no_guard,
    resolve_guard_paths)


def stamp_seconds(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class HjSpeedAgentNode(Node):
    def __init__(self):
        super().__init__('hj_speed_agent_node')
        defaults = {
            'policy_file': '', 'scan_topic': '/scan', 'odom_topic': '/odom',
            'drive_topic': '/drive', 'prediction_topic': '/agent/prediction',
            'command_preview_topic': '/agent/command_preview',
            'dry_run': True, 'device': 'cpu', 'laser_yaw_offset': 0.0,
            'speed_limit': 1.0, 'scan_timeout': 0.2,
            'max_acceleration': 2.0, 'max_deceleration': 3.0,
            'odom_max_skew': 0.1, 'odom_timeout': 0.25,
            'odom_reverse_tolerance': 0.1, 'odom_max_speed_input': 8.0,
            'use_header_stamps': True,
            # speed_source 'odom' (default): /odom twist.linear.x (vesc_to_odom).
            # 'vesc_core': /sensors/core ERPM, v = (erpm - core_speed_offset_erpm)/gain (default
            # offset 0: ERPM 0 = standstill). gain is NOT read from vesc.yaml: keep it equal to
            # the running vesc nodes' speed_to_erpm_gain, or the speed input is wrong.
            # The command offset speed_to_erpm_offset is deliberately not used here.
            'speed_source': 'odom', 'core_topic': '/sensors/core',
            'speed_to_erpm_gain': 2500.0, 'core_speed_offset_erpm': 0.0,
            'core_deadband': 0.05,
            'guard_enabled': False, 'guard_variant': 'a9', 'guard_file': '', 'guard_sha256': '',
            'guard_params_file': '', 'guard_params_sha256': '',
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.params = {key: self.get_parameter(key).value for key in defaults}
        self.dry_run = self.params['dry_run']
        if self.params['speed_source'] not in SPEED_SOURCES:
            raise ValueError('speed_source must be one of %s' % (SPEED_SOURCES,))
        self.speed_source = self.params['speed_source']
        if self.speed_source == 'vesc_core':
            if not (np.isfinite(self.params['speed_to_erpm_gain']) and
                    self.params['speed_to_erpm_gain'] > 0 and
                    np.isfinite(self.params['core_speed_offset_erpm']) and
                    np.isfinite(self.params['core_deadband']) and
                    self.params['core_deadband'] >= 0):
                raise ValueError('invalid speed_to_erpm_gain/core_speed_offset_erpm/core_deadband')
        if not self.params['policy_file']:
            raise ValueError('policy_file (TorchScript 4321 policy) is required')
        if not all(np.isfinite(self.params[k]) and self.params[k] > 0
                   for k in ('speed_limit', 'scan_timeout')):
            raise ValueError('speed_limit and scan_timeout must be finite and positive')
        topics = [self.params[k] for k in (
            'drive_topic', 'prediction_topic', 'command_preview_topic')]
        if len({t.strip('/') for t in topics}) != 3:
            raise ValueError('Drive and diagnostic topics must be distinct')
        device = self.params['device']
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.policy = SpeedPolicy(self.params['policy_file'], device)
        self.assembler = ScanSpeedAssembler(
            4, self.params['scan_timeout'], self.params['odom_max_skew'],
            self.params['odom_timeout'], self.params['odom_reverse_tolerance'],
            self.params['odom_max_speed_input'], self.params['laser_yaw_offset'])
        guard = no_guard
        if self.params['guard_enabled']:
            def config_dir():
                from ament_index_python.packages import get_package_share_directory
                return Path(get_package_share_directory('f1tenth_stack')) / 'config'
            guard_file, guard_sha, params_file, params_sha = resolve_guard_paths(
                self.params['guard_variant'], self.params['guard_file'],
                self.params['guard_sha256'], self.params['guard_params_file'],
                self.params['guard_params_sha256'], config_dir)
            guard = load_guard_hook(guard_file, guard_sha, params_file, params_sha)
        self.guard_enabled = bool(self.params['guard_enabled'])
        self.core = SpeedAgentCore(
            self.assembler, self.policy, self.params['speed_limit'],
            self.params['max_acceleration'], self.params['max_deceleration'],
            self.params['scan_timeout'], guard=guard)
        self.pred_pub = self.create_publisher(
            AckermannDriveStamped, self.params['prediction_topic'], 1)
        self.preview_pub = self.create_publisher(
            AckermannDriveStamped, self.params['command_preview_topic'], 1)
        self.drive_pub = None
        if not self.dry_run:
            self.drive_pub = self.create_publisher(
                AckermannDriveStamped, self.params['drive_topic'], 1)
        self.last_valid_scan = None
        self.last_log = self.last_error_log = 0.0
        self.scan_count = 0
        if self.speed_source == 'vesc_core':
            from vesc_msgs.msg import VescStateStamped  # only needed for this source
            self.create_subscription(
                VescStateStamped, self.params['core_topic'], self.core_callback,
                qos_profile_sensor_data)
        else:
            self.create_subscription(
                Odometry, self.params['odom_topic'], self.odom_callback,
                qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, self.params['scan_topic'], self.scan_callback, qos_profile_sensor_data)
        self.create_timer(0.05, self.watchdog)
        self.get_logger().info(
            'Model %s loaded; dry_run=%s; guard=%s (%s); scan=%s odom=%s speed_limit=%.2f m/s '
            'speed_source=%s' % (
                self.params['policy_file'], self.dry_run, self.guard_enabled,
                self.params['guard_variant'], self.params['scan_topic'],
                self.params['odom_topic'], self.params['speed_limit'], self.speed_source))

    def ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def stop(self):
        # Stops bypass ordinary deceleration limiting, including in dry-run.
        self.core.stop()
        self.last_valid_scan = None
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.preview_pub.publish(msg)
        if self.drive_pub is not None:
            self.drive_pub.publish(msg)

    def watchdog(self):
        now = time.monotonic()
        if (self.last_valid_scan is None or
                now - self.last_valid_scan > self.params['scan_timeout'] or
                not self.assembler.odom_fresh(now)):
            self.stop()

    def message_time(self, header):
        if self.params['use_header_stamps']:
            if not (header.stamp.sec or header.stamp.nanosec):
                raise Rejected('zero header stamp (set use_header_stamps:=false to use receipt time)')
            return stamp_seconds(header)
        return self.ros_now()

    def odom_callback(self, odom):
        try:
            self.assembler.on_odom(odom.twist.twist.linear.x,
                                   self.message_time(odom.header), time.monotonic())
        except Exception as exc:
            self.assembler.odom = None
            self.log_error('Odom rejected: %s' % exc)

    def core_callback(self, msg):
        try:
            vx = erpm_to_speed(msg.state.speed, self.params['speed_to_erpm_gain'],
                               self.params['core_speed_offset_erpm'], self.params['core_deadband'])
            self.assembler.on_odom(vx, self.message_time(msg.header), time.monotonic())
        except Exception as exc:
            self.assembler.odom = None
            self.log_error('VESC core rejected: %s' % exc)

    def scan_callback(self, scan):
        try:
            scan_time = self.message_time(scan.header)
            if self.params['use_header_stamps']:
                age = self.ros_now() - scan_time
                if age > self.params['scan_timeout'] or age < -0.05:
                    raise Rejected('Stale scan or inconsistent ROS clock')
            started = time.monotonic()
            steer, speed, target, command_speed = self.core.step(
                scan.ranges, scan.angle_min, scan.angle_increment,
                scan.range_min, scan.range_max, scan_time, started)
            if time.monotonic() - started > self.params['scan_timeout']:
                raise ValueError('Policy inference exceeded scan_timeout')
            self.last_valid_scan = time.monotonic()
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
                    'scan=%d odom_v=%.3f steer=%.3f rad speed=%.3f target=%.3f '
                    'command=%.3f m/s guard=%s dry_run=%s' % (
                        self.scan_count, self.assembler.odom[0], steer, speed, target,
                        command_speed, self.guard_enabled, self.dry_run))
        except Exception as exc:
            self.last_valid_scan = None
            self.stop()
            self.log_error('Scan rejected: %s' % exc)

    def log_error(self, text):
        if time.monotonic() - self.last_error_log >= 1.0:
            self.last_error_log = time.monotonic()
            self.get_logger().error(text)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HjSpeedAgentNode()
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
