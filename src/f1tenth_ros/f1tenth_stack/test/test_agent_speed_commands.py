"""Verify command routing and stop bypass using ROS stubs and a fixed actor."""
import copy
import importlib.util
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
import yaml

PACKAGE = Path(__file__).parents[1] / 'f1tenth_stack'


@pytest.fixture
def runner(monkeypatch, tmp_path):
    clock = [0.0]
    dry_run = [False]
    for name in ('rclpy', 'rclpy.node', 'rclpy.qos', 'sensor_msgs',
                 'sensor_msgs.msg', 'ackermann_msgs', 'ackermann_msgs.msg'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, msg):
            self.messages.append(copy.deepcopy(msg))

    class Message:
        def __init__(self):
            self.header = NS(stamp=NS(sec=0, nanosec=0), frame_id='laser')
            self.drive = NS(speed=0., steering_angle=0.)

    class Node:
        def __init__(self, name):
            self.parameters, self.publishers = {}, {}

        def declare_parameter(self, key, value):
            self.parameters[key] = value

        def get_parameter(self, key):
            value = self.parameters[key]
            if key == 'project_root':
                value = str(tmp_path)
            elif key == 'dry_run':
                value = dry_run[0]
            return NS(value=value)

        def create_publisher(self, kind, topic, depth):
            publisher = Publisher()
            self.publishers[topic] = publisher
            return publisher

        def create_subscription(self, *args):
            pass

        def create_timer(self, *args):
            pass

        def get_clock(self):
            return NS(now=lambda: NS(nanoseconds=int(clock[0]*1e9),
                                     to_msg=lambda: NS(sec=0, nanosec=0)))

        def get_logger(self):
            return NS(info=lambda msg: None, error=lambda msg: None)

    class Actor(torch.nn.Module):
        def forward(self, observation):
            return NS(loc=torch.tensor([[.5, 1.]], device=observation.device))

    sys.modules['rclpy.node'].Node = Node
    sys.modules['rclpy.qos'].qos_profile_sensor_data = None
    sys.modules['sensor_msgs.msg'].LaserScan = object
    sys.modules['ackermann_msgs.msg'].AckermannDriveStamped = Message
    # Install a local package for the relative helper imports, isolated per test.
    package = types.ModuleType('f1tenth_stack')
    package.__path__ = [str(PACKAGE)]
    monkeypatch.setitem(sys.modules, 'f1tenth_stack', package)
    for name in ('policy_input', 'speed_limiter', 'agent_node'):
        fullname = 'f1tenth_stack.' + name
        spec = importlib.util.spec_from_file_location(fullname, PACKAGE/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, fullname, module)
        spec.loader.exec_module(module)
    module = sys.modules['f1tenth_stack.agent_node']
    monkeypatch.setattr(module, 'time', NS(monotonic=lambda: clock[0]))
    monkeypatch.setattr(module.torch, 'load', lambda *args, **kwargs: Actor())
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: False)
    run = tmp_path/'src/f1tenth/results/imitation/2'
    (run/'backup').mkdir(parents=True)
    (run/'sim2real').mkdir()
    (run/'backup/f1tenth.yaml').write_text(yaml.safe_dump(
        dict(obs_dim=1080, action_dim=2, max_steer=.48, min_speed=1., max_speed=4.)))
    with (run/'sim2real/agent_obs_scale.pkl').open('wb') as stream:
        pickle.dump([np.zeros(1080), np.ones(1080), 100], stream)
    scan = NS(ranges=[5.]*1080, angle_min=-2.35, angle_increment=4.7/1079,
              range_min=.02, range_max=30., header=Message().header)
    return module.AgentNode, clock, dry_run, scan


def test_predictions_preserved_commands_limited_and_invalid_scan_stops_immediately(runner):
    AgentNode, clock, _, scan = runner
    node = AgentNode()
    for value in np.arange(0, .6, .025):
        clock[0] = float(value)
        node.scan_callback(scan)
    assert node.pred_pub.messages[-1].drive.speed == 4
    # Node default cap is 1 m/s; the actor and normalized prediction stay untouched.
    assert node.drive_pub.messages[-1].drive.speed == pytest.approx(1.)
    assert node.preview_pub.messages[-1].drive.speed == pytest.approx(1.)
    assert node.drive_pub.messages[-1].drive.steering_angle == pytest.approx(.24)
    scan.ranges[0] = float('nan')
    clock[0] += .025
    node.scan_callback(scan)
    assert node.drive_pub.messages[-1].drive.speed == 0
    assert node.preview_pub.messages[-1].drive.speed == 0
    assert node.speed_limiter.speed == 0
    scan.ranges[0] = 5.
    clock[0] += .025
    node.scan_callback(scan)
    assert node.drive_pub.messages[-1].drive.speed == 0
    clock[0] += .025
    node.scan_callback(scan)
    assert node.drive_pub.messages[-1].drive.speed == pytest.approx(.05)


def test_watchdog_resets_limiter_and_fresh_scan_after_gap_also_restarts(runner):
    AgentNode, clock, _, scan = runner
    node = AgentNode()
    for value in np.arange(0, .6, .025):
        clock[0] = float(value)
        node.scan_callback(scan)
    clock[0] += .3
    node.watchdog()
    assert node.drive_pub.messages[-1].drive.speed == 0
    assert node.last_valid_scan is None
    node.scan_callback(scan)
    assert node.drive_pub.messages[-1].drive.speed == 0
    clock[0] += .025
    node.scan_callback(scan)
    assert node.drive_pub.messages[-1].drive.speed == pytest.approx(.05)
    # If a fresh scan callback runs before the watchdog, the stale state is reset too.
    clock[0] += .3
    node.scan_callback(scan)
    assert node.drive_pub.messages[-1].drive.speed == 0


def test_dry_run_produces_preview_but_no_vehicle_publisher(runner):
    AgentNode, clock, dry_run, scan = runner
    dry_run[0] = True
    node = AgentNode()
    assert node.drive_pub is None
    assert '/drive' not in node.publishers
    node.scan_callback(scan)
    clock[0] = .025
    node.scan_callback(scan)
    assert node.pred_pub.messages[-1].drive.speed == 4
    assert node.preview_pub.messages[-1].drive.speed == pytest.approx(.05)
    node.stop()
    assert node.preview_pub.messages[-1].drive.speed == 0
