"""Launch only the scan+odom speed agent. Run bringup with enable_agent:=false
so the old agent_node does not also publish to /drive."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch(context):
    config = LaunchConfiguration('hj_agent_config').perform(context)
    dry_run = LaunchConfiguration('dry_run').perform(context).lower()
    if dry_run not in ('true', 'false'):
        raise ValueError('dry_run must be true or false')
    policy_file = LaunchConfiguration('policy_file').perform(context).strip()
    if not policy_file:
        raise ValueError('policy_file:=<path to 4321 driving_policy.ts> is required')
    args = ['--ros-args', '--params-file', config, '-p', 'dry_run:=' + dry_run,
            '-p', 'policy_file:=' + policy_file]
    guard = LaunchConfiguration('guard_enabled').perform(context).strip().lower()
    if guard:
        if guard not in ('true', 'false'):
            raise ValueError('guard_enabled must be true or false')
        args += ['-p', 'guard_enabled:=' + guard]
    variant = LaunchConfiguration('guard_variant').perform(context).strip().lower()
    if variant:
        if variant not in ('a9', 'v2f', 'custom'):
            raise ValueError('guard_variant must be a9, v2f or custom')
        args += ['-p', 'guard_variant:=' + variant]
    source = LaunchConfiguration('speed_source').perform(context).strip().lower()
    if source:
        if source not in ('odom', 'vesc_core'):
            raise ValueError('speed_source must be odom or vesc_core')
        args += ['-p', 'speed_source:=' + source]
    for key in ('speed_limit', 'max_acceleration', 'max_deceleration'):
        value = LaunchConfiguration(key).perform(context).strip()
        if value:
            args += ['-p', '%s:=%s' % (key, float(value))]
    # python -m avoids a setup.py console_script entry for this experimental node.
    return [ExecuteProcess(
        cmd=[LaunchConfiguration('agent_python').perform(context), '-m',
             'f1tenth_stack.hj_speed_agent_node'] + args, output='screen')]


def generate_launch_description():
    default = os.path.join(get_package_share_directory('f1tenth_stack'),
                           'config', 'hj_speed_agent.yaml')
    import sys
    return LaunchDescription([
        DeclareLaunchArgument('hj_agent_config', default_value=default),
        DeclareLaunchArgument('policy_file', default_value=''),
        DeclareLaunchArgument('dry_run', default_value='true'),
        DeclareLaunchArgument('agent_python', default_value=sys.executable),
        DeclareLaunchArgument('guard_enabled', default_value=''),
        DeclareLaunchArgument('guard_variant', default_value=''),
        DeclareLaunchArgument('speed_source', default_value=''),
        DeclareLaunchArgument('speed_limit', default_value=''),
        DeclareLaunchArgument('max_acceleration', default_value=''),
        DeclareLaunchArgument('max_deceleration', default_value=''),
        OpaqueFunction(function=_launch),
    ])
