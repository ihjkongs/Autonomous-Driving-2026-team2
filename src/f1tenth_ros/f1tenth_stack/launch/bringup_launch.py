from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import argparse
import sys

def _launch_agent(context):
    # Merge the named-node YAML and launch overrides before giving ROS one
    # parameter source. Avoid competing named-node and wildcard YAML entries.
    config_path = LaunchConfiguration('agent_config').perform(context)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    parameters = dict(config['agent_node']['ros__parameters'])
    dry_run = LaunchConfiguration('dry_run').perform(context).lower()
    if dry_run not in ('true', 'false'):
        raise ValueError('dry_run must be true or false')
    parameters.update({
        'project_root': LaunchConfiguration('project_root').perform(context),
        'dry_run': dry_run == 'true',
    })
    for key in ('speed_limit', 'max_acceleration', 'max_deceleration'):
        override = LaunchConfiguration(key).perform(context).strip()
        if override:
            parameters[key] = float(override)
    return [Node(
        package='f1tenth_stack', executable='agent_node', name='agent_node',
        prefix=LaunchConfiguration('agent_python'),
        output='screen', parameters=[parameters])]

def generate_launch_description():
    
    vesc_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'vesc.yaml'
    )
    sensors_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'sensors.yaml'
    )
    mux_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'mux.yaml'
    )
    agent_config = os.path.join(
        get_package_share_directory('f1tenth_stack'),
        'config',
        'agent.yaml'
    )

    vesc_la = DeclareLaunchArgument(
        'vesc_config',
        default_value=vesc_config,
        description='Descriptions for vesc configs')
    sensors_la = DeclareLaunchArgument(
        'sensors_config',
        default_value=sensors_config,
        description='Descriptions for sensor configs')
    mux_la = DeclareLaunchArgument(
        'mux_config',
        default_value=mux_config,
        description='Descriptions for ackermann mux configs')
    
    agent_la = DeclareLaunchArgument(
        'agent_config', default_value=agent_config,
        description='Policy model and observation configuration')
    enable_agent_la = DeclareLaunchArgument(
        'enable_agent', default_value='true',
        description='Start the LiDAR policy node')
    dry_run_la = DeclareLaunchArgument(
        'dry_run', default_value='true',
        description='Publish predictions only, without sending drive commands')
    project_root_la = DeclareLaunchArgument(
        'project_root',
        default_value=os.path.expanduser('~/Autonomous-Driving-2026-team2'),
        description='Repository root containing src/f1tenth')
    agent_python_la = DeclareLaunchArgument(
        'agent_python', default_value=sys.executable,
        description='Python interpreter with ROS 2 and PyTorch available')
    speed_limit_la = DeclareLaunchArgument(
        'speed_limit', default_value='', description='Override speed cap in m/s')
    acceleration_la = DeclareLaunchArgument(
        'max_acceleration', default_value='', description='Override command acceleration in m/s^2')
    deceleration_la = DeclareLaunchArgument(
        'max_deceleration', default_value='', description='Override normal command deceleration in m/s^2')
    ld = LaunchDescription([
        vesc_la, sensors_la, mux_la, agent_la, enable_agent_la,
        dry_run_la, project_root_la, agent_python_la,
        speed_limit_la, acceleration_la, deceleration_la])

    agent_node = OpaqueFunction(
        function=_launch_agent,
        condition=IfCondition(LaunchConfiguration('enable_agent')))
    key_to_ackermann_node = Node(
        package='f1tenth_stack',
        executable='keyboard_to_ackermann_drive',
        name='keyboard_to_ackermann_drive'
    )
    ackermann_to_vesc_node = Node(
        package='vesc_ackermann',
        executable='ackermann_to_vesc_node',
        name='ackermann_to_vesc_node',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    vesc_to_odom_node = Node(
        package='vesc_ackermann',
        executable='vesc_to_odom_node',
        name='vesc_to_odom_node',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    vesc_driver_node = Node(
        package='vesc_driver',
        executable='vesc_driver_node',
        name='vesc_driver_node',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    throttle_interpolator_node = Node(
        package='f1tenth_stack',
        executable='throttle_interpolator',
        name='throttle_interpolator',
        parameters=[LaunchConfiguration('vesc_config')]
    )
    urg_node = Node(
        package='urg_node',
        executable='urg_node_driver',
        name='urg_node',
        parameters=[LaunchConfiguration('sensors_config')]
    )
    ackermann_mux_node = Node(
        package='ackermann_mux',
        executable='ackermann_mux',
        name='ackermann_mux',
        parameters=[LaunchConfiguration('mux_config')],
        remappings=[('ackermann_drive_out', 'ackermann_cmd')]
    )
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_baselink_to_laser',
        arguments=['0.13', '0.0', '0.153', '0.0', '0.0', '0.0', 'base_link', 'laser']
    )

    # finalize
    ld.add_action(key_to_ackermann_node)
    ld.add_action(ackermann_to_vesc_node)
    ld.add_action(vesc_to_odom_node)
    ld.add_action(vesc_driver_node)
    ld.add_action(urg_node)
    ld.add_action(ackermann_mux_node)
    ld.add_action(static_tf_node)
    ld.add_action(agent_node)

    return ld
