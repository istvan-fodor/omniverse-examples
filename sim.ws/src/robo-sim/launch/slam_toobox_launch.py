#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    pkg_share = get_package_share_directory('robo-sim')
    default_slam_params_file = os.path.join(pkg_share, 'config', 'mapper_params_online_async.yaml')

    declare_slam_params = DeclareLaunchArgument(
        'slam_params_file',
        default_value=default_slam_params_file,
        description='Full path to the SLAM Toolbox parameters file'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true',
    )

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[LaunchConfiguration('slam_params_file'), LaunchConfiguration('use_sim_time')],
        remappings=[
            ('/scan', '/scan'),
            ('/tf', '/tf'),
            ('/tf_static', '/tf_static'),
            ('/odom', '/odom'),
            ('/imu', '/imu'),
            ('/point_cloud', '/point_cloud')
        ]
    )

    return LaunchDescription([
        declare_slam_params,
        use_sim_time_arg,
        slam_toolbox_node
    ])
