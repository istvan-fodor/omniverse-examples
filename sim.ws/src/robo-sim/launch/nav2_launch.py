#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Get the share directory of your package
    pkg_share = get_package_share_directory('robo-sim')

    # Path to your parameter and map files. Adjust these if needed.
    default_map_file = os.path.join(pkg_share, 'config', 'warehouse.yaml')
    default_nav2_params_file = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    # Declare launch arguments for flexibility
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=default_map_file,
        description='Full path to map file'
    )

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_nav2_params_file,
        description='Full path to the ROS2 parameters file to use for Nav2'
    )

    slam_arg = DeclareLaunchArgument(
        'slam',
        default_value='False',
        description='Whether to run SLAM (True) or not (False)'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true',
    )

    # Launch the Nav2 bringup launch file with our chosen parameters
    # The nav2_bringup package provides a bringup_launch.py we can include,
    # or you can directly start nav2 components with Nodes.
    bringup_launch_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(bringup_launch_dir, 'bringup_launch.py')]),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': LaunchConfiguration('params_file'),
            'slam': LaunchConfiguration('slam')
        }.items()
    )

    return LaunchDescription([
        map_arg,
        params_arg,
        slam_arg,
        use_sim_time_arg,
        nav2_bringup
    ])
