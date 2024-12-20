from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package share directory
    package_dir = get_package_share_directory('robo-sim')

    # Define the paths to the config file
    config_file = os.path.join(package_dir, 'config', 'cartographer_config.lua')

    return LaunchDescription([
        # Launch cartographer_node
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ],
            arguments=[
                '-configuration_directory', os.path.join(package_dir, 'config'),
                '-configuration_basename', 'cartographer_config.lua'
            ],
            remappings=[
                ('points2', 'point_cloud')
            ],
        ),

        # Launch cartographer_occupancy_grid_node
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[
                {'use_sim_time': True}
            ],
            arguments=[
                '-resolution', '0.05',
                '-publish_period_sec', '1.0'
            ]
        ),
    ])
