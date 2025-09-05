from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Teleop Twist Joy Node
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_node',
            output='screen',
            parameters=[
                {'enable_button': 5},
                {'axis_angular.yaw': 3},
                {'axis_linear.x': 4},
                {'scale_linear.x': 2.0},
                {'scale_angular.z': 2.0}
            ],
            remappings=[
                ('joy_vel', 'cmd_vel')
            ]
        ),

        # Joy Node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen'
        ),
    ])
