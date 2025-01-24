This workspace has a collection of launchers for Joystick, Nav2 and SLAM Toolbox.


### Build / Install
1. `colcon build --symlink-install && source ./install/setup.sh`
1. Joystick navigation: `ros2 launch robo-sim joystick_teleop_launch.py`
1. SLAM Toolbox: `ros2 launch robo-sim slam_toobox_launch.py`
1. NAV2 launch: `ros2 launch robo-sim nav2_launch.py slam:=False map:=$(pwd)/src/robo-sim/config/warehouse.yaml`
