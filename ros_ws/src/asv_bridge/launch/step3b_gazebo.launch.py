"""
Step 3b launch: bridge_node (gazebo_mode) + policy_node.

Prerequisites (run FIRST in a separate container terminal):
  ros2 launch vrx_gz vrx_environment.launch.py \
      world:=sydney_regatta sim_mode:=full \
      config_file:=/work/wamv_spawn.yaml

Then in a second container terminal (docker exec into the same container):
  ros2 launch asv_bridge step3b_gazebo.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('seed',          default_value='42'),
        DeclareLaunchArgument('vecnorm_pkl',   default_value=
            '/work/reference/trained_policy/vec_normalize_20km_ciric_version1.pkl'),
        DeclareLaunchArgument('actor_weights', default_value=
            '/work/reference/trained_policy/actor_weights.npz'),
        DeclareLaunchArgument('tick_period',   default_value='300.0'),

        # EKF + obs publisher — reads real WAM-V pose from GPS + IMU
        Node(
            package='asv_bridge',
            executable='bridge_node',
            name='asv_bridge',
            parameters=[{
                'seed':        LaunchConfiguration('seed'),
                'vecnorm_pkl': LaunchConfiguration('vecnorm_pkl'),
                'gazebo_mode': True,
                'tick_period': LaunchConfiguration('tick_period'),
                'use_sim_time': True,
            }],
            output='screen',
        ),

        # SAC actor: obs → action → thruster commands
        Node(
            package='asv_bridge',
            executable='policy_node',
            name='asv_policy',
            parameters=[{
                'actor_weights': LaunchConfiguration('actor_weights'),
                'use_sim_time': True,
            }],
            output='screen',
        ),
    ])
