"""
Step 4 launch: bridge_node + policy_node + glider_viz_node.
Uses the open_ocean world (20 km glider separation, 25 km camera far clip).

Prerequisites — run FIRST in a separate container terminal:
  ros2 launch vrx_gz vrx_environment.launch.py \\
      world:=/work/worlds/open_ocean.sdf \\
      sim_mode:=full \\
      config_file:=/work/wamv_spawn.yaml

Then in a second container terminal (docker exec into the same container):
  ros2 launch asv_bridge step4_ocean.launch.py

Gliders (Slocum models) are spawned automatically once the bridge_node publishes
their initial positions from EkfCore.reset(). They are teleported every 300 s
to match the EkfCore glider dynamics.
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
        DeclareLaunchArgument('world_name',    default_value='open_ocean'),

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
            }],
            output='screen',
        ),

        # Glider visualization: spawns and teleports Slocum models in Gazebo
        Node(
            package='asv_bridge',
            executable='glider_viz_node',
            name='glider_viz',
            parameters=[{
                'world_name': LaunchConfiguration('world_name'),
                'model_sdf':  '/work/models/glider_slocum/model.sdf',
            }],
            output='screen',
        ),
    ])
