from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    mo_slam = Node(
        package='hydrakon_bringup',
        executable='mo_slam',
        output='screen',
    )

    return LaunchDescription([
        mo_slam,
    ])
