from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare('hydrakon_description')
    urdf_file = PathJoinSubstitution([pkg, 'urdf', 'qcar.urdf.xacro'])
    rviz_config = PathJoinSubstitution([pkg, 'rviz', 'qcar.rviz'])

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_jsp_gui = LaunchConfiguration('use_jsp_gui')
    use_rviz = LaunchConfiguration('use_rviz')

    robot_description = Command([FindExecutable(name='xacro'), ' ', urdf_file])

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    # Manual slider GUI for steering/wheel joints — for standalone visualization only.
    jsp_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
        condition=IfCondition(use_jsp_gui),
    )

    # Headless joint state source (publishes zeros) so the TF tree stays complete
    # when the GUI slider isn't used — e.g. when included from another launch file.
    jsp = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen',
        condition=UnlessCondition(use_jsp_gui),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation clock'),
        DeclareLaunchArgument('use_jsp_gui', default_value='true',
                              description='Launch joint_state_publisher_gui with manual sliders'),
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Launch RViz with the default qcar view'),
        rsp,
        jsp_gui,
        jsp,
        rviz,
    ])
