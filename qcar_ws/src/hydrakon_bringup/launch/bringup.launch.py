from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    description_pkg = FindPackageShare('hydrakon_description')
    camera_pkg = FindPackageShare('hydrakon_camera')
    bringup_pkg = FindPackageShare('hydrakon_bringup')

    use_rviz = LaunchConfiguration('use_rviz')
    enable_cone_detection = LaunchConfiguration('enable_cone_detection')
    enable_odom_tf = LaunchConfiguration('enable_odom_tf')

    # GroupAction(scoped=True) isolates the launch_arguments below so that
    # 'use_rviz'/'use_jsp_gui' overrides inside description.launch.py don't
    # leak into this file's own 'use_rviz' LaunchConfiguration — without
    # scoping, IncludeLaunchDescription mutates the LaunchConfiguration
    # globally and silently kills our own RViz node's condition check below.
    description_launch = GroupAction(
        scoped=True,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([description_pkg, 'launch', 'description.launch.py'])
                ),
                launch_arguments={
                    'use_jsp_gui': 'false',
                    'use_rviz': 'false',
                }.items(),
            ),
        ],
    )

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([camera_pkg, 'launch', 'camera.launch.py'])
        ),
        launch_arguments={
            'camera_model': 'zed2',
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([bringup_pkg, 'rviz', 'bringup.rviz'])],
        condition=IfCondition(use_rviz),
    )

    cone_locator = Node(
        package='hydrakon_camera',
        executable='cone_locator',
        output='screen',
        condition=IfCondition(enable_cone_detection),
    )

    odom_tf_bridge = Node(
        package='hydrakon_bringup',
        executable='odom_tf_bridge',
        output='screen',
        condition=IfCondition(enable_odom_tf),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true',
                              description='Launch RViz with the combined robot + camera view'),
        DeclareLaunchArgument('enable_cone_detection', default_value='true',
                              description='Run the YOLO-based cone_locator node alongside the camera'),
        DeclareLaunchArgument('enable_odom_tf', default_value='true',
                              description='Bridge ZED odom -> zed_camera_link onto odom -> base_link '
                                          'so RViz has a usable odom frame'),
        description_launch,
        camera_launch,
        rviz,
        cone_locator,
        odom_tf_bridge,
    ])
