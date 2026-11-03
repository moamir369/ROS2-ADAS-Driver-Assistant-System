import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'driver_assistant'

    # 1. تحديد مسار ملف الخريطة gzharita.world
    world_file = os.path.join(
        get_package_share_directory(pkg_name),
        'worlds',
        'gzharita.world'
    )

    # 2. معالجة ملف الـ Xacro للروبوت
    xacro_file = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf',
        'main.xacro'
    )
    doc = xacro.process_file(xacro_file)
    robot_description_raw = doc.toxml()

    # 3. تشغيل Gazebo واستدعاء الخريطة
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ]),
        launch_arguments={'world': world_file}.items()
    )

    # 4. نشر حالة الروبوت
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_raw, 'use_sim_time': True}]
    )

    # 5. تنزيل السيارة داخل البيئة
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'syrix_robotaksi', '-z', '0.1'],
        output='screen'
    )

    # 6. تشغيل المتحكمات تلقائياً
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    load_ackermann_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'ackermann_steering_controller'],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        node_robot_state_publisher,
        spawn_entity,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_ackermann_controller],
            )
        ),
    ])