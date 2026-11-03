import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'driver_assistant'
    
    # تحديد مسار ملف xacro الرئيسي
    xacro_file = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf',
        'main.xacro'
    )

    # تحديد مسار ملف إعدادات RViz2 المحفوظة
    rviz_config_file = os.path.join(
        get_package_share_directory(pkg_name),
        'rviz',
        'ilk_view_robot.rviz'
    )

    # معالجة ملف الـ Xacro وتحويله إلى URDF مكتمل
    doc = xacro.process_file(xacro_file)
    robot_description_config = doc.toxml()

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description_config}]
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]  # تحميل ملف الإعدادات تلقائياً
        )
    ])