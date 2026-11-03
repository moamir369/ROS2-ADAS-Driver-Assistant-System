from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # فتح نافذة عرض البث المباشر للـ Left Camera
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='left_camera_view',
            arguments=['/front/image_raw'],
            output='screen'
        ),
        # فتح نافذة عرض البث المباشر للـ Back Camera
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='back_camera_view',
            arguments=['/back/image_raw'],
            output='screen'
        )
    ])