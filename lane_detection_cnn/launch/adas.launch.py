
import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    gazebo_launch_file = os.path.join(
        get_package_share_directory('driver_assistant'),
        'launch',
        'gazebo.launch.py'
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file)
    )

    home = os.path.expanduser('~')

    cnn_command = (
        'source /opt/ros/humble/setup.bash && '
        'source "$HOME/adas_ws/install/setup.bash" && '
        'source "$HOME/adas_training/.venv/bin/activate" && '
        'python "$HOME/adas_ws/src/lane_detection_cnn/'
        'lane_detection_cnn/lane_cnn_node.py"'
    )

    cnn = ExecuteProcess(
        cmd=['bash', '-lc', cnn_command],
        output='screen',
        name='lane_cnn'
    )

    return LaunchDescription([
        gazebo,
        cnn
    ])
