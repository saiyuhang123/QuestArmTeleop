import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    oculus_share = get_package_share_directory("oculus_reader")
    bringup_share = get_package_share_directory("canopen_arm_bringup")

    fake_arm = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_share, "launch", "fake_quest_joint_teleop.launch.py"
            )
        )
    )
    ik_node = Node(
        package="oculus_reader",
        executable="arm_ik_pose_node.py",
        name="arm_ik_pose_node",
        output="screen",
        parameters=[
            os.path.join(
                oculus_share, "config", "arm_ik_pose_node.canopen_arm.yaml"
            )
        ],
    )

    return LaunchDescription([fake_arm, ik_node])
