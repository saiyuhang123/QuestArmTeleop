import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    oculus_share = get_package_share_directory("oculus_reader")

    fake_arm_and_ik = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(oculus_share, "launch", "canopen_arm_fake_ik.launch.py")
        )
    )
    pose_node = Node(
        package="oculus_reader",
        executable="pub_pose.py",
        name="pub_pose_node",
        output="screen",
    )
    delta_node = Node(
        package="oculus_reader",
        executable="pub_delta_pose.py",
        name="pub_delta_pose_node",
        output="screen",
        parameters=[
            {
                "hold_to_run": True,
                "delta_pose_frame_id": "base_link",
                "enable_topic": "/canopen_arm/quest_teleop_enable",
                "publish_gripper_when_disabled": False,
            }
        ],
    )

    return LaunchDescription([fake_arm_and_ik, pose_node, delta_node])
