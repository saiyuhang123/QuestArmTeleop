import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Quest-side stack for the REAL arm: Quest reader + deadman enable + IK.
    # Feedback (/feedback/joint_states) comes from the on-arm
    # quest_joint_state_bridge; the fake arm is NOT started here.
    oculus_share = get_package_share_directory("oculus_reader")

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

    return LaunchDescription([pose_node, delta_node, ik_node])
