import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # WebXR chain against the FAKE arm: canopen_arm_fake_ik + webxr node.
    # Use this for direction verification before touching real hardware.
    oculus_share = get_package_share_directory("oculus_reader")

    fake_arm_and_ik = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(oculus_share, "launch", "canopen_arm_fake_ik.launch.py")
        )
    )
    webxr_node = Node(
        package="oculus_reader",
        executable="webxr_teleop_node.py",
        name="webxr_teleop_node",
        output="screen",
        parameters=[
            {
                "hold_to_run": True,
                "delta_pose_frame_id": "base_link",
                "enable_topic": "/canopen_arm/quest_teleop_enable",
                "publish_gripper_when_disabled": False,
                # Measured 2026-09-16: this session's local-floor frame is
                # already aligned with base_link (push fwd -> +x). If a future
                # session starts with a different headset orientation, either
                # recenter facing the arm's forward or re-measure with
                # ~/QuestArmTeleop/tools/measure_session_yaw.py.
                "session_yaw_deg": 0.0,
            }
        ],
    )

    return LaunchDescription([fake_arm_and_ik, webxr_node])
