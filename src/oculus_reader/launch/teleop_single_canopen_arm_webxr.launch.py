import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # WebXR chain for the REAL arm: Quest browser (local-floor) -> webxr node
    # -> /delta_pose -> IK. Feedback (/feedback/joint_states) comes from the
    # on-arm quest_joint_state_bridge; the fake arm is NOT started here.
    # Requires ~/cert.pem and ~/key.pem; open https://<host-ip>:8012 in the
    # Quest browser once and accept the certificate warning.
    oculus_share = get_package_share_directory("oculus_reader")

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
                # Default 0.3 low-passes the handle pose with ~80 ms time
                # constant, which reads as noticeable teleop lag; 0.5 cuts it
                # to ~33 ms at the 30 Hz control rate.
                "smoothing_alpha": 0.5,
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

    return LaunchDescription([webxr_node, ik_node])
