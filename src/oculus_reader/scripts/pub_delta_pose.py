#!/usr/bin/env python3
import math
import time
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from oculus_reader import OculusReader
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header
from transformations import euler_from_quaternion

def xyzrpy_to_mat(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    mat[:3, 3] = np.array([x, y, z])
    return mat

def mat2xyzquat(matrix: np.ndarray):
    pos = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return pos, quat

def calc_pose_incre(start_pose_matrix: np.ndarray, current_pose_matrix, zero_matrix: np.ndarray):
    end_matrix = xyzrpy_to_mat(
        current_pose_matrix[0],
        current_pose_matrix[1],
        current_pose_matrix[2],
        current_pose_matrix[3],
        current_pose_matrix[4],
        current_pose_matrix[5],
    )
    result_matrix = np.dot(zero_matrix, np.dot(np.linalg.inv(start_pose_matrix), end_matrix))
    return mat2xyzquat(result_matrix)

class RosOperator(Node):
    def __init__(self):
        super().__init__("pub_delta_pose_node")

        # Topic parameters
        self.declare_parameter("handle_pose_topic", "/right_handle_pose")
        self.declare_parameter("feedback_tcp_pose_topic", "/feedback/tcp_pose")
        self.declare_parameter("delta_pose_topic", "/delta_pose")
        self.declare_parameter("control_joint_topic", "/control/joint_states")
        self.declare_parameter("enable_topic", "/canopen_arm/quest_teleop_enable")
        self.declare_parameter("delta_pose_frame_id", "base_link")

        # Button mapping parameters
        dynamic_string_param = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("start_button", "A", dynamic_string_param)
        self.declare_parameter("stop_button", "B", dynamic_string_param)
        self.declare_parameter("trigger_axis", "rightTrig")

        # Gripper/control parameters
        self.declare_parameter("gripper_joint_name", "gripper")
        self.declare_parameter("gripper_max_range", 0.07)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("hand_name", "right")
        self.declare_parameter("hold_to_run", True)
        self.declare_parameter("handle_timeout_seconds", 0.25)
        self.declare_parameter("tcp_timeout_seconds", 0.50)
        self.declare_parameter("publish_gripper_when_disabled", False)
        # The arm tracks slower than the handle stream; without saturation the
        # target runs away from the physical arm and the joint-jump guard locks
        # up. Keep the published target inside this sphere around the latest
        # TCP feedback so the arm can always catch up.
        self.declare_parameter("max_target_offset_m", 0.15)
        # Fixed rotation (RPY degrees, 'xyz') from the handle's published frame
        # into base_link. Measured on the real setup 2026-09-15: user forward =
        # handle +Y, user left = handle +X, user up = handle -Z.
        self.declare_parameter("frame_align_rpy_deg", [180.0, 0.0, 90.0])

        handle_pose_topic = str(self.get_parameter("handle_pose_topic").value)
        feedback_tcp_pose_topic = str(self.get_parameter("feedback_tcp_pose_topic").value)
        delta_pose_topic = str(self.get_parameter("delta_pose_topic").value)
        control_joint_topic = str(self.get_parameter("control_joint_topic").value)
        enable_topic = str(self.get_parameter("enable_topic").value)
        self.delta_pose_frame_id = str(self.get_parameter("delta_pose_frame_id").value)

        self.start_button = self._normalize_button_name(self.get_parameter("start_button").value, "start_button")
        self.stop_button = self._normalize_button_name(self.get_parameter("stop_button").value, "stop_button")
        self.trigger_axis = str(self.get_parameter("trigger_axis").value)

        self.gripper_joint_name = str(self.get_parameter("gripper_joint_name").value)
        self.gripper_max_range = float(self.get_parameter("gripper_max_range").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.hand_name = str(self.get_parameter("hand_name").value)
        self.hold_to_run = bool(self.get_parameter("hold_to_run").value)
        self.handle_timeout_seconds = float(self.get_parameter("handle_timeout_seconds").value)
        self.tcp_timeout_seconds = float(self.get_parameter("tcp_timeout_seconds").value)
        self.publish_gripper_when_disabled = bool(
            self.get_parameter("publish_gripper_when_disabled").value
        )
        self.max_target_offset_m = float(self.get_parameter("max_target_offset_m").value)
        self.frame_align_rpy_deg = list(self.get_parameter("frame_align_rpy_deg").value)
        if (
            not self.delta_pose_frame_id
            or self.handle_timeout_seconds <= 0.0
            or self.tcp_timeout_seconds <= 0.0
        ):
            raise ValueError("Frame ID and input timeout parameters must be valid.")

        self.pub_delta_pose = self.create_publisher(PoseStamped, delta_pose_topic, 10)
        self.pub_move_j = self.create_publisher(JointState, control_joint_topic, 10)
        self.pub_enable = self.create_publisher(Bool, enable_topic, 10)

        # Callback inputs
        self.x = None
        self.y = None
        self.z = None
        self.roll = None
        self.pitch = None
        self.yaw = None

        self.tcp_x = None
        self.tcp_y = None
        self.tcp_z = None
        self.tcp_roll = None
        self.tcp_pitch = None
        self.tcp_yaw = None
        self.tcp_quat = None

        self.start_handle_pos = None
        self.anchor_tcp_pos = None
        self.anchor_tcp_quat = None
        self.align_rot = np.eye(3)

        self.flag = False
        self.rearm_required = False
        self.last_handle_time = None
        self.last_tcp_time = None

        self.create_subscription(PoseStamped, handle_pose_topic, self.handle_pose_callback, 1)
        self.create_subscription(PoseStamped, feedback_tcp_pose_topic, self.tcp_pose_callback, 1)

        # Wifi example:
        # self.oculus_reader = OculusReader(ip_address='192.168.124.2')
        self.oculus_reader = OculusReader()

        # Ensure OculusReader has been initialized.
        time.sleep(0.5)

        self.zero_matrix = xyzrpy_to_mat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.start_pose_matrix = None

        self.control_timer = self.create_timer(1.0 / max(control_rate_hz, 1.0), self.control_loop)

        self.get_logger().info(
            f"pub_delta_pose ready ({self.hand_name}). "
            f"handle_topic={handle_pose_topic}, feedback_topic={feedback_tcp_pose_topic}, "
            f"delta_topic={delta_pose_topic}, control_topic={control_joint_topic}, "
            f"buttons=({self.start_button}/{self.stop_button}), trigger={self.trigger_axis}, "
            f"hold_to_run={self.hold_to_run}, enable_topic={enable_topic}"
        )

    def handle_pose_callback(self, msg: PoseStamped):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        )
        self.last_handle_time = time.monotonic()

    def tcp_pose_callback(self, msg: PoseStamped):
        self.tcp_x = msg.pose.position.x
        self.tcp_y = msg.pose.position.y
        self.tcp_z = msg.pose.position.z
        (self.tcp_roll, self.tcp_pitch, self.tcp_yaw) = euler_from_quaternion(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        )
        self.tcp_quat = [
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        ]
        self.last_tcp_time = time.monotonic()

    def _inputs_are_fresh(self) -> bool:
        now = time.monotonic()
        return (
            self.last_handle_time is not None
            and self.last_tcp_time is not None
            and now - self.last_handle_time <= self.handle_timeout_seconds
            and now - self.last_tcp_time <= self.tcp_timeout_seconds
        )

    def _publish_enable(self, enabled: bool) -> None:
        message = Bool()
        message.data = enabled
        self.pub_enable.publish(message)

    def _activate(self) -> None:
        # Anchor in a FIXED frame: handle position deltas (in the handle's
        # published frame) are rotated by a fixed alignment yaw into base_link
        # and added to the TCP at engagement. Orientation is held at the
        # engagement TCP orientation for now (position-first teleop).
        self.start_handle_pos = np.array([self.x, self.y, self.z], dtype=float)
        self.anchor_tcp_pos = np.array([self.tcp_x, self.tcp_y, self.tcp_z], dtype=float)
        self.anchor_tcp_quat = list(self.tcp_quat)
        # keep the legacy matrices for compatibility with any external use
        self.start_pose_matrix = xyzrpy_to_mat(
            self.x, self.y, self.z, self.roll, self.pitch, self.yaw
        )
        self.zero_matrix = xyzrpy_to_mat(
            self.tcp_x, self.tcp_y, self.tcp_z,
            self.tcp_roll, self.tcp_pitch, self.tcp_yaw,
        )
        self.align_rot = Rotation.from_euler(
            "xyz", self.frame_align_rpy_deg, degrees=True).as_matrix()
        self.flag = True
        self.get_logger().info(f"[{self.hand_name}] 开始遥操作")

    def _deactivate(self, reason: str) -> None:
        if self.flag:
            self.get_logger().info(f"[{self.hand_name}] 停止遥操作: {reason}")
        self.flag = False
        self.start_pose_matrix = None

    def _extract_trigger_value(self, buttons: dict) -> float:
        trigger_raw: Any = buttons.get(self.trigger_axis, [0.0])
        if isinstance(trigger_raw, (list, tuple)):
            return float(trigger_raw[0]) if trigger_raw else 0.0
        if isinstance(trigger_raw, (int, float)):
            return float(trigger_raw)
        return 0.0

    def _normalize_button_name(self, raw_value: Any, param_name: str) -> str:
        if isinstance(raw_value, bool):
            normalized_bool = "Y" if raw_value else "N"
            self.get_logger().warn(
                f"Parameter '{param_name}' is BOOL({raw_value}), normalized to '{normalized_bool}'. "
                "Please pass string button names in launch/params."
            )
            return normalized_bool

        normalized = str(raw_value).strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in ("'", '"'):
            normalized = normalized[1:-1].strip()

        normalized = normalized.upper()
        allowed = {"A", "B", "X", "Y", "N"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid {param_name}='{raw_value}'. Expected one of {sorted(allowed)} or BOOL for compatibility."
            )
        return normalized

    def control_loop(self):
        _, buttons = self.oculus_reader.get_transformations_and_buttons()

        start_pressed = bool(buttons.get(self.start_button, False))
        stop_pressed = bool(buttons.get(self.stop_button, False))
        inputs_fresh = self._inputs_are_fresh()

        if self.hold_to_run:
            if stop_pressed:
                self.rearm_required = True
                self._deactivate("stop button")
            elif not inputs_fresh:
                if start_pressed:
                    self.rearm_required = True
                self._deactivate("input timeout")
            elif not start_pressed:
                self._deactivate("button released")
                self.rearm_required = False
            elif not self.rearm_required and not self.flag:
                self._activate()
        else:
            if stop_pressed:
                self.rearm_required = True
                self._deactivate("stop button")
            elif not inputs_fresh:
                self.rearm_required = True
                self._deactivate("input timeout")
            elif not start_pressed:
                self.rearm_required = False
            elif not self.rearm_required and not self.flag:
                self._activate()

        enabled = self.flag and inputs_fresh
        self._publish_enable(enabled)

        # Gripper control
        if enabled or self.publish_gripper_when_disabled:
            trigger_value = self._extract_trigger_value(buttons)
            gripper_value = max(0.0, min(trigger_value, 1.0)) * self.gripper_max_range
            gripper_msg = JointState()
            gripper_msg.header = Header()
            gripper_msg.header.stamp = self.get_clock().now().to_msg()
            gripper_msg.name = [self.gripper_joint_name]
            gripper_msg.position = [gripper_value]
            self.pub_move_j.publish(gripper_msg)

        if not enabled:
            return
        if self.start_handle_pos is None or self.anchor_tcp_quat is None:
            return

        # Fixed-frame position tracking: handle position delta (published
        # frame) rotated into base_link by the fixed alignment yaw, then added
        # to the engagement-time TCP. Orientation is held at engagement.
        handle_now = np.array([self.x, self.y, self.z], dtype=float)
        delta = self.align_rot @ (handle_now - self.start_handle_pos)
        xyz = self._clamp_target_to_tcp((self.anchor_tcp_pos + delta).tolist())
        quat = self.anchor_tcp_quat
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.delta_pose_frame_id
        pose_msg.pose.position.x = float(xyz[0])
        pose_msg.pose.position.y = float(xyz[1])
        pose_msg.pose.position.z = float(xyz[2])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        self.pub_delta_pose.publish(pose_msg)

    def _clamp_target_to_tcp(self, xyz):
        # The arm can only track a target that stays near its current TCP: on
        # slow backends the physical arm lags, and without saturation the
        # target runs away and the joint-jump guard locks up. Clamp the target
        # translation into a sphere around the latest TCP feedback.
        if self.tcp_x is None or self.max_target_offset_m <= 0.0:
            return xyz
        offset = np.array(
            [float(xyz[0]) - self.tcp_x,
             float(xyz[1]) - self.tcp_y,
             float(xyz[2]) - self.tcp_z])
        distance = float(np.linalg.norm(offset))
        if distance <= self.max_target_offset_m:
            return xyz
        clamped = np.array([self.tcp_x, self.tcp_y, self.tcp_z]) + \
            offset / distance * self.max_target_offset_m
        return [float(v) for v in clamped]

    def destroy_node(self):
        self._publish_enable(False)
        self.oculus_reader.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    teleop_node = None
    try:
        teleop_node = RosOperator()
        executor = MultiThreadedExecutor()
        executor.add_node(teleop_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if teleop_node is not None:
            teleop_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
