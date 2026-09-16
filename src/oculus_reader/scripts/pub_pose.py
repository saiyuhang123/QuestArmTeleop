#!/usr/bin/env python3
from oculus_reader import OculusReader
from transformations import quaternion_from_matrix
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped, TransformStamped
from math import sin, cos
import numpy as np

def xyzrpy2Mat(x, y, z, roll, pitch, yaw):
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, sy * sr + cy * sp * cr, x],
        [sy * cp, cy * cr + sy * sp * sr, sy * sp * cr - cy * sr, y],
        [-sp, cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ])
    
class OculusPublisher(Node):
    def __init__(self):
        super().__init__("pub_pose_node")
        self.declare_parameter("publish_rate_hz", 100.0)
        self.declare_parameter("parent_frame_id", "vr_device")
        self.declare_parameter("ros_to_arm_rpy", [-np.pi / 2, 0.0, 0.0])
        self.declare_parameter("ros_to_arm_xyz", [0.0, 0.0, 0.0])

        self.right_handle_pose_pub = self.create_publisher(PoseStamped, "right_handle_pose", 1)
        self.left_handle_pose_pub = self.create_publisher(PoseStamped, "left_handle_pose", 1)
        self.br = TransformBroadcaster(self)
        
        # self.oculus_reader = OculusReader(ip_address='192.168.124.2')    #  WIFI连接
        self.oculus_reader = OculusReader()  # USB连接

        self.parent_frame_id = str(self.get_parameter("parent_frame_id").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        ros_to_arm_rpy = self.get_parameter("ros_to_arm_rpy").value
        ros_to_arm_xyz = self.get_parameter("ros_to_arm_xyz").value
        if len(ros_to_arm_rpy) != 3 or len(ros_to_arm_xyz) != 3:
            raise ValueError("ros_to_arm_rpy/ros_to_arm_xyz must have exactly 3 elements")

        self.adj_mat = np.array([
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
        
        self.r_adj = xyzrpy2Mat(0, 0, 0, -np.pi, 0, -np.pi / 2)  # 将手柄坐标系调整到与ROS一致
        
        self.ros_to_arm_mat = xyzrpy2Mat(
            float(ros_to_arm_xyz[0]),
            float(ros_to_arm_xyz[1]),
            float(ros_to_arm_xyz[2]),
            float(ros_to_arm_rpy[0]),
            float(ros_to_arm_rpy[1]),
            float(ros_to_arm_rpy[2]),
        )

        # 复用消息对象，降低循环内内存分配
        self.right_tf_msg = TransformStamped()
        self.left_tf_msg = TransformStamped()
        self.right_pose_msg = PoseStamped()
        self.left_pose_msg = PoseStamped()
        self.right_tf_msg.header.frame_id = self.parent_frame_id
        self.right_tf_msg.child_frame_id = "right_controller"
        self.left_tf_msg.header.frame_id = self.parent_frame_id
        self.left_tf_msg.child_frame_id = "left_controller"
        self.right_pose_msg.header.frame_id = self.parent_frame_id
        self.left_pose_msg.header.frame_id = self.parent_frame_id

        self.timer = self.create_timer(1.0 / max(publish_rate_hz, 1.0), self._timer_callback)

    def _correct_to_ros(self, transform):
        
        if transform.shape != (4, 4):
            raise ValueError("Input transform must be a 4x4 numpy array.")

        transform = self.adj_mat @ transform
        """OpenXR → ROS 坐标系转换（X前向、Y左向、Z上向）"""
        transform = np.dot(transform, self.r_adj)
        """将ROS坐标系转换为机械臂末端坐标系"""
        transform = np.dot(transform, self.ros_to_arm_mat)
        return transform

    def _publish_transform(self, transform, pose_pub, tf_msg, pose_msg, now):
        """通用发布函数（左右手柄共用）"""
        translation = transform[:3, 3]
        quat = quaternion_from_matrix(transform)

        tf_msg.header.stamp = now
        tf_msg.transform.translation.x = float(translation[0])
        tf_msg.transform.translation.y = float(translation[1])
        tf_msg.transform.translation.z = float(translation[2])
        tf_msg.transform.rotation.x = float(quat[0])
        tf_msg.transform.rotation.y = float(quat[1])
        tf_msg.transform.rotation.z = float(quat[2])
        tf_msg.transform.rotation.w = float(quat[3])
        self.br.sendTransform(tf_msg)

        # 同时发布 PoseStamped
        pose_msg.header.stamp = now
        pose_msg.pose.position.x = float(translation[0])
        pose_msg.pose.position.y = float(translation[1])
        pose_msg.pose.position.z = float(translation[2])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        pose_pub.publish(pose_msg)

    def _timer_callback(self):
        transformations, _ = self.oculus_reader.get_transformations_and_buttons()
        now = self.get_clock().now().to_msg()

        # Publish each hand independently: the other hand may be asleep or out
        # of tracking, which must not silence the hand actually in use.
        for key, pose_pub, tf_msg, pose_msg, last_attr in (
            ('r', self.right_handle_pose_pub, self.right_tf_msg, self.right_pose_msg, '_last_published_r'),
            ('l', self.left_handle_pose_pub, self.left_tf_msg, self.left_pose_msg, '_last_published_l'),
        ):
            transform = transformations.get(key)
            if transform is None:
                continue
            # Never re-stamp and re-publish a cached frame as if it were fresh:
            # downstream deadman/freshness checks rely on silence on stale data.
            last = getattr(self, last_attr, None)
            if last is not None and np.array_equal(last, transform):
                continue
            setattr(self, last_attr, transform)
            self._publish_transform(
                self._correct_to_ros(transform), pose_pub, tf_msg, pose_msg, now
            )

    def destroy_node(self):
        self.oculus_reader.stop()
        return super().destroy_node()

if __name__ == '__main__':
    rclpy.init(args=None)
    publisher = OculusPublisher()
    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()
