#!/usr/bin/env python3
"""Session-yaw calibration: hold A and push the controller toward the arm's
physical front. Prints the session_yaw_deg value to apply.

Usage: python3 ~/measure_session_yaw.py --current <current session_yaw_deg>
"""
import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

parser = argparse.ArgumentParser()
parser.add_argument("--current", type=float, required=True,
                    help="current session_yaw_deg of webxr_teleop_node")
args = parser.parse_args()


class Measurer(Node):
    def __init__(self):
        super().__init__("measure_session_yaw")
        self.anchor_xy = None
        self.last_time = None
        self.done = False
        self.create_subscription(PoseStamped, "/delta_pose", self.cb, 1)

    def cb(self, msg: PoseStamped):
        if self.done:
            return
        now = time.monotonic()
        xy = np.array([msg.pose.position.x, msg.pose.position.y])
        # A gap in the /delta_pose stream means a fresh press (re-anchor).
        if self.last_time is None or now - self.last_time > 0.3:
            self.anchor_xy = xy
        self.last_time = now
        if self.anchor_xy is None:
            return
        d = xy - self.anchor_xy
        if float(np.linalg.norm(d)) < 0.03:
            return
        self.done = True
        phi = float(np.degrees(np.arctan2(d[1], d[0])))
        suggested = args.current - phi
        suggested = (suggested + 180.0) % 360.0 - 180.0
        print(f"\n目标在 base 系的方向: phi = {phi:+.1f} deg  "
              f"(0=正前+x, +90=左, 180/-180=后, -90=右)")
        print(f"如果你刚才是朝机械臂正前方推的, 建议:")
        print(f"  ros2 param set /webxr_teleop_node session_yaw_deg {suggested:.1f}")
        print("改完后松手、重新按住 A 再前推一次验证(重跑本脚本, phi 应接近 0)")


rclpy.init()
node = Measurer()
print("等待: 按住 A, 朝机械臂正前方慢推 ~5cm ...")
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
