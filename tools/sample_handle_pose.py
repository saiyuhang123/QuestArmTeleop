#!/usr/bin/env python3
"""Sample /right_handle_pose at 2 Hz with timestamps, for offline calibration."""
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class Sampler(Node):
    def __init__(self):
        super().__init__("handle_pose_sampler")
        self.latest = None
        self.create_subscription(PoseStamped, "/right_handle_pose", self.cb, 1)
        self.t0 = time.monotonic()
        self.create_timer(0.5, self.tick)

    def cb(self, msg):
        self.latest = msg.pose.position

    def tick(self):
        t = time.monotonic() - self.t0
        if self.latest is None:
            print(f"t={t:5.1f}  (无数据)", flush=True)
            return
        p = self.latest
        print(f"t={t:5.1f}  x={p.x:+.4f} y={p.y:+.4f} z={p.z:+.4f}", flush=True)


rclpy.init()
node = Sampler()
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
