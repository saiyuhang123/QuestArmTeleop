#!/usr/bin/env bash
set -eo pipefail

# Full Cartesian IK -> Quest bridge -> ros2_control fake backend test.
# No Oculus, CAN interface, vendor SDK, or real hardware is used.
INSTALL_PREFIX="${1:?usage: fake_canopen_arm_ik_e2e.sh <colcon_install_prefix>}"
export ROS_DOMAIN_ID=96
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# shellcheck disable=SC1091
source "$INSTALL_PREFIX/setup.bash"
set -u

TMP_DIR="$(mktemp -d)"
LOG="$TMP_DIR/fake_canopen_arm_ik_e2e.log"
LAUNCH_PID=""
cleanup() {
  if [[ -n "$LAUNCH_PID" ]]; then
    kill -- "-$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
  rm -rf -- "$TMP_DIR"
}
trap cleanup EXIT

setsid ros2 launch oculus_reader canopen_arm_fake_ik.launch.py >"$LOG" 2>&1 &
LAUNCH_PID=$!

sleep 3
if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
  echo "FAIL: fake arm/IK launch exited during startup"
  cat "$LOG"
  exit 1
fi

if ! python3 - <<'PY'
import copy
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


class Probe(Node):
    def __init__(self):
        super().__init__("fake_canopen_arm_ik_probe")
        self.positions = None
        self.pose = None
        self.create_subscription(
            JointState, "/joint_states", self.on_joints, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped,
            "/canopen_arm/tool_pose",
            self.on_pose,
            qos_profile_sensor_data,
        )
        self.target_pub = self.create_publisher(PoseStamped, "/delta_pose", 10)
        self.enable_pub = self.create_publisher(Bool, "/canopen_arm/quest_teleop_enable", 10)

    def on_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(name in values for name in ("j1", "j2", "j3", "j4", "j5", "j6")):
            self.positions = [values[f"j{i}"] for i in range(1, 7)]

    def on_pose(self, msg):
        self.pose = msg


rclpy.init()
node = Probe()
deadline = time.monotonic() + 30.0
while time.monotonic() < deadline and (node.positions is None or node.pose is None):
    rclpy.spin_once(node, timeout_sec=0.1)
if node.positions is None or node.pose is None:
    raise SystemExit("FAIL: initial fake arm feedback was not received")

initial_positions = list(node.positions)
initial_xyz = (
    node.pose.pose.position.x,
    node.pose.pose.position.y,
    node.pose.pose.position.z,
)
target = PoseStamped()
target.header.frame_id = "base_link"
target.pose = copy.deepcopy(node.pose.pose)
target.pose.position.y += 0.015

deadline = time.monotonic() + 25.0
joint_motion = 0.0
tcp_motion = 0.0
while time.monotonic() < deadline:
    target.header.stamp = node.get_clock().now().to_msg()
    node.target_pub.publish(target)
    node.enable_pub.publish(Bool(data=True))
    rclpy.spin_once(node, timeout_sec=0.03)
    if node.positions is not None:
        joint_motion = max(
            joint_motion,
            max(abs(a - b) for a, b in zip(node.positions, initial_positions)),
        )
    if node.pose is not None:
        current_xyz = (
            node.pose.pose.position.x,
            node.pose.pose.position.y,
            node.pose.pose.position.z,
        )
        tcp_motion = max(
            tcp_motion,
            math.sqrt(sum((a - b) ** 2 for a, b in zip(current_xyz, initial_xyz))),
        )
    if joint_motion > 0.003 and tcp_motion > 0.002:
        break

node.enable_pub.publish(Bool(data=False))
node.destroy_node()
rclpy.shutdown()
if joint_motion <= 0.003 or tcp_motion <= 0.002:
    raise SystemExit(
        f"FAIL: IK chain did not move (joint={joint_motion:.6f}, tcp={tcp_motion:.6f})"
    )
print(f"PASS: IK chain moved (joint={joint_motion:.6f} rad, tcp={tcp_motion:.6f} m)")
PY
then
  cat "$LOG"
  exit 1
fi

if grep -Eq "FATAL|process has died|Traceback" "$LOG"; then
  echo "FAIL: launch log contains a fatal error"
  cat "$LOG"
  exit 1
fi

echo "PASS: Cartesian IK -> Quest bridge -> fake arm"
