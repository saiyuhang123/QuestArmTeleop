#!/usr/bin/env python3
"""Quest 3 teleop over WebXR (local-floor reference space).

Replaces pub_pose.py + pub_delta_pose.py with a single node whose input comes
from the headset browser: this node serves a minimal WebXR page over HTTPS
(https://<host-ip>:8012). The page requests a `local-floor` reference space —
fixed to the room, NOT moving with the headset — and streams the right
controller's grip pose + buttons back over a WebSocket. No APK, no ADB, no
vuer, no internet access needed on the headset.

Publishes the same interface as the APK chain so everything downstream
(arm_ik_pose_node, quest_joint_state_bridge, vendor_pp) is unchanged:
  /right_handle_pose                  PoseStamped  (ROS frame: +X fwd, +Z up)
  /delta_pose                         PoseStamped  (frame_id base_link)
  /canopen_arm/quest_teleop_enable    Bool
  /control/joint_states               JointState   (gripper from trigger)

Requires ~/cert.pem + ~/key.pem (self-signed is fine; open the page once in
the Quest browser and accept the certificate warning).
"""
import asyncio
import json
import os
import ssl
import threading
import time

import numpy as np
import rclpy
from aiohttp import web
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Header

# WebXR local-floor: +Y up, -Z user-forward, +X right (right-handed).
# ROS convention used by this stack: +Z up, +X forward, +Y left.
# Columns are the WebXR basis vectors expressed in the ROS frame.
WEBXR_TO_ROS = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def _parse_matrix16(raw):
    """Accept a 16-list or a msgpack ExtType binary blob (Float32/Float64).
    Returns a 4x4 row-major ndarray, or None when tracking is invalid."""
    if raw is None:
        return None
    try:
        from msgpack import ExtType
    except ImportError:
        ExtType = ()
    if ExtType and isinstance(raw, ExtType):
        data = bytes(raw.data)
        if len(data) == 64:
            return np.frombuffer(data, dtype="<f4").reshape(4, 4).T.astype(float)
        if len(data) == 128:
            return np.frombuffer(data, dtype="<f8").reshape(4, 4).T
        return None  # 1-byte null marker: controller pose not tracked
    try:
        arr = np.array(raw, dtype=float).reshape(4, 4).T  # column-major on the wire
    except (TypeError, ValueError):
        return None
    return arr


class WebXRTeleopNode(Node):
    def __init__(self):
        super().__init__("webxr_teleop_node")

        self.declare_parameter("handle_pose_topic", "/right_handle_pose")
        self.declare_parameter("feedback_tcp_pose_topic", "/feedback/tcp_pose")
        self.declare_parameter("delta_pose_topic", "/delta_pose")
        self.declare_parameter("control_joint_topic", "/control/joint_states")
        self.declare_parameter("enable_topic", "/canopen_arm/quest_teleop_enable")
        self.declare_parameter("delta_pose_frame_id", "base_link")

        self.declare_parameter("gripper_joint_name", "gripper")
        self.declare_parameter("gripper_max_range", 0.07)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("hold_to_run", True)
        self.declare_parameter("handle_timeout_seconds", 0.25)
        self.declare_parameter("tcp_timeout_seconds", 0.50)
        self.declare_parameter("publish_gripper_when_disabled", False)
        # Same purpose as pub_delta_pose.py: keep the published target near the
        # TCP feedback so slow backends never run past the joint-jump guard.
        self.declare_parameter("max_target_offset_m", 0.05)
        # Extra yaw around the ROS Z axis, applied after WEBXR_TO_ROS. The
        # local-floor yaw zero is set by the headset's recenter pose, so this
        # is the one knob to align "controller forward" with base_link +X.
        # Adjustable at runtime: ros2 param set /webxr_teleop_node session_yaw_deg 90.0
        self.declare_parameter("session_yaw_deg", 0.0)
        self.declare_parameter("debug_events", 5)
        # Hand-tremor suppression: EMA low-pass on the handle position stream
        # plus a deadband on the per-press delta. Raw pose still goes to
        # /right_handle_pose for calibration; only the teleop target is smoothed.
        self.declare_parameter("smoothing_alpha", 0.3)
        self.declare_parameter("deadband_m", 0.002)

        handle_pose_topic = str(self.get_parameter("handle_pose_topic").value)
        feedback_tcp_pose_topic = str(self.get_parameter("feedback_tcp_pose_topic").value)
        delta_pose_topic = str(self.get_parameter("delta_pose_topic").value)
        control_joint_topic = str(self.get_parameter("control_joint_topic").value)
        enable_topic = str(self.get_parameter("enable_topic").value)
        self.delta_pose_frame_id = str(self.get_parameter("delta_pose_frame_id").value)

        self.gripper_joint_name = str(self.get_parameter("gripper_joint_name").value)
        self.gripper_max_range = float(self.get_parameter("gripper_max_range").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.hold_to_run = bool(self.get_parameter("hold_to_run").value)
        self.handle_timeout_seconds = float(self.get_parameter("handle_timeout_seconds").value)
        self.tcp_timeout_seconds = float(self.get_parameter("tcp_timeout_seconds").value)
        self.publish_gripper_when_disabled = bool(
            self.get_parameter("publish_gripper_when_disabled").value
        )
        self.max_target_offset_m = float(self.get_parameter("max_target_offset_m").value)
        if self.handle_timeout_seconds <= 0.0 or self.tcp_timeout_seconds <= 0.0:
            raise ValueError("Input timeout parameters must be positive.")

        self._yaw_rot = Rotation.from_euler(
            "z", float(self.get_parameter("session_yaw_deg").value), degrees=True
        ).as_matrix()
        self._debug_remaining = int(self.get_parameter("debug_events").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        if not 0.0 < self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1].")
        self.deadband_m = float(self.get_parameter("deadband_m").value)
        self._pos_ema = None

        self.pub_handle_pose = self.create_publisher(PoseStamped, handle_pose_topic, 10)
        self.pub_delta_pose = self.create_publisher(PoseStamped, delta_pose_topic, 10)
        self.pub_move_j = self.create_publisher(JointState, control_joint_topic, 10)
        self.pub_enable = self.create_publisher(Bool, enable_topic, 10)
        self.create_subscription(PoseStamped, feedback_tcp_pose_topic, self.tcp_pose_callback, 1)

        # Shared with the Vuer asyncio thread; guarded by _lock.
        self._lock = threading.Lock()
        self._handle_pos = None
        self._handle_quat = None
        self._a_pressed = False
        self._b_pressed = False
        self._trigger_value = 0.0
        self._last_handle_time = None

        # TCP feedback state
        self.tcp_pos = None
        self.tcp_quat = None
        self.last_tcp_time = None

        # Engagement state
        self.flag = False
        self.rearm_required = False
        self.start_handle_pos = None
        self.anchor_tcp_pos = None
        self.anchor_tcp_quat = None

        # Diagnostics counters
        self._event_count = 0
        self._pub_count = 0
        self._tick_count = 0
        self._last_pos_log = 0.0
        self._last_buttons_logged = None

        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.create_timer(1.0 / max(control_rate_hz, 1.0), self.control_loop)

        self.get_logger().info(
            f"webxr_teleop ready. handle={handle_pose_topic}, delta={delta_pose_topic}, "
            f"enable={enable_topic}, hold_to_run={self.hold_to_run}, "
            f"max_target_offset_m={self.max_target_offset_m}"
        )

    # ------------------------------------------------------------------ Vuer
    def on_controller_event(self, value: dict):
        mat_info = "None"
        m = _parse_matrix16(value.get("right"))
        if m is not None:
            mat_info = f"pos_xr={[round(float(v), 4) for v in m[:3, 3]]}"
        if self._debug_remaining > 0:
            self._debug_remaining -= 1
            self.get_logger().info(
                f"CONTROLLER_MOVE keys={sorted(value.keys())} right={mat_info} rightState={value.get('rightState')}"
            )
        state = value.get("rightState") or {}

        buttons_snapshot = (
            bool(state.get("aButton")), bool(state.get("bButton")),
            bool(state.get("trigger")), bool(state.get("squeeze")),
            round(float(state.get("triggerValue", 0.0) or 0.0), 2),
        )
        if buttons_snapshot != self._last_buttons_logged:
            self._last_buttons_logged = buttons_snapshot
            self.get_logger().info(
                f"buttons: A={buttons_snapshot[0]} B={buttons_snapshot[1]} "
                f"trigger={buttons_snapshot[2]}({buttons_snapshot[4]}) squeeze={buttons_snapshot[3]}"
            )
        if m is not None:
            p = m[:3, 3]
            now = time.monotonic()
            if np.abs(p).max() > 1e-4 and now - self._last_pos_log > 0.5:
                self._last_pos_log = now
                self.get_logger().info(f"pos_xr nonzero: {[round(float(v), 4) for v in p]}")

        pos = quat = None
        if m is not None:
            fix = self._yaw_rot @ WEBXR_TO_ROS
            pos = fix @ m[:3, 3]
            rot = fix @ m[:3, :3] @ WEBXR_TO_ROS.T
            quat = Rotation.from_matrix(rot).as_quat()

        with self._lock:
            self._event_count += 1
            if pos is not None:
                if self._pos_ema is None:
                    self._pos_ema = pos
                else:
                    a = self.smoothing_alpha
                    self._pos_ema = a * pos + (1.0 - a) * self._pos_ema
                self._handle_pos = pos          # raw, for /right_handle_pose
                self._handle_quat = quat
            self._a_pressed = bool(state.get("aButton", False))
            self._b_pressed = bool(state.get("bButton", False))
            self._trigger_value = float(state.get("triggerValue", 0.0) or 0.0)
            self._last_handle_time = time.monotonic()

    # ------------------------------------------------------------------ ROS
    def tcp_pose_callback(self, msg: PoseStamped):
        self.tcp_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        self.tcp_quat = [
            msg.pose.orientation.x, msg.pose.orientation.y,
            msg.pose.orientation.z, msg.pose.orientation.w,
        ]
        self.last_tcp_time = time.monotonic()

    def _on_set_parameters(self, params):
        for param in params:
            if param.name == "session_yaw_deg":
                self._yaw_rot = Rotation.from_euler(
                    "z", float(param.value), degrees=True
                ).as_matrix()
                self.get_logger().info(f"session_yaw_deg -> {float(param.value)}")
            elif param.name == "smoothing_alpha" and 0.0 < float(param.value) <= 1.0:
                self.smoothing_alpha = float(param.value)
                self.get_logger().info(f"smoothing_alpha -> {self.smoothing_alpha}")
            elif param.name == "deadband_m" and float(param.value) >= 0.0:
                self.deadband_m = float(param.value)
                self.get_logger().info(f"deadband_m -> {self.deadband_m}")
        return SetParametersResult(successful=True)

    def _inputs_fresh(self) -> bool:
        now = time.monotonic()
        return (
            self._last_handle_time is not None
            and self.last_tcp_time is not None
            and now - self._last_handle_time <= self.handle_timeout_seconds
            and now - self.last_tcp_time <= self.tcp_timeout_seconds
        )

    def _publish_enable(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self.pub_enable.publish(msg)

    def _publish_handle_pose(self, pos, quat) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "quest_world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
            float(pos[0]), float(pos[1]), float(pos[2]),
        )
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = (
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]),
        )
        self.pub_handle_pose.publish(msg)

    def _activate(self, pos) -> None:
        # Anchor in the room-fixed frame: position deltas rotate into base_link
        # by the fixed WEBXR_TO_ROS (+ session yaw) conversion already applied
        # at ingest. Orientation held at the engagement TCP orientation.
        self.start_handle_pos = pos.copy()
        self.anchor_tcp_pos = self.tcp_pos.copy()
        self.anchor_tcp_quat = list(self.tcp_quat)
        self.flag = True
        self.get_logger().info("开始遥操作")

    def _deactivate(self, reason: str) -> None:
        if self.flag:
            self.get_logger().info(f"停止遥操作: {reason}")
        self.flag = False
        self.start_handle_pos = None

    def control_loop(self):
        with self._lock:
            pos = None if self._handle_pos is None else self._handle_pos.copy()
            smooth = None if self._pos_ema is None else self._pos_ema.copy()
            quat = self._handle_quat
            start_pressed = self._a_pressed
            stop_pressed = self._b_pressed
            trigger_value = self._trigger_value

        if pos is not None and quat is not None:
            self._pub_count += 1
            self._publish_handle_pose(pos, quat)

        self._tick_count += 1
        if self._tick_count % 150 == 0:
            self.get_logger().info(
                f"stats: events={self._event_count} handle_pub={self._pub_count} "
                f"pos={'set' if pos is not None else 'NONE'} engaged={self.flag}"
            )

        inputs_fresh = self._inputs_fresh() and pos is not None and self.tcp_pos is not None

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
                self._activate(smooth)
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
                self._activate(smooth)

        enabled = self.flag and inputs_fresh
        self._publish_enable(enabled)

        if enabled or self.publish_gripper_when_disabled:
            gripper_value = max(0.0, min(trigger_value, 1.0)) * self.gripper_max_range
            gripper_msg = JointState()
            gripper_msg.header = Header()
            gripper_msg.header.stamp = self.get_clock().now().to_msg()
            gripper_msg.name = [self.gripper_joint_name]
            gripper_msg.position = [gripper_value]
            self.pub_move_j.publish(gripper_msg)

        if not enabled:
            return
        if self.start_handle_pos is None or self.anchor_tcp_pos is None:
            return
        if smooth is None:
            return

        delta = smooth - self.start_handle_pos
        norm = float(np.linalg.norm(delta))
        if norm <= self.deadband_m:
            delta = np.zeros(3)
        elif self.deadband_m > 0.0:
            delta = delta * ((norm - self.deadband_m) / norm)
        xyz = self._clamp_target_to_tcp((self.anchor_tcp_pos + delta).tolist())
        quat_msg = self.anchor_tcp_quat
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.delta_pose_frame_id
        pose_msg.pose.position.x = float(xyz[0])
        pose_msg.pose.position.y = float(xyz[1])
        pose_msg.pose.position.z = float(xyz[2])
        pose_msg.pose.orientation.x = float(quat_msg[0])
        pose_msg.pose.orientation.y = float(quat_msg[1])
        pose_msg.pose.orientation.z = float(quat_msg[2])
        pose_msg.pose.orientation.w = float(quat_msg[3])
        self.pub_delta_pose.publish(pose_msg)

    def _clamp_target_to_tcp(self, xyz):
        if self.tcp_pos is None or self.max_target_offset_m <= 0.0:
            return xyz
        offset = np.array(
            [float(xyz[0]) - self.tcp_pos[0],
             float(xyz[1]) - self.tcp_pos[1],
             float(xyz[2]) - self.tcp_pos[2]]
        )
        distance = float(np.linalg.norm(offset))
        if distance <= self.max_target_offset_m:
            return xyz
        clamped = self.tcp_pos + offset / distance * self.max_target_offset_m
        return [float(v) for v in clamped]


WEBXR_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>WebXR Teleop</title>
<style>body{font-family:sans-serif;text-align:center;padding-top:40px}
button{font-size:2em;padding:0.6em 1.2em}#st{margin-top:1em;color:#666}</style>
</head><body>
<h2>Quest WebXR Teleop (local-floor)</h2>
<button id="enter">Enter VR</button>
<div id="st">connecting...</div>
<script>
const ws = new WebSocket("wss://" + location.host + "/ws");
const st = document.getElementById("st");
ws.onopen = () => st.textContent = "ws connected";
ws.onclose = () => st.textContent = "ws closed";
ws.onerror = () => st.textContent = "ws error";

document.getElementById("enter").onclick = async () => {
  try {
    const session = await navigator.xr.requestSession("immersive-vr",
      {requiredFeatures: ["local-floor"]});
    const refSpace = await session.requestReferenceSpace("local-floor");
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl");
    session.updateRenderState({baseLayer: new XRWebGLLayer(session, gl)});
    st.textContent = "in VR (local-floor)";
    session.requestAnimationFrame(function onFrame(t, frame) {
      if (session) session.requestAnimationFrame(onFrame);
      if (ws.readyState !== 1) return;
      const out = {};
      for (const src of session.inputSources) {
        if (src.handedness !== "right") continue;
        if (src.gripSpace) {
          const pose = frame.getPose(src.gripSpace, refSpace);
          if (pose) out.right = Array.from(pose.transform.matrix);
        }
        if (src.gamepad) {
          const b = src.gamepad.buttons, ax = src.gamepad.axes;
          out.rightState = {
            trigger: !!(b[0] && b[0].pressed),
            squeeze: !!(b[1] && b[1].pressed),
            touchpad: !!(b[2] && b[2].pressed),
            thumbstick: !!(b[3] && b[3].pressed),
            aButton: !!(b[4] && b[4].pressed),
            bButton: !!(b[5] && b[5].pressed),
            triggerValue: b[0] ? b[0].value : 0,
            squeezeValue: b[1] ? b[1].value : 0,
            thumbstickValue: [ax[2] || 0, ax[3] || 0],
          };
        }
      }
      ws.send(JSON.stringify(out));
    });
    session.addEventListener("end", () => st.textContent = "session ended");
  } catch (e) { st.textContent = "ERR: " + e; }
};
</script></body></html>
"""


def _free_port(port: int, logger) -> None:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) != 0:
            return
    logger.warn(f"port {port} is already bound; killing the stale process (leftover server instance)")
    from killport import kill_ports
    kill_ports(ports=[port])
    time.sleep(1.0)


def _run_web_server(node, port: int = 8012) -> None:
    async def index(request):
        return web.Response(text=WEBXR_PAGE, content_type="text/html")

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        node.get_logger().info("headset websocket connected")
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    node.on_controller_event(json.loads(msg.data))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
        node.get_logger().info("headset websocket closed")
        return ws

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(os.path.expanduser("~/cert.pem"), os.path.expanduser("~/key.pem"))
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    web.run_app(app, host="0.0.0.0", port=port, ssl_context=ssl_ctx,
                print=None, handle_signals=False)


def main(args=None):
    rclpy.init(args=args)
    node = WebXRTeleopNode()

    _free_port(8012, node.get_logger())

    server_thread = threading.Thread(target=_run_web_server, args=(node,), daemon=True)
    server_thread.start()

    node.get_logger().info("Quest page: https://<this-host-ip>:8012 (accept cert, then Enter VR)")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_enable(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
