#!/usr/bin/env python3
import math
import os
import tempfile
import time
from typing import List

import numpy as np
import pinocchio as pin
import rclpy
import xacro
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

try:
    import casadi
    from pinocchio import casadi as cpin
except ImportError:
    casadi = None
    cpin = None
    

def xyzrpy_to_mat(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    mat[:3, 3] = np.array([x, y, z])
    return mat


def pose_to_mat(pose: Pose) -> np.ndarray:
    mat = np.eye(4)
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    mat[:3, :3] = Rotation.from_quat(quat).as_matrix()
    mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return mat


def materialize_urdf(model_path: str) -> tuple[str, str]:
    """Return a Pinocchio-readable URDF path and an optional temporary path."""
    if not model_path.endswith(".xacro"):
        return model_path, ""

    document = xacro.process_file(model_path, mappings={"backend": "fake"})
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix="quest_arm_model_", suffix=".urdf"
    )
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as urdf_file:
        urdf_file.write(document.toxml())
    return temporary_path, temporary_path


class ArmIK:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: List[str],
        locked_joints: List[str],
        ee_parent_joint: str,
        ee_frame_name: str,
        tool_pre_rot_rpy: List[float],
        tool_translation_xyz: List[float],
        collision_pairs_flat: List[int],
        w_pos: float,
        w_ori: float,
        w_reg: float,
        w_smooth: float,
        ipopt_max_iter: int,
        ipopt_tol: float,
        enable_visualization: bool,
        viewer_open_browser: bool,
        viewer_model_name: str,
        viewer_target_frame_name: str,
        viewer_axis_length: float,
        viewer_axis_width: float,
    ):
        if len(tool_pre_rot_rpy) != 3 or len(tool_translation_xyz) != 3:
            raise ValueError("Tool rotation and translation must each contain three values.")
        if (
            w_pos <= 0.0
            or w_ori <= 0.0
            or w_reg < 0.0
            or w_smooth < 0.0
            or ipopt_max_iter <= 0
            or ipopt_tol <= 0.0
        ):
            raise ValueError("IK weights, iteration count, and tolerance are invalid.")
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs=package_dirs)
        unique_locked_joints = self._deduplicate_locked_joints(locked_joints)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=unique_locked_joints,
            reference_configuration=np.zeros(self.robot.model.nq),
        )

        first = xyzrpy_to_mat(0.0, 0.0, 0.0, tool_pre_rot_rpy[0], tool_pre_rot_rpy[1], tool_pre_rot_rpy[2])
        second = xyzrpy_to_mat(tool_translation_xyz[0], tool_translation_xyz[1], tool_translation_xyz[2], 0.0, 0.0, 0.0)
        ee_mat = first @ second
        quat = Rotation.from_matrix(ee_mat[:3, :3]).as_quat()  # x y z w
        if not self.reduced_robot.model.existJointName(ee_parent_joint):
            raise ValueError(f"ee_parent_joint '{ee_parent_joint}' is not in the IK model.")

        self.reduced_robot.model.addFrame(
            pin.Frame(
                ee_frame_name,
                self.reduced_robot.model.getJointId(ee_parent_joint),
                pin.SE3(
                    pin.Quaternion(quat[3], quat[0], quat[1], quat[2]),
                    np.array(ee_mat[:3, 3]),
                ),
                pin.FrameType.OP_FRAME,
            )
        )
        # buildReducedRobot creates Data before the custom tool frame exists.
        # Recreate it so frame placements include the newly-added ee frame.
        self.reduced_robot.data = self.reduced_robot.model.createData()

        self.geom_model = self.reduced_robot.collision_model
        geometry_count = len(self.geom_model.geometryObjects)
        for i in range(0, len(collision_pairs_flat), 2):
            a = collision_pairs_flat[i]
            b = collision_pairs_flat[i + 1]
            if a < 0 or b < 0 or a >= geometry_count or b >= geometry_count or a == b:
                raise ValueError(
                    f"Invalid collision pair ({a}, {b}) for {geometry_count} geometry objects."
                )
            self.geom_model.addCollisionPair(pin.CollisionPair(a, b))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.ee_id = self.reduced_robot.model.getFrameId(ee_frame_name)
        self.w_pos = w_pos
        self.w_ori = w_ori
        self.w_reg = w_reg
        self.w_smooth = w_smooth
        self.max_iterations = ipopt_max_iter
        self.solver_tolerance = ipopt_tol
        self.solver_backend = "scipy_least_squares"
        self.opti = None
        if casadi is not None and cpin is not None:
            self._init_casadi_solver(ipopt_max_iter, ipopt_tol)
            self.solver_backend = "casadi_ipopt"

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)
        self.enable_visualization = enable_visualization
        self.vis = None
        self.viewer_target_frame_name = viewer_target_frame_name
        if self.enable_visualization:
            self._init_visualizer(
                open_browser=viewer_open_browser,
                viewer_model_name=viewer_model_name,
                target_frame_name=viewer_target_frame_name,
                axis_length=viewer_axis_length,
                axis_width=viewer_axis_width,
            )

    def _deduplicate_locked_joints(self, locked_joints: List[str]) -> List[str]:
        unique_names: List[str] = []
        seen_joint_ids = set()
        for joint_name in locked_joints:
            if not self.robot.model.existJointName(joint_name):
                continue
            try:
                joint_id = self.robot.model.getJointId(joint_name)
            except Exception:
                continue
            if joint_id <= 0:
                continue
            if joint_id in seen_joint_ids:
                continue
            seen_joint_ids.add(joint_id)
            unique_names.append(joint_name)
        return unique_names

    @property
    def nq(self) -> int:
        return self.reduced_robot.model.nq

    def active_joint_names(self) -> List[str]:
        names = [n for n in self.reduced_robot.model.names if n != "universe"]
        return names

    def sync_state(self, q_current: List[float]) -> None:
        q = np.array(q_current, dtype=float)
        if q.shape[0] == self.nq:
            self.init_data = q
            self.history_data = q

    def solve(self, target_pose: np.ndarray) -> np.ndarray:
        if self.opti is None:
            return self._solve_with_scipy(target_pose)

        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_q_prev, self.history_data)
        self.opti.set_value(self.param_tf, target_pose)
        self.display_target(target_pose)

        sol = self.opti.solve_limited()
        sol_q = np.array(self.opti.value(self.var_q)).reshape(-1)
        self.init_data = sol_q
        self.history_data = sol_q
        self.display_solution(sol_q)
        return sol_q

    def _init_casadi_solver(self, max_iterations: int, tolerance: float) -> None:
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.ctf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        self.error = casadi.Function(
            "error",
            [self.cq, self.ctf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.ee_id].inverse() * cpin.SE3(self.ctf)
                    ).vector
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.param_q_prev = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        error_vec = self.error(self.var_q, self.param_tf)
        total_cost = casadi.sumsqr(self.w_pos * error_vec[:3]) + casadi.sumsqr(
            self.w_ori * error_vec[3:]
        )
        self.opti.minimize(
            total_cost
            + self.w_reg * casadi.sumsqr(self.var_q)
            + self.w_smooth * casadi.sumsqr(self.var_q - self.param_q_prev)
        )
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "print_level": 0,
                    "max_iter": max_iterations,
                    "tol": tolerance,
                },
                "print_time": False,
            },
        )

    def _solve_with_scipy(self, target_pose: np.ndarray) -> np.ndarray:
        target = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        previous = self.history_data.copy()

        def residual(q: np.ndarray) -> np.ndarray:
            pin.framesForwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
            error = np.asarray(
                pin.log6(self.reduced_robot.data.oMf[self.ee_id].inverse() * target).vector
            ).reshape(-1)
            return np.concatenate(
                (
                    self.w_pos * error[:3],
                    self.w_ori * error[3:],
                    math.sqrt(self.w_reg) * q,
                    math.sqrt(self.w_smooth) * (q - previous),
                )
            )

        solution = least_squares(
            residual,
            self.init_data,
            bounds=(
                self.reduced_robot.model.lowerPositionLimit,
                self.reduced_robot.model.upperPositionLimit,
            ),
            max_nfev=max(self.max_iterations, 1),
            ftol=self.solver_tolerance,
            xtol=self.solver_tolerance,
            gtol=self.solver_tolerance,
        )
        sol_q = np.asarray(solution.x).reshape(-1)
        self.init_data = sol_q
        self.history_data = sol_q
        self.display_target(target_pose)
        self.display_solution(sol_q)
        return sol_q

    def check_self_collision(self, q: np.ndarray) -> bool:
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        pin.updateGeometryPlacements(self.reduced_robot.model, self.reduced_robot.data, self.geom_model, self.geometry_data)
        return pin.computeCollisions(self.geom_model, self.geometry_data, False)

    def _init_visualizer(
        self,
        open_browser: bool,
        viewer_model_name: str,
        target_frame_name: str,
        axis_length: float,
        axis_width: float,
    ) -> None:
        import meshcat.geometry as mg
        from pinocchio.visualize import MeshcatVisualizer

        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=open_browser)
        self.vis.loadViewerModel(viewer_model_name)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        frame_axis_positions = (
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        frame_axis_colors = (
            np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        self.vis.viewer[target_frame_name].set_object(
            mg.LineSegments(
                mg.PointsGeometry(position=axis_length * frame_axis_positions, color=frame_axis_colors),
                mg.LineBasicMaterial(linewidth=axis_width, vertexColors=True),
            )
        )

    def display_target(self, target_pose: np.ndarray) -> None:
        if self.vis is not None:
            self.vis.viewer[self.viewer_target_frame_name].set_transform(target_pose)

    def display_solution(self, q: np.ndarray) -> None:
        if self.vis is not None:
            self.vis.display(q)


class ArmIKPoseNode(Node):
    def __init__(self):
        super().__init__("arm_ik_pose_node")

        self.declare_parameter("robot_description_package", "nero_description")
        self.declare_parameter("urdf_relative_path", "urdf/nero.urdf")
        self.declare_parameter("locked_joints", ["joint8"])
        self.declare_parameter("ee_parent_joint", "joint7")
        self.declare_parameter("ee_frame_name", "ee")
        self.declare_parameter("tool_pre_rot_rpy", [-1.57, 0.0, -1.57])
        self.declare_parameter("tool_translation_xyz", [0.0, 0.023, 0.064])
        self.declare_parameter("collision_pairs_flat", [5, 0, 5, 1, 5, 2, 5, 3])
        self.declare_parameter("enable_collision_check", False)

        self.declare_parameter("w_pos", 20.0)
        self.declare_parameter("w_ori", 2.0)
        self.declare_parameter("w_reg", 0.01)
        self.declare_parameter("w_smooth", 2.0)
        self.declare_parameter("ipopt_max_iter", 50)
        self.declare_parameter("ipopt_tol", 1e-4)
        self.declare_parameter("enable_visualization", False)
        self.declare_parameter("viewer_open_browser", True)
        self.declare_parameter("viewer_model_name", "pinocchio")
        self.declare_parameter("viewer_target_frame_name", "ee_target")
        self.declare_parameter("viewer_axis_length", 0.1)
        self.declare_parameter("viewer_axis_width", 10.0)

        self.declare_parameter("pose_stamped_topic", "")
        self.declare_parameter("feedback_joint_topic", "")
        self.declare_parameter("pin_joint_status_topic", "pin_joint_status")
        # NOTE: Empty list default is inferred as BYTE_ARRAY in rclpy.
        # Use string array default to keep YAML STRING_ARRAY override compatible.
        self.declare_parameter("output_joint_names", [""])

        package_name = self.get_parameter("robot_description_package").value
        urdf_rel = self.get_parameter("urdf_relative_path").value
        locked_joints = list(self.get_parameter("locked_joints").value)
        ee_parent_joint = self.get_parameter("ee_parent_joint").value
        ee_frame_name = self.get_parameter("ee_frame_name").value
        tool_pre_rot_rpy = list(self.get_parameter("tool_pre_rot_rpy").value)
        tool_translation_xyz = list(self.get_parameter("tool_translation_xyz").value)
        collision_pairs_flat = [int(v) for v in self.get_parameter("collision_pairs_flat").value]
        if len(collision_pairs_flat) % 2 != 0:
            raise ValueError("collision_pairs_flat length must be even, e.g. [5,0,5,1].")
        enable_collision_check = bool(self.get_parameter("enable_collision_check").value)
        self.declare_parameter("target_frame_id", "base_link")
        self.target_frame_id = str(self.get_parameter("target_frame_id").value).strip()
        if not self.target_frame_id:
            raise ValueError("target_frame_id cannot be empty.")

        w_pos = float(self.get_parameter("w_pos").value)
        w_ori = float(self.get_parameter("w_ori").value)
        w_reg = float(self.get_parameter("w_reg").value)
        w_smooth = float(self.get_parameter("w_smooth").value)
        ipopt_max_iter = int(self.get_parameter("ipopt_max_iter").value)
        ipopt_tol = float(self.get_parameter("ipopt_tol").value)
        enable_visualization = bool(self.get_parameter("enable_visualization").value)
        viewer_open_browser = bool(self.get_parameter("viewer_open_browser").value)
        viewer_model_name = self.get_parameter("viewer_model_name").value
        viewer_target_frame_name = self.get_parameter("viewer_target_frame_name").value
        viewer_axis_length = float(self.get_parameter("viewer_axis_length").value)
        viewer_axis_width = float(self.get_parameter("viewer_axis_width").value)

        pose_stamped_topic = str(self.get_parameter("pose_stamped_topic").value).strip()
        feedback_joint_topic = self.get_parameter("feedback_joint_topic").value
        pin_joint_status_topic = self.get_parameter("pin_joint_status_topic").value

        package_path = get_package_share_directory(package_name)
        model_source_path = os.path.join(package_path, urdf_rel)
        urdf_path, temporary_urdf_path = materialize_urdf(model_source_path)
        try:
            self.ik = ArmIK(
                urdf_path=urdf_path,
                package_dirs=[package_path],
                locked_joints=locked_joints,
                ee_parent_joint=ee_parent_joint,
                ee_frame_name=ee_frame_name,
                tool_pre_rot_rpy=tool_pre_rot_rpy,
                tool_translation_xyz=tool_translation_xyz,
                collision_pairs_flat=(collision_pairs_flat if enable_collision_check else []),
                w_pos=w_pos,
                w_ori=w_ori,
                w_reg=w_reg,
                w_smooth=w_smooth,
                ipopt_max_iter=ipopt_max_iter,
                ipopt_tol=ipopt_tol,
                enable_visualization=enable_visualization,
                viewer_open_browser=viewer_open_browser,
                viewer_model_name=viewer_model_name,
                viewer_target_frame_name=viewer_target_frame_name,
                viewer_axis_length=viewer_axis_length,
                viewer_axis_width=viewer_axis_width,
            )
        finally:
            if temporary_urdf_path:
                os.unlink(temporary_urdf_path)
        dedup_locked = self.ik._deduplicate_locked_joints(locked_joints)
        self.get_logger().info(f"locked_joints(raw)={locked_joints}, locked_joints(dedup)={dedup_locked}")

        output_joint_names = list(self.get_parameter("output_joint_names").value)
        if (
            len(output_joint_names) != self.ik.nq
            or any(not name for name in output_joint_names)
            or len(set(output_joint_names)) != len(output_joint_names)
        ):
            raise ValueError(
                f"output_joint_names must contain exactly {self.ik.nq} unique, non-empty names."
            )
        self.output_joint_names = output_joint_names
        
        self.pub_joint = self.create_publisher(JointState, pin_joint_status_topic, 10)
        self.pub_collision = self.create_publisher(Bool, f"{pin_joint_status_topic}_collision", 10)
        self.enable_collision_check = enable_collision_check
        self.last_solve_warning_time = 0.0

        if pose_stamped_topic:
            self.create_subscription(PoseStamped, pose_stamped_topic, self.pose_stamped_callback, 10)
        if feedback_joint_topic:
            self.create_subscription(JointState, feedback_joint_topic, self.feedback_joint_callback, 10)
        if not pose_stamped_topic:
            raise ValueError("pose_stamped_topic cannot be empty.")

        self.get_logger().info(
            f"IK node ready. model={model_source_path}, input=({pose_stamped_topic}), "
            f"target_frame={self.target_frame_id}, output={pin_joint_status_topic}, "
            f"nq={self.ik.nq}, solver={self.ik.solver_backend}"
        )

    def feedback_joint_callback(self, msg: JointState) -> None:
        if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
            return
        positions_by_name = dict(zip(msg.name, msg.position))
        if not all(name in positions_by_name for name in self.output_joint_names):
            return
        ordered = [positions_by_name[name] for name in self.output_joint_names]
        if np.all(np.isfinite(ordered)):
            self.ik.sync_state(ordered)

    def pose_stamped_callback(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != self.target_frame_id:
            self.get_logger().warning(
                f"Ignoring IK target in frame '{msg.header.frame_id}'; expected '{self.target_frame_id}'."
            )
            return
        pose_values = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=float,
        )
        quaternion_norm = np.linalg.norm(pose_values[3:])
        if not np.all(np.isfinite(pose_values)) or abs(quaternion_norm - 1.0) > 1e-3:
            self.get_logger().warning(
                "Ignoring a non-finite or non-normalized-quaternion IK target."
            )
            return
        self._solve_and_publish(pose_to_mat(msg.pose), stamp=msg.header.stamp)

    def _solve_and_publish(self, target_pose: np.ndarray, stamp) -> None:
        try:
            sol_q = self.ik.solve(target_pose)
            if not np.all(np.isfinite(sol_q)):
                raise ValueError("IK returned a non-finite joint solution.")

            if self.enable_collision_check:
                col_msg = Bool()
                col_msg.data = self.ik.check_self_collision(sol_q)
                self.pub_collision.publish(col_msg)
                if col_msg.data:
                    self.get_logger().warning("IK solution rejected because it is in self-collision.")
                    return

            joint_msg = JointState()
            if stamp.sec == 0 and stamp.nanosec == 0:
                joint_msg.header.stamp = self.get_clock().now().to_msg()
            else:
                joint_msg.header.stamp = stamp
            joint_msg.name = self.output_joint_names
            joint_msg.position = sol_q.tolist()
            self.pub_joint.publish(joint_msg)
        except Exception as e:
            warning_time = time.monotonic()
            if warning_time - self.last_solve_warning_time >= 1.0:
                self.get_logger().warning(f"IK solve failed: {e}")
                self.last_solve_warning_time = warning_time


def main(args=None):
    rclpy.init(args=args)
    node = ArmIKPoseNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
