#!/usr/bin/env python3
import json
import cv2
import signal
from pathlib import Path
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage, JointState, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

import tf2_ros
from tf_transformations import euler_from_quaternion


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def quat_to_rpy(x, y, z, w):
    roll, pitch, yaw = euler_from_quaternion([x, y, z, w])
    return roll, pitch, yaw


def make_sensor_qos(depth: int = 5) -> QoSProfile:
    """
    Camera, depth and PointCloud2 topics from RealSense / Orbbec often use BEST_EFFORT QoS.
    If the subscriber uses the default RELIABLE QoS, messages may not arrive.
    """
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class EpisodeBuffer:
    def __init__(self):
        self.timestamps: List[float] = []
        self.joint_pos: List[np.ndarray] = []
        self.eef_pose: List[np.ndarray] = []
        self.gripper_state: List[np.ndarray] = []

        # Main camera, normally wrist/front camera: /camera/color/image_raw
        self.rgb_frames: List[np.ndarray] = []

        # Environment camera, normally Astra: /astra/color/image_raw
        self.env_rgb_frames: List[np.ndarray] = []

        # Main camera depth image: /camera/depth/image_raw
        self.depth_frames: List[np.ndarray] = []

        # Environment camera depth image: /astra/depth/image_raw
        self.env_depth_frames: List[np.ndarray] = []

        # Optional fixed-size point cloud frames, shape per frame = [num_points, 3]
        self.pointcloud_frames: List[np.ndarray] = []

    def clear(self):
        self.timestamps.clear()
        self.joint_pos.clear()
        self.eef_pose.clear()
        self.gripper_state.clear()
        self.rgb_frames.clear()
        self.env_rgb_frames.clear()
        self.depth_frames.clear()
        self.env_depth_frames.clear()
        self.pointcloud_frames.clear()

    def __len__(self):
        return len(self.timestamps)


class FR5DPMultimodalRecorderNode(Node):
    def __init__(self):
        super().__init__("fr5_dp_multimodal_recorder_node")

        # ======================
        # Parameters
        # ======================
        self.declare_parameter("save_root", "/home/xjtlu/xbox_control/fr5_dp_data_pile")

        # RGB topics
        self.declare_parameter("image_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("env_image_topic", "/env_camera/color/image_raw")

        # Depth image topics. These are image-like depth maps, not point clouds.
        self.declare_parameter("depth_topic", "/camera/depth/image_raw/compressedDepth")
        self.declare_parameter("env_depth_topic", "/env_camera/depth/image_raw")

        # Optional point cloud topic. Keep this if you still want to store xyz point cloud as backup.
        self.declare_parameter("pointcloud_topic", "/camera/depth/points")

        self.declare_parameter("joint_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("eef_frame", "wrist3_link")

        self.declare_parameter("sample_rate", 10.0)
        self.declare_parameter("motion_threshold", 0.002)
        self.declare_parameter("gripper_threshold", 0.01)
        self.declare_parameter("idle_timeout_sec", 5.0)
        self.declare_parameter("min_episode_len", 20)

        # Depth / point cloud saving control
        self.declare_parameter("require_env_image", True)
        self.declare_parameter("require_depth", True)
        self.declare_parameter("require_env_depth", True)
        self.declare_parameter("save_pointcloud", False)
        self.declare_parameter("require_pointcloud", False)
        self.declare_parameter("num_points", 4096)

        # Resize switches
        self.declare_parameter("resize_rgb", True)
        self.declare_parameter("resize_env_rgb", True)
        self.declare_parameter("resize_depth", True)
        self.declare_parameter("resize_env_depth", True)

        # Unified output size
        self.declare_parameter("image_width", 320)
        self.declare_parameter("image_height", 240)
        self.declare_parameter("depth_width", 320)
        self.declare_parameter("depth_height", 240)


        self.save_root = Path(self.get_parameter("save_root").value)

        self.image_topic = self.get_parameter("image_topic").value
        self.env_image_topic = self.get_parameter("env_image_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.env_depth_topic = self.get_parameter("env_depth_topic").value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.joint_topic = self.get_parameter("joint_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.eef_frame = self.get_parameter("eef_frame").value

        self.sample_rate = float(self.get_parameter("sample_rate").value)
        self.motion_threshold = float(self.get_parameter("motion_threshold").value)
        self.gripper_threshold = float(self.get_parameter("gripper_threshold").value)
        self.idle_timeout_sec = float(self.get_parameter("idle_timeout_sec").value)
        self.min_episode_len = int(self.get_parameter("min_episode_len").value)

        self.require_env_image = bool(self.get_parameter("require_env_image").value)
        self.require_depth = bool(self.get_parameter("require_depth").value)
        self.require_env_depth = bool(self.get_parameter("require_env_depth").value)
        self.save_pointcloud = bool(self.get_parameter("save_pointcloud").value)
        self.require_pointcloud = bool(self.get_parameter("require_pointcloud").value)
        self.num_points = int(self.get_parameter("num_points").value)

        self.resize_rgb = bool(self.get_parameter("resize_rgb").value)
        self.resize_env_rgb = bool(self.get_parameter("resize_env_rgb").value)
        self.resize_depth = bool(self.get_parameter("resize_depth").value)
        self.resize_env_depth = bool(self.get_parameter("resize_env_depth").value)

        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.depth_width = int(self.get_parameter("depth_width").value)
        self.depth_height = int(self.get_parameter("depth_height").value)

        ensure_dir(self.save_root)
        self.bridge = CvBridge()
        self.sensor_qos = make_sensor_qos(depth=5)

        # ======================
        # Latest cache
        # ======================
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_rgb_stamp: Optional[float] = None
        self.latest_rgb_frame_id: Optional[str] = None

        self.latest_env_rgb: Optional[np.ndarray] = None
        self.latest_env_rgb_stamp: Optional[float] = None
        self.latest_env_rgb_frame_id: Optional[str] = None

        self.latest_depth: Optional[np.ndarray] = None
        self.latest_depth_stamp: Optional[float] = None
        self.latest_depth_frame_id: Optional[str] = None
        self.latest_depth_encoding: Optional[str] = None

        self.latest_env_depth: Optional[np.ndarray] = None
        self.latest_env_depth_stamp: Optional[float] = None
        self.latest_env_depth_frame_id: Optional[str] = None
        self.latest_env_depth_encoding: Optional[str] = None

        self.latest_pointcloud: Optional[np.ndarray] = None
        self.latest_pointcloud_stamp: Optional[float] = None
        self.latest_pointcloud_frame_id: Optional[str] = None

        self.latest_joint: Optional[np.ndarray] = None
        self.latest_gripper: Optional[float] = None

        self.prev_eef_pose_for_motion: Optional[np.ndarray] = None

        # Per-sample sensor timestamps. These are useful for checking loose synchronization.
        self.rgb_timestamps: List[float] = []
        self.env_rgb_timestamps: List[float] = []
        self.depth_timestamps: List[float] = []
        self.env_depth_timestamps: List[float] = []
        self.pointcloud_timestamps: List[float] = []

        # ======================
        # Episode state
        # ======================
        self.current_episode = EpisodeBuffer()
        self.recording = False
        self.last_motion_time = None
        self.episode_idx = self._find_next_episode_idx()

        # ======================
        # TF
        # ======================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ======================
        # Subscriptions
        # ======================
        self.create_subscription(CompressedImage, self.image_topic, self.image_callback, self.sensor_qos)
        self.create_subscription(Image, self.env_image_topic, self.env_image_callback, self.sensor_qos)
        self.create_subscription(CompressedImage, self.depth_topic, self.depth_callback, self.sensor_qos)
        self.create_subscription(Image, self.env_depth_topic, self.env_depth_callback, self.sensor_qos)

        if self.save_pointcloud or self.require_pointcloud:
            self.create_subscription(PointCloud2, self.pointcloud_topic, self.pointcloud_callback, self.sensor_qos)

        self.create_subscription(JointState, self.joint_topic, self.joint_callback, 10)

        self.timer = self.create_timer(1.0 / self.sample_rate, self.sample_timer_callback)

        self.get_logger().info("FR5 multimodal DP recorder node started.")
        self.get_logger().info(f"Main image topic: {self.image_topic}")
        self.get_logger().info(f"Environment image topic: {self.env_image_topic}")
        self.get_logger().info(f"Main depth topic: {self.depth_topic}")
        self.get_logger().info(f"Environment depth topic: {self.env_depth_topic}")
        self.get_logger().info(f"Save pointcloud: {self.save_pointcloud}, topic: {self.pointcloud_topic}")
        self.get_logger().info(f"Joint topic: {self.joint_topic}")
        self.get_logger().info(f"TF: {self.base_frame} -> {self.eef_frame}")

    def _find_next_episode_idx(self) -> int:
        if not self.save_root.exists():
            return 1
        indices = []
        for p in self.save_root.iterdir():
            if p.is_dir() and p.name.startswith("episode_"):
                try:
                    indices.append(int(p.name.split("_")[1]))
                except Exception:
                    pass
        return max(indices) + 1 if indices else 1

    @staticmethod
    def _stamp_to_sec(msg) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def image_callback(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                raise RuntimeError("cv2.imdecode failed for compressed RGB image")

            if self.resize_rgb:
                img = cv2.resize(
                    img,
                    (self.image_width, self.image_height),
                    interpolation=cv2.INTER_AREA
                )

            self.latest_rgb = img.astype(np.uint8)
            self.latest_rgb_stamp = self._stamp_to_sec(msg)
            self.latest_rgb_frame_id = msg.header.frame_id

        except Exception as e:
            self.get_logger().warn(f"Main compressed image conversion failed: {e}")

    def env_image_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.resize_env_rgb:
                img = cv2.resize(
                    img,
                    (self.image_width, self.image_height),
                    interpolation=cv2.INTER_AREA
                )
            self.latest_env_rgb = img
            self.latest_env_rgb_stamp = self._stamp_to_sec(msg)
            self.latest_env_rgb_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"Environment image conversion failed: {e}")

    def _convert_depth_msg(self, msg: Image) -> np.ndarray:
        """
        Convert ROS depth image to a float32 depth map.

        - 16UC1 is commonly depth in millimeters, so it is converted to meters.
        - 32FC1 is commonly depth in meters, so it is kept as float32.
        - Other encodings are passed through and converted to float32 without unit scaling.
        """
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)

        if msg.encoding == "16UC1" or depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)

        # Replace inf with nan first, then fill nan with 0.0 for stable saving/training.
        depth[~np.isfinite(depth)] = np.nan
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth

    def depth_callback(self, msg: CompressedImage):
        try:
            raw = np.frombuffer(msg.data, np.uint8)

            # compressedDepth 前面有额外 header，真正的 PNG 从 PNG signature 开始。
            png_signature = np.array(
                [137, 80, 78, 71, 13, 10, 26, 10],
                dtype=np.uint8
            )

            start_idx = -1
            max_search = min(64, len(raw) - len(png_signature))
            for i in range(max_search):
                if np.array_equal(raw[i:i + len(png_signature)], png_signature):
                    start_idx = i
                    break

            if start_idx < 0:
                raise RuntimeError("PNG signature not found in compressedDepth data")

            depth_raw = cv2.imdecode(raw[start_idx:], cv2.IMREAD_UNCHANGED)

            if depth_raw is None:
                raise RuntimeError("cv2.imdecode failed for compressedDepth image")

            if "16UC1" in msg.format or depth_raw.dtype == np.uint16:
                depth = depth_raw.astype(np.float32) / 1000.0
            else:
                depth = depth_raw.astype(np.float32)

            depth[~np.isfinite(depth)] = np.nan
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            if self.resize_depth:
                depth = cv2.resize(
                    depth,
                    (self.depth_width, self.depth_height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.float32)

            self.latest_depth = depth
            self.latest_depth_stamp = self._stamp_to_sec(msg)
            self.latest_depth_frame_id = msg.header.frame_id
            self.latest_depth_encoding = msg.format

        except Exception as e:
            self.get_logger().warn(f"Main compressed depth conversion failed: {e}")

    def env_depth_callback(self, msg: Image):
        try:
            depth = self._convert_depth_msg(msg)
            if self.resize_env_depth:
                depth = cv2.resize(
                    depth,
                    (self.depth_width, self.depth_height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.float32)
            self.latest_env_depth = depth
            self.latest_env_depth_stamp = self._stamp_to_sec(msg)
            self.latest_env_depth_frame_id = msg.header.frame_id
            self.latest_env_depth_encoding = msg.encoding
        except Exception as e:
            self.get_logger().warn(f"Environment depth conversion failed: {e}")

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            points_iter = pc2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            )
            points = np.asarray(list(points_iter), dtype=np.float32)
            if points.ndim != 2 or points.shape[0] == 0:
                return

            if points.dtype.fields is not None:
                points = np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float32)

            points = self._fixed_size_pointcloud(points, self.num_points)
            self.latest_pointcloud = points
            self.latest_pointcloud_stamp = self._stamp_to_sec(msg)
            self.latest_pointcloud_frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f"PointCloud2 conversion failed: {e}")

    def _fixed_size_pointcloud(self, points: np.ndarray, n: int) -> np.ndarray:
        if points.shape[0] >= n:
            idx = np.linspace(0, points.shape[0] - 1, n).astype(np.int64)
            return points[idx, :3].astype(np.float32)

        pad_count = n - points.shape[0]
        pad = np.repeat(points[-1:, :3], pad_count, axis=0)
        return np.concatenate([points[:, :3], pad], axis=0).astype(np.float32)

    def joint_callback(self, msg: JointState):
        if len(msg.position) >= 6:
            self.latest_joint = np.array(msg.position[:6], dtype=np.float32)
            if len(msg.position) >= 7:
                self.latest_gripper = float(msg.position[6])
            else:
                self.latest_gripper = 0.0

    def get_current_eef_pose(self) -> Optional[np.ndarray]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.eef_frame,
                rclpy.time.Time(),
            )
            t = tf_msg.transform.translation
            q = tf_msg.transform.rotation
            roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
            return np.array([t.x, t.y, t.z, roll, pitch, yaw], dtype=np.float32)
        except Exception:
            return None

    def is_motion_detected(self, eef_pose: np.ndarray) -> bool:
        motion = False
        if self.prev_eef_pose_for_motion is not None:
            diff = np.abs(eef_pose - self.prev_eef_pose_for_motion)
            if float(np.sum(diff)) > self.motion_threshold:
                motion = True
        self.prev_eef_pose_for_motion = eef_pose.copy()
        return motion

    def _ready_to_sample(self) -> bool:
        self.get_logger().debug(
            f"rgb:{self.latest_rgb is not None} joint:{self.latest_joint is not None} "
            f"env_rgb:{self.latest_env_rgb is not None if self.require_env_image else True} ..."
        )
        if self.latest_rgb is None or self.latest_joint is None:
            return False
        if self.require_env_image and self.latest_env_rgb is None:
            return False
        if self.require_depth and self.latest_depth is None:
            return False
        if self.require_env_depth and self.latest_env_depth is None:
            return False
        if self.require_pointcloud and self.latest_pointcloud is None:
            return False
        return True

    def _clear_sensor_timestamp_buffers(self):
        self.rgb_timestamps.clear()
        self.env_rgb_timestamps.clear()
        self.depth_timestamps.clear()
        self.env_depth_timestamps.clear()
        self.pointcloud_timestamps.clear()

    def sample_timer_callback(self):
        if not self._ready_to_sample():
            return

        eef_pose = self.get_current_eef_pose()
        if eef_pose is None:
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        moving = self.is_motion_detected(eef_pose)

        if not self.recording and moving:
            self.recording = True
            self.last_motion_time = now_sec
            self.current_episode.clear()
            self._clear_sensor_timestamp_buffers()
            self.get_logger().info(f"Start recording episode_{self.episode_idx:06d}")

        if self.recording:
            self.current_episode.timestamps.append(now_sec)
            self.current_episode.joint_pos.append(self.latest_joint.copy())
            self.current_episode.eef_pose.append(eef_pose.copy())
            self.current_episode.gripper_state.append(np.array([self.latest_gripper], dtype=np.float32))

            self.current_episode.rgb_frames.append(self.latest_rgb.copy())
            self.rgb_timestamps.append(float(self.latest_rgb_stamp or 0.0))

            if self.latest_env_rgb is not None:
                self.current_episode.env_rgb_frames.append(self.latest_env_rgb.copy())
                self.env_rgb_timestamps.append(float(self.latest_env_rgb_stamp or 0.0))
            else:
                self.current_episode.env_rgb_frames.append(np.zeros_like(self.latest_rgb, dtype=np.uint8))
                self.env_rgb_timestamps.append(0.0)

            if self.latest_depth is not None:
                self.current_episode.depth_frames.append(self.latest_depth.copy())
                self.depth_timestamps.append(float(self.latest_depth_stamp or 0.0))
            else:
                # Placeholder only used when require_depth=False.
                h, w = self.latest_rgb.shape[:2]
                self.current_episode.depth_frames.append(np.zeros((h, w), dtype=np.float32))
                self.depth_timestamps.append(0.0)

            if self.latest_env_depth is not None:
                self.current_episode.env_depth_frames.append(self.latest_env_depth.copy())
                self.env_depth_timestamps.append(float(self.latest_env_depth_stamp or 0.0))
            else:
                h, w = self.latest_env_rgb.shape[:2] if self.latest_env_rgb is not None else self.latest_rgb.shape[:2]
                self.current_episode.env_depth_frames.append(np.zeros((h, w), dtype=np.float32))
                self.env_depth_timestamps.append(0.0)

            if self.save_pointcloud:
                if self.latest_pointcloud is not None:
                    self.current_episode.pointcloud_frames.append(self.latest_pointcloud.copy())
                    self.pointcloud_timestamps.append(float(self.latest_pointcloud_stamp or 0.0))
                else:
                    self.current_episode.pointcloud_frames.append(np.zeros((self.num_points, 3), dtype=np.float32))
                    self.pointcloud_timestamps.append(0.0)

            if moving:
                self.last_motion_time = now_sec

            if self.last_motion_time is not None and (now_sec - self.last_motion_time) > self.idle_timeout_sec:
                self.finish_current_episode()

    def finish_current_episode(self):
        if not self.recording:
            return

        ep_len = len(self.current_episode)
        if ep_len < self.min_episode_len:
            self.get_logger().warn(f"Episode too short ({ep_len}), discarded.")
            self.current_episode.clear()
            self._clear_sensor_timestamp_buffers()
            self.recording = False
            return

        ep_dir = self.save_root / f"episode_{self.episode_idx:06d}"
        ensure_dir(ep_dir)

        timestamps = np.array(self.current_episode.timestamps, dtype=np.float64)
        joint_pos = np.stack(self.current_episode.joint_pos, axis=0)
        eef_pose = np.stack(self.current_episode.eef_pose, axis=0)
        gripper_state = np.stack(self.current_episode.gripper_state, axis=0)
        rgb_frames = np.stack(self.current_episode.rgb_frames, axis=0)
        env_rgb_frames = np.stack(self.current_episode.env_rgb_frames, axis=0)
        depth_frames = np.stack(self.current_episode.depth_frames, axis=0)
        env_depth_frames = np.stack(self.current_episode.env_depth_frames, axis=0)

        pointcloud_frames = None
        if self.save_pointcloud and len(self.current_episode.pointcloud_frames) > 0:
            pointcloud_frames = np.stack(self.current_episode.pointcloud_frames, axis=0)

        # action = adjacent EEF pose delta + gripper delta
        delta_eef = eef_pose[1:] - eef_pose[:-1]
        delta_gripper = gripper_state[1:] - gripper_state[:-1]
        action_delta = np.concatenate([delta_eef, delta_gripper], axis=1)

        # Observations are aligned with action, so save T-1 observations.
        np.save(ep_dir / "timestamps.npy", timestamps[:-1])
        np.save(ep_dir / "joint_pos.npy", joint_pos[:-1])
        np.save(ep_dir / "eef_pose.npy", eef_pose[:-1])
        np.save(ep_dir / "gripper_state.npy", gripper_state[:-1])
        np.save(ep_dir / "rgb_frames.npy", rgb_frames[:-1])
        np.save(ep_dir / "env_rgb_frames.npy", env_rgb_frames[:-1])
        np.save(ep_dir / "depth_frames.npy", depth_frames[:-1])
        np.save(ep_dir / "env_depth_frames.npy", env_depth_frames[:-1])
        np.save(ep_dir / "action_delta.npy", action_delta)

        np.save(ep_dir / "rgb_timestamps.npy", np.array(self.rgb_timestamps[:-1], dtype=np.float64))
        np.save(ep_dir / "env_rgb_timestamps.npy", np.array(self.env_rgb_timestamps[:-1], dtype=np.float64))
        np.save(ep_dir / "depth_timestamps.npy", np.array(self.depth_timestamps[:-1], dtype=np.float64))
        np.save(ep_dir / "env_depth_timestamps.npy", np.array(self.env_depth_timestamps[:-1], dtype=np.float64))

        if pointcloud_frames is not None:
            np.save(ep_dir / "pointcloud_frames.npy", pointcloud_frames[:-1])
            np.save(ep_dir / "pointcloud_timestamps.npy", np.array(self.pointcloud_timestamps[:-1], dtype=np.float64))

        meta = {
            "episode_idx": self.episode_idx,
            "num_steps_raw": int(ep_len),
            "num_steps_saved": int(len(action_delta)),
            "action_definition": "[dx,dy,dz,droll,dpitch,dyaw,dgripper]",
            "base_frame": self.base_frame,
            "eef_frame": self.eef_frame,
            "sample_rate": self.sample_rate,
            "topics": {
                "image_topic": self.image_topic,
                "env_image_topic": self.env_image_topic,
                "depth_topic": self.depth_topic,
                "env_depth_topic": self.env_depth_topic,
                "pointcloud_topic": self.pointcloud_topic if self.save_pointcloud else None,
                "joint_topic": self.joint_topic,
            },
            "resize": {
                "rgb": {
                    "enabled": self.resize_rgb,
                    "width": self.image_width,
                    "height": self.image_height,
                    "interpolation": "INTER_AREA"
                },
                "env_rgb": {
                    "enabled": self.resize_env_rgb,
                    "width": self.image_width,
                    "height": self.image_height,
                    "interpolation": "INTER_AREA"
                },
                "depth": {
                    "enabled": self.resize_depth,
                    "width": self.depth_width,
                    "height": self.depth_height,
                    "interpolation": "INTER_NEAREST"
                },
                "env_depth": {
                    "enabled": self.resize_env_depth,
                    "width": self.depth_width,
                    "height": self.depth_height,
                    "interpolation": "INTER_NEAREST"
                }
            },
            "frames": {
                "rgb_frame_id": self.latest_rgb_frame_id,
                "env_rgb_frame_id": self.latest_env_rgb_frame_id,
                "depth_frame_id": self.latest_depth_frame_id,
                "env_depth_frame_id": self.latest_env_depth_frame_id,
                "pointcloud_frame_id": self.latest_pointcloud_frame_id if self.save_pointcloud else None,
            },
            "depth": {
                "depth_encoding_original": self.latest_depth_encoding,
                "env_depth_encoding_original": self.latest_env_depth_encoding,
                "saved_dtype": "float32",
                "unit": "meters",
                "invalid_values": "nan/inf converted to 0.0",
            },
            "pointcloud": {
                "enabled": self.save_pointcloud,
                "num_points": self.num_points if self.save_pointcloud else None,
                "saved_shape": "[T, num_points, 3]" if self.save_pointcloud else None,
            },
            "saved_files": {
                "timestamps": "timestamps.npy, shape [T]",
                "joint_pos": "joint_pos.npy, shape [T, 6]",
                "eef_pose": "eef_pose.npy, shape [T, 6]",
                "gripper_state": "gripper_state.npy, shape [T, 1]",
                "rgb_frames": "rgb_frames.npy, shape [T, H, W, 3], BGR format",
                "env_rgb_frames": "env_rgb_frames.npy, shape [T, H, W, 3], BGR format",
                "depth_frames": "depth_frames.npy, shape [T, H, W], float32 meters",
                "env_depth_frames": "env_depth_frames.npy, shape [T, H, W], float32 meters",
                "pointcloud_frames": "pointcloud_frames.npy, shape [T, num_points, 3], optional",
                "action_delta": "action_delta.npy, shape [T, 7]",
            },
        }
        with open(ep_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        msg = (
            f"Saved episode_{self.episode_idx:06d}, steps={len(action_delta)}, "
            f"rgb={rgb_frames[:-1].shape}, env_rgb={env_rgb_frames[:-1].shape}, "
            f"depth={depth_frames[:-1].shape}, env_depth={env_depth_frames[:-1].shape}, "
            f"action={action_delta.shape}"
        )
        if pointcloud_frames is not None:
            msg += f", pc={pointcloud_frames[:-1].shape}"
        self.get_logger().info(msg)

        self.episode_idx += 1
        self.current_episode.clear()
        self._clear_sensor_timestamp_buffers()
        self.recording = False

    def close_and_flush(self):
        if self.recording and len(self.current_episode) >= self.min_episode_len:
            self.get_logger().info("Shutting down, flushing current episode...")
            self.finish_current_episode()


def main():
    rclpy.init()
    node = FR5DPMultimodalRecorderNode()

    def shutdown_handler(sig, frame):
        node.get_logger().info("Received shutdown signal.")
        node.close_and_flush()
        node.destroy_node()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_and_flush()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()