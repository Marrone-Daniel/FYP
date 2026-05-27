import time
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge


# ============================================================
# Legacy OpenCV camera controller
# ============================================================

class CameraController:
    """
    原始 OpenCV USB RGB camera controller。
    只适合 /dev/videoX 形式的普通 RGB 相机。
    """

    def __init__(
        self,
        index_or_path=0,
        width=96,
        height=96,
        capture_width=640,
        capture_height=480,
        fps=30,
    ):
        self.index_or_path = index_or_path
        self.width = width
        self.height = height
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.fps = fps
        self.cap = None
        self._is_ready = False

    @property
    def is_ready(self):
        return self._is_ready

    def start(self, wait=True):
        self.cap = cv2.VideoCapture(self.index_or_path, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.index_or_path}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        for _ in range(50):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._is_ready = True
                return
            time.sleep(0.05)

        self.stop()
        raise RuntimeError("Camera opened but failed to read frames.")

    def stop(self, wait=True):
        self._is_ready = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_image(self):
        if not self._is_ready:
            raise RuntimeError("CameraController is not ready.")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from camera.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA
        )

        return frame.astype(np.uint8)


# ============================================================
# Dual RGB-D camera controller from ROS topics
# Supports raw Image and CompressedImage
# ============================================================

class _DualRGBDSubscriberNode(Node):
    """
    一个 ROS2 node 同时订阅两个 RGB-D 相机，避免多个线程同时 spin 默认 executor。

    输出:
        image:      wrist RGB, (H,W,3), uint8 RGB
        depth:      wrist depth, (H,W), float32 meter
        env_image:  env RGB, (H,W,3), uint8 RGB
        env_depth:  env depth, (H,W), float32 meter
    """

    def __init__(
        self,
        wrist_color_topic,
        env_color_topic,
        env_depth_topic,
        width=160,
        height=160,
        depth_scale=0.001,
        max_depth=5.0,
        node_name="dual_rgbd_camera_controller",
    ):
        super().__init__(node_name)

        self.wrist_color_topic = wrist_color_topic
        self.env_color_topic = env_color_topic
        self.env_depth_topic = env_depth_topic

        self.width = width
        self.height = height
        self.depth_scale = depth_scale
        self.max_depth = max_depth

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest = {
            "image": None,
            "env_image": None,
            "env_depth": None,

            "image_time": None,
            "env_image_time": None,
            "env_depth_time": None,
        }

        # 根据 topic 名自动选择 Image 或 CompressedImage
        wrist_color_type = self._infer_msg_type(wrist_color_topic)
        env_color_type = self._infer_msg_type(env_color_topic)
        env_depth_type = self._infer_msg_type(env_depth_topic)

        self.wrist_color_sub = self.create_subscription(
            wrist_color_type,
            wrist_color_topic,
            self._wrist_color_callback,
            10
        )

        self.env_color_sub = self.create_subscription(
            env_color_type,
            env_color_topic,
            self._env_color_callback,
            10
        )

        self.env_depth_sub = self.create_subscription(
            env_depth_type,
            env_depth_topic,
            self._env_depth_callback,
            10
        )

        self.get_logger().info(f"Subscribed wrist color topic: {wrist_color_topic}, type={wrist_color_type.__name__}")
        self.get_logger().info(f"Subscribed env color topic: {env_color_topic}, type={env_color_type.__name__}")
        self.get_logger().info(f"Subscribed env depth topic: {env_depth_topic}, type={env_depth_type.__name__}")

    @staticmethod
    def _infer_msg_type(topic_name):
        """
        如果 topic 名包含 compressed, 则使用 CompressedImage。
        否则使用普通 Image。
        """
        if "compressed" in topic_name:
            return CompressedImage
        return Image

    @staticmethod
    def _stamp_to_float(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    # ============================================================
    # Color decoding
    # ============================================================

    def _decode_color_msg(self, msg):
        """
        Return RGB uint8 image, shape (H,W,3).
        Supports:
            sensor_msgs/Image
            sensor_msgs/CompressedImage
        """
        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise RuntimeError(
                    f"Failed to decode compressed color image. format={msg.format}"
                )

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        elif isinstance(msg, Image):
            encoding = msg.encoding.lower()

            if encoding == "rgb8":
                img_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            elif encoding == "rgba8":
                img_rgba = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgba8")
                img_rgb = img_rgba[:, :, :3]

            else:
                # 常见 bgr8
                img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        else:
            raise TypeError(f"Unsupported color msg type: {type(msg)}")

        img_rgb = cv2.resize(
            img_rgb,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA
        ).astype(np.uint8)

        return img_rgb

    # ============================================================
    # Depth decoding
    # ============================================================

    def _decode_depth_msg(self, msg):
        """
        Return depth float32 image in meters, shape (H,W).

        Supports:
            sensor_msgs/Image:
                16UC1 -> mm, convert to meter by depth_scale
                32FC1 -> meter
            sensor_msgs/CompressedImage:
                common compressedDepth PNG.
                For 16UC1 compressedDepth, usually skip 12-byte header then imdecode PNG.
        """
        if isinstance(msg, Image):
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * self.depth_scale
            else:
                depth = depth.astype(np.float32)


        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0

        # clip depth to valid range
        depth = np.clip(depth, 0.0, self.max_depth)

        depth = cv2.resize(
            depth,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

        return depth



    # ============================================================
    # Callbacks
    # ============================================================

    def _wrist_color_callback(self, msg):
        try:
            img = self._decode_color_msg(msg)
            t = self._stamp_to_float(msg)

            with self.lock:
                self.latest["image"] = img
                self.latest["image_time"] = t

        except Exception as e:
            self.get_logger().error(f"Failed to process wrist color: {e}")


    def _env_color_callback(self, msg):
        try:
            img = self._decode_color_msg(msg)
            t = self._stamp_to_float(msg)

            with self.lock:
                self.latest["env_image"] = img
                self.latest["env_image_time"] = t

        except Exception as e:
            self.get_logger().error(f"Failed to process env color: {e}")

    def _env_depth_callback(self, msg):

        try:
            depth = self._decode_depth_msg(msg)
            t = self._stamp_to_float(msg)

            with self.lock:
                self.latest["env_depth"] = depth
                self.latest["env_depth_time"] = t

        except Exception as e:
            self.get_logger().error(f"Failed to process env depth: {e}")

    def get_latest(self):
        with self.lock:
            required = ["image", "env_image", "env_depth"]
            for key in required:
                if self.latest[key] is None:
                    return None

            return {
                "image": self.latest["image"].copy(),
                "env_image": self.latest["env_image"].copy(),
                "env_depth": self.latest["env_depth"].copy(),

                "image_time": self.latest["image_time"],
                "env_image_time": self.latest["env_image_time"],
                "env_depth_time": self.latest["env_depth_time"],
            }


class DualRosRGBDCameraController:
    """
    双 RGB-D 相机控制器，支持 raw topic 和 compressed topic。

    推荐 compressed topic 配置例子:
        wrist_color_topic="/camera/color/image_raw/compressed"
        env_color_topic="/env_camera/color/image_raw"
        env_depth_topic="/env_camera/depth/image_raw"

    如果 env 相机使用 raw topic, 也可以:
        env_color_topic="/env_camera/color/image_raw"
        env_depth_topic="/env_camera/depth/image_raw"
    """

    def __init__(
        self,
        wrist_color_topic="/camera/color/image_raw/compressed",
        env_color_topic="/env_camera/color/image_raw",
        env_depth_topic="/env_camera/depth/image_raw",
        width=160,
        height=160,
        depth_scale=0.001,
        max_depth=10.0,
        timeout=8.0,
    ):
        self.wrist_color_topic = wrist_color_topic
        self.env_color_topic = env_color_topic
        self.env_depth_topic = env_depth_topic

        self.width = width
        self.height = height
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.timeout = timeout

        self.node = None
        self.executor = None
        self.executor_thread = None
        self._running = False
        self._is_ready = False

    @property
    def is_ready(self):
        return self._is_ready

    def start(self, wait=True):
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = _DualRGBDSubscriberNode(
            node_name="dual_rgbd_camera_controller",
            wrist_color_topic=self.wrist_color_topic,
            env_color_topic=self.env_color_topic,
            env_depth_topic=self.env_depth_topic,
            width=self.width,
            height=self.height,
            depth_scale=self.depth_scale,
            max_depth=self.max_depth,
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._running = True
        self.executor_thread = threading.Thread(
            target=self._spin,
            daemon=True
        )
        self.executor_thread.start()

        if wait:
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                data = self.node.get_latest()
                if data is not None:
                    self._is_ready = True
                    return
                time.sleep(0.05)

            missing = self._get_missing_keys()
            self.stop()
            raise RuntimeError(
                "Dual RGB-D camera not ready. "
                f"missing={missing}, "
                f"topics=[{self.wrist_color_topic}, "
                f"{self.env_color_topic}, {self.env_depth_topic}]"
            )
        else:
            self._is_ready = True

    def _spin(self):
        while self._running and rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception as e:
                if self.node is not None:
                    self.node.get_logger().error(f"Dual RGBD executor spin failed: {e}")
                break

    def _get_missing_keys(self):
        if self.node is None:
            return ["node_none"]

        with self.node.lock:
            return [
                key for key in ["image", "env_image", "env_depth"]
                if self.node.latest[key] is None
            ]

    def stop(self, wait=True):
        self._is_ready = False
        self._running = False

        if self.executor_thread is not None:
            self.executor_thread.join(timeout=1.0)
            self.executor_thread = None

        if self.executor is not None and self.node is not None:
            try:
                self.executor.remove_node(self.node)
            except Exception:
                pass

        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None

        self.executor = None

    def get_obs(self):
        if not self._is_ready:
            raise RuntimeError("DualRosRGBDCameraController is not ready.")

        data = self.node.get_latest()
        if data is None:
            raise RuntimeError("No complete dual RGB-D observation received yet.")

        return data

    def get_image(self):
        """
        兼容旧接口，只返回 wrist RGB。
        """
        return self.get_obs()["image"]

    def get_depth(self):
        return self.get_obs()["depth"]