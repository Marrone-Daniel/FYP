import time
import numpy as np
from collections import deque


class FR5RealEnv:
    def __init__(
        self,
        robot_controller,
        camera_controller,
        gripper_controller=None,
        frequency=10,
        n_obs_steps=2,
        max_obs_buffer_size=30,
    ):
        self.robot = robot_controller
        self.camera = camera_controller
        self.gripper = gripper_controller

        self.frequency = frequency
        self.n_obs_steps = n_obs_steps

        # ===== image/depth buffers =====
        self.image_buffer = deque(maxlen=max_obs_buffer_size)

        self.env_image_buffer = deque(maxlen=max_obs_buffer_size)
        self.env_depth_buffer = deque(maxlen=max_obs_buffer_size)

        # ===== robot state buffers =====
        self.agent_pos_buffer = deque(maxlen=max_obs_buffer_size)
        self.timestamp_buffer = deque(maxlen=max_obs_buffer_size)

        # optional: camera timestamp buffers
        self.image_time_buffer = deque(maxlen=max_obs_buffer_size)
        self.env_image_time_buffer = deque(maxlen=max_obs_buffer_size)
        self.env_depth_time_buffer = deque(maxlen=max_obs_buffer_size)

    @property
    def is_ready(self):
        ready = self.robot.is_ready and self.camera.is_ready
        if self.gripper is not None:
            ready = ready and self.gripper.is_ready
        return ready

    def start(self):
        self.robot.start()
        self.camera.start()

        if self.gripper is not None:
            self.gripper.start()

        # 等待 FR5ServoJController 产生第一帧机器人状态
        if hasattr(self.robot, "wait_for_state"):
            self.robot.wait_for_state(timeout=3.0)

        time.sleep(0.5)

        # warm up observation buffer
        for _ in range(self.n_obs_steps):
            self._append_current_obs()
            time.sleep(1.0 / self.frequency)

    def stop(self):
        if self.gripper is not None:
            self.gripper.stop()

        self.camera.stop()
        self.robot.stop()

    def _get_camera_obs(self):
        """
        支持新版 DualRosRGBDCameraController。

        期望 camera.get_obs() 返回:
            image:      (H,W,3), uint8 RGB
            env_image:  (H,W,3), uint8 RGB
            env_depth:  (H,W), float32

        为了兼容旧 CameraController，如果没有 get_obs()，则只返回旧 camera_0。
        """
        if hasattr(self.camera, "get_obs"):
            cam_obs = self.camera.get_obs()

            required_keys = ["image", "env_image", "env_depth"]
            for key in required_keys:
                if key not in cam_obs:
                    raise KeyError(f"camera.get_obs() missing key: {key}")

            image = cam_obs["image"]
            env_image = cam_obs["env_image"]
            env_depth = cam_obs["env_depth"]

            image_time = cam_obs.get("image_time", time.time())
            env_image_time = cam_obs.get("env_image_time", time.time())
            env_depth_time = cam_obs.get("env_depth_time", time.time())

            return {
                "image": image,
                "env_image": env_image,
                "env_depth": env_depth,
                "image_time": image_time,
                "env_image_time": env_image_time,
                "env_depth_time": env_depth_time,
            }

        # ===== fallback: old RGB-only CameraController =====
        image = self.camera.get_image()
        h, w = image.shape[:2]
        dummy_depth = np.zeros((h, w), dtype=np.float32)

        return {
            "image": image,
            "env_image": image.copy(),
            "env_depth": dummy_depth.copy(),
            "image_time": time.time(),
            "env_image_time": time.time(),
            "env_depth_time": time.time(),
        }

    def _append_current_obs(self):
        # ===== camera obs =====
        cam_obs = self._get_camera_obs()

        image = cam_obs["image"]              # (H,W,3), uint8          
        env_image = cam_obs["env_image"]      # (H,W,3), uint8
        env_depth = cam_obs["env_depth"]      # (H,W), float32

        # ===== basic validation =====
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"image should be (H,W,3), got {image.shape}")

        if env_image.ndim != 3 or env_image.shape[-1] != 3:
            raise ValueError(f"env_image should be (H,W,3), got {env_image.shape}")


        if env_depth.ndim != 2:
            raise ValueError(f"env_depth should be (H,W), got {env_depth.shape}")

        image = image.astype(np.uint8)
        env_image = env_image.astype(np.uint8)
        env_depth = env_depth.astype(np.float32)

        # ===== robot state =====
        state = self.robot.get_state()
        if state is None:
            if hasattr(self.robot, "wait_for_state"):
                state = self.robot.wait_for_state(timeout=2.0)
            else:
                raise RuntimeError("Robot state is None.")

        joint_pos = state["ActualQ"]  # (6,), rad

        if self.gripper is not None:
            gripper_pos = self.gripper.get_gripper_state()  # normalized [0, 1]
        else:
            gripper_pos = 0.0

        agent_pos = np.concatenate([
            joint_pos.astype(np.float64),
            np.array([gripper_pos], dtype=np.float64)
        ])

        timestamp = time.time()

        # ===== append buffers =====
        self.image_buffer.append(image)
        self.env_image_buffer.append(env_image)
        self.env_depth_buffer.append(env_depth)

        self.agent_pos_buffer.append(agent_pos)
        self.timestamp_buffer.append(timestamp)

        self.image_time_buffer.append(cam_obs["image_time"])
        self.env_image_time_buffer.append(cam_obs["env_image_time"])
        self.env_depth_time_buffer.append(cam_obs["env_depth_time"])

    def get_obs(self):
        assert self.is_ready

        self._append_current_obs()

        while len(self.image_buffer) < self.n_obs_steps:
            self._append_current_obs()
            time.sleep(1.0 / self.frequency)

        return {
            # 新模型需要的四个视觉模态
            "image": np.stack(
                list(self.image_buffer)[-self.n_obs_steps:],
                axis=0
            ),  # (T,H,W,3), uint8

            "env_image": np.stack(
                list(self.env_image_buffer)[-self.n_obs_steps:],
                axis=0
            ),  # (T,H,W,3), uint8

            "env_depth": np.stack(
                list(self.env_depth_buffer)[-self.n_obs_steps:],
                axis=0
            ),  # (T,H,W), float32

            "agent_pos": np.stack(
                list(self.agent_pos_buffer)[-self.n_obs_steps:],
                axis=0
            ),  # (T,7)

            "timestamp": np.asarray(
                list(self.timestamp_buffer)[-self.n_obs_steps:],
                dtype=np.float64
            ),

            # optional camera timestamps, useful for checking lag
            "image_time": np.asarray(
                list(self.image_time_buffer)[-self.n_obs_steps:],
                dtype=np.float64
            ),
            "env_image_time": np.asarray(
                list(self.env_image_time_buffer)[-self.n_obs_steps:],
                dtype=np.float64
            ),
            "env_depth_time": np.asarray(
                list(self.env_depth_time_buffer)[-self.n_obs_steps:],
                dtype=np.float64
            ),
        }

    def exec_actions(self, actions, timestamps):
        assert self.is_ready

        actions = np.asarray(actions, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)

        now = time.time()
        is_new = timestamps > now

        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        for action, ts in zip(new_actions, new_timestamps):
            joint_target = action[:6]
            gripper_target = float(np.clip(action[6], 0.0, 1.0))

            # 机械臂 6 关节
            self.robot.schedule_joint_waypoint(
                joint_pos=joint_target,
                gripper=0.0,
                target_time=ts
            )

            # Robotiq 夹爪独立控制
            if self.gripper is not None:
                self.gripper.set_gripper_state(gripper_target)

    def get_robot_state(self):
        return self.robot.get_state()