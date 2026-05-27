#!/usr/bin/env python3

import os
import sys
import time
import threading
import traceback
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

import torch
import dill
import hydra
from omegaconf import OmegaConf

from camera_controller import DualRosRGBDCameraController
from fr5_controller import FR5ServoJController
from gripper_controller import RobotiqGripperController
from fr5_env import FR5RealEnv
from fr5_inference_util import get_fr5_obs_dict


class FR5LocalPolicyControlNode(Node):
    def __init__(self):
        super().__init__("fr5_local_policy_control_node")

        # ============================================================
        # Parameters
        # ============================================================
        self.declare_parameter(
            "ckpt_path",
            "/home/xjtlu/diffusion_policy/output/checkpoints/epoch=0133-val_loss=0.046767.ckpt"
        )
        self.declare_parameter("diffusion_policy_root", "/home/xjtlu/diffusion_policy")

        self.declare_parameter("robot_ip", "192.168.58.2")
        self.declare_parameter("gripper_port", "/dev/ttyUSB0")

        # 控制频率：第一次建议 2.5Hz，确认稳定后再尝试 5Hz / 10Hz
        self.declare_parameter("control_frequency", 2.5)

        # 模型输出 n_action_steps，一次推理后只执行前几步
        self.declare_parameter("steps_per_inference", 5)
        self.declare_parameter("action_offset", 0)

        self.declare_parameter("n_obs_steps", 2)

        # 是否真正控制机械臂 / 夹爪
        self.declare_parameter("enable_robot_control", True)
        self.declare_parameter("enable_gripper_control", True)

        # 夹爪控制
        self.declare_parameter("max_gripper_delta", 0.04)
        self.declare_parameter("gripper_command_interval", 2)

        # replay-style ServoJ 执行参数
        self.declare_parameter("direct_servoj_duration_scale", 1.0)
        self.declare_parameter("max_policy_delta_warn_deg", 5.0)

        # 如果太久没有新推理，则清空旧动作
        self.declare_parameter("action_timeout", 2.0)

        # depth 与训练数据一致：meter float32，不归一化
        self.declare_parameter("normalize_depth", False)
        self.declare_parameter("max_depth", 10.0)

        # 本地推理设备
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("num_inference_steps", 16)
        self.declare_parameter("print_full_action", True)

        # 双相机 topic
        self.declare_parameter("wrist_color_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("env_color_topic", "/env_camera/color/image_raw")
        self.declare_parameter("env_depth_topic", "/env_camera/depth/image_raw")

        # ============================================================
        # Read parameters
        # ============================================================
        self.ckpt_path = self.get_parameter("ckpt_path").value
        self.diffusion_policy_root = self.get_parameter("diffusion_policy_root").value

        self.robot_ip = self.get_parameter("robot_ip").value
        self.gripper_port = self.get_parameter("gripper_port").value

        self.control_frequency = float(self.get_parameter("control_frequency").value)
        self.steps_per_inference = int(self.get_parameter("steps_per_inference").value)
        self.action_offset = int(self.get_parameter("action_offset").value)
        self.n_obs_steps = int(self.get_parameter("n_obs_steps").value)

        self.enable_robot_control = bool(self.get_parameter("enable_robot_control").value)
        self.enable_gripper_control = bool(self.get_parameter("enable_gripper_control").value)

        self.max_gripper_delta = float(self.get_parameter("max_gripper_delta").value)
        self.gripper_command_interval = int(self.get_parameter("gripper_command_interval").value)

        self.direct_servoj_duration_scale = float(
            self.get_parameter("direct_servoj_duration_scale").value
        )
        self.max_policy_delta_warn_deg = float(
            self.get_parameter("max_policy_delta_warn_deg").value
        )

        self.action_timeout = float(self.get_parameter("action_timeout").value)

        self.normalize_depth = bool(self.get_parameter("normalize_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)

        self.device_str = self.get_parameter("device").value
        self.num_inference_steps = int(self.get_parameter("num_inference_steps").value)
        self.print_full_action = bool(self.get_parameter("print_full_action").value)

        self.wrist_color_topic = self.get_parameter("wrist_color_topic").value
        self.env_color_topic = self.get_parameter("env_color_topic").value
        self.env_depth_topic = self.get_parameter("env_depth_topic").value

        self.get_logger().info(f"ckpt_path: {self.ckpt_path}")
        self.get_logger().info(f"diffusion_policy_root: {self.diffusion_policy_root}")
        self.get_logger().info(f"control_frequency: {self.control_frequency} Hz")
        self.get_logger().info(f"steps_per_inference: {self.steps_per_inference}")
        self.get_logger().info(f"action_offset: {self.action_offset}")
        self.get_logger().info(f"n_obs_steps: {self.n_obs_steps}")
        self.get_logger().info(f"enable_robot_control: {self.enable_robot_control}")
        self.get_logger().info(f"enable_gripper_control: {self.enable_gripper_control}")
        self.get_logger().info(f"max_gripper_delta: {self.max_gripper_delta}")
        self.get_logger().info(f"gripper_command_interval: {self.gripper_command_interval}")
        self.get_logger().info(
            f"direct_servoj_duration_scale: {self.direct_servoj_duration_scale}"
        )
        self.get_logger().info(
            f"max_policy_delta_warn_deg: {self.max_policy_delta_warn_deg}"
        )
        self.get_logger().info(f"action_timeout: {self.action_timeout}")
        self.get_logger().info(f"normalize_depth: {self.normalize_depth}")
        self.get_logger().info(f"max_depth: {self.max_depth}")
        self.get_logger().info(f"device: {self.device_str}")
        self.get_logger().info(f"num_inference_steps: {self.num_inference_steps}")

        # ============================================================
        # Debug publishers
        # ============================================================
        self.latency_pub = self.create_publisher(
            Float64,
            "/fr5/policy_latency_ms",
            10
        )

        self.raw_action_pub = self.create_publisher(
            Float64MultiArray,
            "/fr5/raw_delta_action",
            10
        )

        self.safe_target_pub = self.create_publisher(
            Float64MultiArray,
            "/fr5/safe_target_action",
            10
        )

        self.current_agent_pub = self.create_publisher(
            Float64MultiArray,
            "/fr5/current_agent_pos",
            10
        )

        self.full_action_seq_pub = self.create_publisher(
            Float64MultiArray,
            "/fr5/full_action_seq",
            10
        )

        # ============================================================
        # Policy holders
        # ============================================================
        self.policy = None
        self.device = None
        self.cfg = None
        self.policy_lock = threading.Lock()

        # ============================================================
        # Start FR5RealEnv
        # ============================================================
        # 注意：
        # 这里的 FR5ServoJController 必须是你 replay 已经成功验证的版本，
        # 需要提供 servoj_joint_target(q_target_rad, duration) 方法。
        robot = FR5ServoJController(
            robot_ip=self.robot_ip,
            frequency=100,
            joint_unit="rad",
            verbose=True,
        )

        camera = DualRosRGBDCameraController(
            wrist_color_topic=self.wrist_color_topic,
            env_color_topic=self.env_color_topic,
            env_depth_topic=self.env_depth_topic,
            width=160,
            height=160,
            depth_scale=0.001,
            max_depth=self.max_depth,
            timeout=8.0,
        )

        gripper = RobotiqGripperController(
            com_port=self.gripper_port,
            device_id=9,
            gripper_type="2F85",
            initial_pos=128,
            min_pos=0,
            max_pos=255,
        )

        self.env = FR5RealEnv(
            robot_controller=robot,
            camera_controller=camera,
            gripper_controller=gripper,
            frequency=10,
            n_obs_steps=self.n_obs_steps,
        )

        self.get_logger().info("Starting FR5RealEnv...")
        self.env.start()
        self.get_logger().info(f"env ready: {self.env.is_ready}")

        if not hasattr(self.env.robot, "servoj_joint_target"):
            raise AttributeError(
                "self.env.robot does not provide servoj_joint_target(). "
                "Please use the fr5_controller version that worked in replay."
            )

        # ============================================================
        # Load local policy
        # ============================================================
        self.load_local_policy()

        # ============================================================
        # Shared state
        # ============================================================
        self.action_buffer = deque()
        self.buffer_lock = threading.Lock()

        self.running = True
        self.inference_in_progress = False

        self.infer_count = 0
        self.fail_count = 0
        self.control_count = 0

        self.last_inference_success_time = 0.0
        self.last_action_time = 0.0

        # ============================================================
        # Inference thread
        # ============================================================
        self.infer_thread = threading.Thread(
            target=self.strict_receding_horizon_loop,
            daemon=True
        )
        self.infer_thread.start()

        # ============================================================
        # Control timer
        # ============================================================
        control_period = 1.0 / self.control_frequency
        self.control_timer = self.create_timer(
            control_period,
            self.control_timer_callback
        )

    # ============================================================
    # Load local policy
    # ============================================================
    def load_local_policy(self):
        if self.diffusion_policy_root and os.path.isdir(self.diffusion_policy_root):
            if self.diffusion_policy_root not in sys.path:
                sys.path.insert(0, self.diffusion_policy_root)
                self.get_logger().info(
                    f"Added to sys.path: {self.diffusion_policy_root}"
                )

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        if torch.cuda.is_available() and "cuda" in self.device_str:
            self.device = torch.device(self.device_str)
        else:
            self.device = torch.device("cpu")

        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Loading local checkpoint: {self.ckpt_path}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info("=" * 80)

        try:
            OmegaConf.register_new_resolver("eval", eval, replace=True)
        except Exception:
            pass

        payload = torch.load(
            open(self.ckpt_path, "rb"),
            pickle_module=dill,
            map_location="cpu"
        )

        cfg = payload["cfg"]
        self.cfg = cfg

        self.get_logger().info(f"Workspace target: {cfg._target_}")
        self.get_logger().info(
            f"Config name: {cfg.name if 'name' in cfg else 'unknown'}"
        )

        if hasattr(cfg, "task") and hasattr(cfg.task, "shape_meta"):
            self.get_logger().info("shape_meta:")
            self.get_logger().info(OmegaConf.to_yaml(cfg.task.shape_meta))

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        if cfg.training.use_ema:
            self.get_logger().info("Using EMA model.")
            policy = workspace.ema_model
        else:
            self.get_logger().info("Using raw model.")
            policy = workspace.model

        policy.eval().to(self.device)
        policy.reset()

        if "diffusion" in cfg.name:
            if hasattr(policy, "num_inference_steps"):
                policy.num_inference_steps = self.num_inference_steps
                self.get_logger().info(
                    f"Using num_inference_steps: {policy.num_inference_steps}"
                )

            if hasattr(policy, "n_action_steps"):
                if hasattr(cfg, "policy") and hasattr(cfg.policy, "n_action_steps"):
                    policy.n_action_steps = int(cfg.policy.n_action_steps)
                elif hasattr(cfg, "n_action_steps"):
                    policy.n_action_steps = int(cfg.n_action_steps)

                self.get_logger().info(
                    f"Using n_action_steps: {policy.n_action_steps}"
                )

        self.policy = policy
        self.get_logger().info("Local policy loaded.")

    # ============================================================
    # Strict receding horizon loop
    # ============================================================
    def strict_receding_horizon_loop(self):
        """
        严格 receding horizon：

        buffer 为空：
            获取最新 obs
            policy.predict_action()
            取 action_seq 中 selected steps 放入 buffer

        control_timer：
            逐步弹出 buffer 里的 action 执行
        """
        while self.running and rclpy.ok():
            try:
                with self.buffer_lock:
                    buffer_empty = len(self.action_buffer) == 0

                if not buffer_empty:
                    time.sleep(0.01)
                    continue

                if self.inference_in_progress:
                    time.sleep(0.01)
                    continue

                self.inference_in_progress = True
                self.infer_count += 1

                env_obs = self.env.get_obs()

                obs_dict_np = get_fr5_obs_dict(
                    env_obs,
                    normalize_depth=self.normalize_depth,
                    max_depth=self.max_depth,
                )

                image = obs_dict_np["image"].astype(np.float32)
                env_image = obs_dict_np["env_image"].astype(np.float32)
                env_depth = obs_dict_np["env_depth"].astype(np.float32)
                agent_pos = obs_dict_np["agent_pos"].astype(np.float32)

                obs_dict = {
                    "image": torch.from_numpy(image).unsqueeze(0).to(self.device),
                    "env_image": torch.from_numpy(env_image).unsqueeze(0).to(self.device),
                    "env_depth": torch.from_numpy(env_depth).unsqueeze(0).to(self.device),
                    "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
                }

                infer_start = time.time()

                with self.policy_lock:
                    with torch.no_grad():
                        result = self.policy.predict_action(obs_dict)

                infer_end = time.time()

                if "action" not in result:
                    raise KeyError(
                        f"predict_action result keys={result.keys()}, no 'action'"
                    )

                action_seq = result["action"][0].detach().cpu().numpy().astype(np.float64)

                if action_seq.ndim != 2 or action_seq.shape[1] != 7:
                    raise ValueError(f"Invalid action_seq shape: {action_seq.shape}")

                if not np.all(np.isfinite(action_seq)):
                    raise ValueError("action_seq contains NaN or Inf")

                self.publish_full_action_seq(action_seq)

                if self.print_full_action:
                    self.print_action_seq(action_seq)

                start = self.action_offset
                end = self.action_offset + self.steps_per_inference

                if start >= len(action_seq):
                    raise ValueError(
                        f"action_offset out of range. "
                        f"action_seq len={len(action_seq)}, offset={self.action_offset}"
                    )

                selected = action_seq[start:min(end, len(action_seq))]

                if len(selected) == 0:
                    raise ValueError(
                        f"No action selected. action_seq len={len(action_seq)}, "
                        f"offset={self.action_offset}, steps={self.steps_per_inference}"
                    )

                with self.buffer_lock:
                    self.action_buffer.clear()
                    for a in selected:
                        self.action_buffer.append(a.copy())

                self.last_inference_success_time = time.time()

                latency_ms = (infer_end - infer_start) * 1000.0
                latency_msg = Float64()
                latency_msg.data = latency_ms
                self.latency_pub.publish(latency_msg)

                self.get_logger().info(
                    f"[local infer {self.infer_count}] "
                    f"action_seq={action_seq.shape}, "
                    f"selected={selected.shape}, "
                    f"infer={latency_ms:.1f} ms, "
                    f"buffer={len(self.action_buffer)}"
                )

                self.get_logger().info(
                    f"[local infer {self.infer_count}] obs_shapes="
                    f"image={list(image.shape)}, "
                    f"env_image={list(env_image.shape)}, "
                    f"env_depth={list(env_depth.shape)}, "
                    f"agent_pos={list(agent_pos.shape)}"
                )

                self.get_logger().info(
                    f"[local infer {self.infer_count}] obs_ranges="
                    f"image=[{float(image.min()):.4f}, {float(image.max()):.4f}], "
                    f"env_image=[{float(env_image.min()):.4f}, {float(env_image.max()):.4f}], "
                    f"env_depth=[{float(env_depth.min()):.4f}, {float(env_depth.max()):.4f}]"
                )

            except Exception as e:
                self.fail_count += 1
                self.get_logger().error(
                    f"Local inference failed. fail_count={self.fail_count}, error={e}"
                )
                self.get_logger().debug(traceback.format_exc())
                time.sleep(0.1)

            finally:
                self.inference_in_progress = False

    # ============================================================
    # Control loop
    # ============================================================
    def control_timer_callback(self):
        self.control_count += 1

        # 如果太久没有新推理，清空旧动作，避免执行 stale action
        if self.last_inference_success_time > 0:
            if time.time() - self.last_inference_success_time > self.action_timeout:
                with self.buffer_lock:
                    self.action_buffer.clear()
                return

        with self.buffer_lock:
            if len(self.action_buffer) == 0:
                return
            raw_delta_action = self.action_buffer.popleft()

        try:
            if self.enable_robot_control:
                self.execute_policy_delta_like_replay(raw_delta_action)
            else:
                self.get_logger().info(
                    f"[control {self.control_count}] robot control disabled. "
                    f"raw_delta_deg={np.rad2deg(raw_delta_action[:6])}, "
                    f"gripper_raw={raw_delta_action[6]:.6f}, "
                    f"remaining_buffer={len(self.action_buffer)}"
                )

        except Exception as e:
            self.get_logger().error(f"Direct ServoJ control step failed: {e}")
            self.get_logger().debug(traceback.format_exc())

    # ============================================================
    # Replay-style policy action execution
    # ============================================================
    def execute_policy_delta_like_replay(self, raw_delta_action):
        """
        使用 replay 成功的方法执行模型输出：

            current_joint + model_delta -> q_target_rad
            self.env.robot.servoj_joint_target(q_target_rad, duration)

        第 7 维用于夹爪：
            target_gripper = current_gripper + gripper_delta
        """
        raw_delta_action = np.asarray(raw_delta_action, dtype=np.float64)

        if raw_delta_action.shape != (7,):
            raise ValueError(
                f"raw_delta_action should be shape (7,), got {raw_delta_action.shape}"
            )

        if not np.all(np.isfinite(raw_delta_action)):
            raise ValueError("raw_delta_action contains NaN or Inf")

        # ============================================================
        # 1. 当前机器人状态
        # ============================================================
        current_agent_pos = self.get_current_agent_pos()
        current_joint = current_agent_pos[:6].astype(np.float64)
        current_gripper = float(current_agent_pos[6])

        # ============================================================
        # 2. 模型输出：前 6 维为 joint delta，单位 rad
        # ============================================================
        joint_delta = raw_delta_action[:6].astype(np.float64)

        max_delta_deg = float(np.max(np.abs(np.rad2deg(joint_delta))))
        if max_delta_deg > self.max_policy_delta_warn_deg:
            self.get_logger().warn(
                f"[DirectServoJ] large model delta detected: "
                f"max_delta={max_delta_deg:.3f} deg, "
                f"delta_deg={np.rad2deg(joint_delta)}"
            )

        # ============================================================
        # 3. replay 成功的核心：
        #    当前关节 + 模型 delta = ServoJ target
        # ============================================================
        q_target_rad = current_joint + joint_delta

        # ============================================================
        # 4. 夹爪控制：第 7 维作为 gripper delta
        # ============================================================
        if self.enable_gripper_control:
            gripper_delta = float(raw_delta_action[6])

            gripper_delta = float(np.clip(
                gripper_delta,
                -self.max_gripper_delta,
                self.max_gripper_delta
            ))

            target_gripper = float(np.clip(
                current_gripper + gripper_delta,
                0.0,
                1.0
            ))

            if self.env.gripper is None:
                self.get_logger().warn(
                    "[Gripper] enable_gripper_control=True, "
                    "but self.env.gripper is None."
                )
            else:
                # 降低夹爪串口发送频率，避免串口压力过大
                if self.control_count % max(1, self.gripper_command_interval) == 0:
                    self.env.gripper.set_gripper_state(target_gripper)
        else:
            gripper_delta = 0.0
            target_gripper = current_gripper

        # ============================================================
        # 5. 通过 replay 已验证成功的 fr5_controller 接口执行 ServoJ
        # ============================================================
        duration = (1.0 / self.control_frequency) * self.direct_servoj_duration_scale

        ret = self.env.robot.servoj_joint_target(
            q_target_rad=q_target_rad,
            duration=duration
        )

        # ============================================================
        # 6. 发布 debug topic
        # ============================================================
        target_action = np.concatenate([
            q_target_rad.astype(np.float64),
            np.array([target_gripper], dtype=np.float64)
        ])

        current_msg = Float64MultiArray()
        current_msg.data = current_agent_pos.tolist()
        self.current_agent_pub.publish(current_msg)

        raw_msg = Float64MultiArray()
        raw_msg.data = raw_delta_action.tolist()
        self.raw_action_pub.publish(raw_msg)

        target_msg = Float64MultiArray()
        target_msg.data = target_action.tolist()
        self.safe_target_pub.publish(target_msg)

        self.last_action_time = time.time()

        self.get_logger().info(
            f"[DirectServoJ control {self.control_count}] "
            f"raw_delta_deg={np.rad2deg(joint_delta)}, "
            f"current_joint_deg={np.rad2deg(current_joint)}, "
            f"target_joint_deg={np.rad2deg(q_target_rad)}, "
            f"duration={duration:.4f}, "
            f"gripper_delta={gripper_delta:.4f}, "
            f"gripper={target_gripper:.3f}, "
            f"ret={ret}, "
            f"remaining_buffer={len(self.action_buffer)}"
        )

    # ============================================================
    # Helpers
    # ============================================================
    def get_current_agent_pos(self):
        state = self.env.get_robot_state()
        if state is None:
            raise RuntimeError("Robot state is None")

        joint_pos = state["ActualQ"].astype(np.float64)

        if self.env.gripper is not None:
            gripper_pos = float(self.env.gripper.get_gripper_state())
        else:
            gripper_pos = 0.0

        return np.concatenate([
            joint_pos,
            np.array([gripper_pos], dtype=np.float64)
        ])

    def publish_full_action_seq(self, action_seq):
        msg = Float64MultiArray()
        msg.data = [
            float(action_seq.shape[0]),
            float(action_seq.shape[1]),
        ] + action_seq.reshape(-1).astype(np.float64).tolist()

        self.full_action_seq_pub.publish(msg)

    def print_action_seq(self, action_seq):
        np.set_printoptions(
            precision=6,
            suppress=True,
            linewidth=220
        )

        self.get_logger().info("=" * 100)
        self.get_logger().info(f"[MODEL OUTPUT] action_seq shape={action_seq.shape}")
        self.get_logger().info(f"action_seq rad/gripper:\n{action_seq}")
        self.get_logger().info(f"joint_delta_deg:\n{np.rad2deg(action_seq[:, :6])}")
        self.get_logger().info(f"gripper_delta_or_cmd:\n{action_seq[:, 6]}")
        self.get_logger().info("=" * 100)

    def destroy_node(self):
        self.get_logger().info("Stopping node...")

        self.running = False

        try:
            if hasattr(self, "infer_thread") and self.infer_thread.is_alive():
                self.infer_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            self.env.stop()
        except Exception as e:
            self.get_logger().error(f"Failed to stop env: {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FR5LocalPolicyControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()