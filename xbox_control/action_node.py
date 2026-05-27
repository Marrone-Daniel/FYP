# action_monitor_node.py

import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64


class ActionMonitorNode(Node):
    def __init__(self):
        super().__init__("action_monitor_node")

        self.action_sub = self.create_subscription(
            Float64MultiArray,
            "/fr5/policy_action_seq",
            self.action_callback,
            10
        )

        self.latency_sub = self.create_subscription(
            Float64,
            "/fr5/policy_latency_ms",
            self.latency_callback,
            10
        )

        self.last_action_time = None
        self.last_latency_ms = None
        self.action_count = 0

        self.get_logger().info("ActionMonitorNode started. It will NOT control the robot.")

    def latency_callback(self, msg: Float64):
        self.last_latency_ms = msg.data

    def action_callback(self, msg: Float64MultiArray):
        now = time.time()
        self.action_count += 1

        data = np.asarray(msg.data, dtype=np.float64)

        if data.shape[0] < 2:
            self.get_logger().error("Invalid action message: too short")
            return

        n_actions = int(data[0])
        action_dim = int(data[1])
        flat = data[2:]

        expected = n_actions * action_dim
        if flat.shape[0] != expected:
            self.get_logger().error(
                f"Invalid action length: got {flat.shape[0]}, expected {expected}"
            )
            return

        action_seq = flat.reshape(n_actions, action_dim)

        if action_dim != 7:
            self.get_logger().error(f"Expected action_dim=7, got {action_dim}")
            return

        if not np.all(np.isfinite(action_seq)):
            self.get_logger().error("action_seq contains NaN or Inf")
            return

        # frequency estimate
        if self.last_action_time is None:
            hz = 0.0
        else:
            dt = now - self.last_action_time
            hz = 1.0 / dt if dt > 0 else 0.0
        self.last_action_time = now

        first_action = action_seq[0]
        joint_min = np.min(action_seq[:, :6])
        joint_max = np.max(action_seq[:, :6])
        gripper_min = np.min(action_seq[:, 6])
        gripper_max = np.max(action_seq[:, 6])

        latency_text = (
            f"{self.last_latency_ms:.1f} ms"
            if self.last_latency_ms is not None
            else "unknown"
        )

        self.get_logger().info(
            f"[{self.action_count}] recv action_seq={action_seq.shape}, "
            f"hz={hz:.2f}, latency={latency_text}\n"
            f"  first_action={first_action}\n"
            f"  joint range=[{joint_min:.4f}, {joint_max:.4f}], "
            f"gripper range=[{gripper_min:.4f}, {gripper_max:.4f}]"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ActionMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()