#!/usr/bin/env python3
import time
import math
import numpy as np
from pathlib import Path
import sys

sys.path.append('/home/xjtlu/xbox_control/xbox_control')

from fairino import Robot
import pyrobotiqgripper as rq


EP_DIR = Path("/home/xjtlu/xbox_control/fr5_dp_data_area_A/episode_000040")

ROBOT_IP = "192.168.58.2"

GRIPPER_PORT = "/dev/ttyUSB0"
GRIPPER_ID = 9

# 安全限制：单步最大关节增量，单位 rad
# 如果某一步 action_delta 超过这个值，说明数据可能异常，直接停止
MAX_DELTA_RAD = 1

# 如果没有 timestamps.npy，就用这个默认频率
DEFAULT_SAMPLE_RATE = 10.0


def rad_to_deg(q_rad):
    return [math.degrees(float(x)) for x in q_rad]


def gripper_rad_to_raw(g_rad, open_rad=0.0, closed_rad=0.8):
    ratio = (float(g_rad) - open_rad) / (closed_rad - open_rad)
    ratio = max(0.0, min(1.0, ratio))
    return int(ratio * 255)


def estimate_dt(ep_dir: Path, default_sample_rate=10.0):
    ts_path = ep_dir / "timestamps.npy"
    if not ts_path.exists():
        return 1.0 / default_sample_rate

    timestamps = np.load(ts_path)
    if len(timestamps) < 2:
        return 1.0 / default_sample_rate

    dt_list = np.diff(timestamps)
    dt = float(np.median(dt_list))

    if not np.isfinite(dt) or dt <= 0:
        return 1.0 / default_sample_rate

    return dt


def infer_delta_alignment(joint_pos, action_delta):
    """
    判断 action_delta 的对齐方式。

    常见两种：
    A: action_delta[i] ≈ joint_pos[i+1] - joint_pos[i]
    B: action_delta[i] ≈ joint_pos[i] - joint_pos[i-1]

    返回：
    "current_to_next" 或 "prev_to_current"
    """

    q = joint_pos[:, :6]
    d = action_delta[:, :6]

    T = min(len(q), len(d))

    if T < 3:
        return "current_to_next"

    # Candidate A:
    # q[1:] ≈ q[0] + cumsum(d[0:T-1])
    q_recon_a = [q[0].copy()]
    for i in range(T - 1):
        q_recon_a.append(q_recon_a[-1] + d[i])
    q_recon_a = np.asarray(q_recon_a)
    err_a = np.mean(np.linalg.norm(q_recon_a[:T] - q[:T], axis=1))

    # Candidate B:
    # q[1:] ≈ q[0] + cumsum(d[1:T])
    q_recon_b = [q[0].copy()]
    for i in range(1, T):
        q_recon_b.append(q_recon_b[-1] + d[i])
    q_recon_b = np.asarray(q_recon_b)
    err_b = np.mean(np.linalg.norm(q_recon_b[:T] - q[:T], axis=1))

    print("\nDelta alignment check:")
    print(f"  err if action_delta[i]   = q[i+1] - q[i]:   {err_a:.8f}")
    print(f"  err if action_delta[i]   = q[i]   - q[i-1]: {err_b:.8f}")

    if err_a <= err_b:
        print("  Using alignment: current_to_next")
        return "current_to_next"
    else:
        print("  Using alignment: prev_to_current")
        return "prev_to_current"


def build_replay_joints_from_delta(joint_pos, action_delta, alignment):
    """
    用 action_delta 积分还原关节轨迹。
    起点固定为 joint_pos[0]。
    """

    q0 = joint_pos[0, :6].astype(np.float64)
    d = action_delta[:, :6].astype(np.float64)

    T = len(joint_pos)
    q_replay = np.zeros((T, 6), dtype=np.float64)
    q_replay[0] = q0

    if alignment == "current_to_next":
        # q[i+1] = q[i] + action_delta[i]
        for i in range(T - 1):
            delta = d[i]
            q_replay[i + 1] = q_replay[i] + delta

    elif alignment == "prev_to_current":
        # q[i] = q[i-1] + action_delta[i]
        for i in range(1, T):
            delta = d[i]
            q_replay[i] = q_replay[i - 1] + delta

    else:
        raise ValueError(f"Unknown alignment: {alignment}")

    return q_replay


def check_delta_safety(action_delta):
    d = action_delta[:, :6]
    per_step_max = np.max(np.abs(d), axis=1)
    global_max = float(np.max(per_step_max))

    print("\nAction delta safety check:")
    print(f"  max abs joint delta per scalar: {global_max:.6f} rad")
    print(f"  max abs joint delta per scalar: {math.degrees(global_max):.4f} deg")

    bad_idx = np.where(per_step_max > MAX_DELTA_RAD)[0]
    if len(bad_idx) > 0:
        print(f"  [WARN] Found {len(bad_idx)} large delta steps.")
        print(f"  First bad step: {bad_idx[0]}, value={per_step_max[bad_idx[0]]:.6f} rad")
        return False

    print("  [OK] action_delta looks safe.")
    return True


def main():
    joint_pos = np.load(EP_DIR / "joint_pos.npy")              # [T, 6], rad
    gripper_state = np.load(EP_DIR / "gripper_state.npy")      # [T, 1], rad
    action_delta = np.load(EP_DIR / "action_delta.npy")        # [T, 7] or [T, ?]

    print("joint_pos shape:", joint_pos.shape)
    print("gripper_state shape:", gripper_state.shape)
    print("action_delta shape:", action_delta.shape)

    if action_delta.shape[1] < 6:
        raise ValueError("action_delta must contain at least 6 joint delta dimensions.")

    T = min(len(joint_pos), len(gripper_state), len(action_delta))
    joint_pos = joint_pos[:T]
    gripper_state = gripper_state[:T]
    action_delta = action_delta[:T]

    dt = estimate_dt(EP_DIR, default_sample_rate=DEFAULT_SAMPLE_RATE)
    sample_rate = 1.0 / dt

    print(f"Estimated dt: {dt:.4f}s")
    print(f"Estimated sample rate: {sample_rate:.2f} Hz")

    # 检查 action_delta 是否存在异常大跳变
    safe = check_delta_safety(action_delta)
    if not safe:
        ans = input("Large action delta detected. Continue anyway? [y/N]: ")
        if ans.lower() != "y":
            print("Abort for safety.")
            return

    # 自动判断 action_delta 和 joint_pos 的对齐方式
    alignment = infer_delta_alignment(joint_pos, action_delta)

    # 用关节增量积分还原轨迹
    q_replay = build_replay_joints_from_delta(
        joint_pos=joint_pos,
        action_delta=action_delta,
        alignment=alignment
    )

    # 和原始 joint_pos 比较一下
    recon_err = np.linalg.norm(q_replay - joint_pos[:, :6], axis=1)
    print("\nReconstruction error compared with recorded joint_pos:")
    print(f"  mean error: {np.mean(recon_err):.8f} rad")
    print(f"  max error : {np.max(recon_err):.8f} rad")
    print(f"  mean error: {math.degrees(float(np.mean(recon_err))):.6f} deg")
    print(f"  max error : {math.degrees(float(np.max(recon_err))):.6f} deg")

    robot = Robot.RPC(ROBOT_IP)
    print("\nFR5 connected")

    robot.RobotEnable(1)
    robot.ServoMoveStart()
    print("Servo mode started")

    gripper = rq.RobotiqGripper(
        com_port=GRIPPER_PORT,
        device_id=GRIPPER_ID,
        gripper_type="2F85",
        connection_type=rq.GRIPPER_MODE_RTU,
        debug=False
    )
    gripper.connect()
    gripper.activate()
    print("Gripper connected and activated")

    try:
        input("\nPress Enter to move to the first recorded joint position...")

        q0_deg = rad_to_deg(joint_pos[0])
        print("Move to first pose:", q0_deg)

        # 先回到数据集起点
        robot.MoveJ(q0_deg, 0, 0)
        time.sleep(1.0)

        input("\nPress Enter to start delta-based replay...")

        for i in range(T):
            q_target_rad = q_replay[i]
            q_target_deg = rad_to_deg(q_target_rad)

            g_raw = gripper_rad_to_raw(gripper_state[i, 0])

            axis_pos = [0.0, 0.0, 0.0, 0.0]

            ret = robot.ServoJ(
                q_target_deg,
                axis_pos,
                0.0,
                0.0,
                dt,
                0.0,
                0.0
            )

            if ret != 0:
                print(f"[Step {i}] ServoJ failed, ret={ret}")
                break

            # 夹爪不需要每一帧都发，降低串口压力
            if i % 2 == 0:
                gripper.move(g_raw)

            time.sleep(dt)

        print("\nDelta-based replay finished")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        try:
            robot.ServoMoveEnd()
        except Exception:
            pass

        try:
            gripper.disconnect()
        except Exception:
            pass

        print("Exit")


if __name__ == "__main__":
    main()














# #!/usr/bin/env python3
# import time
# import math
# import numpy as np
# from pathlib import Path
# import sys


# # ============================================================
# # Project path
# # ============================================================

# sys.path.append("/home/xjtlu/xbox_control/xbox_control")

# from fr5_controller import FR5ServoJController


# # ============================================================
# # Config
# # ============================================================

# EP_DIR = Path("/home/xjtlu/xbox_control/fr5_dp_data_pile/episode_000040")

# ROBOT_IP = "192.168.58.2"

# DEFAULT_SAMPLE_RATE = 10.0

# # 是否使用 timestamps.npy 中的 dt
# USE_DATASET_TIMESTAMPS = True

# # replay 放慢倍率
# # 1.0 = 原始速度
# # 1.5 = 放慢 1.5 倍
# # 2.0 = 放慢 2 倍
# TIME_SCALE = 1.5

# # action_delta 安全检查阈值
# # 你的数据之前最大大约 2.087 deg，因此 2 deg 比较宽松
# MAX_DELTA_CHECK_RAD = np.deg2rad(2.0)

# # 起点误差容忍
# START_POSE_TOL_RAD = np.deg2rad(1.5)

# # 等待到达起点最长时间
# START_POSE_TIMEOUT = 8.0

# # tracking error warning
# TRACKING_WARN_DEG = 10.0

# # tracking error stop
# TRACKING_STOP_DEG = 30.0

# # 是否 tracking error 过大时停止
# STOP_ON_LARGE_TRACKING_ERROR = False


# # ============================================================
# # Utils
# # ============================================================

# def rad_to_deg(x):
#     return np.rad2deg(np.asarray(x, dtype=np.float64))


# def estimate_dt(ep_dir: Path, default_sample_rate=10.0):
#     ts_path = ep_dir / "timestamps.npy"

#     if not USE_DATASET_TIMESTAMPS:
#         return 1.0 / default_sample_rate

#     if not ts_path.exists():
#         print("[WARN] timestamps.npy not found. Use default sample rate.")
#         return 1.0 / default_sample_rate

#     timestamps = np.load(ts_path)

#     if len(timestamps) < 2:
#         print("[WARN] timestamps.npy length < 2. Use default sample rate.")
#         return 1.0 / default_sample_rate

#     dt_list = np.diff(timestamps)
#     dt = float(np.median(dt_list))

#     if not np.isfinite(dt) or dt <= 0:
#         print("[WARN] invalid dt from timestamps.npy. Use default sample rate.")
#         return 1.0 / default_sample_rate

#     return dt


# def infer_delta_alignment(joint_pos, action_delta):
#     """
#     判断 action_delta 对齐方式：

#     current_to_next:
#         action_delta[i] ≈ joint_pos[i+1] - joint_pos[i]

#     prev_to_current:
#         action_delta[i] ≈ joint_pos[i] - joint_pos[i-1]
#     """

#     q = joint_pos[:, :6].astype(np.float64)
#     d = action_delta[:, :6].astype(np.float64)

#     T = min(len(q), len(d))

#     if T < 3:
#         return "current_to_next"

#     # A: q[i+1] = q[i] + d[i]
#     q_recon_a = [q[0].copy()]
#     for i in range(T - 1):
#         q_recon_a.append(q_recon_a[-1] + d[i])
#     q_recon_a = np.asarray(q_recon_a)

#     err_a = np.mean(np.linalg.norm(q_recon_a[:T] - q[:T], axis=1))

#     # B: q[i] = q[i-1] + d[i]
#     q_recon_b = [q[0].copy()]
#     for i in range(1, T):
#         q_recon_b.append(q_recon_b[-1] + d[i])
#     q_recon_b = np.asarray(q_recon_b)

#     err_b = np.mean(np.linalg.norm(q_recon_b[:T] - q[:T], axis=1))

#     print("\n" + "=" * 80)
#     print("Delta alignment check")
#     print("=" * 80)
#     print(f"err current_to_next: {err_a:.10f} rad")
#     print(f"err prev_to_current: {err_b:.10f} rad")

#     if err_a <= err_b:
#         print("Using alignment: current_to_next")
#         return "current_to_next"
#     else:
#         print("Using alignment: prev_to_current")
#         return "prev_to_current"


# def build_q_replay_from_delta(joint_pos, action_delta, alignment):
#     q0 = joint_pos[0, :6].astype(np.float64)
#     d = action_delta[:, :6].astype(np.float64)

#     T = len(joint_pos)

#     q_replay = np.zeros((T, 6), dtype=np.float64)
#     q_replay[0] = q0

#     if alignment == "current_to_next":
#         for i in range(T - 1):
#             q_replay[i + 1] = q_replay[i] + d[i]

#     elif alignment == "prev_to_current":
#         for i in range(1, T):
#             q_replay[i] = q_replay[i - 1] + d[i]

#     else:
#         raise ValueError(f"Unknown alignment: {alignment}")

#     return q_replay


# def check_delta_safety(action_delta):
#     d = action_delta[:, :6].astype(np.float64)

#     per_step_max = np.max(np.abs(d), axis=1)
#     global_max = float(np.max(per_step_max))
#     global_idx = int(np.argmax(per_step_max))

#     print("\n" + "=" * 80)
#     print("Action delta safety check")
#     print("=" * 80)
#     print(f"max abs joint delta: {global_max:.8f} rad")
#     print(f"max abs joint delta: {math.degrees(global_max):.4f} deg")
#     print(f"max delta index: {global_idx}")

#     bad_idx = np.where(per_step_max > MAX_DELTA_CHECK_RAD)[0]

#     if len(bad_idx) > 0:
#         print(f"[WARN] Found {len(bad_idx)} large delta steps.")
#         print(f"First bad index: {bad_idx[0]}")
#         print(f"First bad value: {per_step_max[bad_idx[0]]:.8f} rad")
#         return False

#     print("[OK] action_delta looks safe.")
#     return True


# def print_reconstruction_error(joint_pos, q_replay):
#     err = np.linalg.norm(q_replay - joint_pos[:, :6], axis=1)

#     print("\n" + "=" * 80)
#     print("Reconstruction error compared with recorded joint_pos")
#     print("=" * 80)
#     print(f"mean error: {float(np.mean(err)):.10f} rad")
#     print(f"max error : {float(np.max(err)):.10f} rad")
#     print(f"mean error: {math.degrees(float(np.mean(err))):.6f} deg")
#     print(f"max error : {math.degrees(float(np.max(err))):.6f} deg")


# def wait_for_controller_state(controller, timeout=5.0):
#     start = time.time()

#     while time.time() - start < timeout:
#         state = controller.get_state()
#         if state is not None:
#             return state
#         time.sleep(0.05)

#     raise RuntimeError("Timeout waiting for controller state.")


# def get_actual_q(controller):
#     state = controller.get_state()

#     if state is None:
#         state = wait_for_controller_state(controller, timeout=3.0)

#     if "ActualQ" not in state:
#         raise KeyError("controller state missing key: ActualQ")

#     return state["ActualQ"].astype(np.float64)


# def wait_until_close(controller, q_target, timeout=8.0, tol_rad=np.deg2rad(1.5)):
#     start = time.time()
#     last_err_vec = None
#     last_err_norm = None

#     while time.time() - start < timeout:
#         q_now = get_actual_q(controller)
#         err_vec = q_target - q_now
#         err_norm = float(np.linalg.norm(err_vec))

#         last_err_vec = err_vec
#         last_err_norm = err_norm

#         if err_norm < tol_rad:
#             return True, err_norm, err_vec

#         time.sleep(0.05)

#     return False, last_err_norm, last_err_vec


# def call_controller_servoj(controller, q_target_rad, duration):
#     """
#     通过 fr5_controller 调用 ServoJ。

#     你的 fr5_controller 里建议实现这个方法：

#         def servoj_joint_target(self, q_target_rad, duration=0.1):
#             ...

#     如果你的方法名不同，可以在这里改。
#     """

#     if hasattr(controller, "servoj_joint_target"):
#         return controller.servoj_joint_target(
#             q_target_rad=np.asarray(q_target_rad, dtype=np.float64),
#             duration=float(duration)
#         )

#     if hasattr(controller, "direct_servoj"):
#         return controller.direct_servoj(
#             q_target_rad=np.asarray(q_target_rad, dtype=np.float64),
#             duration=float(duration)
#         )

#     if hasattr(controller, "send_servoj_joint_target"):
#         return controller.send_servoj_joint_target(
#             q_target_rad=np.asarray(q_target_rad, dtype=np.float64),
#             duration=float(duration)
#         )

#     raise AttributeError(
#         "Your fr5_controller does not provide a direct ServoJ method. "
#         "Please add one method, for example: "
#         "servoj_joint_target(q_target_rad, duration). "
#         "Do not use schedule_joint_waypoint here, because it may clip or reshape the target."
#     )


# # ============================================================
# # Main
# # ============================================================

# def main():
#     print("\n" + "=" * 80)
#     print("Replay delta_action through fr5_controller")
#     print("=" * 80)
#     print(f"Episode dir: {EP_DIR}")

#     joint_pos_path = EP_DIR / "joint_pos.npy"
#     action_delta_path = EP_DIR / "action_delta.npy"

#     if not joint_pos_path.exists():
#         raise FileNotFoundError(f"Missing joint_pos.npy: {joint_pos_path}")

#     if not action_delta_path.exists():
#         raise FileNotFoundError(f"Missing action_delta.npy: {action_delta_path}")

#     joint_pos = np.load(joint_pos_path).astype(np.float64)
#     action_delta = np.load(action_delta_path).astype(np.float64)

#     print("joint_pos shape:", joint_pos.shape)
#     print("action_delta shape:", action_delta.shape)

#     if joint_pos.ndim != 2 or joint_pos.shape[1] < 6:
#         raise ValueError(f"joint_pos should be [T, >=6], got {joint_pos.shape}")

#     if action_delta.ndim != 2 or action_delta.shape[1] < 6:
#         raise ValueError(f"action_delta should be [T, >=6], got {action_delta.shape}")

#     T = min(len(joint_pos), len(action_delta))
#     joint_pos = joint_pos[:T]
#     action_delta = action_delta[:T]

#     dt = estimate_dt(EP_DIR, DEFAULT_SAMPLE_RATE)
#     replay_dt = dt * TIME_SCALE

#     print(f"Estimated dataset dt: {dt:.4f}s")
#     print(f"Replay dt: {replay_dt:.4f}s")
#     print(f"Replay sample rate: {1.0 / replay_dt:.2f} Hz")

#     safe = check_delta_safety(action_delta)
#     if not safe:
#         ans = input("Large delta detected. Continue anyway? [y/N]: ")
#         if ans.lower() != "y":
#             print("Abort for safety.")
#             return

#     alignment = infer_delta_alignment(joint_pos, action_delta)

#     q_replay = build_q_replay_from_delta(
#         joint_pos=joint_pos,
#         action_delta=action_delta,
#         alignment=alignment
#     )

#     print_reconstruction_error(joint_pos, q_replay)

#     print("\n" + "=" * 80)
#     print("Starting fr5_controller")
#     print("=" * 80)

#     controller = FR5ServoJController(
#         robot_ip=ROBOT_IP,
#         frequency=100,
#         joint_unit="rad",
#         verbose=True,
#     )

#     controller.start(wait=True)
#     print("fr5_controller ready.")

#     try:
#         q0 = q_replay[0].astype(np.float64)

#         print("\n" + "=" * 80)
#         print("Move to first recorded joint position")
#         print("=" * 80)
#         print("q0 deg:")
#         print(rad_to_deg(q0))

#         input("\nPress Enter to move to q0...")

#         # 这里也通过 fr5_controller 执行 ServoJ
#         # 连续发一小段时间，让机械臂靠近起点
#         move_start = time.time()
#         while time.time() - move_start < START_POSE_TIMEOUT:
#             call_controller_servoj(
#                 controller=controller,
#                 q_target_rad=q0,
#                 duration=5
#             )

#             ok, err_norm, err_vec = wait_until_close(
#                 controller=controller,
#                 q_target=q0,
#                 timeout=0.5,
#                 tol_rad=START_POSE_TOL_RAD
#             )

#             if ok:
#                 break

#             time.sleep(dt)

#         ok, err_norm, err_vec = wait_until_close(
#             controller=controller,
#             q_target=q0,
#             timeout=0.5,
#             tol_rad=START_POSE_TOL_RAD
#         )

#         print(f"Move to q0 ok: {ok}")
#         print(f"q0 error norm: {math.degrees(float(err_norm)):.4f} deg")
#         print("q0 error vec deg:")
#         print(rad_to_deg(err_vec))

#         if not ok:
#             ans = input("q0 error is still large. Continue replay? [y/N]: ")
#             if ans.lower() != "y":
#                 print("Abort.")
#                 return

#         input("\nPress Enter to start replay through fr5_controller...")

#         print("\n" + "=" * 80)
#         print("Start replay")
#         print("=" * 80)

#         start_time = time.time()

#         # current_to_next:
#         # q_replay[0] 是起点，后续从 q_replay[1] 开始执行
#         if alignment == "current_to_next":
#             indices = range(1, T)
#         else:
#             indices = range(1, T)

#         executed = 0

#         for step, i in enumerate(indices):
#             q_target = q_replay[i].astype(np.float64)

#             ret = call_controller_servoj(
#                 controller=controller,
#                 q_target_rad=q_target,
#                 duration=replay_dt
#             )

#             executed += 1

#             if step % 10 == 0:
#                 q_now = get_actual_q(controller)
#                 err_vec = q_target - q_now
#                 err_deg = rad_to_deg(err_vec)
#                 max_abs_err = float(np.max(np.abs(err_deg)))

#                 print(f"\n[Step {step:04d} | data index {i:04d}]")
#                 print("target deg:")
#                 print(rad_to_deg(q_target))
#                 print("actual deg:")
#                 print(rad_to_deg(q_now))
#                 print("tracking error deg:")
#                 print(err_deg)
#                 print(f"max abs tracking error: {max_abs_err:.4f} deg")
#                 print(f"ServoJ ret: {ret}")

#                 if max_abs_err > TRACKING_WARN_DEG:
#                     print(f"[WARN] tracking error > {TRACKING_WARN_DEG:.1f} deg")

#                 if STOP_ON_LARGE_TRACKING_ERROR and max_abs_err > TRACKING_STOP_DEG:
#                     print(f"[STOP] tracking error > {TRACKING_STOP_DEG:.1f} deg")
#                     break

#             time.sleep(replay_dt)

#         total_time = time.time() - start_time

#         print("\n" + "=" * 80)
#         print("Replay finished")
#         print("=" * 80)
#         print(f"Executed steps: {executed}")
#         print(f"Total time: {total_time:.2f}s")

#         q_final_target = q_replay[min(T - 1, executed)].astype(np.float64)
#         q_final_now = get_actual_q(controller)
#         final_err_deg = rad_to_deg(q_final_target - q_final_now)

#         print("Final tracking error deg:")
#         print(final_err_deg)
#         print(f"Final max abs tracking error: {float(np.max(np.abs(final_err_deg))):.4f} deg")

#     except KeyboardInterrupt:
#         print("\nInterrupted by user.")

#     finally:
#         print("\nStopping fr5_controller...")
#         controller.stop(wait=True)
#         print("fr5_controller stopped.")


# if __name__ == "__main__":
#     main()