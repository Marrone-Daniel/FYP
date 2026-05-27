import pygame
import time
import threading
import pyrobotiqgripper as rq
import sys
sys.path.append('/home/xjtlu/xbox_control/xbox_control')
from fairino import Robot
import math
import rclpy
import os
import signal
import shutil
import subprocess
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time as RosTime

class JointStatePublisherNode(Node):
    def __init__(self):
        super().__init__('fr5_gripper_joint_state_publisher')
        self.pub = self.create_publisher(JointState, '/joint_states', 20)

        # ===== 这里一定要改成你 URDF 里的真实关节名 =====
        self.arm_joint_names = [
            'j1',
            'j2',
            'j3',
            'j4',
            'j5',
            'j6'
        ]

        # 如果你的夹爪 URDF 使用 mimic，通常发布主驱动关节即可；
        # 但前提是这个 joint 名必须和 URDF 一致
        self.gripper_joint_name = 'gripper_robotiq_85_left_knuckle_joint'

        # 根据你的 URDF 调整这个值：夹爪从全开到全闭对应的弧度范围
        self.gripper_open_rad = 0.0
        self.gripper_closed_rad = 0.8

    def publish_joint_states(self, arm_deg, arm_vel_deg=None, gripper_raw=None):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        # 1) 机械臂：deg -> rad
        arm_rad = [math.radians(x) for x in arm_deg]

        msg.name = self.arm_joint_names.copy()
        msg.position = arm_rad

        # 2) 速度：deg/s -> rad/s
        if arm_vel_deg is not None and len(arm_vel_deg) == 6:
            msg.velocity = [math.radians(v) for v in arm_vel_deg]
        else:
            msg.velocity = []

        msg.effort = []

        # 3) 夹爪：把 0~255 映射到 URDF 里的关节弧度
        if gripper_raw is not None:
            g = max(0, min(255, int(gripper_raw)))
            ratio = g / 255.0
            gripper_joint = self.gripper_open_rad + \
                ratio * (self.gripper_closed_rad - self.gripper_open_rad)

            msg.name.append(self.gripper_joint_name)
            msg.position.append(gripper_joint)

            if msg.velocity is not None and len(msg.velocity) == 6:
                msg.velocity.append(0.0)

        self.pub.publish(msg)


# ===== 连接机器人 =====
robot = Robot.RPC('192.168.58.2')
print("FR5已连接")

robot.RobotEnable(1)
robot.ServoMoveStart()
print("Servo模式已开启")

# ===== 初始化手柄 =====
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("未检测到手柄")

js = pygame.joystick.Joystick(0)
js.init()
print("手柄已连接:", js.get_name())

# ===== 初始化夹爪（pyrobotiqgripper）=====
gripper = rq.RobotiqGripper(
    com_port="/dev/ttyUSB0",
    device_id=9,
    gripper_type="2F85",
    connection_type=rq.GRIPPER_MODE_RTU,
    debug=False
)

gripper.connect()
print("夹爪连接成功")

gripper.activate()
print("夹爪激活成功")

rclpy.init()
joint_state_node = JointStatePublisherNode()


# ===== 参数 =====
deadzone = 0.12
alpha = 0.25

scale_xy_slow = 3.0
scale_z_slow = 2.5
scale_rot_slow = 1.0
scale_pitch_slow = 1.2
scale_euler_slow = 1.0

scale_xy_fast = 8.0
scale_z_fast = 6.0
scale_rot_fast = 2.5
scale_pitch_fast = 3.0
scale_euler_fast = 2.5

cmdT = 0.01

# ===== 滤波状态 =====
vx = 0.0
vy = 0.0
vz = 0.0
vrz = 0.0
vpitch = 0.0
vroll = 0.0

# ===== 夹爪共享状态 =====
gripper_lock = threading.Lock()
gripper_running = True

gripper_target_pos = 128
gripper_min = 0
gripper_max = 255
gripper_step_slow = 4
gripper_step_fast = 10

gripper_last_sent_pos = None
gripper_status = None
gripper_status_time = 0.0

# 初始化到中间位置
gripper.move(gripper_target_pos)
time.sleep(0.5)


def dz(x, deadzone=0.12):
    return 0.0 if abs(x) < deadzone else x


def start_robot_state_publisher():
    xacro_path = os.path.expanduser(
        "~/fr5_ws/src/fr5_camera_gripper_moveit_config/config/fr5_gripper.urdf.xacro"
    )

    xacro_exe = shutil.which("xacro")
    if xacro_exe is None:
        raise RuntimeError("未找到 xacro, 请先 source ROS2 环境")

    result = subprocess.run(
        [xacro_exe, xacro_path],
        capture_output=True,
        text=True,
        check=True
    )

    robot_description = result.stdout

    cmd = [
        "ros2",
        "run",
        "robot_state_publisher",
        "robot_state_publisher",
        "--ros-args",
        "-p",
        f"robot_description:={robot_description}"
    ]

    return subprocess.Popen(cmd)


joint_state_running = True
def joint_state_worker():
    global joint_state_running, gripper_status

    publish_interval = 1.0 / 30.0   # 30 Hz

    while joint_state_running and rclpy.ok():
        try:
            # 机械臂关节角（degree）
            ret_j, joint_deg = robot.GetActualJointPosDegree(1)
            if ret_j != 0:
                time.sleep(publish_interval)
                continue

            # 机械臂关节速度（deg/s），失败了也没关系
            arm_vel_deg = None
            try:
                ret_v, vel_deg = robot.GetActualJointSpeedsDegree(1)
                if ret_v == 0 and len(vel_deg) == 6:
                    arm_vel_deg = vel_deg
            except Exception:
                arm_vel_deg = None

            # 夹爪当前状态：尽量从 status 里取当前位置
            gripper_raw = None
            with gripper_lock:
                local_status = gripper_status
                local_target = gripper_target_pos

            # 你这个 pyrobotiqgripper 库的 status() 返回字段，可能因版本不同而不同
            # 所以这里做一个兼容处理：优先取状态里的当前位置，取不到就退回 target_pos
            if isinstance(local_status, dict):
                for k in ['position', 'pos', 'requested_position', 'current_position']:
                    if k in local_status:
                        gripper_raw = local_status[k]
                        break

            if gripper_raw is None:
                gripper_raw = local_target

            joint_state_node.publish_joint_states(
                arm_deg=joint_deg,
                arm_vel_deg=arm_vel_deg,
                gripper_raw=gripper_raw
            )

            rclpy.spin_once(joint_state_node, timeout_sec=0.0)

        except Exception as e:
            print(f"[JointState Thread] 发布失败: {e}")

        time.sleep(publish_interval)


def gripper_worker():
    """
    后台线程：
    1. 只在目标位置变化时发送 move()
    2. 低频读取 status()
    """
    global gripper_running, gripper_last_sent_pos, gripper_status, gripper_status_time

    send_interval = 0.08      # 约 12.5 Hz
    status_interval = 0.2     # 约 5 Hz

    last_send_time = 0.0
    last_status_time = 0.0

    while gripper_running:
        now = time.time()

        # 读取共享目标位置
        with gripper_lock:
            target_pos = gripper_target_pos

        # 只有变化了才发命令，减少串口占用
        if (now - last_send_time >= send_interval) and (target_pos != gripper_last_sent_pos):
            try:
                gripper.move(target_pos)
                gripper_last_sent_pos = target_pos
            except Exception as e:
                print(f"[Gripper Thread] move失败: {e}")
            last_send_time = now

        # 低频读取状态
        if now - last_status_time >= status_interval:
            try:
                status = gripper.status()
                with gripper_lock:
                    gripper_status = status
                    gripper_status_time = now
            except Exception as e:
                print(f"[Gripper Thread] status读取失败: {e}")
            last_status_time = now

        time.sleep(0.01)


# ===== 启动夹爪后台线程 =====
gripper_thread = threading.Thread(target=gripper_worker, daemon=True)
gripper_thread.start()
print("夹爪后台线程已启动")

joint_state_thread = threading.Thread(target=joint_state_worker, daemon=True)
joint_state_thread.start()
print("/joint_states 后台发布线程已启动")


rsp_proc = None

try:
    rsp_proc = start_robot_state_publisher()
    print("robot_state_publisher 已启动")
except Exception as e:
    print(f"启动 robot_state_publisher 失败: {e}")


# ===== 获取初始位姿 =====
ret, pose = robot.GetActualTCPPose()
if ret != 0:
    raise RuntimeError(f"获取初始位姿失败，ret={ret}")

print("初始位姿:", pose)

try:
    while True:
        pygame.event.pump()

        # -----------------------------
        # 读取摇杆
        # -----------------------------
        lx = dz(js.get_axis(0), deadzone)
        ly = dz(js.get_axis(1), deadzone)
        rz_axis = dz(js.get_axis(4), deadzone) if js.get_numaxes() > 4 else 0.0

        # LB / RB 控制绕 Z 旋转
        lb = js.get_button(4) if js.get_numbuttons() > 4 else 0
        rb = js.get_button(5) if js.get_numbuttons() > 5 else 0

        rot_cmd = 0.0
        if lb and not rb:
            rot_cmd = -1.0
        elif rb and not lb:
            rot_cmd = 1.0

        # A/B 切换速度
        a_btn = js.get_button(0) if js.get_numbuttons() > 0 else 0
        b_btn = js.get_button(1) if js.get_numbuttons() > 1 else 0

        # X / Y 控制夹爪
        # 常见 Xbox 映射：X=2, Y=3
        x_btn = js.get_button(2) if js.get_numbuttons() > 2 else 0
        y_btn = js.get_button(3) if js.get_numbuttons() > 3 else 0

        # LT / RT -> ry
        lt_raw = js.get_axis(2) if js.get_numaxes() > 2 else -1.0
        rt_raw = js.get_axis(5) if js.get_numaxes() > 5 else -1.0

        lt_val = (lt_raw + 1.0) / 2.0
        rt_val = (rt_raw + 1.0) / 2.0
        pitch_cmd = rt_val - lt_val

        if b_btn:
            scale_xy = scale_xy_fast
            scale_z = scale_z_fast
            scale_rot = scale_rot_fast
            scale_pitch = scale_pitch_fast
            scale_euler = scale_euler_fast
            gripper_step = gripper_step_fast
        else:
            scale_xy = scale_xy_slow
            scale_z = scale_z_slow
            scale_rot = scale_rot_slow
            scale_pitch = scale_pitch_slow
            scale_euler = scale_euler_slow
            gripper_step = gripper_step_slow

        # 十字键
        hat_x, hat_y = (0, 0)
        if js.get_numhats() > 0:
            hat_x, hat_y = js.get_hat(0)

        roll_cmd = 0.0
        if hat_y == 1:
            roll_cmd = 1.0
        elif hat_y == -1:
            roll_cmd = -1.0

        # ===== 低通滤波 =====
        vx = -alpha * lx + (1 - alpha) * vx
        vy = alpha * ly + (1 - alpha) * vy
        vz = alpha * (-rz_axis) + (1 - alpha) * vz
        vrz = alpha * rot_cmd + (1 - alpha) * vrz
        vpitch = alpha * pitch_cmd + (1 - alpha) * vpitch
        vroll = alpha * roll_cmd + (1 - alpha) * vroll

        # ==================================================
        # 主线程只更新夹爪目标，不直接调 gripper.move()
        # 长按 X 持续关闭，长按 Y 持续打开
        # ==================================================
        with gripper_lock:
            if x_btn and not y_btn:
                gripper_target_pos = min(gripper_max, gripper_target_pos + gripper_step)
            elif y_btn and not x_btn:
                gripper_target_pos = max(gripper_min, gripper_target_pos - gripper_step)

        # ===== 如果机械臂没有输入，也允许夹爪单独工作 =====
        motion_small = (
            abs(vx) < 1e-3 and
            abs(vy) < 1e-3 and
            abs(vz) < 1e-3 and
            abs(vrz) < 1e-3 and
            abs(vpitch) < 1e-3 and
            abs(vroll) < 1e-3
        )

        # 可选：按 A 打印一次夹爪状态
        if a_btn:
            with gripper_lock:
                local_status = gripper_status
                local_target = gripper_target_pos
            print(f"[Gripper] target={local_target}, status={local_status}")

        if motion_small:
            time.sleep(cmdT)
            continue

        # ===== 获取当前位姿 =====
        ret, pose = robot.GetActualTCPPose()
        if ret != 0:
            print(f"获取当前位姿失败 ret={ret}")
            time.sleep(cmdT)
            continue

        target = pose.copy()

        # ===== 笛卡尔位置控制 =====
        target[0] += vx * scale_xy
        target[1] += vy * scale_xy
        target[2] += vz * scale_z

        # ===== 姿态控制 =====
        target[3] += vroll * scale_euler
        target[4] += vpitch * scale_pitch
        target[5] += vrz * scale_rot

        # ===== 发机械臂控制命令 =====
        robot.robot.ServoCart(
            0,
            target,
            [0.0, 0.0, 0.0, 0.0],
            0.0,
            0.0,
            cmdT,
            0.0,
            0.0
        )

        time.sleep(cmdT)

except KeyboardInterrupt:
    print("程序结束")

finally:
    gripper_running = False
    try:
        gripper_thread.join(timeout=1.0)
    except Exception:
        pass

    joint_state_running = False
    try:
        joint_state_thread.join(timeout=1.0)
    except Exception:
        pass

    try:
        joint_state_node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass

    robot.ServoMoveEnd()
    pygame.quit()

    if rsp_proc is not None:
        try:
            rsp_proc.send_signal(signal.SIGINT)
            rsp_proc.wait(timeout=2.0)
        except Exception:
            try:
                rsp_proc.kill()
            except Exception:
                pass
            
    try:
        gripper.disconnect()
    except Exception:
        pass

    print("Servo模式关闭，程序退出")