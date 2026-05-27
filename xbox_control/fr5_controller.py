import time
import enum
import multiprocessing as mp
import numpy as np
from fairino import Robot


class Command(enum.Enum):
    STOP = 0
    SERVOJ_TARGET = 1


class FR5ServoJController(mp.Process):
    """
    功能：
    1. 持续读取 FR5 当前关节角 ActualQ，供 replay / policy observation 使用
    2. 提供 servoj_joint_target(q_target_rad, duration) 接口
    3. replay 脚本不直接 Robot.RPC，而是通过这个 controller 调用 ServoJ
    """

    def __init__(
        self,
        robot_ip,
        frequency=100,
        joint_unit="rad",
        launch_timeout=30,
        verbose=False,
    ):
        super().__init__(name="FR5ServoJController")

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.joint_unit = joint_unit
        self.launch_timeout = launch_timeout
        self.verbose = verbose

        self.input_queue = mp.Queue()
        self.state_queue = mp.Queue(maxsize=1)

        self.ready_event = mp.Event()
        self._last_state = None

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self):
        ok = self.ready_event.wait(self.launch_timeout)
        if not ok:
            raise RuntimeError("FR5ServoJController failed to start.")
        if not self.is_alive():
            raise RuntimeError("FR5ServoJController process died.")

    def stop(self, wait=True):
        self.input_queue.put({
            "cmd": Command.STOP.value
        })

        if wait:
            self.join()

    def get_state(self):
        latest = None

        while not self.state_queue.empty():
            latest = self.state_queue.get()

        if latest is not None:
            self._last_state = latest

        return self._last_state

    def wait_for_state(self, timeout=3.0):
        start = time.time()

        while time.time() - start < timeout:
            state = self.get_state()
            if state is not None:
                return state
            time.sleep(0.02)

        raise RuntimeError("Timeout waiting for FR5 state.")

    # ============================================================
    # 这个就是 replay 脚本需要调用的接口
    # ============================================================

    def servoj_joint_target(self, q_target_rad, duration=0.1):
        """
        直接 ServoJ 到目标关节角。

        参数：
            q_target_rad: shape=(6,), 单位 rad
            duration: ServoJ 单步时间，通常等于数据集 dt，例如 0.1s

        注意：
            这里不做 max_joint_delta
            这里不做 max_servo_step
            这里不改变目标轨迹
        """
        q_target_rad = np.asarray(q_target_rad, dtype=np.float64)

        if q_target_rad.shape != (6,):
            raise ValueError(f"q_target_rad should be shape (6,), got {q_target_rad.shape}")

        if not np.all(np.isfinite(q_target_rad)):
            raise ValueError("q_target_rad contains NaN or Inf")

        self.input_queue.put({
            "cmd": Command.SERVOJ_TARGET.value,
            "q_target_rad": q_target_rad,
            "duration": float(duration),
        })

        return 0

    # ============================================================
    # SDK helpers
    # ============================================================

    def _parse_fairino_ret(self, ret, expected_len=None, name="SDK call"):
        """
        FAIRINO SDK 常见返回：
        (0, data) 正常
        err_code 异常
        """

        if isinstance(ret, tuple):
            err_code = ret[0]

            if err_code != 0:
                raise RuntimeError(f"{name} failed, err_code={err_code}, ret={ret}")

            if len(ret) < 2:
                raise RuntimeError(f"{name} returned tuple without data: ret={ret}")

            data = ret[1]

        else:
            if isinstance(ret, (int, float, np.integer, np.floating)):
                raise RuntimeError(f"{name} failed or returned error code: ret={ret}")

            data = ret

        data = np.asarray(data, dtype=np.float64)

        if expected_len is not None and data.shape != (expected_len,):
            raise ValueError(
                f"{name} expected shape ({expected_len},), got {data.shape}, ret={ret}"
            )

        return data

    def _get_joint_positions(self, robot):
        ret = robot.GetActualJointPosDegree()

        joint_deg = self._parse_fairino_ret(
            ret,
            expected_len=6,
            name="GetActualJointPosDegree"
        )

        if self.joint_unit == "rad":
            return np.deg2rad(joint_deg)

        return joint_deg

    def _get_tcp_pose(self, robot):
        ret = robot.GetActualTCPPose()

        tcp_pose = self._parse_fairino_ret(
            ret,
            expected_len=6,
            name="GetActualTCPPose"
        )

        return tcp_pose

    def _put_latest_state(self, state):
        """
        state_queue 只保留最新状态，避免队列堆积。
        """
        try:
            while not self.state_queue.empty():
                self.state_queue.get_nowait()
        except Exception:
            pass

        try:
            self.state_queue.put_nowait(state)
        except Exception:
            pass

    # ============================================================
    # Process main loop
    # ============================================================

    def run(self):
        robot = Robot.RPC(self.robot_ip)
        time.sleep(2.0)

        try:
            ret = robot.RobotEnable(1)
            if self.verbose:
                print("[FR5StateController] RobotEnable ret:", ret)

            ret = robot.ServoMoveStart()
            if self.verbose:
                print("[FR5ServoJController] ServoMoveStart ret:", ret)

            time.sleep(1.0)

            epos = [0.0, 0.0, 0.0, 0.0]

            dt = 1.0 / self.frequency
            next_tick = time.perf_counter()

            self.ready_event.set()

            keep_running = True

            while keep_running:
                # ====================================================
                # 1. 处理外部 ServoJ target 命令
                # ====================================================
                try:
                    while True:
                        msg = self.input_queue.get_nowait()
                        cmd = msg["cmd"]

                        if cmd == Command.STOP.value:
                            keep_running = False
                            break

                        elif cmd == Command.SERVOJ_TARGET.value:
                            q_target_rad = np.asarray(
                                msg["q_target_rad"],
                                dtype=np.float64
                            )

                            duration = float(msg.get("duration", 0.1))

                            if self.joint_unit == "rad":
                                q_target_deg = np.rad2deg(q_target_rad)
                            else:
                                q_target_deg = q_target_rad

                            ret = robot.ServoJ(
                                q_target_deg.tolist(),
                                epos,
                                0.0,
                                0.0,
                                duration,
                                0.0,
                                0.0
                            )

                            if self.verbose:
                                print("[FR5ServoJController] ServoJ target deg:", q_target_deg)
                                print("[FR5ServoJController] ServoJ ret:", ret)

                except Exception:
                    pass

                if not keep_running:
                    break

                # ====================================================
                # 2. 持续更新机械臂状态
                # ====================================================
                try:
                    q = self._get_joint_positions(robot)
                    tcp_pose = self._get_tcp_pose(robot)

                    state = {
                        "ActualQ": q,
                        "ActualTCPPose": tcp_pose,
                        "robot_receive_timestamp": time.time(),
                    }

                    self._put_latest_state(state)

                except Exception as e:
                    if self.verbose:
                        print("[FR5ServoJController] state update failed:", e)

                next_tick += dt
                sleep_time = next_tick - time.perf_counter()

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_tick = time.perf_counter()

        finally:
            try:
                robot.ServoMoveEnd()
            except Exception as e:
                print("[FR5ServoJController] ServoMoveEnd failed:", e)

            self.ready_event.set()

            if self.verbose:
                print("[FR5ServoJController] stopped.")
