import time
import threading
import numpy as np
import pyrobotiqgripper as rq


class RobotiqGripperController:
    def __init__(
        self,
        com_port="/dev/ttyUSB0",
        device_id=9,
        gripper_type="2F85",
        min_pos=0,
        max_pos=255,
        initial_pos=None,
        send_interval=0.08,
        status_interval=0.2,
        debug=False,
    ):
        self.com_port = com_port
        self.device_id = device_id
        self.gripper_type = gripper_type
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.initial_pos = initial_pos
        self.send_interval = send_interval
        self.status_interval = status_interval
        self.debug = debug

        self.gripper = None
        self._is_ready = False

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._target_pos = initial_pos if initial_pos is not None else min_pos
        self._last_sent_pos = None
        self._status = None
        self._status_time = 0.0

    @property
    def is_ready(self):
        return self._is_ready

    def start(self, wait=True):
        self.gripper = rq.RobotiqGripper(
            com_port=self.com_port,
            device_id=self.device_id,
            gripper_type=self.gripper_type,
            connection_type=rq.GRIPPER_MODE_RTU,
            debug=self.debug
        )

        self.gripper.connect()
        self.gripper.activate()

        if self.initial_pos is not None:
            self.gripper.move(int(self.initial_pos))
            time.sleep(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._is_ready = True

        print("[RobotiqGripperController] started.")

    def stop(self, wait=True):
        self._running = False
        self._is_ready = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except Exception:
                pass

        print("[RobotiqGripperController] stopped.")

    def set_gripper_state(self, value):
        """
        value: [0, 1]
            0.0 = open
            1.0 = close
        """
        value = float(np.clip(value, 0.0, 1.0))
        pos = int(self.min_pos + value * (self.max_pos - self.min_pos))

        with self._lock:
            self._target_pos = pos

    def set_gripper_raw(self, pos):
        pos = int(np.clip(pos, self.min_pos, self.max_pos))
        with self._lock:
            self._target_pos = pos

    def get_gripper_state(self):
        """
        返回归一化状态 [0, 1]。
        优先从 status 中取当前位置，取不到就返回 target。
        """
        with self._lock:
            status = self._status
            target_pos = self._target_pos

        raw = None
        if isinstance(status, dict):
            for k in ["position", "pos", "requested_position", "current_position"]:
                if k in status:
                    raw = status[k]
                    break

        if raw is None:
            raw = target_pos

        raw = int(np.clip(raw, self.min_pos, self.max_pos))
        return float((raw - self.min_pos) / (self.max_pos - self.min_pos))

    def get_gripper_raw(self):
        with self._lock:
            status = self._status
            target_pos = self._target_pos

        raw = None
        if isinstance(status, dict):
            for k in ["position", "pos", "requested_position", "current_position"]:
                if k in status:
                    raw = status[k]
                    break

        if raw is None:
            raw = target_pos

        return int(np.clip(raw, self.min_pos, self.max_pos))

    def get_status(self):
        with self._lock:
            return self._status

    def _worker(self):
        last_send_time = 0.0
        last_status_time = 0.0

        while self._running:
            now = time.time()

            with self._lock:
                target_pos = self._target_pos

            if (now - last_send_time >= self.send_interval) and (target_pos != self._last_sent_pos):
                try:
                    self.gripper.move(int(target_pos))
                    self._last_sent_pos = target_pos
                except Exception as e:
                    print(f"[RobotiqGripperController] move failed: {e}")
                last_send_time = now

            if now - last_status_time >= self.status_interval:
                try:
                    status = self.gripper.status()
                    with self._lock:
                        self._status = status
                        self._status_time = now
                except Exception as e:
                    print(f"[RobotiqGripperController] status failed: {e}")
                last_status_time = now

            time.sleep(0.01)