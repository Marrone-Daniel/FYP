import time
import numpy as np
from fr5_controller import FR5ServoJController

ROBOT_IP = "192.168.58.2"

def main():
    ctrl = FR5ServoJController(
        robot_ip=ROBOT_IP,
        frequency=100,
        joint_unit="rad",
        max_joint_delta=np.deg2rad(0.5),
        max_servo_step=np.deg2rad(0.05),
        vel=20,
        verbose=True,
    )

    ctrl.start()
    print("子进程 PID:", ctrl.pid)  
    print("is_ready:", ctrl.is_ready)

    time.sleep(0.5)
    state = ctrl.get_state()
    print("state:", state)

    if state is None:
        print("No state yet, wait more...")
        time.sleep(1.0)
        state = ctrl.get_state()

    current = state["ActualQ"]
    print("current deg:", np.rad2deg(current))

    target = current.copy()
    target[0] += np.deg2rad(0.5)

    input("确认安全后，按 Enter 发送非阻塞目标...")

    t0 = time.time()
    ctrl.schedule_joint_waypoint(
        joint_pos=target,
        gripper=0.0,
        target_time=time.time() + 0.1
    )
    print("schedule call cost:", time.time() - t0)

    # 注意：主线程没有被 ServoJ 循环卡住
    for i in range(10):
        time.sleep(0.2)
        state = ctrl.get_state()
        if state is not None:
            print(i, "actual deg:", np.rad2deg(state["ActualQ"]))

    ctrl.stop()
    print("done")

if __name__ == "__main__":
    main()