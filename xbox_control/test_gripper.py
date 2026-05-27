import time
from gripper_controller import RobotiqGripperController


def main():
    gripper = RobotiqGripperController(
        com_port="/dev/ttyUSB0",
        device_id=9,
        gripper_type="2F85",
        initial_pos=128,
        min_pos=0,
        max_pos=255,
    )

    gripper.start()
    time.sleep(1.0)

    print("state:", gripper.get_gripper_state())
    print("raw:", gripper.get_gripper_raw())
    print("status:", gripper.get_status())

    input("按 Enter 打开夹爪...")
    gripper.set_gripper_state(0.0)
    time.sleep(1.5)
    print("state:", gripper.get_gripper_state())
    print("raw:", gripper.get_gripper_raw())
    print("status:", gripper.get_status())

    input("按 Enter 关闭夹爪...")
    gripper.set_gripper_state(1.0)
    time.sleep(1.5)
    print("state:", gripper.get_gripper_state())
    print("raw:", gripper.get_gripper_raw())
    print("status:", gripper.get_status())

    input("按 Enter 回到中间...")
    gripper.set_gripper_raw(128)
    time.sleep(1.0)

    gripper.stop()


if __name__ == "__main__":
    main()