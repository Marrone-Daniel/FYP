import time
import numpy as np

from fr5_controller import FR5ServoJController
from gripper_controller import RobotiqGripperController
from camera_controller import DualRosRGBDCameraController
from fr5_env import FR5RealEnv
from fr5_inference_util import get_fr5_obs_dict


def main():
    robot = FR5ServoJController(
        robot_ip="192.168.58.2",
        frequency=100,
        joint_unit="rad",
        max_joint_delta=np.deg2rad(0.1),
        max_servo_step=np.deg2rad(0.02),
        vel=20,
        verbose=False,
    )

    camera = DualRosRGBDCameraController(
        wrist_color_topic="/camera/color/image_raw/compressed",
        env_color_topic="/env_camera/color/image_raw",
        env_depth_topic="/env_camera/depth/image_raw",
        width=160,
        height=160,
        depth_scale=0.001,
        max_depth=5.0,
        timeout=8.0
    )

    gripper = RobotiqGripperController(
        com_port="/dev/ttyUSB0",
        device_id=9,
        gripper_type="2F85",
        initial_pos=128,
        min_pos=0,
        max_pos=255,
    )

    env = FR5RealEnv(
        robot_controller=robot,
        camera_controller=camera,
        gripper_controller=gripper,
        frequency=10,
        n_obs_steps=2,
    )

    try:
        env.start()
        print("env ready:", env.is_ready)

        for i in range(5):
            obs = env.get_obs()

            print(f"\n--- obs {i} ---")
            print("image:", obs["image"].shape, obs["image"].dtype, obs["image"].min(), obs["image"].max())
            print("env_image:", obs["env_image"].shape, obs["env_image"].dtype, obs["env_image"].min(), obs["env_image"].max())
            print("env_depth:", obs["env_depth"].shape, obs["env_depth"].dtype, obs["env_depth"].min(), obs["env_depth"].max())
            print("agent_pos:", obs["agent_pos"].shape, obs["agent_pos"].dtype)
            print("agent_pos last:", obs["agent_pos"][-1])
            print("timestamp:", obs["timestamp"])

            obs_dict = get_fr5_obs_dict(obs, normalize_depth=False, max_depth=5.0)
            print("policy image:", obs_dict["image"].shape, obs_dict["image"].dtype)
            print("policy depth:", obs_dict["depth"].shape, obs_dict["depth"].dtype)
            print("policy env_image:", obs_dict["env_image"].shape, obs_dict["env_image"].dtype)
            print("policy env_depth:", obs_dict["env_depth"].shape, obs_dict["env_depth"].dtype)
            print("policy agent_pos:", obs_dict["agent_pos"].shape, obs_dict["agent_pos"].dtype)

            time.sleep(0.2)

    finally:
        env.stop()
        print("done")


if __name__ == "__main__":
    main()