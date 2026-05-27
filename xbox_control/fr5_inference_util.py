import numpy as np



def get_fr5_obs_dict(env_obs, normalize_depth=False, max_depth=5.0):
    """
    Convert FR5RealEnv obs to policy input dict.

    Input from FR5RealEnv:
        image:      (T,H,W,3), uint8 RGB
        env_image:  (T,H,W,3), uint8 RGB
        env_depth:  (T,H,W), float32 meter
        agent_pos:  (T,7), float

    Output for policy:
        image:      (T,3,H,W), float32, [0,1]
        env_image:  (T,3,H,W), float32, [0,1]
        env_depth:  (T,1,H,W), float32
        agent_pos:  (T,7), float32
    """

    image = env_obs["image"]
    env_image = env_obs["env_image"]
    env_depth = env_obs["env_depth"]
    agent_pos = env_obs["agent_pos"]

    # RGB: uint8 HWC -> float32 CHW [0,1]
    image = image.astype(np.float32) / 255.0
    image = np.moveaxis(image, -1, 1)  # (T,H,W,3) -> (T,3,H,W)

    env_image = env_image.astype(np.float32) / 255.0
    env_image = np.moveaxis(env_image, -1, 1)

    # Depth: (T,H,W) -> (T,1,H,W)
    env_depth = env_depth.astype(np.float32)

    env_depth[~np.isfinite(env_depth)] = 0.0

    env_depth = np.clip(env_depth, 0.0, max_depth)

    if normalize_depth:
        depth = depth / max_depth
        env_depth = env_depth / max_depth

    env_depth = env_depth[:, None, :, :]  # (T,1,H,W)

    return {
        "image": image.astype(np.float32),
        "env_image": env_image.astype(np.float32),
        "env_depth": env_depth.astype(np.float32),
        "agent_pos": agent_pos.astype(np.float32),
    }


def make_safe_action(
    raw_action,
    current_agent_pos,
    max_joint_delta=0.003,
    gripper_binary=False,
    gripper_threshold=0.5,
):
    raw_action = np.asarray(raw_action, dtype=np.float64)
    current_agent_pos = np.asarray(current_agent_pos, dtype=np.float64)

    assert raw_action.shape == (7,)
    assert current_agent_pos.shape == (7,)

    current_joint = current_agent_pos[:6]
    raw_joint = raw_action[:6]

    safe_joint = current_joint + np.clip(
        raw_joint - current_joint,
        -max_joint_delta,
        max_joint_delta
    )

    raw_gripper = float(np.clip(raw_action[6], 0.0, 1.0))

    if gripper_binary:
        safe_gripper = 1.0 if raw_gripper >= gripper_threshold else 0.0
    else:
        safe_gripper = raw_gripper

    return np.concatenate([
        safe_joint,
        np.array([safe_gripper], dtype=np.float64)
    ])