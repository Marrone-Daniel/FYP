#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import numpy as np


def build_agent_pos(ep_dir: Path) -> np.ndarray:
    joint_path = ep_dir / "joint_pos.npy"
    grip_path = ep_dir / "gripper_state.npy"

    if not joint_path.exists():
        raise FileNotFoundError(f"Missing joint_pos.npy: {joint_path}")

    joint_pos = np.load(joint_path, allow_pickle=False).astype(np.float32)

    if joint_pos.ndim != 2:
        raise ValueError(f"{ep_dir.name}: joint_pos should be 2D, got {joint_pos.shape}")

    # Case 1: joint_pos already includes gripper, shape [T, 7]
    if joint_pos.shape[1] == 7:
        return joint_pos

    # Case 2: joint_pos is [T, 6], need gripper_state.npy
    if joint_pos.shape[1] == 6:
        if not grip_path.exists():
            raise FileNotFoundError(
                f"{ep_dir.name}: joint_pos is [T,6], but gripper_state.npy is missing."
            )

        gripper_state = np.load(grip_path, allow_pickle=False).astype(np.float32)

        if gripper_state.ndim == 1:
            gripper_state = gripper_state[:, None]

        if gripper_state.ndim != 2 or gripper_state.shape[1] != 1:
            raise ValueError(
                f"{ep_dir.name}: gripper_state should be [T,1], got {gripper_state.shape}"
            )

        T = min(len(joint_pos), len(gripper_state))
        return np.concatenate([joint_pos[:T], gripper_state[:T]], axis=1).astype(np.float32)

    raise ValueError(
        f"{ep_dir.name}: joint_pos should be [T,6] or [T,7], got {joint_pos.shape}"
    )


def make_action_delta(agent_pos: np.ndarray) -> np.ndarray:
    """
    action_delta[t] = agent_pos[t+1] - agent_pos[t]
    action_delta[-1] = 0
    """
    action_delta = np.zeros_like(agent_pos, dtype=np.float32)
    action_delta[:-1] = agent_pos[1:] - agent_pos[:-1]
    action_delta[-1] = 0.0
    return action_delta


def verify_reconstruction(agent_pos: np.ndarray, action_delta: np.ndarray):
    recon = np.zeros_like(agent_pos, dtype=np.float32)
    recon[0] = agent_pos[0]

    for i in range(len(agent_pos) - 1):
        recon[i + 1] = recon[i] + action_delta[i]

    err = recon - agent_pos
    joint_l2 = np.linalg.norm(err[:, :6], axis=1)
    max_abs = np.max(np.abs(err))

    return float(np.mean(joint_l2)), float(np.max(joint_l2)), float(max_abs)


def backup_old_action(ep_dir: Path):
    old_path = ep_dir / "action_delta.npy"
    if not old_path.exists():
        return None

    backup_path = ep_dir / "action_delta_old.npy"

    # 避免覆盖已有备份
    if backup_path.exists():
        idx = 2
        while True:
            candidate = ep_dir / f"action_delta_old_{idx}.npy"
            if not candidate.exists():
                backup_path = candidate
                break
            idx += 1

    shutil.copy2(old_path, backup_path)
    return backup_path


def process_episode(ep_dir: Path, no_backup: bool = False):
    print("\n" + "=" * 80)
    print(f"Episode: {ep_dir.name}")
    print("=" * 80)

    agent_pos = build_agent_pos(ep_dir)
    action_delta = make_action_delta(agent_pos)

    mean_l2, max_l2, max_abs = verify_reconstruction(agent_pos, action_delta)

    print(f"agent_pos shape: {agent_pos.shape}")
    print(f"new action_delta shape: {action_delta.shape}")
    print(f"reconstruction mean joint L2 error: {mean_l2:.8e} rad")
    print(f"reconstruction max  joint L2 error: {max_l2:.8e} rad")
    print(f"reconstruction max abs all-dim error: {max_abs:.8e}")

    if max_abs > 1e-5:
        raise RuntimeError(
            f"{ep_dir.name}: reconstruction error too large, max_abs={max_abs}"
        )

    if not no_backup:
        backup_path = backup_old_action(ep_dir)
        if backup_path is not None:
            print(f"[BACKUP] old action_delta.npy -> {backup_path.name}")
        else:
            print("[INFO] no old action_delta.npy found, no backup created.")

    out_path = ep_dir / "action_delta.npy"
    np.save(out_path, action_delta)

    print(f"[OK] replaced: {out_path}")

    joint_delta = action_delta[:, :6]
    grip_delta = action_delta[:, 6]

    print("new action stats:")
    print(f"  joint abs mean: {np.mean(np.abs(joint_delta)):.8e}")
    print(f"  joint abs max : {np.max(np.abs(joint_delta)):.8e}")
    print(f"  grip  abs mean: {np.mean(np.abs(grip_delta)):.8e}")
    print(f"  grip  abs max : {np.max(np.abs(grip_delta)):.8e}")


def find_episodes(root: Path):
    if root.name.startswith("episode_"):
        return [root]

    episodes = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("episode_")
    )
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=False,
        help="Dataset root or one episode dir, e.g. /home/xjtlu/xbox_control/fr5_dp_data_pile",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not backup old action_delta.npy.",
    )
    args = parser.parse_args()

    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
    else:
        root = Path("/home/xjtlu/xbox_control/fr5_dp_data_pile")

    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    episodes = find_episodes(root)

    if not episodes:
        raise RuntimeError(f"No episode_xxxxxx directories found in {root}")

    print(f"Found {len(episodes)} episode(s).")

    ok = 0
    failed = 0

    for ep in episodes:
        try:
            process_episode(ep, no_backup=args.no_backup)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAILED] {ep}: {e}")

    print("\n" + "#" * 80)
    print("Final Summary")
    print("#" * 80)
    print(f"OK episodes    : {ok}")
    print(f"Failed episodes: {failed}")


if __name__ == "__main__":
    main()