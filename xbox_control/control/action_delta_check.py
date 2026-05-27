#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def load_episode(ep_dir: Path):
    joint_pos = np.load(ep_dir / "joint_pos.npy")          # [T, 6], rad
    action_delta = np.load(ep_dir / "action_delta.npy")    # [T, 7] or [T, 6/7]

    print("=" * 80)
    print(f"Episode: {ep_dir}")
    print("=" * 80)
    print("joint_pos shape:", joint_pos.shape, "dtype:", joint_pos.dtype)
    print("action_delta shape:", action_delta.shape, "dtype:", action_delta.dtype)

    T = min(len(joint_pos), len(action_delta))
    joint_pos = joint_pos[:T, :6].astype(np.float64)
    action_delta = action_delta[:T, :6].astype(np.float64)

    return joint_pos, action_delta


def print_stats(name, x):
    x = np.asarray(x)
    print(f"\n{name}:")
    print(f"  shape : {x.shape}")
    print(f"  mean  : {np.mean(x):.8e}")
    print(f"  std   : {np.std(x):.8e}")
    print(f"  min   : {np.min(x):.8e}")
    print(f"  max   : {np.max(x):.8e}")
    print(f"  abs mean : {np.mean(np.abs(x)):.8e}")
    print(f"  abs max  : {np.max(np.abs(x)):.8e}")


def reconstruct_current_to_next(joint_pos, action_delta):
    """
    假设：
    action_delta[i] = joint_pos[i+1] - joint_pos[i]

    所以：
    q_recon[0] = joint_pos[0]
    q_recon[i+1] = q_recon[i] + action_delta[i]
    """
    T = len(joint_pos)
    q_recon = np.zeros_like(joint_pos)
    q_recon[0] = joint_pos[0]

    for i in range(T - 1):
        q_recon[i + 1] = q_recon[i] + action_delta[i]

    return q_recon


def reconstruct_prev_to_current(joint_pos, action_delta):
    """
    假设：
    action_delta[i] = joint_pos[i] - joint_pos[i-1]

    所以：
    q_recon[0] = joint_pos[0]
    q_recon[i] = q_recon[i-1] + action_delta[i]
    """
    T = len(joint_pos)
    q_recon = np.zeros_like(joint_pos)
    q_recon[0] = joint_pos[0]

    for i in range(1, T):
        q_recon[i] = q_recon[i - 1] + action_delta[i]

    return q_recon


def reconstruction_error(joint_pos, q_recon):
    """
    返回每一帧的关节向量 L2 误差，以及每一维绝对误差。
    """
    err = q_recon - joint_pos
    err_l2 = np.linalg.norm(err, axis=1)
    err_abs = np.abs(err)
    return err, err_l2, err_abs


def compare_direct_delta(joint_pos, action_delta):
    """
    直接比较 action_delta 和 joint_pos 相邻差分。
    """
    true_delta_forward = joint_pos[1:] - joint_pos[:-1]   # [T-1, 6]

    # 对齐方式 A:
    # action_delta[0:T-1] 应该等于 joint_pos[1:] - joint_pos[:-1]
    pred_a = action_delta[:-1]
    diff_a = pred_a - true_delta_forward

    # 对齐方式 B:
    # action_delta[1:T] 应该等于 joint_pos[1:] - joint_pos[:-1]
    pred_b = action_delta[1:]
    diff_b = pred_b - true_delta_forward

    return true_delta_forward, diff_a, diff_b


def evaluate(ep_dir: Path, save_recon: bool = True):
    joint_pos, action_delta = load_episode(ep_dir)

    print_stats("joint_pos", joint_pos)
    print_stats("action_delta[:, :6]", action_delta)

    true_delta_forward, diff_a, diff_b = compare_direct_delta(joint_pos, action_delta)

    print_stats("true joint delta = joint_pos[i+1] - joint_pos[i]", true_delta_forward)

    print("\n" + "=" * 80)
    print("Direct delta comparison")
    print("=" * 80)

    print("\nCase A: action_delta[i] ?= joint_pos[i+1] - joint_pos[i]")
    print_stats("diff_a = action_delta[:-1] - true_delta_forward", diff_a)
    print("  mean L2 error:", np.mean(np.linalg.norm(diff_a, axis=1)))
    print("  max  L2 error:", np.max(np.linalg.norm(diff_a, axis=1)))

    print("\nCase B: action_delta[i+1] ?= joint_pos[i+1] - joint_pos[i]")
    print_stats("diff_b = action_delta[1:] - true_delta_forward", diff_b)
    print("  mean L2 error:", np.mean(np.linalg.norm(diff_b, axis=1)))
    print("  max  L2 error:", np.max(np.linalg.norm(diff_b, axis=1)))

    # 积分还原
    q_recon_a = reconstruct_current_to_next(joint_pos, action_delta)
    q_recon_b = reconstruct_prev_to_current(joint_pos, action_delta)

    err_a, err_l2_a, err_abs_a = reconstruction_error(joint_pos, q_recon_a)
    err_b, err_l2_b, err_abs_b = reconstruction_error(joint_pos, q_recon_b)

    print("\n" + "=" * 80)
    print("Reconstruction comparison")
    print("=" * 80)

    print("\nCase A reconstruction:")
    print("  q[0] = joint_pos[0]")
    print("  q[i+1] = q[i] + action_delta[i]")
    print(f"  mean L2 error: {np.mean(err_l2_a):.8e} rad")
    print(f"  max  L2 error: {np.max(err_l2_a):.8e} rad")
    print(f"  mean abs joint error: {np.mean(err_abs_a):.8e} rad")
    print(f"  max  abs joint error: {np.max(err_abs_a):.8e} rad")

    print("\nCase B reconstruction:")
    print("  q[0] = joint_pos[0]")
    print("  q[i] = q[i-1] + action_delta[i]")
    print(f"  mean L2 error: {np.mean(err_l2_b):.8e} rad")
    print(f"  max  L2 error: {np.max(err_l2_b):.8e} rad")
    print(f"  mean abs joint error: {np.mean(err_abs_b):.8e} rad")
    print(f"  max  abs joint error: {np.max(err_abs_b):.8e} rad")

    if np.mean(err_l2_a) < np.mean(err_l2_b):
        best_case = "A"
        best_err = np.mean(err_l2_a)
    else:
        best_case = "B"
        best_err = np.mean(err_l2_b)

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)

    print(f"Best alignment: Case {best_case}")
    print(f"Best mean L2 reconstruction error: {best_err:.8e} rad")

    if best_err < 1e-4:
        print("[PASS] action_delta can accurately reconstruct joint_pos.")
        print("       你的 action_delta 基本就是关节增量。")
    elif best_err < 1e-2:
        print("[WARN] action_delta can roughly reconstruct joint_pos, but there is noticeable drift.")
        print("       可能存在单位误差、对齐误差、采样丢帧，或者 action_delta 不是严格 joint diff。")
    else:
        print("[FAIL] action_delta cannot reconstruct joint_pos well.")
        print("       高概率说明 action_delta 不是关节增量，或者构建 action_delta 的逻辑有问题。")

    # 保存还原结果，方便后续可视化/对比
    if save_recon:
        out_a = ep_dir / "joint_recon_from_action_caseA.npy"
        out_b = ep_dir / "joint_recon_from_action_caseB.npy"
        np.save(out_a, q_recon_a)
        np.save(out_b, q_recon_b)
        print("\nSaved:")
        print(" ", out_a)
        print(" ", out_b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode_dir",
        type=str,
        default="/home/xjtlu/xbox_control/fr5_dp_data_pile/episode_000040",
        help="Path to one episode directory"
    )
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir).expanduser().resolve()
    if not ep_dir.exists():
        raise FileNotFoundError(f"Episode not found: {ep_dir}")

    evaluate(ep_dir)


if __name__ == "__main__":
    main()