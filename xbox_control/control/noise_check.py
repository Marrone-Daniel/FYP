#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Basic utils
# ============================================================

def load_npy(path: Path):
    if path.exists():
        return np.load(path, allow_pickle=True)
    return None


def to_float_array(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def metric_stats(x):
    if x is None:
        return None

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None

    valid = np.isfinite(x)
    if not np.any(valid):
        return None

    x = x[valid]
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
    }


# ============================================================
# Timestamp lag
# ============================================================

def compute_lag(main_ts, sensor_ts):
    """
    lag = timestamps.npy - sensor_timestamp.npy

    Positive lag means sensor frame is older than sample time.
    """
    if main_ts is None or sensor_ts is None:
        return None

    main_ts = to_float_array(main_ts)
    sensor_ts = to_float_array(sensor_ts)

    n = min(len(main_ts), len(sensor_ts))
    if n == 0:
        return None

    return main_ts[:n] - sensor_ts[:n]


# ============================================================
# Spike detection
# ============================================================

def robust_spike_mask(x, z_th=4.0):
    """
    Use robust z-score based on MAD to detect spikes.
    """
    if x is None:
        return None

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros_like(x, dtype=bool)

    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=bool)

    x_valid = x[valid]
    median = np.median(x_valid)
    mad = np.median(np.abs(x_valid - median))

    mask = np.zeros_like(x, dtype=bool)

    if mad < 1e-12:
        std = np.std(x_valid)
        if std < 1e-12:
            return mask
        z = np.abs((x - np.mean(x_valid)) / std)
    else:
        z = np.abs(0.6745 * (x - median) / mad)

    mask[np.isfinite(z)] = z[np.isfinite(z)] > z_th
    return mask


def vector_norm(x):
    if x is None:
        return None

    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        return np.abs(x)

    return np.linalg.norm(x, axis=-1)


def step_norm(x):
    """
    Compute ||x[t] - x[t-1]||.
    Return length T, with first value = 0.
    """
    if x is None:
        return None

    x = np.asarray(x, dtype=np.float64)

    if len(x) < 2:
        return np.zeros(len(x), dtype=np.float64)

    diff = np.diff(x, axis=0)

    if diff.ndim == 1:
        step = np.abs(diff)
    else:
        step = np.linalg.norm(diff, axis=-1)

    return np.concatenate([[0.0], step])


# ============================================================
# Depth quality
# ============================================================

def squeeze_depth(depth):
    if depth is None:
        return None

    depth = np.asarray(depth)

    # [T, H, W, 1] -> [T, H, W]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    # [T, 1, H, W] -> [T, H, W]
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]

    return depth


def depth_valid_ratio(depth):
    """
    Per-frame valid depth ratio.
    valid = finite and depth > 0
    """
    depth = squeeze_depth(depth)
    if depth is None:
        return None

    if depth.ndim != 3:
        return None

    valid = np.isfinite(depth) & (depth > 0)
    return valid.reshape(valid.shape[0], -1).mean(axis=1)


def depth_frame_jump(depth):
    """
    Per-frame mean absolute depth difference.
    Compare depth[t] and depth[t-1] only on pixels valid in both frames.
    """
    depth = squeeze_depth(depth)
    if depth is None:
        return None

    if depth.ndim != 3 or len(depth) < 2:
        return None

    depth = depth.astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0)

    jumps = []
    for i in range(1, len(depth)):
        mask = valid[i] & valid[i - 1]
        if np.any(mask):
            jump = np.mean(np.abs(depth[i][mask] - depth[i - 1][mask]))
        else:
            jump = np.nan
        jumps.append(jump)

    return np.asarray(jumps, dtype=np.float64)


# ============================================================
# Image/depth frame preview
# ============================================================

def prepare_rgb_frame(rgb):
    if rgb is None:
        return None

    rgb = np.asarray(rgb)
    if len(rgb) == 0:
        return None

    frame = rgb[len(rgb) // 2]

    # CHW -> HWC
    if frame.ndim == 3 and frame.shape[0] in [1, 3] and frame.shape[-1] not in [1, 3]:
        frame = np.transpose(frame, (1, 2, 0))

    # BGR/RGB 判断这里不强制转换，主要用于质量预览
    if frame.dtype != np.uint8:
        f = frame.astype(np.float32)
        vmin, vmax = np.nanmin(f), np.nanmax(f)
        if vmax > vmin:
            frame = ((f - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(f, dtype=np.uint8)

    return frame


def prepare_depth_frame(depth):
    depth = squeeze_depth(depth)
    if depth is None:
        return None

    if len(depth) == 0:
        return None

    frame = depth[len(depth) // 2].astype(np.float64)
    valid = np.isfinite(frame) & (frame > 0)

    if not np.any(valid):
        return frame

    values = frame[valid]
    lo = np.percentile(values, 2)
    hi = np.percentile(values, 98)

    frame_vis = frame.copy()
    frame_vis[~valid] = np.nan
    frame_vis = np.clip(frame_vis, lo, hi)

    return frame_vis


# ============================================================
# Plot helpers
# ============================================================

def plot_series(ax, x, title, ylabel, spike_mask=None):
    if x is None:
        ax.set_title(title + " (missing)")
        ax.axis("off")
        return

    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        ax.set_title(title + " (empty)")
        ax.axis("off")
        return

    ax.plot(x, linewidth=1.2)

    if spike_mask is not None and len(spike_mask) == len(x):
        idx = np.where(spike_mask)[0]
        if len(idx) > 0:
            ax.scatter(idx, x[idx], s=20, marker="x", label="spike")
            ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


# ============================================================
# Main analysis
# ============================================================

def analyze_episode(episode_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Load arrays
    # --------------------------
    timestamps = load_npy(episode_dir / "timestamps.npy")

    rgb_timestamps = load_npy(episode_dir / "rgb_timestamps.npy")
    depth_timestamps = load_npy(episode_dir / "depth_timestamps.npy")
    env_rgb_timestamps = load_npy(episode_dir / "env_rgb_timestamps.npy")
    env_depth_timestamps = load_npy(episode_dir / "env_depth_timestamps.npy")

    action_delta = load_npy(episode_dir / "action_delta.npy")
    joint_pos = load_npy(episode_dir / "joint_pos.npy")
    eef_pose = load_npy(episode_dir / "eef_pose.npy")
    gripper_state = load_npy(episode_dir / "gripper_state.npy")

    rgb_frames = load_npy(episode_dir / "rgb_frames.npy")
    depth_frames = load_npy(episode_dir / "depth_frames.npy")
    env_rgb_frames = load_npy(episode_dir / "env_rgb_frames.npy")
    env_depth_frames = load_npy(episode_dir / "env_depth_frames.npy")

    depth_frames = squeeze_depth(depth_frames)
    env_depth_frames = squeeze_depth(env_depth_frames)

    # --------------------------
    # Time lag
    # --------------------------
    rgb_lag = compute_lag(timestamps, rgb_timestamps)
    depth_lag = compute_lag(timestamps, depth_timestamps)
    env_rgb_lag = compute_lag(timestamps, env_rgb_timestamps)
    env_depth_lag = compute_lag(timestamps, env_depth_timestamps)

    # --------------------------
    # Action / state noise
    # --------------------------
    action_norm = vector_norm(action_delta)
    action_change = step_norm(action_delta)
    action_spikes = robust_spike_mask(action_change, z_th=4.0)

    joint_step = step_norm(joint_pos)
    joint_spikes = robust_spike_mask(joint_step, z_th=4.0)

    eef_step = step_norm(eef_pose)
    eef_spikes = robust_spike_mask(eef_step, z_th=4.0)

    # --------------------------
    # Depth quality
    # --------------------------
    depth_valid = depth_valid_ratio(depth_frames)
    env_depth_valid = depth_valid_ratio(env_depth_frames)

    depth_jump = depth_frame_jump(depth_frames)
    env_depth_jump = depth_frame_jump(env_depth_frames)

    # --------------------------
    # Summary
    # --------------------------
    summary = {
        "episode_dir": str(episode_dir),
        "num_steps": int(len(timestamps)) if timestamps is not None else None,
        "shapes": {
            "timestamps": list(timestamps.shape) if timestamps is not None else None,
            "rgb_frames": list(rgb_frames.shape) if rgb_frames is not None else None,
            "depth_frames": list(depth_frames.shape) if depth_frames is not None else None,
            "env_rgb_frames": list(env_rgb_frames.shape) if env_rgb_frames is not None else None,
            "env_depth_frames": list(env_depth_frames.shape) if env_depth_frames is not None else None,
            "joint_pos": list(joint_pos.shape) if joint_pos is not None else None,
            "eef_pose": list(eef_pose.shape) if eef_pose is not None else None,
            "gripper_state": list(gripper_state.shape) if gripper_state is not None else None,
            "action_delta": list(action_delta.shape) if action_delta is not None else None,
        },
        "lag_stats_sec": {
            "rgb_lag": metric_stats(rgb_lag),
            "depth_lag": metric_stats(depth_lag),
            "env_rgb_lag": metric_stats(env_rgb_lag),
            "env_depth_lag": metric_stats(env_depth_lag),
        },
        "action_noise": {
            "action_norm": metric_stats(action_norm),
            "action_change": metric_stats(action_change),
            "action_spike_count": int(np.sum(action_spikes)) if action_spikes is not None else None,
        },
        "state_noise": {
            "joint_step": metric_stats(joint_step),
            "joint_spike_count": int(np.sum(joint_spikes)) if joint_spikes is not None else None,
            "eef_step": metric_stats(eef_step),
            "eef_spike_count": int(np.sum(eef_spikes)) if eef_spikes is not None else None,
        },
        "depth_quality": {
            "depth_valid_ratio": metric_stats(depth_valid),
            "depth_jump": metric_stats(depth_jump),
            "env_depth_valid_ratio": metric_stats(env_depth_valid),
            "env_depth_jump": metric_stats(env_depth_jump),
        }
    }

    save_json(summary, out_dir / "summary.json")

    # --------------------------
    # Figure 1: noise report
    # --------------------------
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    axes = axes.ravel()

    # Sensor lag
    ax = axes[0]
    has_lag = False

    if rgb_lag is not None:
        ax.plot(rgb_lag, label="rgb")
        has_lag = True
    if depth_lag is not None:
        ax.plot(depth_lag, label="depth")
        has_lag = True
    if env_rgb_lag is not None:
        ax.plot(env_rgb_lag, label="env_rgb")
        has_lag = True
    if env_depth_lag is not None:
        ax.plot(env_depth_lag, label="env_depth")
        has_lag = True

    if has_lag:
        ax.set_title("Sensor lag relative to timestamps.npy")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Lag (sec)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.set_title("Sensor lag (missing)")
        ax.axis("off")

    plot_series(axes[1], action_norm, "Action delta norm", "Norm")
    plot_series(axes[2], action_change, "Action delta change", "Step change", action_spikes)

    plot_series(axes[3], joint_step, "Joint position step norm", "Step norm", joint_spikes)
    plot_series(axes[4], eef_step, "EEF pose step norm", "Step norm", eef_spikes)

    plot_series(axes[5], depth_valid, "Hand depth valid ratio", "Valid ratio")
    plot_series(axes[6], depth_jump, "Hand depth frame-to-frame jump", "Mean abs diff")

    plot_series(axes[7], env_depth_valid, "Env depth valid ratio", "Valid ratio")

    # Text summary
    axes[8].axis("off")

    text_lines = [
        f"Episode: {episode_dir.name}",
        f"num_steps: {summary['num_steps']}",
        "",
        f"action_spikes: {summary['action_noise']['action_spike_count']}",
        f"joint_spikes: {summary['state_noise']['joint_spike_count']}",
        f"eef_spikes: {summary['state_noise']['eef_spike_count']}",
        "",
    ]

    for key in ["rgb_lag", "depth_lag", "env_rgb_lag", "env_depth_lag"]:
        stats = summary["lag_stats_sec"][key]
        if stats is not None:
            text_lines.append(
                f"{key}: mean={stats['mean']:.4f}s, max={stats['max']:.4f}s"
            )

    if summary["depth_quality"]["depth_valid_ratio"] is not None:
        stats = summary["depth_quality"]["depth_valid_ratio"]
        text_lines.append(
            f"hand depth valid: mean={stats['mean']:.3f}, min={stats['min']:.3f}"
        )

    if summary["depth_quality"]["env_depth_valid_ratio"] is not None:
        stats = summary["depth_quality"]["env_depth_valid_ratio"]
        text_lines.append(
            f"env depth valid: mean={stats['mean']:.3f}, min={stats['min']:.3f}"
        )

    axes[8].text(
        0.02,
        0.98,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=12,
    )

    fig.suptitle(f"Episode noise report: {episode_dir.name}", fontsize=18)
    plt.tight_layout()
    plt.savefig(out_dir / "episode_noise_report.png", dpi=180)
    plt.close(fig)

    # --------------------------
    # Figure 2: sample frames
    # --------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    hand_rgb = prepare_rgb_frame(rgb_frames)
    hand_depth = prepare_depth_frame(depth_frames)
    env_rgb = prepare_rgb_frame(env_rgb_frames)
    env_depth = prepare_depth_frame(env_depth_frames)

    if hand_rgb is not None:
        axes2[0, 0].imshow(hand_rgb)
        axes2[0, 0].set_title("Hand RGB middle frame")
    else:
        axes2[0, 0].set_title("Hand RGB missing")
    axes2[0, 0].axis("off")

    if hand_depth is not None:
        im = axes2[0, 1].imshow(hand_depth)
        axes2[0, 1].set_title("Hand depth middle frame")
        fig2.colorbar(im, ax=axes2[0, 1], fraction=0.046, pad=0.04)
    else:
        axes2[0, 1].set_title("Hand depth missing")
    axes2[0, 1].axis("off")

    if env_rgb is not None:
        axes2[1, 0].imshow(env_rgb)
        axes2[1, 0].set_title("Env RGB middle frame")
    else:
        axes2[1, 0].set_title("Env RGB missing")
    axes2[1, 0].axis("off")

    if env_depth is not None:
        im = axes2[1, 1].imshow(env_depth)
        axes2[1, 1].set_title("Env depth middle frame")
        fig2.colorbar(im, ax=axes2[1, 1], fraction=0.046, pad=0.04)
    else:
        axes2[1, 1].set_title("Env depth missing")
    axes2[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "sample_frames.png", dpi=180)
    plt.close(fig2)

    print("=" * 80)
    print(f"Episode: {episode_dir}")
    print("=" * 80)
    print(f"[OK] summary saved: {out_dir / 'summary.json'}")
    print(f"[OK] noise report saved: {out_dir / 'episode_noise_report.png'}")
    print(f"[OK] sample frames saved: {out_dir / 'sample_frames.png'}")

    print("\nKey summary:")
    print(f"  num_steps: {summary['num_steps']}")
    print(f"  action_spikes: {summary['action_noise']['action_spike_count']}")
    print(f"  joint_spikes: {summary['state_noise']['joint_spike_count']}")
    print(f"  eef_spikes: {summary['state_noise']['eef_spike_count']}")

    for key, stats in summary["lag_stats_sec"].items():
        if stats is not None:
            print(f"  {key}: mean={stats['mean']:.4f}s, max={stats['max']:.4f}s")

    hand_valid_stats = summary["depth_quality"]["depth_valid_ratio"]
    if hand_valid_stats is not None:
        print(
            f"  hand depth valid ratio: "
            f"mean={hand_valid_stats['mean']:.3f}, min={hand_valid_stats['min']:.3f}"
        )

    env_valid_stats = summary["depth_quality"]["env_depth_valid_ratio"]
    if env_valid_stats is not None:
        print(
            f"  env depth valid ratio: "
            f"mean={env_valid_stats['mean']:.3f}, min={env_valid_stats['min']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode_dir",
        type=str,
        required=False,
        default="/home/xjtlu/xbox_control/fr5_dp_data_area_A/episode_000030",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/xjtlu/xbox_control/xbox_control/control/episode_noise_output",
        help="Output directory for summary and figures.",
    )
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    analyze_episode(episode_dir, out_dir)


if __name__ == "__main__":
    main()