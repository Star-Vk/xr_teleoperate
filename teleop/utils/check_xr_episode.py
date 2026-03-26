#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect a recorded XR episode and verify bridge pose fields."
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("teleop/utils/data"),
        help="Base task directory that contains task-name folders.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task folder name under task-dir.",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Episode folder name like episode_0008. If omitted, use the latest one.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="How many sample frames to print.",
    )
    return parser.parse_args()


def find_episode_dir(task_root: Path, episode_name: str | None) -> Path:
    if not task_root.exists():
        raise FileNotFoundError(f"Task directory does not exist: {task_root}")

    if episode_name:
        episode_dir = task_root / episode_name
        if not episode_dir.exists():
            raise FileNotFoundError(f"Episode directory does not exist: {episode_dir}")
        return episode_dir

    episode_dirs = sorted(
        p for p in task_root.iterdir() if p.is_dir() and p.name.startswith("episode_")
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* directories found under: {task_root}")
    return episode_dirs[-1]


def pose_xyz(pose):
    if not isinstance(pose, list) or len(pose) < 4:
        return None
    try:
        return [float(pose[0][3]), float(pose[1][3]), float(pose[2][3])]
    except (TypeError, IndexError, ValueError):
        return None


def main():
    args = parse_args()
    task_root = args.task_dir / args.task_name
    episode_dir = find_episode_dir(task_root, args.episode)
    frames_path = episode_dir / "frames.jsonl"
    data_path = episode_dir / "data.json"

    if not frames_path.exists():
        raise FileNotFoundError(f"frames.jsonl not found: {frames_path}")

    frames = []
    with frames_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))

    if not frames:
        raise RuntimeError(f"No frames found in: {frames_path}")

    tele_keys = sorted(frames[0].get("tele_data", {}).keys())
    left_rw_count = 0
    right_rw_count = 0
    left_valid_count = 0
    right_valid_count = 0
    head_valid_count = 0
    sample_rows = []

    for frame in frames:
        tele = frame.get("tele_data", {})
        left_rw = tele.get("left_robot_world_pose")
        right_rw = tele.get("right_robot_world_pose")
        if left_rw:
            left_rw_count += 1
        if right_rw:
            right_rw_count += 1
        if tele.get("left_arm_pose_valid"):
            left_valid_count += 1
        if tele.get("right_arm_pose_valid"):
            right_valid_count += 1
        if tele.get("head_pose_valid"):
            head_valid_count += 1

        if len(sample_rows) < args.sample_count:
            sample_rows.append(
                {
                    "frame_idx": frame.get("frame_idx"),
                    "left_robot_world_xyz": pose_xyz(left_rw),
                    "right_robot_world_xyz": pose_xyz(right_rw),
                    "left_wrist_xyz": pose_xyz(tele.get("left_wrist_pose")),
                    "right_wrist_xyz": pose_xyz(tele.get("right_wrist_pose")),
                    "head_valid": tele.get("head_pose_valid"),
                    "left_valid": tele.get("left_arm_pose_valid"),
                    "right_valid": tele.get("right_arm_pose_valid"),
                }
            )

    print(f"episode_dir: {episode_dir}")
    print(f"frames_path: {frames_path}")
    print(f"data_json_exists: {data_path.exists()}")
    print(f"frame_count: {len(frames)}")
    print(f"input_mode: {frames[0].get('input_mode')}")
    print(f"tele_data_keys: {', '.join(tele_keys)}")
    print(f"has_left_robot_world_pose: {'left_robot_world_pose' in tele_keys}")
    print(f"has_right_robot_world_pose: {'right_robot_world_pose' in tele_keys}")
    print(f"has_head_pose_valid: {'head_pose_valid' in tele_keys}")
    print(f"has_left_arm_pose_valid: {'left_arm_pose_valid' in tele_keys}")
    print(f"has_right_arm_pose_valid: {'right_arm_pose_valid' in tele_keys}")
    print(f"frames_with_left_robot_world_pose: {left_rw_count}/{len(frames)}")
    print(f"frames_with_right_robot_world_pose: {right_rw_count}/{len(frames)}")
    print(f"frames_with_head_valid: {head_valid_count}/{len(frames)}")
    print(f"frames_with_left_valid: {left_valid_count}/{len(frames)}")
    print(f"frames_with_right_valid: {right_valid_count}/{len(frames)}")
    print("samples:")
    for row in sample_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
