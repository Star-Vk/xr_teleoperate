#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from xr_retarget.bridge import load_bridge_frames
from xr_retarget.pose_processing import StablePoseConfig, StableRelativePoseFilter, pose_is_valid


def parse_args():
    defaults = StablePoseConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess XR bridge frames into stable baseline-relative poses using "
            "EMA smoothing, soft deadzone, and static lock."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to source frames.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Path to output stabilized frames.jsonl")
    parser.add_argument(
        "--sides",
        choices=["left", "right", "both"],
        default="both",
        help="Which controller sides to preprocess.",
    )
    parser.add_argument("--baseline-seconds", type=float, default=defaults.baseline_seconds)
    parser.add_argument("--smoothing-alpha", type=float, default=defaults.smoothing_alpha)
    parser.add_argument("--position-deadzone-m", type=float, default=defaults.position_deadzone_m)
    parser.add_argument("--rotation-deadzone-deg", type=float, default=defaults.rotation_deadzone_deg)
    parser.add_argument("--static-position-speed-m-s", type=float, default=defaults.static_position_speed_m_s)
    parser.add_argument("--static-rotation-speed-deg-s", type=float, default=defaults.static_rotation_speed_deg_s)
    parser.add_argument("--static-hold-seconds", type=float, default=defaults.static_hold_seconds)
    parser.add_argument("--unlock-position-speed-m-s", type=float, default=defaults.unlock_position_speed_m_s)
    parser.add_argument("--unlock-rotation-speed-deg-s", type=float, default=defaults.unlock_rotation_speed_deg_s)
    parser.add_argument(
        "--ignore-trigger-recenter",
        action="store_true",
        help="Do not reset the relative-pose baseline when the recorded controller trigger has a rising edge.",
    )
    return parser.parse_args()


def build_config(args) -> StablePoseConfig:
    return StablePoseConfig(
        baseline_seconds=args.baseline_seconds,
        smoothing_alpha=args.smoothing_alpha,
        position_deadzone_m=args.position_deadzone_m,
        rotation_deadzone_deg=args.rotation_deadzone_deg,
        static_position_speed_m_s=args.static_position_speed_m_s,
        static_rotation_speed_deg_s=args.static_rotation_speed_deg_s,
        static_hold_seconds=args.static_hold_seconds,
        unlock_position_speed_m_s=args.unlock_position_speed_m_s,
        unlock_rotation_speed_deg_s=args.unlock_rotation_speed_deg_s,
    )


def select_sides(sides_arg: str) -> list[str]:
    if sides_arg == "both":
        return ["left", "right"]
    return [sides_arg]


def extract_pose_matrix(payload: dict, side: str):
    tele = payload.get("tele_data", payload)
    for key in (
        f"{side}_robot_relative_pose",
        f"{side}_robot_world_pose",
        f"{side}_robot_world",
        f"{side}_wrist_pose",
        f"{side}_wrist",
    ):
        pose = tele.get(key)
        if pose is None or pose == []:
            continue
        pose_array = np.asarray(pose, dtype=float)
        if pose_array.shape == (4, 4):
            return pose_array, key
    return None, None


def get_trigger_pressed(payload: dict, side: str) -> bool:
    tele = payload.get("tele_data", payload)
    return bool(tele.get(f"{side}_ctrl_trigger", False))


def main() -> int:
    args = parse_args()
    config = build_config(args)
    frames = load_bridge_frames(args.input)
    sides = select_sides(args.sides)
    filters = {side: StableRelativePoseFilter(config) for side in sides}
    stats = {
        side: {
            "updated": 0,
            "missing": 0,
            "locked_frames": 0,
            "recenter_events": 0,
            "last_source": None,
        }
        for side in sides
    }
    prev_trigger_state = {side: False for side in sides}

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for frame in frames:
            payload = copy.deepcopy(frame.raw)
            tele = payload.setdefault("tele_data", {})

            for side in sides:
                trigger_pressed = get_trigger_pressed(payload, side)
                if (
                    not args.ignore_trigger_recenter
                    and trigger_pressed
                    and not prev_trigger_state[side]
                ):
                    filters[side].reset_baseline(int(payload["timestamp_ms"]))
                    stats[side]["recenter_events"] += 1
                prev_trigger_state[side] = trigger_pressed

                pose_matrix, source_key = extract_pose_matrix(payload, side)
                if pose_matrix is None or not pose_is_valid(pose_matrix):
                    stats[side]["missing"] += 1
                    continue

                state = filters[side].update(int(payload["timestamp_ms"]), pose_matrix)
                tele[f"{side}_robot_relative_pose"] = state.output_pose_matrix.tolist()
                tele[f"{side}_robot_relative_xyz"] = state.output_relative_position.tolist()
                tele[f"{side}_robot_relative_rpy_deg"] = state.output_relative_rpy_deg.tolist()
                tele[f"{side}_robot_relative_locked"] = bool(state.locked)
                tele[f"{side}_robot_relative_calibrating"] = bool(state.calibrating)
                tele[f"{side}_robot_relative_source"] = source_key

                stats[side]["updated"] += 1
                stats[side]["last_source"] = source_key
                if state.locked:
                    stats[side]["locked_frames"] += 1

            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"[ok] wrote stabilized frames to {args.output}")
    print(f"[ok] input_frames={len(frames)}")
    for side in sides:
        side_stats = stats[side]
        print(
            "[ok] "
            f"{side}: updated={side_stats['updated']} "
            f"missing={side_stats['missing']} "
            f"locked_frames={side_stats['locked_frames']} "
            f"recenter_events={side_stats['recenter_events']} "
            f"source={side_stats['last_source']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
