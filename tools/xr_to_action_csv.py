#!/usr/bin/env python3
import argparse
from pathlib import Path

from xr_retarget.csv_export import export_action_csv
from xr_retarget.retarget import XRRetargeter, load_frames_from_jsonl, load_retarget_config


def select_frame_slice(frames, start, end, label):
    total = len(frames)
    start_idx = 0 if start is None else start
    end_idx = total - 1 if end is None else end

    if total == 0:
        raise ValueError(f"{label}: no frames available")
    if start_idx < 0 or end_idx < 0 or start_idx >= total or end_idx >= total:
        raise ValueError(
            f"{label}: frame range [{start_idx}, {end_idx}] is outside valid range [0, {total - 1}]"
        )
    if start_idx > end_idx:
        raise ValueError(f"{label}: start frame {start_idx} is greater than end frame {end_idx}")

    indices = list(range(start_idx, end_idx + 1))
    return [frames[index] for index in indices], (start_idx, end_idx), indices


def format_vector(vector):
    if vector is None:
        return "None"
    return "[" + ", ".join(f"{value:.4f}" for value in vector.tolist()) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert XR bridge JSONL frames into action-library CSV."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to bridge frames.jsonl")
    parser.add_argument("--config", required=True, type=Path, help="Path to xr_retarget.yaml")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Optional URDF override. If omitted, use the path from config.",
    )
    parser.add_argument(
        "--arms",
        choices=("left", "right", "both"),
        default="both",
        help="Which arms to export. Disabled arms will hold home/previous positions.",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="Inclusive start frame for conversion.")
    parser.add_argument("--end-frame", type=int, default=None, help="Inclusive end frame for conversion.")
    parser.add_argument(
        "--calib-start-frame",
        type=int,
        default=None,
        help="Inclusive start frame for neutral-pose calibration. Defaults to conversion start.",
    )
    parser.add_argument(
        "--calib-end-frame",
        type=int,
        default=None,
        help="Inclusive end frame for neutral-pose calibration. Defaults to conversion end.",
    )
    args = parser.parse_args()

    config = load_retarget_config(args.config, urdf_override=args.urdf)
    all_frames = load_frames_from_jsonl(args.input)

    conversion_frames, conversion_range, conversion_indices = select_frame_slice(
        all_frames,
        args.start_frame,
        args.end_frame,
        label="conversion",
    )
    if args.calib_start_frame is None and args.calib_end_frame is None and config.calibration.default_frame_count is not None:
        calib_start = conversion_range[0]
        calib_end = min(
            conversion_range[0] + config.calibration.default_frame_count - 1,
            conversion_range[1],
        )
    else:
        calib_start = args.calib_start_frame if args.calib_start_frame is not None else conversion_range[0]
        calib_end = args.calib_end_frame if args.calib_end_frame is not None else conversion_range[1]
    calibration_frames, calibration_range, calibration_indices = select_frame_slice(
        all_frames,
        calib_start,
        calib_end,
        label="calibration",
    )

    retargeter = XRRetargeter(config)
    retargeted_frames = retargeter.retarget_frames(
        conversion_frames,
        arms_mode=args.arms,
        calibration_frames=calibration_frames,
        source_frame_indices=conversion_indices,
        calibration_frame_indices=calibration_indices,
    )

    export_stats = export_action_csv(
        frames=retargeted_frames,
        output_path=args.output,
        motor_order=retargeter.motor_order,
        export_cfg=config.csv_export,
    )

    print("[ok] conversion complete")
    print(f"[ok] urdf={config.urdf_path}")
    print(f"[ok] input_frames_total={len(all_frames)} conversion_frames={len(conversion_frames)} output_csv={args.output}")
    print(
        "[ok] export_stats="
        f"input_frames={export_stats['input_frames']} "
        f"output_frames={export_stats['output_frames']} "
        f"rows={export_stats['row_count']}"
    )
    print(
        "[ok] ranges="
        f"conversion={conversion_range} "
        f"calibration={calibration_range} "
        f"inactive_arms={retargeter.summary.inactive_arms}"
    )
    print(
        "[ok] ik_summary="
        f"left_failures={retargeter.summary.left_failures} "
        f"right_failures={retargeter.summary.right_failures} "
        f"auto_translation={retargeter.summary.used_auto_translation}"
    )
    print(
        "[ok] workspace_clamp="
        f"left_frames={retargeter.summary.left_clamped_frames} "
        f"right_frames={retargeter.summary.right_clamped_frames}"
    )
    print(
        "[ok] calibration="
        f"left_neutral_xr={format_vector(retargeter.summary.left_neutral_xr_position)} "
        f"right_neutral_xr={format_vector(retargeter.summary.right_neutral_xr_position)}"
    )
    print(
        "[ok] arm_translation_xyz="
        f"left={format_vector(retargeter.summary.left_translation_xyz)} "
        f"right={format_vector(retargeter.summary.right_translation_xyz)}"
    )
    print(
        "[ok] home_ee="
        f"left={format_vector(retargeter.summary.left_home_ee_position)} "
        f"right={format_vector(retargeter.summary.right_home_ee_position)}"
    )
    print(f"[ok] translation_xyz={format_vector(retargeter.summary.translation_xyz)}")
    print(
        "[ok] failure_samples="
        f"left={retargeter.summary.left_failure_samples} "
        f"right={retargeter.summary.right_failure_samples}"
    )
    print(
        "[ok] clamp_samples="
        f"left={retargeter.summary.left_clamp_samples} "
        f"right={retargeter.summary.right_clamp_samples}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
