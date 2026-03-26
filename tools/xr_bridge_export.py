#!/usr/bin/env python3
import argparse
from pathlib import Path

from xr_retarget.bridge import JsonlBridgeRecorder, load_bridge_frames


def validate_jsonl(path: Path) -> int:
    frames = load_bridge_frames(path)
    print(f"[ok] validated {len(frames)} frames from {path}")
    if frames:
        print(f"[ok] first timestamp_ms={frames[0].timestamp_ms} last timestamp_ms={frames[-1].timestamp_ms}")
    return 0


def write_example(output_dir: Path) -> int:
    recorder = JsonlBridgeRecorder(output_dir)
    episode_dir = recorder.start_episode(name="example_episode", metadata={"source": "example"})
    recorder.append_frame(
        timestamp_ms=0,
        left_wrist={"position": [0.10, 0.20, 0.30], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        right_wrist={"position": [0.10, -0.20, 0.30], "quaternion": [0.0, 0.0, 0.0, 1.0]},
    )
    recorder.append_frame(
        timestamp_ms=33,
        left_wrist={"position": [0.12, 0.22, 0.33], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        right_wrist={"position": [0.12, -0.22, 0.33], "quaternion": [0.0, 0.0, 0.0, 1.0]},
    )
    recorder.stop_episode(summary={"frame_count": 2})
    print(f"[ok] wrote example bridge data to {episode_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XR bridge JSONL helper for integrating with xr_teleoperate bridge data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a frames.jsonl file.")
    validate_parser.add_argument("input", type=Path, help="Path to frames.jsonl")

    example_parser = subparsers.add_parser("example", help="Write a minimal example bridge episode.")
    example_parser.add_argument("--output-dir", type=Path, default=Path("xr_bridge_examples"))

    args = parser.parse_args()

    if args.command == "validate":
        return validate_jsonl(args.input)
    if args.command == "example":
        return write_example(args.output_dir)

    parser.error(f"unsupported command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
