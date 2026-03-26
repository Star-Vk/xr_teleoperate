import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


MotorKey = Tuple[str, int]


@dataclass
class RetargetedFrame:
    timestamp_ms: int
    positions: Dict[MotorKey, float]


def compress_frames(
    frames: Sequence[RetargetedFrame],
    position_epsilon_rad: float,
    min_interval_ms: int,
) -> List[RetargetedFrame]:
    if not frames:
        return []

    compressed = [frames[0]]
    last_kept = frames[0]

    for frame in frames[1:]:
        keep = False
        elapsed_ms = frame.timestamp_ms - last_kept.timestamp_ms
        if elapsed_ms >= min_interval_ms:
            for key, position in frame.positions.items():
                previous = last_kept.positions.get(key, position)
                if abs(position - previous) >= position_epsilon_rad:
                    keep = True
                    break
        if keep:
            compressed.append(frame)
            last_kept = frame

    if compressed[-1].timestamp_ms != frames[-1].timestamp_ms:
        compressed.append(frames[-1])

    return compressed


def retime_and_densify_frames(
    frames: Sequence[RetargetedFrame],
    motor_order: Iterable[MotorKey],
    max_position_step_rad: float,
    trajectory_max_velocity_rad_s: float,
    trajectory_target_interval_ms: int,
) -> List[RetargetedFrame]:
    if not frames:
        return []

    motor_keys = list(motor_order)
    if max_position_step_rad <= 0.0:
        max_position_step_rad = float("inf")
    if trajectory_max_velocity_rad_s <= 0.0:
        trajectory_max_velocity_rad_s = float("inf")
    if trajectory_target_interval_ms <= 0:
        trajectory_target_interval_ms = 20

    dense: List[RetargetedFrame] = [
        RetargetedFrame(timestamp_ms=frames[0].timestamp_ms, positions=dict(frames[0].positions))
    ]
    current_time_ms = float(frames[0].timestamp_ms)

    for previous, current in zip(frames, frames[1:]):
        max_delta = max(abs(current.positions[key] - previous.positions[key]) for key in motor_keys)
        raw_dt_ms = max(1, current.timestamp_ms - previous.timestamp_ms)
        required_dt_ms = int(math.ceil(1000.0 * max_delta / trajectory_max_velocity_rad_s))
        steps_by_step = int(math.ceil(max_delta / max_position_step_rad))
        effective_dt_ms = max(raw_dt_ms, required_dt_ms, steps_by_step * trajectory_target_interval_ms)
        steps = max(
            1,
            steps_by_step,
            int(math.ceil(effective_dt_ms / trajectory_target_interval_ms)),
        )
        step_dt_ms = effective_dt_ms / steps

        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            current_time_ms += step_dt_ms
            timestamp_ms = int(round(current_time_ms))
            if timestamp_ms <= dense[-1].timestamp_ms:
                timestamp_ms = dense[-1].timestamp_ms + 1
                current_time_ms = float(timestamp_ms)

            positions = {
                key: previous.positions[key] + alpha * (current.positions[key] - previous.positions[key])
                for key in motor_keys
            }
            dense.append(RetargetedFrame(timestamp_ms=timestamp_ms, positions=positions))

    return dense


def compute_export_speed(
    previous_pos: float,
    current_pos: float,
    dt_sec: float,
    previous_speed: float,
    min_speed: float,
    max_speed: float,
    accel_limit: float,
) -> float:
    if dt_sec <= 1.0e-3:
        return previous_speed

    target_speed = abs(current_pos - previous_pos) / dt_sec
    target_speed = max(min_speed, min(max_speed, target_speed))

    max_speed_delta = accel_limit * dt_sec
    speed_delta = target_speed - previous_speed
    if speed_delta > max_speed_delta:
        target_speed = previous_speed + max_speed_delta
    elif speed_delta < -max_speed_delta:
        target_speed = previous_speed - max_speed_delta

    return max(min_speed, min(max_speed, target_speed))


def export_action_csv(
    frames: Sequence[RetargetedFrame],
    output_path: Path,
    motor_order: Iterable[MotorKey],
    export_cfg: Dict[str, float],
) -> Dict[str, int]:
    motor_keys = list(motor_order)
    compressed = compress_frames(
        frames,
        position_epsilon_rad=float(export_cfg["position_epsilon_rad"]),
        min_interval_ms=int(export_cfg["min_interval_ms"]),
    )
    dense = retime_and_densify_frames(
        compressed,
        motor_order=motor_keys,
        max_position_step_rad=float(export_cfg.get("max_position_step_rad", 0.0)),
        trajectory_max_velocity_rad_s=float(export_cfg.get("trajectory_max_velocity_rad_s", 0.0)),
        trajectory_target_interval_ms=int(export_cfg.get("trajectory_target_interval_ms", 20)),
    )

    last_positions: Dict[MotorKey, float] = {}
    last_speeds: Dict[MotorKey, float] = {}
    last_accels: Dict[MotorKey, float] = {}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_timestamp = dense[0].timestamp_ms if dense else 0

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "can_iface",
                "motor_id",
                "position_rad",
                "elapsed_ms",
                "speed_rad_s",
                "accel_rad_s2",
            ]
        )

        for frame_index, frame in enumerate(dense):
            elapsed_ms = frame.timestamp_ms - first_timestamp
            previous_elapsed_ms = dense[frame_index - 1].timestamp_ms - first_timestamp if frame_index > 0 else 0
            delta_ms = max(1, elapsed_ms - previous_elapsed_ms)
            dt_sec = delta_ms / 1000.0

            for key in motor_keys:
                can_iface, motor_id = key
                position = frame.positions[key]
                previous_speed = last_speeds.get(key, float(export_cfg["default_speed"]))
                previous_accel = last_accels.get(key, float(export_cfg["accel_limit"]))

                if key in last_positions:
                    speed = compute_export_speed(
                        previous_pos=last_positions[key],
                        current_pos=position,
                        dt_sec=dt_sec,
                        previous_speed=previous_speed,
                        min_speed=float(export_cfg["min_speed"]),
                        max_speed=float(export_cfg["max_speed"]),
                        accel_limit=float(export_cfg["accel_limit"]),
                    )
                else:
                    speed = float(export_cfg["default_speed"])

                speed = max(
                    float(export_cfg["min_speed"]),
                    min(float(export_cfg["max_speed"]), speed * float(export_cfg["speed_scale"])),
                )

                accel = abs(speed - previous_speed) / max(dt_sec, 1.0e-3)
                accel = max(
                    0.5,
                    min(float(export_cfg["accel_limit"]), accel * float(export_cfg["accel_scale"])),
                )
                accel = 0.5 * accel + 0.5 * previous_accel

                writer.writerow(
                    [
                        frame_index,
                        can_iface,
                        motor_id,
                        f"{position:.6f}",
                        elapsed_ms,
                        f"{speed:.3f}",
                        f"{accel:.3f}",
                    ]
                )

                last_positions[key] = position
                last_speeds[key] = speed
                last_accels[key] = accel

    return {
        "input_frames": len(frames),
        "output_frames": len(dense),
        "row_count": len(dense) * len(motor_keys),
    }
