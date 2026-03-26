from dataclasses import dataclass
from typing import Optional

import numpy as np

from .math_utils import rpy_matrix


def normalize_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def align_quaternion_sign(reference_xyzw: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion_xyzw(quaternion_xyzw)
    ref = normalize_quaternion_xyzw(reference_xyzw)
    if float(np.dot(ref, quat)) < 0.0:
        quat = -quat
    return quat


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    else:
        if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif rotation[1, 1] > rotation[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    return normalize_quaternion_xyzw(np.array([x, y, z, w], dtype=float))


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def quaternion_xyzw_conjugate(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)
    return np.array([-x, -y, -z, w], dtype=float)


def quaternion_xyzw_multiply(lhs_xyzw: np.ndarray, rhs_xyzw: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion_xyzw(lhs_xyzw)
    x2, y2, z2, w2 = normalize_quaternion_xyzw(rhs_xyzw)
    return normalize_quaternion_xyzw(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )
    )


def quaternion_xyzw_nlerp(previous_xyzw: np.ndarray, current_xyzw: np.ndarray, alpha: float) -> np.ndarray:
    prev = normalize_quaternion_xyzw(previous_xyzw)
    current = align_quaternion_sign(prev, current_xyzw)
    return normalize_quaternion_xyzw((1.0 - alpha) * prev + alpha * current)


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1.0e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def soft_deadzone(values: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    threshold_array = np.asarray(threshold, dtype=float)
    return np.sign(array) * np.maximum(np.abs(array) - threshold_array, 0.0)


def quaternion_angle_deg(delta_quaternion_xyzw: np.ndarray) -> float:
    quat = normalize_quaternion_xyzw(delta_quaternion_xyzw)
    angle = 2.0 * np.arccos(np.clip(abs(quat[3]), -1.0, 1.0))
    return float(np.rad2deg(angle))


def pose_is_valid(pose_matrix: np.ndarray) -> bool:
    pose = np.asarray(pose_matrix, dtype=float)
    if pose.shape != (4, 4):
        return False
    det = np.linalg.det(pose[:3, :3])
    return bool(np.isfinite(det) and not np.isclose(det, 0.0, atol=1.0e-6))


def make_pose_matrix(position_xyz: np.ndarray, rpy_deg: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rpy_matrix(np.deg2rad(np.asarray(rpy_deg, dtype=float)))
    matrix[:3, 3] = np.asarray(position_xyz, dtype=float).reshape(3)
    return matrix


@dataclass
class StablePoseConfig:
    baseline_seconds: float = 2.0
    smoothing_alpha: float = 0.18
    position_deadzone_m: float = 0.004
    rotation_deadzone_deg: float = 1.2
    static_position_speed_m_s: float = 0.015
    static_rotation_speed_deg_s: float = 8.0
    static_hold_seconds: float = 0.35
    unlock_position_speed_m_s: float = 0.030
    unlock_rotation_speed_deg_s: float = 16.0


@dataclass
class StablePoseState:
    pose_valid: bool
    calibrating: bool
    locked: bool
    progress: float
    filtered_position: np.ndarray
    filtered_quaternion: np.ndarray
    baseline_position: Optional[np.ndarray]
    baseline_quaternion: Optional[np.ndarray]
    raw_relative_position: np.ndarray
    raw_relative_rpy_deg: np.ndarray
    output_relative_position: np.ndarray
    output_relative_rpy_deg: np.ndarray
    output_pose_matrix: np.ndarray


class StableRelativePoseFilter:
    def __init__(self, config: StablePoseConfig):
        self.config = config
        self.filtered_position: Optional[np.ndarray] = None
        self.filtered_quaternion: Optional[np.ndarray] = None
        self.prev_filtered_position: Optional[np.ndarray] = None
        self.prev_filtered_quaternion: Optional[np.ndarray] = None
        self.baseline_position: Optional[np.ndarray] = None
        self.baseline_quaternion: Optional[np.ndarray] = None
        self.baseline_started_at_ms: Optional[int] = None
        self.baseline_position_samples: list[np.ndarray] = []
        self.baseline_quaternion_samples: list[np.ndarray] = []
        self.static_elapsed_seconds: float = 0.0
        self.locked: bool = False
        self.locked_position: np.ndarray = np.zeros(3, dtype=float)
        self.locked_rpy_deg: np.ndarray = np.zeros(3, dtype=float)
        self.last_timestamp_ms: Optional[int] = None

    def reset_baseline(self, timestamp_ms: int):
        self.baseline_position = None
        self.baseline_quaternion = None
        self.baseline_started_at_ms = int(timestamp_ms)
        self.baseline_position_samples = []
        self.baseline_quaternion_samples = []
        self.static_elapsed_seconds = 0.0
        self.locked = False

    def _finalize_baseline(self):
        if not self.baseline_position_samples or not self.baseline_quaternion_samples:
            return
        self.baseline_position = np.mean(np.asarray(self.baseline_position_samples), axis=0)
        reference_quaternion = self.baseline_quaternion_samples[0]
        aligned_quaternions = [
            align_quaternion_sign(reference_quaternion, sample)
            for sample in self.baseline_quaternion_samples
        ]
        self.baseline_quaternion = normalize_quaternion_xyzw(
            np.mean(np.asarray(aligned_quaternions), axis=0)
        )

    def _delta_seconds(self, timestamp_ms: int) -> float:
        if self.last_timestamp_ms is None:
            return 0.0
        return max(1.0e-3, (int(timestamp_ms) - self.last_timestamp_ms) / 1000.0)

    def update(self, timestamp_ms: int, pose_matrix: np.ndarray) -> StablePoseState:
        pose_matrix = np.asarray(pose_matrix, dtype=float)
        pose_valid = pose_is_valid(pose_matrix)
        timestamp_ms = int(timestamp_ms)
        dt_seconds = self._delta_seconds(timestamp_ms)

        if pose_valid:
            raw_position = pose_matrix[:3, 3].copy()
            raw_quaternion = rotation_matrix_to_quaternion_xyzw(pose_matrix[:3, :3])
            if self.filtered_position is None:
                self.filtered_position = raw_position
                self.filtered_quaternion = raw_quaternion
            else:
                self.prev_filtered_position = self.filtered_position.copy()
                self.prev_filtered_quaternion = self.filtered_quaternion.copy()
                alpha = self.config.smoothing_alpha
                self.filtered_position = alpha * raw_position + (1.0 - alpha) * self.filtered_position
                self.filtered_quaternion = quaternion_xyzw_nlerp(
                    self.filtered_quaternion,
                    raw_quaternion,
                    alpha,
                )
        elif self.filtered_position is None:
            self.filtered_position = np.zeros(3, dtype=float)
            self.filtered_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        if self.baseline_started_at_ms is None:
            self.baseline_started_at_ms = timestamp_ms

        if self.baseline_position is None or self.baseline_quaternion is None:
            self.baseline_position_samples.append(self.filtered_position.copy())
            self.baseline_quaternion_samples.append(self.filtered_quaternion.copy())
            elapsed_seconds = (timestamp_ms - self.baseline_started_at_ms) / 1000.0
            if self.config.baseline_seconds <= 0.0 or elapsed_seconds >= self.config.baseline_seconds:
                self._finalize_baseline()
            progress = (
                1.0
                if self.config.baseline_seconds <= 0.0
                else min(1.0, elapsed_seconds / self.config.baseline_seconds)
            )
            raw_relative_position = np.zeros(3, dtype=float)
            raw_relative_rpy_deg = np.zeros(3, dtype=float)
            output_relative_position = np.zeros(3, dtype=float)
            output_relative_rpy_deg = np.zeros(3, dtype=float)
            calibrating = self.baseline_position is None or self.baseline_quaternion is None
            self.static_elapsed_seconds = 0.0
            self.locked = False
        else:
            progress = 1.0
            raw_relative_position = self.filtered_position - self.baseline_position
            raw_relative_quaternion = quaternion_xyzw_multiply(
                quaternion_xyzw_conjugate(self.baseline_quaternion),
                self.filtered_quaternion,
            )
            raw_relative_rpy_deg = rotation_matrix_to_rpy_deg(
                quaternion_xyzw_to_rotation_matrix(raw_relative_quaternion)
            )

            candidate_position = soft_deadzone(
                raw_relative_position,
                self.config.position_deadzone_m,
            )
            candidate_rpy_deg = soft_deadzone(
                raw_relative_rpy_deg,
                self.config.rotation_deadzone_deg,
            )

            if self.prev_filtered_position is None:
                linear_speed = 0.0
            else:
                linear_speed = float(
                    np.linalg.norm(self.filtered_position - self.prev_filtered_position) / max(dt_seconds, 1.0e-3)
                )

            if self.prev_filtered_quaternion is None:
                angular_speed_deg = 0.0
            else:
                delta_quaternion = quaternion_xyzw_multiply(
                    quaternion_xyzw_conjugate(self.prev_filtered_quaternion),
                    self.filtered_quaternion,
                )
                angular_speed_deg = quaternion_angle_deg(delta_quaternion) / max(dt_seconds, 1.0e-3)

            if self.locked:
                if (
                    linear_speed >= self.config.unlock_position_speed_m_s
                    or angular_speed_deg >= self.config.unlock_rotation_speed_deg_s
                ):
                    self.locked = False
                    self.static_elapsed_seconds = 0.0
                else:
                    output_relative_position = self.locked_position.copy()
                    output_relative_rpy_deg = self.locked_rpy_deg.copy()
                    self.last_timestamp_ms = timestamp_ms
                    return StablePoseState(
                        pose_valid=pose_valid,
                        calibrating=False,
                        locked=True,
                        progress=progress,
                        filtered_position=self.filtered_position.copy(),
                        filtered_quaternion=self.filtered_quaternion.copy(),
                        baseline_position=self.baseline_position.copy(),
                        baseline_quaternion=self.baseline_quaternion.copy(),
                        raw_relative_position=raw_relative_position,
                        raw_relative_rpy_deg=raw_relative_rpy_deg,
                        output_relative_position=output_relative_position,
                        output_relative_rpy_deg=output_relative_rpy_deg,
                        output_pose_matrix=make_pose_matrix(output_relative_position, output_relative_rpy_deg),
                    )

            if (
                linear_speed <= self.config.static_position_speed_m_s
                and angular_speed_deg <= self.config.static_rotation_speed_deg_s
            ):
                self.static_elapsed_seconds += dt_seconds
            else:
                self.static_elapsed_seconds = 0.0

            if self.static_elapsed_seconds >= self.config.static_hold_seconds:
                self.locked = True
                self.locked_position = candidate_position.copy()
                self.locked_rpy_deg = candidate_rpy_deg.copy()
                output_relative_position = self.locked_position.copy()
                output_relative_rpy_deg = self.locked_rpy_deg.copy()
            else:
                output_relative_position = candidate_position
                output_relative_rpy_deg = candidate_rpy_deg

            calibrating = False

        self.last_timestamp_ms = timestamp_ms
        return StablePoseState(
            pose_valid=pose_valid,
            calibrating=calibrating,
            locked=self.locked,
            progress=progress,
            filtered_position=self.filtered_position.copy(),
            filtered_quaternion=self.filtered_quaternion.copy(),
            baseline_position=None if self.baseline_position is None else self.baseline_position.copy(),
            baseline_quaternion=None if self.baseline_quaternion is None else self.baseline_quaternion.copy(),
            raw_relative_position=raw_relative_position,
            raw_relative_rpy_deg=raw_relative_rpy_deg,
            output_relative_position=output_relative_position,
            output_relative_rpy_deg=output_relative_rpy_deg,
            output_pose_matrix=make_pose_matrix(output_relative_position, output_relative_rpy_deg),
        )
