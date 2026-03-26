import math
from typing import Iterable

import numpy as np


def as_vec3(values: Iterable[float]) -> np.ndarray:
    vec = np.asarray(list(values), dtype=float).reshape(3)
    return vec


def as_quat_xyzw(values: Iterable[float]) -> np.ndarray:
    quat = np.asarray(list(values), dtype=float).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return quat / norm


def translation_matrix(xyz: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = as_vec3(xyz)
    return matrix


def rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=float,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=float,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rz @ ry @ rx


def transform_matrix(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rpy_matrix(rpy)
    matrix[:3, 3] = as_vec3(xyz)
    return matrix


def axis_angle_matrix(axis: Iterable[float], angle_rad: float) -> np.ndarray:
    axis_vec = np.asarray(list(axis), dtype=float).reshape(3)
    norm = np.linalg.norm(axis_vec)
    if norm <= 1e-12:
        raise ValueError("joint axis norm is zero")
    x, y, z = axis_vec / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c

    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )


def homogeneous_rotation(axis: Iterable[float], angle_rad: float) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = axis_angle_matrix(axis, angle_rad)
    return matrix


def quaternion_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = as_quat_xyzw(quaternion_xyzw)
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


def apply_transform(transform: np.ndarray, point_xyz: Iterable[float]) -> np.ndarray:
    point = np.ones(4, dtype=float)
    point[:3] = as_vec3(point_xyz)
    return (transform @ point)[:3]


def clamp_array(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(values, lower), upper)
