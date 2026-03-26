import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .math_utils import as_quat_xyzw, as_vec3


@dataclass
class BridgeFrame:
    timestamp_ms: int
    left_position: Optional[np.ndarray]
    left_quaternion: Optional[np.ndarray]
    right_position: Optional[np.ndarray]
    right_quaternion: Optional[np.ndarray]
    raw: Dict[str, Any]


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
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

    quat = np.asarray([x, y, z, w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("invalid rotation matrix for quaternion conversion")
    return quat / norm


def _normalize_pose(pose: Optional[Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if pose is None:
        return None, None

    if isinstance(pose, (list, tuple)):
        pose_array = np.asarray(pose, dtype=float)
        if pose_array.shape == (4, 4):
            position = pose_array[:3, 3]
            quaternion = _rotation_matrix_to_quaternion_xyzw(pose_array[:3, :3])
            return position, quaternion
        if pose_array.shape == (3,):
            return as_vec3(pose_array), None
        raise ValueError(f"unsupported pose array shape: {pose_array.shape}")

    if not isinstance(pose, dict):
        raise ValueError(f"unsupported pose payload type: {type(pose).__name__}")

    position = pose.get("position", pose.get("pos"))
    quaternion = pose.get("quaternion", pose.get("quat"))

    if position is None:
        return None, None

    pos = as_vec3(position)
    quat = as_quat_xyzw(quaternion) if quaternion is not None else None
    return pos, quat


def _extract_pose(payload: Dict[str, Any], preferred_keys: Iterable[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    for key in preferred_keys:
        pose = payload.get(key)
        if pose is not None:
            return _normalize_pose(pose)
    return None, None


def parse_bridge_frame(payload: Dict[str, Any]) -> BridgeFrame:
    if "timestamp_ms" not in payload:
        raise ValueError("bridge frame missing timestamp_ms")

    pose_payload = payload
    if isinstance(payload.get("tele_data"), dict):
        pose_payload = payload["tele_data"]

    # Prefer the robot-world poses from xr_teleoperate.
    left_position, left_quaternion = _extract_pose(
        pose_payload,
        (
            "left_robot_relative",
            "left_robot_relative_pose",
            "left_robot_world",
            "left_robot_world_pose",
            "left_wrist",
            "left_wrist_pose",
        ),
    )
    right_position, right_quaternion = _extract_pose(
        pose_payload,
        (
            "right_robot_relative",
            "right_robot_relative_pose",
            "right_robot_world",
            "right_robot_world_pose",
            "right_wrist",
            "right_wrist_pose",
        ),
    )

    return BridgeFrame(
        timestamp_ms=int(payload["timestamp_ms"]),
        left_position=left_position,
        left_quaternion=left_quaternion,
        right_position=right_position,
        right_quaternion=right_quaternion,
        raw=payload,
    )


def load_bridge_frames(path: Path) -> List[BridgeFrame]:
    frames: List[BridgeFrame] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            frames.append(parse_bridge_frame(payload))

    if not frames:
        raise ValueError(f"no frames found in {path}")

    return frames


class JsonlBridgeRecorder:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._frames_handle = None
        self._episode_dir: Optional[Path] = None

    @property
    def episode_dir(self) -> Optional[Path]:
        return self._episode_dir

    def start_episode(self, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Path:
        if self._frames_handle is not None:
            raise RuntimeError("episode already started")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_name = name or f"episode_{stamp}"
        episode_dir = self.output_root / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        if metadata is not None:
            metadata_path = episode_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        self._frames_handle = (episode_dir / "frames.jsonl").open("w", encoding="utf-8")
        self._episode_dir = episode_dir
        return episode_dir

    def append_frame(
        self,
        timestamp_ms: int,
        left_wrist: Optional[Dict[str, Any]] = None,
        right_wrist: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._frames_handle is None:
            raise RuntimeError("episode not started")

        payload: Dict[str, Any] = {"timestamp_ms": int(timestamp_ms)}
        if left_wrist is not None:
            payload["left_wrist"] = left_wrist
        if right_wrist is not None:
            payload["right_wrist"] = right_wrist
        if extra:
            payload.update(extra)

        parse_bridge_frame(payload)
        self._frames_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._frames_handle.flush()

    def stop_episode(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if self._frames_handle is None:
            return

        self._frames_handle.close()
        self._frames_handle = None

        if self._episode_dir is not None and summary is not None:
            summary_path = self._episode_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self._episode_dir = None
