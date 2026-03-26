import json
import os
from dataclasses import asdict
from typing import Any

import numpy as np

import logging_mp

logger_mp = logging_mp.getLogger(__name__)


def _to_builtin(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


class XRBridgeWriter:
    """Write XR teleoperation frames as JSONL for downstream retargeting."""

    def __init__(self, episode_dir: str, input_mode: str):
        self.episode_dir = episode_dir
        self.input_mode = input_mode
        self.output_path = os.path.join(self.episode_dir, "frames.jsonl")
        self.frame_idx = 0
        self._file = open(self.output_path, "w", encoding="utf-8")
        logger_mp.info(f"[XRBridgeWriter] Writing XR bridge frames to: {self.output_path}")

    def add_frame(self, tele_data, timestamp_ms: int):
        frame = {
            "frame_idx": self.frame_idx,
            "timestamp_ms": int(timestamp_ms),
            "input_mode": self.input_mode,
            "tele_data": _to_builtin(asdict(tele_data)),
        }
        self._file.write(json.dumps(frame, ensure_ascii=False) + "\n")
        self._file.flush()
        self.frame_idx += 1

    def close(self):
        if getattr(self, "_file", None) and not self._file.closed:
            self._file.close()
            logger_mp.info(f"[XRBridgeWriter] Closed: {self.output_path}")
