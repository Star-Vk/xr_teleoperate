import argparse
import os
import site
import socket
import sys
import time
from pathlib import Path

import numpy as np

try:
    import logging_mp
except ImportError:  # pragma: no cover - fallback for lightweight environments
    import logging as logging_mp


logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)


THIS_DIR = Path(__file__).resolve().parent
TELEVUER_SRC = THIS_DIR / "televuer" / "src"
TOOLS_DIR = THIS_DIR.parent / "tools"
if str(TELEVUER_SRC) not in sys.path:
    sys.path.insert(0, str(TELEVUER_SRC))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from xr_retarget.pose_processing import StablePoseConfig, StableRelativePoseFilter


XR_PORT = 8012
T_ROBOT_OPENXR = None
T_OPENXR_ROBOT = None
cv2 = None


def remove_user_site_packages():
    removed_paths = []
    user_sites = site.getusersitepackages()
    if isinstance(user_sites, str):
        user_sites = [user_sites]
    resolved_user_sites = {str(Path(path).resolve()) for path in user_sites}
    kept_paths = []
    for path in sys.path:
        try:
            resolved_path = str(Path(path).resolve())
        except Exception:
            kept_paths.append(path)
            continue
        if resolved_path in resolved_user_sites:
            removed_paths.append(path)
        else:
            kept_paths.append(path)
    sys.path[:] = kept_paths
    return removed_paths


REMOVED_USER_SITE_PATHS = remove_user_site_packages()


def parse_args():
    defaults = StablePoseConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Display stable controller pose values directly in the XR browser. "
            "Values are shown as baseline-relative xyz and Euler angles under robot convention."
        )
    )
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="left",
        help="Which controller pose to display.",
    )
    parser.add_argument(
        "--display-mode",
        choices=["ego", "immersive"],
        default="immersive",
        help="XR display mode. Default is a full-screen data panel.",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=15.0,
        help="Refresh rate for the pose panel in XR.",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=1040,
        help="Pose panel height in pixels.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=960,
        help="Per-eye pose panel width in pixels.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=defaults.baseline_seconds,
        help="Initial calibration duration. Hold the selected controller still during this time.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=defaults.smoothing_alpha,
        help="EMA smoothing factor. Higher means more reactive, lower means more stable.",
    )
    parser.add_argument(
        "--position-deadzone-m",
        type=float,
        default=defaults.position_deadzone_m,
        help="Soft deadzone applied to relative xyz in meters.",
    )
    parser.add_argument(
        "--rotation-deadzone-deg",
        type=float,
        default=defaults.rotation_deadzone_deg,
        help="Soft deadzone applied to relative roll/pitch/yaw in degrees.",
    )
    parser.add_argument(
        "--static-position-speed-m-s",
        type=float,
        default=defaults.static_position_speed_m_s,
        help="Position speed threshold below which static lock timing starts.",
    )
    parser.add_argument(
        "--static-rotation-speed-deg-s",
        type=float,
        default=defaults.static_rotation_speed_deg_s,
        help="Rotation speed threshold below which static lock timing starts.",
    )
    parser.add_argument(
        "--static-hold-seconds",
        type=float,
        default=defaults.static_hold_seconds,
        help="How long the controller must stay slow before the output locks.",
    )
    parser.add_argument(
        "--unlock-position-speed-m-s",
        type=float,
        default=defaults.unlock_position_speed_m_s,
        help="Position speed threshold that releases the static lock.",
    )
    parser.add_argument(
        "--unlock-rotation-speed-deg-s",
        type=float,
        default=defaults.unlock_rotation_speed_deg_s,
        help="Rotation speed threshold that releases the static lock.",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Seconds between console diagnostic prints. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--monocular",
        action="store_true",
        help="Use a single panel instead of duplicating the panel for both eyes.",
    )
    parser.add_argument(
        "--host-ip",
        type=str,
        default="",
        help="Host IP shown in the browser URL hint. Auto-detected if omitted.",
    )
    parser.add_argument("--cert-file", type=str, default=None, help="Optional SSL cert path.")
    parser.add_argument("--key-file", type=str, default=None, help="Optional SSL key path.")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    return args


def detect_host_ip(explicit_ip: str) -> str:
    if explicit_ip:
        return explicit_ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def build_browser_url(host_ip: str) -> str:
    return f"https://{host_ip}:{XR_PORT}/?ws=wss://{host_ip}:{XR_PORT}"


def warn_if_nested_virtualenv():
    if REMOVED_USER_SITE_PATHS:
        logger_mp.info("Removed user site-packages from sys.path: %s", REMOVED_USER_SITE_PATHS)
    venv_path = os.environ.get("VIRTUAL_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if venv_path and conda_prefix and Path(venv_path).resolve() != Path(conda_prefix).resolve():
        logger_mp.warning(
            "Detected both CONDA_PREFIX=%s and VIRTUAL_ENV=%s. "
            "A nested virtualenv can mix packages and make XR websocket behavior unstable.",
            conda_prefix,
            venv_path,
        )


def install_vuer_websocket_patch():
    import traceback
    from aiohttp import web
    from concurrent.futures import CancelledError
    import vuer.base as vuer_base
    import vuer.server as vuer_server

    async def safe_websocket_handler(request, handler, **ws_kwargs):
        ws = web.WebSocketResponse(**ws_kwargs)
        await ws.prepare(request)

        try:
            await handler(request, ws)
        except ConnectionResetError:
            print("Connection reset")
        except CancelledError:
            print("WebSocket Canceled")
        except Exception as exp:
            print(f"Error:\n{exp}\n{traceback.print_exc()}")
        finally:
            if not ws.closed:
                await ws.close()
            print("WebSocket connection closed")

        return ws

    vuer_base.websocket_handler = safe_websocket_handler
    vuer_server.websocket_handler = safe_websocket_handler


def install_aiohttp_resume_patch():
    import aiohttp.base_protocol as aiohttp_base_protocol

    def safe_resume_writing(self):
        if not getattr(self, "_paused", False):
            return
        self._paused = False

        waiter = getattr(self, "_drain_waiter", None)
        if waiter is not None:
            self._drain_waiter = None
            if not waiter.done():
                waiter.set_result(None)

    aiohttp_base_protocol.BaseProtocol.resume_writing = safe_resume_writing


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


def fit_text_scale(text: str, max_width: int, base_scale: float, thickness: int) -> float:
    scale = base_scale
    for _ in range(20):
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale *= 0.92
    return scale


def draw_text_block(image, lines, origin, font_scale=0.72, color=(235, 240, 245), thickness=1, line_gap=14):
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1]
        y += text_height + line_gap
    return y


def draw_metric_card(image, label: str, value_text: str, top_left, size, accent_color):
    x, y = top_left
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), (27, 35, 45), thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), accent_color, thickness=3)
    cv2.rectangle(image, (x + 18, y + 18), (x + width - 18, y + height - 18), (20, 27, 35), thickness=2)

    label_scale = fit_text_scale(label, width - 50, 0.90, 2)
    value_scale = fit_text_scale(value_text, width - 60, 1.90, 3)

    cv2.putText(
        image,
        label,
        (x + 28, y + 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        label_scale,
        (196, 212, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value_text,
        (x + 28, y + height - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        value_scale,
        (245, 250, 255),
        3,
        cv2.LINE_AA,
    )


def draw_metric_section(
    image: np.ndarray,
    title: str,
    subtitle_lines: list[str],
    metrics: list[tuple[str, str, tuple[int, int, int]]],
    info_lines: list[str],
    top: int,
    width: int,
    height: int,
):
    section_left = 48
    section_right = width - 48
    section_bottom = top + height
    cv2.rectangle(image, (section_left, top), (section_right, section_bottom), (18, 24, 31), thickness=-1)
    cv2.rectangle(image, (section_left, top), (section_right, section_bottom), (56, 72, 92), thickness=2)

    draw_text_block(
        image,
        [title],
        origin=(section_left + 22, top + 42),
        font_scale=0.76,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )
    draw_text_block(
        image,
        subtitle_lines,
        origin=(section_left + 22, top + 80),
        font_scale=0.46,
        color=(163, 188, 210),
        thickness=1,
        line_gap=6,
    )

    cards_top = top + 118
    gap = 18
    inner_width = section_right - section_left - 44
    card_width = (inner_width - gap * 2) // 3
    card_height = 132

    for idx, (label, value_text, accent_color) in enumerate(metrics):
        draw_metric_card(
            image=image,
            label=label,
            value_text=value_text,
            top_left=(section_left + 22 + idx * (card_width + gap), cards_top),
            size=(card_width, card_height),
            accent_color=accent_color,
        )

    draw_text_block(
        image,
        info_lines,
        origin=(section_left + 22, cards_top + card_height + 42),
        font_scale=0.50,
        color=(163, 188, 210),
        thickness=1,
        line_gap=8,
    )


def draw_status_chip(image, label: str, value_text: str, top_left, size, accent_color, fill_color):
    x, y = top_left
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), fill_color, thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), accent_color, thickness=2)
    cv2.putText(
        image,
        label,
        (x + 16, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (168, 190, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value_text,
        (x + 16, y + height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        fit_text_scale(value_text, width - 30, 0.78, 2),
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )


def draw_axis_bar(
    image: np.ndarray,
    label: str,
    value: float,
    unit: str,
    top_left,
    size,
    accent_color,
    limit: float,
    decimals: int,
):
    x, y = top_left
    width, height = size
    track_top = y + 34
    track_bottom = y + height - 16
    track_left = x + 18
    track_right = x + width - 18
    track_center = (track_left + track_right) // 2

    cv2.rectangle(image, (x, y), (x + width, y + height), (22, 30, 39), thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (58, 76, 98), thickness=2)
    cv2.putText(
        image,
        label,
        (x + 18, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (188, 206, 223),
        1,
        cv2.LINE_AA,
    )

    value_text = f"{value:+.{decimals}f} {unit}"
    cv2.putText(
        image,
        value_text,
        (x + width - 18 - cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)[0][0], y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(image, (track_left, track_top), (track_right, track_bottom), (12, 18, 24), thickness=-1)
    cv2.rectangle(image, (track_left, track_top), (track_right, track_bottom), (45, 60, 78), thickness=1)
    cv2.line(image, (track_center, track_top - 2), (track_center, track_bottom + 2), (198, 208, 220), 1, cv2.LINE_AA)

    normalized = float(np.clip(value / max(limit, 1.0e-6), -1.0, 1.0))
    fill_half = int((track_right - track_left) * 0.5 * abs(normalized))
    if normalized >= 0.0:
        fill_left = track_center
        fill_right = min(track_right, track_center + fill_half)
    else:
        fill_left = max(track_left, track_center - fill_half)
        fill_right = track_center
    if fill_right > fill_left:
        cv2.rectangle(image, (fill_left, track_top + 4), (fill_right, track_bottom - 4), accent_color, thickness=-1)

    cv2.putText(
        image,
        f"-{limit:g}",
        (track_left, track_bottom + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (126, 144, 162),
        1,
        cv2.LINE_AA,
    )
    pos_label = f"+{limit:g}"
    cv2.putText(
        image,
        pos_label,
        (track_right - cv2.getTextSize(pos_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0], track_bottom + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (126, 144, 162),
        1,
        cv2.LINE_AA,
    )


def draw_step_cards(image: np.ndarray, steps: list[tuple[str, str, tuple[int, int, int], bool]], top: int, width: int):
    left = 60
    right = width - 60
    gap = 16
    card_width = (right - left - gap * 3) // 4
    card_height = 124
    for idx, (title, detail, accent_color, active) in enumerate(steps):
        x = left + idx * (card_width + gap)
        fill = (32, 44, 58) if active else (22, 30, 39)
        border = accent_color if active else (61, 76, 92)
        cv2.rectangle(image, (x, top), (x + card_width, top + card_height), fill, thickness=-1)
        cv2.rectangle(image, (x, top), (x + card_width, top + card_height), border, thickness=2)
        cv2.putText(
            image,
            title,
            (x + 16, top + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            fit_text_scale(title, card_width - 30, 0.60, 2),
            (245, 250, 255),
            2,
            cv2.LINE_AA,
        )
        draw_text_block(
            image,
            [detail],
            origin=(x + 16, top + 60),
            font_scale=0.46,
            color=(182, 202, 220),
            thickness=1,
            line_gap=6,
        )


def fmt_vec3(vec: np.ndarray) -> str:
    return f"[{vec[0]:+.4f}, {vec[1]:+.4f}, {vec[2]:+.4f}]"


def pose_is_valid(pose: np.ndarray) -> bool:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        return False
    det = np.linalg.det(pose[:3, :3])
    return bool(np.isfinite(det) and not np.isclose(det, 0.0, atol=1.0e-6))


def compute_robot_world_arm_pose(raw_arm_pose: np.ndarray) -> np.ndarray:
    return T_ROBOT_OPENXR @ raw_arm_pose @ T_OPENXR_ROBOT


def get_pose_from_tele_data(tele_data, side: str):
    if side == "left":
        return getattr(tele_data, "left_robot_world_pose", None)
    return getattr(tele_data, "right_robot_world_pose", None)


def get_pose_from_raw_tv(tv_wrapper, side: str) -> np.ndarray:
    if side == "left":
        return compute_robot_world_arm_pose(tv_wrapper.tvuer.left_arm_pose)
    return compute_robot_world_arm_pose(tv_wrapper.tvuer.right_arm_pose)


def get_recenter_pressed(tele_data, side: str) -> bool:
    if side == "left":
        return bool(getattr(tele_data, "left_ctrl_trigger", False))
    return bool(getattr(tele_data, "right_ctrl_trigger", False))


def get_recenter_hint(side: str) -> str:
    if side == "left":
        return "Left trigger: recenter the baseline."
    return "Right trigger: recenter the baseline."


class PoseTracker:
    def __init__(self, smoothing_alpha: float, baseline_seconds: float):
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.baseline_seconds = max(0.0, baseline_seconds)
        self.filtered_position = None
        self.filtered_quaternion = None
        self.baseline_position = None
        self.baseline_quaternion = None
        self.calibration_started_at = None
        self.baseline_position_samples = []
        self.baseline_quaternion_samples = []
        self.last_recenter_at = 0.0

    def reset_baseline(self, now: float):
        self.baseline_position = None
        self.baseline_quaternion = None
        self.calibration_started_at = now
        self.baseline_position_samples = []
        self.baseline_quaternion_samples = []
        self.last_recenter_at = now

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

    def update(self, pose_matrix: np.ndarray, now: float) -> dict:
        pose_matrix = np.asarray(pose_matrix, dtype=float)
        pose_valid = pose_is_valid(pose_matrix)

        if pose_valid:
            raw_position = pose_matrix[:3, 3].copy()
            raw_quaternion = rotation_matrix_to_quaternion_xyzw(pose_matrix[:3, :3])

            if self.filtered_position is None:
                self.filtered_position = raw_position
                self.filtered_quaternion = raw_quaternion
            else:
                alpha = self.smoothing_alpha
                self.filtered_position = alpha * raw_position + (1.0 - alpha) * self.filtered_position
                self.filtered_quaternion = quaternion_xyzw_nlerp(
                    self.filtered_quaternion,
                    raw_quaternion,
                    alpha,
                )
        elif self.filtered_position is None:
            self.filtered_position = np.zeros(3, dtype=float)
            self.filtered_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

        if self.baseline_position is None or self.baseline_quaternion is None:
            if self.calibration_started_at is None:
                self.calibration_started_at = now
            self.baseline_position_samples.append(self.filtered_position.copy())
            self.baseline_quaternion_samples.append(self.filtered_quaternion.copy())
            elapsed = now - self.calibration_started_at
            if self.baseline_seconds <= 0.0 or elapsed >= self.baseline_seconds:
                self._finalize_baseline()
            progress = 1.0 if self.baseline_seconds <= 0.0 else min(1.0, elapsed / self.baseline_seconds)
            relative_position = np.zeros(3, dtype=float)
            relative_rpy_deg = np.zeros(3, dtype=float)
            calibrating = self.baseline_position is None or self.baseline_quaternion is None
        else:
            progress = 1.0
            relative_position = self.filtered_position - self.baseline_position
            relative_quaternion = quaternion_xyzw_multiply(
                quaternion_xyzw_conjugate(self.baseline_quaternion),
                self.filtered_quaternion,
            )
            relative_rpy_deg = rotation_matrix_to_rpy_deg(
                quaternion_xyzw_to_rotation_matrix(relative_quaternion)
            )
            calibrating = False

        absolute_rpy_deg = rotation_matrix_to_rpy_deg(
            quaternion_xyzw_to_rotation_matrix(self.filtered_quaternion)
        )

        return {
            "pose_valid": pose_valid,
            "filtered_position": self.filtered_position.copy(),
            "filtered_quaternion": self.filtered_quaternion.copy(),
            "absolute_rpy_deg": absolute_rpy_deg,
            "relative_position": relative_position,
            "relative_rpy_deg": relative_rpy_deg,
            "baseline_position": None if self.baseline_position is None else self.baseline_position.copy(),
            "baseline_quaternion": None if self.baseline_quaternion is None else self.baseline_quaternion.copy(),
            "calibrating": calibrating,
            "progress": progress,
        }


def maybe_log_diagnostics(
    now: float,
    last_log_time: float,
    log_interval: float,
    side: str,
    state,
):
    if log_interval <= 0.0 or (now - last_log_time) < log_interval:
        return last_log_time

    logger_mp.info(
        "[diag][%s] valid=%s locked=%s abs_xyz=%s raw_rel_xyz=%s out_rel_xyz=%s raw_rel_rpy_deg=%s out_rel_rpy_deg=%s calibrating=%s",
        side,
        state.pose_valid,
        state.locked,
        fmt_vec3(state.filtered_position),
        fmt_vec3(state.raw_relative_position),
        fmt_vec3(state.output_relative_position),
        fmt_vec3(state.raw_relative_rpy_deg),
        fmt_vec3(state.output_relative_rpy_deg),
        state.calibrating,
    )
    return now


def render_pose_panel(
    side: str,
    state,
    recenter_pressed: bool,
    fps: float,
    uptime_s: float,
    width: int,
    height: int,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (12, 18, 26)
    cv2.rectangle(image, (0, 0), (width, 130), (20, 30, 42), thickness=-1)
    cv2.rectangle(image, (0, height - 110), (width, height), (14, 20, 28), thickness=-1)
    cv2.circle(image, (width - 120, 118), 108, (22, 40, 54), thickness=-1)
    cv2.circle(image, (120, height - 132), 92, (20, 32, 46), thickness=-1)
    cv2.rectangle(image, (22, 22), (width - 22, height - 22), (42, 55, 72), thickness=3)
    cv2.rectangle(image, (44, 44), (width - 44, height - 44), (18, 25, 34), thickness=-1)

    draw_text_block(
        image,
        [f"{side.upper()} CONTROLLER XR MONITOR"],
        origin=(70, 84),
        font_scale=1.04,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )
    draw_text_block(
        image,
        [
            "Live stabilized pose for zeroing and data verification before conversion.",
            "Robot convention: x forward, y left, z up. Origin = startup or trigger recenter pose.",
        ],
        origin=(70, 118),
        font_scale=0.52,
        color=(176, 198, 218),
        thickness=1,
        line_gap=8,
    )

    banner_top = 160
    banner_left = 60
    banner_right = width - 60
    banner_height = 94
    if not state.pose_valid:
        banner_fill = (54, 42, 26)
        banner_border = (255, 183, 84)
        status_title = "WAITING FOR CONTROLLER POSE"
        status_hint = "Wear the headset, wake the controller, and keep it in view of XR tracking."
    elif state.calibrating:
        banner_fill = (38, 54, 78)
        banner_border = (96, 176, 255)
        status_title = f"CALIBRATING {int(round(state.progress * 100.0)):d}%"
        status_hint = f"Hold the {side} controller still until calibration reaches 100%."
    elif recenter_pressed:
        banner_fill = (50, 44, 84)
        banner_border = (198, 154, 255)
        status_title = "RECENTER SIGNAL RECEIVED"
        status_hint = "Zero point is being reset from the current controller pose."
    elif state.locked:
        banner_fill = (42, 58, 30)
        banner_border = (154, 224, 108)
        status_title = "LOCKED AND STABLE"
        status_hint = "Static lock is holding the processed output steady while the controller stays still."
    else:
        banner_fill = (26, 58, 43)
        banner_border = (96, 222, 166)
        status_title = "LIVE TRACKING"
        status_hint = "Move the controller and confirm xyz / rpy increase and decrease as expected."

    cv2.rectangle(image, (banner_left, banner_top), (banner_right, banner_top + banner_height), banner_fill, thickness=-1)
    cv2.rectangle(image, (banner_left, banner_top), (banner_right, banner_top + banner_height), banner_border, thickness=3)
    draw_text_block(
        image,
        [status_title],
        origin=(banner_left + 22, banner_top + 34),
        font_scale=1.00,
        color=(245, 250, 255),
        thickness=2,
        line_gap=6,
    )
    draw_text_block(
        image,
        [status_hint],
        origin=(banner_left + 22, banner_top + 68),
        font_scale=0.58,
        color=(220, 234, 245),
        thickness=1,
        line_gap=6,
    )

    chip_top = 274
    chip_gap = 14
    chip_width = (width - 120 - chip_gap * 3) // 4
    draw_status_chip(
        image,
        "pose stream",
        "valid" if state.pose_valid else "waiting",
        (60, chip_top),
        (chip_width, 74),
        (96, 222, 166) if state.pose_valid else (255, 183, 84),
        (26, 40, 52),
    )
    draw_status_chip(
        image,
        "zero reset",
        "pressed" if recenter_pressed else "ready",
        (60 + (chip_width + chip_gap), chip_top),
        (chip_width, 74),
        (198, 154, 255) if recenter_pressed else (104, 154, 206),
        (26, 40, 52),
    )
    draw_status_chip(
        image,
        "static lock",
        "locked" if state.locked else "tracking",
        (60 + 2 * (chip_width + chip_gap), chip_top),
        (chip_width, 74),
        (154, 224, 108) if state.locked else (96, 176, 255),
        (26, 40, 52),
    )
    draw_status_chip(
        image,
        "runtime",
        f"{uptime_s:.1f}s @ {fps:.1f} fps",
        (60 + 3 * (chip_width + chip_gap), chip_top),
        (chip_width, 74),
        (255, 189, 89),
        (26, 40, 52),
    )

    steps = [
        (
            "1. Wear headset",
            "Open the XR page and wait until the pose stream becomes valid.",
            (96, 176, 255),
            not state.pose_valid,
        ),
        (
            "2. Hold still",
            f"Keep the {side} controller still for startup calibration.",
            (96, 222, 166),
            state.pose_valid and state.calibrating,
        ),
        (
            "3. Pull trigger",
            get_recenter_hint(side).replace(" recenter the baseline.", " to zero the current pose."),
            (198, 154, 255),
            state.pose_valid and (recenter_pressed or not state.calibrating),
        ),
        (
            "4. Move controller",
            "Watch the bars below. Forward/back/up and rotations should respond immediately.",
            (255, 189, 89),
            state.pose_valid and not state.calibrating,
        ),
    ]
    draw_step_cards(image, steps=steps, top=368, width=width)

    section_left = 60
    section_right = width - 60
    section_width = section_right - section_left
    section_height = 214

    cv2.rectangle(image, (section_left, 522), (section_right, 522 + section_height), (18, 24, 31), thickness=-1)
    cv2.rectangle(image, (section_left, 522), (section_right, 522 + section_height), (56, 72, 92), thickness=2)
    draw_text_block(
        image,
        [
            "POSITION BARS",
            "Processed relative xyz values after EMA, soft deadzone, and static lock.",
        ],
        origin=(section_left + 20, 556),
        font_scale=0.66,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )
    bar_y = 596
    bar_gap = 14
    bar_height = 46
    bar_width = section_width - 40
    draw_axis_bar(image, "x", state.output_relative_position[0], "m", (section_left + 20, bar_y), (bar_width, bar_height), (80, 160, 255), 0.20, 4)
    draw_axis_bar(image, "y", state.output_relative_position[1], "m", (section_left + 20, bar_y + (bar_height + bar_gap)), (bar_width, bar_height), (103, 214, 186), 0.20, 4)
    draw_axis_bar(image, "z", state.output_relative_position[2], "m", (section_left + 20, bar_y + 2 * (bar_height + bar_gap)), (bar_width, bar_height), (255, 189, 89), 0.20, 4)

    cv2.rectangle(image, (section_left, 760), (section_right, 760 + section_height), (18, 24, 31), thickness=-1)
    cv2.rectangle(image, (section_left, 760), (section_right, 760 + section_height), (56, 72, 92), thickness=2)
    draw_text_block(
        image,
        [
            "ROTATION BARS",
            "Euler order = roll, pitch, yaw. These are relative to the frozen zero orientation.",
        ],
        origin=(section_left + 20, 794),
        font_scale=0.66,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )
    rot_bar_y = 834
    draw_axis_bar(image, "roll", state.output_relative_rpy_deg[0], "deg", (section_left + 20, rot_bar_y), (bar_width, bar_height), (255, 126, 126), 60.0, 1)
    draw_axis_bar(image, "pitch", state.output_relative_rpy_deg[1], "deg", (section_left + 20, rot_bar_y + (bar_height + bar_gap)), (bar_width, bar_height), (126, 208, 255), 60.0, 1)
    draw_axis_bar(image, "yaw", state.output_relative_rpy_deg[2], "deg", (section_left + 20, rot_bar_y + 2 * (bar_height + bar_gap)), (bar_width, bar_height), (182, 146, 255), 60.0, 1)

    footer_lines = [
        f"raw relative xyz = {fmt_vec3(state.raw_relative_position)} m",
        f"raw relative rpy = {fmt_vec3(state.raw_relative_rpy_deg)} deg",
        "Source = tele_data.<side>_robot_world_pose; press Ctrl+C in the terminal to stop this viewer.",
    ]
    draw_text_block(
        image,
        footer_lines,
        origin=(70, height - 64),
        font_scale=0.49,
        color=(163, 188, 210),
        thickness=1,
        line_gap=7,
    )

    return image


def make_xr_frame(panel: np.ndarray, binocular: bool) -> np.ndarray:
    if binocular:
        return np.concatenate([panel, panel], axis=1)
    return panel


def main():
    global T_ROBOT_OPENXR, T_OPENXR_ROBOT, cv2
    args = parse_args()
    import cv2 as cv2_module

    cv2 = cv2_module
    warn_if_nested_virtualenv()
    install_aiohttp_resume_patch()
    from televuer import TeleVuerWrapper
    from televuer.tv_wrapper import T_OPENXR_ROBOT as TV_T_OPENXR_ROBOT
    from televuer.tv_wrapper import T_ROBOT_OPENXR as TV_T_ROBOT_OPENXR

    install_vuer_websocket_patch()
    T_ROBOT_OPENXR = TV_T_ROBOT_OPENXR
    T_OPENXR_ROBOT = TV_T_OPENXR_ROBOT

    host_ip = detect_host_ip(args.host_ip)
    binocular = not args.monocular
    image_shape = (args.panel_height, args.panel_width * 2 if binocular else args.panel_width)

    logger_mp.info("Launching controller pose XR viewer...")
    logger_mp.info("XR browser URL: %s", build_browser_url(host_ip))
    logger_mp.info(
        "side=%s display_mode=%s binocular=%s image_shape=%s",
        args.side,
        args.display_mode,
        binocular,
        image_shape,
    )

    tv_wrapper = TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=binocular,
        img_shape=image_shape,
        display_fps=args.display_fps,
        display_mode=args.display_mode,
        zmq=True,
        webrtc=False,
        cert_file=args.cert_file,
        key_file=args.key_file,
        show_hud=False,
        show_controller_models=False,
    )

    start_time = time.time()
    target_dt = 1.0 / max(args.display_fps, 1.0)
    last_log_time = 0.0
    prev_recenter_pressed = False

    stable_filter = StableRelativePoseFilter(
        StablePoseConfig(
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
    )

    try:
        while True:
            loop_start = time.time()
            tele_data = tv_wrapper.get_tele_data()
            pose = get_pose_from_tele_data(tele_data, args.side)
            if pose is None or not pose_is_valid(pose):
                pose = get_pose_from_raw_tv(tv_wrapper, args.side)

            recenter_pressed = get_recenter_pressed(tele_data, args.side)
            if recenter_pressed and not prev_recenter_pressed:
                stable_filter.reset_baseline(int(loop_start * 1000))
                logger_mp.info("Recenter signal detected, baseline recalibration started.")
            prev_recenter_pressed = recenter_pressed

            state = stable_filter.update(int(loop_start * 1000), pose)
            last_log_time = maybe_log_diagnostics(
                now=loop_start,
                last_log_time=last_log_time,
                log_interval=args.log_interval,
                side=args.side,
                state=state,
            )

            panel = render_pose_panel(
                side=args.side,
                state=state,
                recenter_pressed=recenter_pressed,
                fps=args.display_fps,
                uptime_s=loop_start - start_time,
                width=args.panel_width,
                height=args.panel_height,
            )
            tv_wrapper.render_to_xr(make_xr_frame(panel, binocular=binocular))

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, target_dt - elapsed))
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt received, stopping controller pose XR viewer...")
    finally:
        tv_wrapper.close()
        logger_mp.info("XR viewer closed.")


if __name__ == "__main__":
    main()
