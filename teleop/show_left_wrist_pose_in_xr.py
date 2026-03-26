import argparse
import os
import site
import socket
import sys
import time
from pathlib import Path

import cv2
import logging_mp
import numpy as np


logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)


THIS_DIR = Path(__file__).resolve().parent
TELEVUER_SRC = THIS_DIR / "televuer" / "src"
if str(TELEVUER_SRC) not in sys.path:
    sys.path.insert(0, str(TELEVUER_SRC))


XR_PORT = 8012
T_ROBOT_OPENXR = None
T_OPENXR_ROBOT = None
T_TO_UNITREE_HUMANOID_LEFT_ARM = None


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
    parser = argparse.ArgumentParser(
        description="Display raw left_arm_pose and processed left_wrist_pose directly inside the XR browser page."
    )
    parser.add_argument(
        "--tracking-mode",
        choices=["hand", "controller"],
        default="hand",
        help="XR input mode used by TeleVuerWrapper. Use 'controller' when testing physical controllers.",
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
        default=900,
        help="Pose panel height in pixels.",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=900,
        help="Per-eye pose panel width in pixels.",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Also render the full 4x4 matrix below the large xyz cards.",
    )
    parser.add_argument(
        "--baseline-seconds",
        type=float,
        default=2.0,
        help="Initial calibration duration. Hold the left hand still during this time.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor for the displayed position. Higher means more reactive.",
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
    # vuer/params-proto inspects sys.argv during import. Keep only this tool's
    # already-parsed arguments so our custom CLI stays stable.
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


class PositionTracker:
    def __init__(self, smoothing_alpha: float, baseline_seconds: float):
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.baseline_seconds = max(0.0, baseline_seconds)
        self.filtered_position = None
        self.baseline_position = None
        self.calibration_started_at = None
        self.baseline_samples = []
        self.last_recenter_at = 0.0

    def reset_baseline(self, now: float):
        self.baseline_position = None
        self.calibration_started_at = now
        self.baseline_samples = []
        self.last_recenter_at = now

    def update(self, raw_position: np.ndarray, now: float):
        raw_position = np.asarray(raw_position, dtype=float)
        if self.filtered_position is None:
            self.filtered_position = raw_position.copy()
        else:
            alpha = self.smoothing_alpha
            self.filtered_position = alpha * raw_position + (1.0 - alpha) * self.filtered_position

        if self.baseline_position is None:
            if self.calibration_started_at is None:
                self.calibration_started_at = now
            self.baseline_samples.append(self.filtered_position.copy())
            elapsed = now - self.calibration_started_at
            if self.baseline_seconds <= 0.0 or elapsed >= self.baseline_seconds:
                self.baseline_position = np.mean(self.baseline_samples, axis=0)
            progress = 1.0 if self.baseline_seconds <= 0.0 else min(1.0, elapsed / self.baseline_seconds)
            delta = np.zeros(3)
            calibrating = self.baseline_position is None
        else:
            progress = 1.0
            delta = self.filtered_position - self.baseline_position
            calibrating = False

        return {
            "filtered_position": self.filtered_position.copy(),
            "baseline_position": None if self.baseline_position is None else self.baseline_position.copy(),
            "delta_position": delta,
            "calibrating": calibrating,
            "progress": progress,
        }


def format_matrix_lines(matrix: np.ndarray) -> list[str]:
    return [
        "[" + "  ".join(f"{value: .4f}" for value in row) + "]"
        for row in matrix
    ]


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


def fit_text_scale(text: str, max_width: int, base_scale: float, thickness: int) -> float:
    scale = base_scale
    for _ in range(20):
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale *= 0.92
    return scale


def draw_metric_card(image, label: str, value_text: str, top_left, size, accent_color):
    x, y = top_left
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), (27, 35, 45), thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), accent_color, thickness=3)
    cv2.rectangle(image, (x + 18, y + 18), (x + width - 18, y + height - 18), (20, 27, 35), thickness=2)

    label_scale = fit_text_scale(label, width - 50, 0.95, 2)
    value_scale = fit_text_scale(value_text, width - 70, 2.15, 3)

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


def draw_pose_section(
    image: np.ndarray,
    title: str,
    subtitle_lines: list[str],
    delta_position: np.ndarray,
    absolute_position: np.ndarray,
    top: int,
    width: int,
    height: int,
    calibrating: bool,
    calibration_progress: float,
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

    cards_top = top + 120
    gap = 18
    inner_width = section_right - section_left - 44
    card_width = (inner_width - gap * 2) // 3
    card_height = 132
    metrics = [
        ("dx", f"{delta_position[0]: .4f} m", (80, 160, 255)),
        ("dy", f"{delta_position[1]: .4f} m", (103, 214, 186)),
        ("dz", f"{delta_position[2]: .4f} m", (255, 189, 89)),
    ]
    for idx, (label, value_text, accent_color) in enumerate(metrics):
        draw_metric_card(
            image,
            label=label,
            value_text=value_text,
            top_left=(section_left + 22 + idx * (card_width + gap), cards_top),
            size=(card_width, card_height),
            accent_color=accent_color,
        )

    info_lines = [
        f"filtered abs x/y/z = {absolute_position[0]: .4f}, {absolute_position[1]: .4f}, {absolute_position[2]: .4f} m",
        "Displayed values are relative to the startup baseline.",
    ]
    if calibrating:
        calibration_percent = int(round(calibration_progress * 100.0))
        info_lines = [
            f"Calibrating baseline... {calibration_percent}%",
            "Hold the left source still until calibration reaches 100%.",
        ]
    draw_text_block(
        image,
        info_lines,
        origin=(section_left + 22, cards_top + card_height + 42),
        font_scale=0.52,
        color=(163, 188, 210),
        thickness=1,
        line_gap=8,
    )


def render_matrix_block(
    image: np.ndarray,
    title: str,
    matrix: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
):
    x, y = top_left
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), (18, 24, 31), thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (56, 72, 92), thickness=2)
    title_y = draw_text_block(
        image,
        [title],
        origin=(x + 16, y + 30),
        font_scale=0.5,
        color=(235, 240, 245),
        thickness=1,
        line_gap=6,
    )
    draw_text_block(
        image,
        format_matrix_lines(matrix),
        origin=(x + 16, title_y + 4),
        font_scale=0.38,
        color=(140, 255, 199),
        thickness=1,
        line_gap=9,
    )


def get_recenter_pressed(tele_data, tracking_mode: str) -> bool:
    if tracking_mode == "hand":
        return bool(tele_data.left_hand_pinch)
    return bool(tele_data.left_ctrl_trigger)


def get_recenter_hint(tracking_mode: str) -> str:
    if tracking_mode == "hand":
        return "Left-hand pinch will recenter the baseline."
    return "Left controller trigger will recenter the baseline."


def fmt_vec3(vec: np.ndarray) -> str:
    return f"[{vec[0]: .4f}, {vec[1]: .4f}, {vec[2]: .4f}]"


def pose_is_valid(pose: np.ndarray) -> bool:
    det = np.linalg.det(pose)
    return bool(np.isfinite(det) and not np.isclose(det, 0.0, atol=1e-6))


def compute_robot_world_arm_pose(raw_arm_pose: np.ndarray, tracking_mode: str) -> np.ndarray:
    robot_world_pose = T_ROBOT_OPENXR @ raw_arm_pose @ T_OPENXR_ROBOT
    if tracking_mode == "hand" and pose_is_valid(raw_arm_pose):
        robot_world_pose = robot_world_pose @ T_TO_UNITREE_HUMANOID_LEFT_ARM
    return robot_world_pose


def compute_robot_world_head_pose(raw_head_pose: np.ndarray) -> np.ndarray:
    return T_ROBOT_OPENXR @ raw_head_pose @ T_OPENXR_ROBOT


def maybe_log_diagnostics(
    now: float,
    last_log_time: float,
    log_interval: float,
    tracking_mode: str,
    raw_arm_pose: np.ndarray,
    raw_head_pose: np.ndarray,
    robot_world_arm_pose: np.ndarray,
    robot_world_head_pose: np.ndarray,
    processed_pose: np.ndarray,
    head_pose_valid: bool,
    arm_pose_valid: bool,
    raw_state: dict,
    processed_state: dict,
):
    if log_interval <= 0.0 or (now - last_log_time) < log_interval:
        return last_log_time

    logger_mp.info(
        "[diag][%s] raw_arm_xyz=%s raw_head_xyz=%s robot_world_arm_xyz=%s robot_world_head_xyz=%s processed_xyz=%s raw_delta=%s processed_delta=%s raw_cal=%s processed_cal=%s",
        tracking_mode,
        fmt_vec3(raw_arm_pose[:3, 3]),
        fmt_vec3(raw_head_pose[:3, 3]),
        fmt_vec3(robot_world_arm_pose[:3, 3]),
        fmt_vec3(robot_world_head_pose[:3, 3]),
        fmt_vec3(processed_pose[:3, 3]),
        fmt_vec3(raw_state["delta_position"]),
        fmt_vec3(processed_state["delta_position"]),
        raw_state["calibrating"],
        processed_state["calibrating"],
    )
    logger_mp.info(
        "[diag][%s] valid arm=%s head=%s robot_world_minus_head=%s waist_offset_plus=%s",
        tracking_mode,
        arm_pose_valid,
        head_pose_valid,
        fmt_vec3(robot_world_arm_pose[:3, 3] - robot_world_head_pose[:3, 3]),
        fmt_vec3(processed_pose[:3, 3] - (robot_world_arm_pose[:3, 3] - robot_world_head_pose[:3, 3])),
    )
    return now


def render_pose_panel(
    tracking_mode: str,
    raw_pose: np.ndarray,
    raw_state: dict,
    processed_pose: np.ndarray,
    processed_state: dict,
    fps: float,
    uptime_s: float,
    width: int,
    height: int,
    show_matrix: bool,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (16, 22, 30)

    cv2.rectangle(image, (22, 22), (width - 22, height - 22), (42, 55, 72), thickness=3)
    cv2.rectangle(image, (44, 44), (width - 44, height - 44), (21, 29, 38), thickness=-1)

    draw_text_block(
        image,
        [f"left pose diagnostic panel ({tracking_mode})"],
        origin=(70, 96),
        font_scale=1.1,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )

    overall_calibrating = bool(raw_state["calibrating"] or processed_state["calibrating"])
    overall_progress = min(float(raw_state["progress"]), float(processed_state["progress"]))
    banner_top = 132
    banner_height = 88
    banner_left = 60
    banner_right = width - 60
    if overall_calibrating:
        banner_fill = (38, 54, 78)
        banner_border = (96, 176, 255)
        status_title = f"CALIBRATING {int(round(overall_progress * 100.0)):d}%"
        status_hint = "Hold head and left controller still for about 2 seconds."
    else:
        banner_fill = (26, 58, 43)
        banner_border = (96, 222, 166)
        status_title = "READY"
        status_hint = "Start the test now. Press left trigger once to recenter."

    cv2.rectangle(image, (banner_left, banner_top), (banner_right, banner_top + banner_height), banner_fill, thickness=-1)
    cv2.rectangle(image, (banner_left, banner_top), (banner_right, banner_top + banner_height), banner_border, thickness=3)
    draw_text_block(
        image,
        [status_title],
        origin=(banner_left + 22, banner_top + 34),
        font_scale=1.05,
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

    meta_lines = [
        f"fps = {fps: .1f}",
        f"uptime = {uptime_s: .1f} s",
        "Top block: raw left_arm_pose under OpenXR world",
        "Bottom block: processed left_wrist_pose under robot convention",
    ]
    draw_text_block(
        image,
        meta_lines,
        origin=(70, 244),
        font_scale=0.50,
        color=(163, 188, 210),
        thickness=1,
        line_gap=6,
    )

    section_height = 240 if show_matrix else 255
    first_top = 334
    second_top = first_top + section_height + 22
    draw_pose_section(
        image=image,
        title="RAW left_arm_pose",
        subtitle_lines=[
            "basis = OpenXR convention",
            "origin = XR world frame",
        ],
        delta_position=raw_state["delta_position"],
        absolute_position=raw_state["filtered_position"],
        top=first_top,
        width=width,
        height=section_height,
        calibrating=raw_state["calibrating"],
        calibration_progress=raw_state["progress"],
    )
    draw_pose_section(
        image=image,
        title="PROCESSED left_wrist_pose",
        subtitle_lines=[
            "basis = Robot convention",
            "origin = head-relative + waist offset",
        ],
        delta_position=processed_state["delta_position"],
        absolute_position=processed_state["filtered_position"],
        top=second_top,
        width=width,
        height=section_height,
        calibrating=processed_state["calibrating"],
        calibration_progress=processed_state["progress"],
    )

    footer_y = height - 48

    if show_matrix:
        matrix_top = second_top + section_height + 18
        matrix_gap = 16
        matrix_width = (width - 96 - matrix_gap) // 2
        matrix_height = max(150, height - matrix_top - 76)
        render_matrix_block(
            image=image,
            title="RAW 4x4 matrix",
            matrix=raw_pose,
            top_left=(48, matrix_top),
            size=(matrix_width, matrix_height),
        )
        render_matrix_block(
            image=image,
            title="PROCESSED 4x4 matrix",
            matrix=processed_pose,
            top_left=(48 + matrix_width + matrix_gap, matrix_top),
            size=(matrix_width, matrix_height),
        )
        footer_y = matrix_top - 40

    footer_lines = [
        get_recenter_hint(tracking_mode),
        "Press Ctrl+C in the terminal to stop.",
    ]
    draw_text_block(
        image,
        footer_lines,
        origin=(70, footer_y),
        font_scale=0.62,
        color=(163, 188, 210),
        thickness=1,
        line_gap=8,
    )

    return image


def make_xr_frame(panel: np.ndarray, binocular: bool) -> np.ndarray:
    if binocular:
        return np.concatenate([panel, panel], axis=1)
    return panel


def main():
    global T_ROBOT_OPENXR, T_OPENXR_ROBOT, T_TO_UNITREE_HUMANOID_LEFT_ARM
    args = parse_args()
    warn_if_nested_virtualenv()
    install_aiohttp_resume_patch()
    from televuer import TeleVuerWrapper
    from televuer.tv_wrapper import T_OPENXR_ROBOT as TV_T_OPENXR_ROBOT
    from televuer.tv_wrapper import T_ROBOT_OPENXR as TV_T_ROBOT_OPENXR
    from televuer.tv_wrapper import T_TO_UNITREE_HUMANOID_LEFT_ARM as TV_T_TO_UNITREE_HUMANOID_LEFT_ARM
    install_vuer_websocket_patch()
    T_ROBOT_OPENXR = TV_T_ROBOT_OPENXR
    T_OPENXR_ROBOT = TV_T_OPENXR_ROBOT
    T_TO_UNITREE_HUMANOID_LEFT_ARM = TV_T_TO_UNITREE_HUMANOID_LEFT_ARM

    host_ip = detect_host_ip(args.host_ip)
    binocular = not args.monocular
    image_shape = (args.panel_height, args.panel_width * 2 if binocular else args.panel_width)

    logger_mp.info("Launching left_wrist_pose XR viewer...")
    logger_mp.info("XR browser URL: %s", build_browser_url(host_ip))
    logger_mp.info(
        "Display mode=%s, tracking_mode=%s, binocular=%s, image_shape=%s",
        args.display_mode,
        args.tracking_mode,
        binocular,
        image_shape,
    )

    tv_wrapper = TeleVuerWrapper(
        use_hand_tracking=(args.tracking_mode == "hand"),
        binocular=binocular,
        img_shape=image_shape,
        display_fps=args.display_fps,
        display_mode=args.display_mode,
        zmq=True,
        webrtc=False,
        cert_file=args.cert_file,
        key_file=args.key_file,
    )

    start_time = time.time()
    target_dt = 1.0 / max(args.display_fps, 1.0)
    last_log_time = 0.0
    raw_tracker = PositionTracker(
        smoothing_alpha=args.smoothing_alpha,
        baseline_seconds=args.baseline_seconds,
    )
    processed_tracker = PositionTracker(
        smoothing_alpha=args.smoothing_alpha,
        baseline_seconds=args.baseline_seconds,
    )
    prev_recenter_pressed = False

    try:
        while True:
            loop_start = time.time()
            tele_data = tv_wrapper.get_tele_data()
            recenter_pressed = get_recenter_pressed(tele_data, args.tracking_mode)
            if recenter_pressed and not prev_recenter_pressed:
                raw_tracker.reset_baseline(loop_start)
                processed_tracker.reset_baseline(loop_start)
                logger_mp.info("Recenter signal detected, baseline recalibration started.")
            prev_recenter_pressed = recenter_pressed
            raw_pose = tv_wrapper.tvuer.left_arm_pose
            raw_head_pose = tv_wrapper.tvuer.head_pose
            robot_world_arm_pose = (
                tele_data.left_robot_world_pose
                if getattr(tele_data, "left_robot_world_pose", None) is not None
                else compute_robot_world_arm_pose(raw_pose, args.tracking_mode)
            )
            robot_world_head_pose = compute_robot_world_head_pose(raw_head_pose)
            processed_pose = tele_data.left_wrist_pose
            head_pose_valid = bool(getattr(tele_data, "head_pose_valid", pose_is_valid(raw_head_pose)))
            arm_pose_valid = bool(getattr(tele_data, "left_arm_pose_valid", pose_is_valid(raw_pose)))
            raw_state = raw_tracker.update(raw_pose[:3, 3], loop_start)
            processed_state = processed_tracker.update(processed_pose[:3, 3], loop_start)
            last_log_time = maybe_log_diagnostics(
                now=loop_start,
                last_log_time=last_log_time,
                log_interval=args.log_interval,
                tracking_mode=args.tracking_mode,
                raw_arm_pose=raw_pose,
                raw_head_pose=raw_head_pose,
                robot_world_arm_pose=robot_world_arm_pose,
                robot_world_head_pose=robot_world_head_pose,
                processed_pose=processed_pose,
                head_pose_valid=head_pose_valid,
                arm_pose_valid=arm_pose_valid,
                raw_state=raw_state,
                processed_state=processed_state,
            )
            panel = render_pose_panel(
                tracking_mode=args.tracking_mode,
                raw_pose=raw_pose,
                raw_state=raw_state,
                processed_pose=processed_pose,
                processed_state=processed_state,
                fps=args.display_fps,
                uptime_s=loop_start - start_time,
                width=args.panel_width,
                height=args.panel_height,
                show_matrix=args.show_matrix,
            )
            tv_wrapper.render_to_xr(make_xr_frame(panel, binocular=binocular))

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, target_dt - elapsed))
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt received, stopping left_wrist_pose XR viewer...")
    finally:
        tv_wrapper.close()
        logger_mp.info("XR viewer closed.")


if __name__ == "__main__":
    main()
