import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import cv2
import logging_mp
import numpy as np
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.xr_bridge_writer import XRBridgeWriter
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
RECORD_START_DELAY_SEC = 3.0
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


MISSING_FRAME_WARNINGS = set()


def warn_missing_frame_once(key: str, message: str):
    if key in MISSING_FRAME_WARNINGS:
        return
    MISSING_FRAME_WARNINGS.add(key)
    logger_mp.warning(message)


def fit_text_scale(text: str, max_width: int, base_scale: float, thickness: int) -> float:
    scale = base_scale
    for _ in range(20):
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if width <= max_width:
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


def draw_hud_chip(image, label, value, top_left, size, accent, fill):
    x, y = top_left
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), fill, thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), accent, thickness=2)
    cv2.putText(
        image,
        label,
        (x + 16, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (168, 190, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value,
        (x + 16, y + height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        fit_text_scale(value, width - 32, 0.82, 2),
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )


def draw_step_card(image, title, detail, top_left, size, accent, active):
    x, y = top_left
    width, height = size
    fill = (34, 47, 62) if active else (22, 30, 39)
    border = accent if active else (61, 76, 92)
    cv2.rectangle(image, (x, y), (x + width, y + height), fill, thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), border, thickness=2)
    cv2.putText(
        image,
        title,
        (x + 16, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        fit_text_scale(title, width - 32, 0.58, 2),
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )
    draw_text_block(
        image,
        [detail],
        origin=(x + 16, y + 58),
        font_scale=0.44,
        color=(182, 202, 220),
        thickness=1,
        line_gap=6,
    )


def make_xr_frame(panel: np.ndarray, binocular: bool) -> np.ndarray:
    if binocular:
        return np.concatenate([panel, panel], axis=1)
    return panel


def should_use_controller_record_panel(args, collector_only: bool) -> bool:
    return bool(collector_only and args.input_mode == "controller")


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1.0e-6

    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0

    return np.rad2deg(np.asarray([roll, pitch, yaw], dtype=float))


def controller_pose_summary(pose: np.ndarray):
    if pose is None:
        return None
    xyz = np.asarray(pose[:3, 3], dtype=float)
    rpy_deg = rotation_matrix_to_rpy_deg(np.asarray(pose[:3, :3], dtype=float))
    return xyz, rpy_deg


def draw_compact_chip(image, label, value, top_left, size, accent, fill):
    x, y = top_left
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), fill, thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), accent, thickness=2)
    cv2.putText(
        image,
        label.upper(),
        (x + 18, y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (162, 186, 208),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        value,
        (x + 18, y + height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        fit_text_scale(value, width - 36, 0.76, 2),
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )


def draw_controller_pose_card(image, title, pose, valid, top_left, size, accent):
    x, y = top_left
    width, height = size
    fill = (14, 18, 26)
    border = accent if valid else (88, 110, 128)
    cv2.rectangle(image, (x, y), (x + width, y + height), fill, thickness=-1)
    cv2.rectangle(image, (x, y), (x + width, y + height), border, thickness=2)
    cv2.putText(
        image,
        title,
        (x + 22, y + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 250, 255),
        2,
        cv2.LINE_AA,
    )
    status_text = "tracking ok" if valid and pose is not None else "tracking lost"
    cv2.putText(
        image,
        status_text,
        (x + 22, y + 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        accent if valid else (170, 120, 120),
        1,
        cv2.LINE_AA,
    )

    if pose is None:
        draw_text_block(
            image,
            ["xyz: unavailable", "rpy: unavailable"],
            origin=(x + 22, y + 118),
            font_scale=0.72,
            color=(188, 200, 214),
            thickness=1,
            line_gap=18,
        )
        return

    xyz, rpy_deg = pose
    draw_text_block(
        image,
        [
            f"x     {xyz[0]:+0.3f} m",
            f"y     {xyz[1]:+0.3f} m",
            f"z     {xyz[2]:+0.3f} m",
            "",
            f"roll  {rpy_deg[0]:+0.1f} deg",
            f"pitch {rpy_deg[1]:+0.1f} deg",
            f"yaw   {rpy_deg[2]:+0.1f} deg",
        ],
        origin=(x + 22, y + 108),
        font_scale=0.70,
        color=(236, 242, 248),
        thickness=1,
        line_gap=14,
    )


def build_controller_record_panel(args, tele_data, record_start_deadline=None, status_override=None, detail_override=None):
    panel = np.zeros((1040, 960, 3), dtype=np.uint8)
    panel[:] = (4, 7, 12)

    cv2.rectangle(panel, (0, 0), (960, 116), (10, 18, 30), thickness=-1)
    cv2.rectangle(panel, (28, 24), (932, 1012), (28, 40, 54), thickness=2)
    cv2.rectangle(panel, (44, 136), (916, 278), (9, 14, 22), thickness=-1)
    cv2.rectangle(panel, (44, 300), (916, 742), (8, 12, 18), thickness=-1)
    cv2.rectangle(panel, (44, 764), (916, 1000), (9, 14, 22), thickness=-1)

    if status_override is not None:
        status_text = status_override
    elif record_start_deadline is not None and not RECORD_RUNNING:
        status_text = f"COUNTDOWN {max(0.0, record_start_deadline - time.time()):0.1f}s"
    elif RECORD_RUNNING:
        status_text = "RECORDING"
    elif START:
        status_text = "SESSION RUNNING"
    else:
        status_text = "WAITING TO START"

    can_record = bool(args.record and START and not RECORD_RUNNING and record_start_deadline is None and READY)
    if detail_override is not None:
        next_step = detail_override
    elif not START:
        next_step = "Next: Press Left X to start session."
    elif not args.record:
        next_step = "Next: Tracking only. Recording is disabled."
    elif record_start_deadline is not None and not RECORD_RUNNING:
        next_step = "Next: Hold still. Recording starts after countdown."
    elif RECORD_RUNNING:
        next_step = "Next: Do the action, then press Right B to save."
    elif can_record:
        next_step = "Next: Press Right B to start recording."
    else:
        next_step = "Next: Wait until recorder is ready."

    draw_text_block(
        panel,
        ["XR CONTROLLER RECORDER"],
        origin=(58, 64),
        font_scale=1.02,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )
    draw_text_block(
        panel,
        ["Only the clean status panel is rendered in XR. No extra HUD or controller models."],
        origin=(58, 96),
        font_scale=0.48,
        color=(172, 192, 214),
        thickness=1,
        line_gap=8,
    )

    cv2.rectangle(panel, (56, 152), (904, 260), (14, 34, 48), thickness=-1)
    cv2.rectangle(panel, (56, 152), (904, 260), (78, 156, 255), thickness=2)
    draw_text_block(
        panel,
        [status_text],
        origin=(82, 192),
        font_scale=0.96,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )
    draw_text_block(
        panel,
        [next_step],
        origin=(82, 232),
        font_scale=0.56,
        color=(224, 233, 244),
        thickness=1,
        line_gap=8,
    )

    chip_y = 784
    chip_w = 202
    chip_h = 82
    chip_gap = 16
    draw_compact_chip(panel, "session", "running" if START else "waiting", (56, chip_y), (chip_w, chip_h), (94, 176, 255), (16, 23, 34))
    if record_start_deadline is not None and not RECORD_RUNNING:
        record_value = "countdown"
        record_accent = (255, 186, 84)
    else:
        record_value = "yes" if RECORD_RUNNING else "no"
        record_accent = (80, 94, 255) if RECORD_RUNNING else (104, 154, 206)
    draw_compact_chip(panel, "recording", record_value, (56 + chip_w + chip_gap, chip_y), (chip_w, chip_h), record_accent, (16, 23, 34))
    draw_compact_chip(panel, "can record", "yes" if can_record else "no", (56 + 2 * (chip_w + chip_gap), chip_y), (chip_w, chip_h), (102, 220, 170), (16, 23, 34))
    draw_compact_chip(panel, "input", args.input_mode, (56 + 3 * (chip_w + chip_gap), chip_y), (chip_w, chip_h), (255, 188, 96), (16, 23, 34))

    left_pose = controller_pose_summary(getattr(tele_data, "left_robot_world_pose", None)) if tele_data is not None else None
    right_pose = controller_pose_summary(getattr(tele_data, "right_robot_world_pose", None)) if tele_data is not None else None
    left_valid = bool(getattr(tele_data, "left_arm_pose_valid", False)) if tele_data is not None else False
    right_valid = bool(getattr(tele_data, "right_arm_pose_valid", False)) if tele_data is not None else False
    draw_controller_pose_card(panel, "LEFT CONTROLLER", left_pose, left_valid, (56, 324), (414, 398), (90, 178, 255))
    draw_controller_pose_card(panel, "RIGHT CONTROLLER", right_pose, right_valid, (490, 324), (414, 398), (96, 222, 166))

    draw_text_block(
        panel,
        [
            "Keys",
            "Left X: start session",
            "Right B: start/save recording",
            "Trigger: mark zero pose",
            "Keyboard Q: exit",
        ],
        origin=(72, 886),
        font_scale=0.58,
        color=(220, 230, 240),
        thickness=1,
        line_gap=12,
    )
    return panel


def build_record_colors(camera_config, head_img, left_wrist_img, right_wrist_img):
    colors = {}
    depths = {}
    head_camera = camera_config.get('head_camera', {})
    left_wrist_camera = camera_config.get('left_wrist_camera', {})
    right_wrist_camera = camera_config.get('right_wrist_camera', {})

    head_bgr = getattr(head_img, "bgr", None)
    left_wrist_bgr = getattr(left_wrist_img, "bgr", None)
    right_wrist_bgr = getattr(right_wrist_img, "bgr", None)

    if head_camera.get('binocular', False):
        if head_bgr is not None:
            image_width = head_camera.get('image_shape', [0, head_bgr.shape[1]])[1]
            half_width = image_width // 2
            colors[f"color_{0}"] = head_bgr[:, :half_width]
            colors[f"color_{1}"] = head_bgr[:, half_width:]
        else:
            warn_missing_frame_once("head_binocular", "Head image is None, skip recording head camera frames.")
        if left_wrist_camera.get('enable_zmq', False):
            if left_wrist_bgr is not None:
                colors[f"color_{2}"] = left_wrist_bgr
            else:
                warn_missing_frame_once("left_wrist", "Left wrist image is None, skip recording left wrist camera frame.")
        if right_wrist_camera.get('enable_zmq', False):
            if right_wrist_bgr is not None:
                colors[f"color_{3}"] = right_wrist_bgr
            else:
                warn_missing_frame_once("right_wrist", "Right wrist image is None, skip recording right wrist camera frame.")
    else:
        if head_bgr is not None:
            colors[f"color_{0}"] = head_bgr
        else:
            warn_missing_frame_once("head_mono", "Head image is None, skip recording head camera frame.")
        if left_wrist_camera.get('enable_zmq', False):
            if left_wrist_bgr is not None:
                colors[f"color_{1}"] = left_wrist_bgr
            else:
                warn_missing_frame_once("left_wrist", "Left wrist image is None, skip recording left wrist camera frame.")
        if right_wrist_camera.get('enable_zmq', False):
            if right_wrist_bgr is not None:
                colors[f"color_{2}"] = right_wrist_bgr
            else:
                warn_missing_frame_once("right_wrist", "Right wrist image is None, skip recording right wrist camera frame.")
    return colors, depths


def build_collector_states(tele_data):
    return {
        "xr": {
            "head_pose": tele_data.head_pose.tolist(),
            "left_wrist_pose": tele_data.left_wrist_pose.tolist(),
            "right_wrist_pose": tele_data.right_wrist_pose.tolist(),
            "head_pose_valid": tele_data.head_pose_valid,
            "left_arm_pose_valid": tele_data.left_arm_pose_valid,
            "right_arm_pose_valid": tele_data.right_arm_pose_valid,
            "left_robot_world_pose": [] if tele_data.left_robot_world_pose is None else tele_data.left_robot_world_pose.tolist(),
            "right_robot_world_pose": [] if tele_data.right_robot_world_pose is None else tele_data.right_robot_world_pose.tolist(),
            "left_hand_pos": [] if tele_data.left_hand_pos is None else tele_data.left_hand_pos.tolist(),
            "right_hand_pos": [] if tele_data.right_hand_pos is None else tele_data.right_hand_pos.tolist(),
            "left_hand_rot": [] if tele_data.left_hand_rot is None else tele_data.left_hand_rot.tolist(),
            "right_hand_rot": [] if tele_data.right_hand_rot is None else tele_data.right_hand_rot.tolist(),
            "left_hand_pinch": tele_data.left_hand_pinch,
            "left_hand_pinchValue": tele_data.left_hand_pinchValue,
            "left_hand_squeeze": tele_data.left_hand_squeeze,
            "left_hand_squeezeValue": tele_data.left_hand_squeezeValue,
            "right_hand_pinch": tele_data.right_hand_pinch,
            "right_hand_pinchValue": tele_data.right_hand_pinchValue,
            "right_hand_squeeze": tele_data.right_hand_squeeze,
            "right_hand_squeezeValue": tele_data.right_hand_squeezeValue,
            "left_ctrl_trigger": tele_data.left_ctrl_trigger,
            "left_ctrl_triggerValue": tele_data.left_ctrl_triggerValue,
            "left_ctrl_squeeze": tele_data.left_ctrl_squeeze,
            "left_ctrl_squeezeValue": tele_data.left_ctrl_squeezeValue,
            "left_ctrl_aButton": tele_data.left_ctrl_aButton,
            "left_ctrl_bButton": tele_data.left_ctrl_bButton,
            "left_ctrl_thumbstick": tele_data.left_ctrl_thumbstick,
            "left_ctrl_thumbstickValue": tele_data.left_ctrl_thumbstickValue.tolist(),
            "right_ctrl_trigger": tele_data.right_ctrl_trigger,
            "right_ctrl_triggerValue": tele_data.right_ctrl_triggerValue,
            "right_ctrl_squeeze": tele_data.right_ctrl_squeeze,
            "right_ctrl_squeezeValue": tele_data.right_ctrl_squeezeValue,
            "right_ctrl_aButton": tele_data.right_ctrl_aButton,
            "right_ctrl_bButton": tele_data.right_ctrl_bButton,
            "right_ctrl_thumbstick": tele_data.right_ctrl_thumbstick,
            "right_ctrl_thumbstickValue": tele_data.right_ctrl_thumbstickValue.tolist(),
        }
    }


def controller_button_edge(current, previous):
    return bool(current) and not bool(previous)


def apply_controller_shortcuts(args, tele_data, prev_button_state):
    global START, RECORD_TOGGLE

    if args.input_mode != "controller" or tele_data is None:
        return

    current_button_state = {
        "start_button": bool(tele_data.left_ctrl_aButton),
        "record_button": bool(tele_data.right_ctrl_bButton),
    }

    if controller_button_edge(current_button_state["start_button"], prev_button_state["start_button"]):
        if not START:
            START = True
            logger_mp.info("[XR shortcut] Left X pressed, session started.")
        else:
            logger_mp.info("[XR shortcut] Left X pressed, session is already running.")

    if controller_button_edge(current_button_state["record_button"], prev_button_state["record_button"]):
        if not START:
            logger_mp.info("[XR shortcut] Right B pressed before session start, ignored.")
        elif not args.record:
            logger_mp.info("[XR shortcut] Right B pressed, but recording is disabled (--record not set).")
        else:
            RECORD_TOGGLE = True
            logger_mp.info("[XR shortcut] Right B pressed, toggling recording state.")

    prev_button_state.update(current_button_state)


def build_status_hud(args, tele_data, record_start_deadline=None, status_override=None, detail_override=None):
    hud = np.zeros((720, 1280, 3), dtype=np.uint8)
    hud[:] = (12, 18, 26)
    cv2.rectangle(hud, (0, 0), (1280, 124), (20, 30, 42), thickness=-1)
    cv2.rectangle(hud, (0, 610), (1280, 720), (14, 20, 28), thickness=-1)
    cv2.circle(hud, (1130, 102), 128, (22, 40, 54), thickness=-1)
    cv2.circle(hud, (120, 642), 96, (20, 32, 46), thickness=-1)
    cv2.rectangle(hud, (20, 20), (1260, 700), (42, 55, 72), 3)
    cv2.rectangle(hud, (40, 40), (1240, 680), (18, 25, 34), -1)

    if status_override is not None:
        status_text = status_override
        status_color = (80, 190, 255)
    elif record_start_deadline is not None and not RECORD_RUNNING:
        remain = max(0.0, record_start_deadline - time.time())
        status_text = f"RECORD IN {remain:0.1f}s"
        status_color = (80, 190, 255)
    elif RECORD_RUNNING:
        status_text = "RECORDING"
        status_color = (70, 90, 255)
    elif START:
        status_text = "TRACKING"
        status_color = (60, 205, 120)
    else:
        status_text = "READY TO START"
        status_color = (0, 200, 255)

    if detail_override is not None:
        detail_text = detail_override
    elif args.input_mode == "controller":
        if args.record:
            detail_text = "Left X: start session   Right B: record / save   Keyboard Q: exit"
        else:
            detail_text = "Left X: start session   Recording disabled   Keyboard Q: exit"
    else:
        if args.record:
            detail_text = "Keyboard R: start   Keyboard S: record / save   Keyboard Q: exit"
        else:
            detail_text = "Keyboard R: start   Recording disabled   Keyboard Q: exit"

    head_valid = bool(getattr(tele_data, "head_pose_valid", False))
    left_valid = bool(getattr(tele_data, "left_arm_pose_valid", False))
    right_valid = bool(getattr(tele_data, "right_arm_pose_valid", False))
    validity_text = (
        f"Tracking  Head:{'OK' if head_valid else '--'}   "
        f"Left:{'OK' if left_valid else '--'}   Right:{'OK' if right_valid else '--'}"
    )
    all_valid = head_valid and left_valid and right_valid
    validity_color = (110, 225, 130) if all_valid else (60, 180, 255)

    zero_hint = "Trigger is recorded as zero mark for later stabilization."
    if args.input_mode == "controller" and tele_data is not None:
        robot_world_pose = tele_data.left_robot_world_pose
        if robot_world_pose is not None:
            pose = robot_world_pose[:3, 3]
            pose_text = f"Left robot_world xyz  x:{pose[0]:+.3f}  y:{pose[1]:+.3f}  z:{pose[2]:+.3f}"
        else:
            pose_text = "Left robot_world xyz  unavailable"
        shortcut_text = "Left X = start session    Right B = start/save recording    Trigger = mark zero pose"
    else:
        pose_text = "XR HUD active"
        shortcut_text = "Keyboard R = start    Keyboard S = start/save recording    Keyboard Q = exit"

    draw_text_block(
        hud,
        ["XR TELEOP RECORDER"],
        origin=(68, 82),
        font_scale=1.10,
        color=(245, 250, 255),
        thickness=2,
        line_gap=8,
    )
    draw_text_block(
        hud,
        [
            "Collector / teleop status panel for headset-side operation.",
            "If you cannot see the main camera image, use this panel to know exactly which step the session is in.",
        ],
        origin=(68, 116),
        font_scale=0.50,
        color=(176, 198, 218),
        thickness=1,
        line_gap=8,
    )

    cv2.rectangle(hud, (60, 150), (1220, 246), (26, 58, 43), thickness=-1)
    cv2.rectangle(hud, (60, 150), (1220, 246), status_color, thickness=3)
    draw_text_block(
        hud,
        [status_text],
        origin=(84, 188),
        font_scale=1.05,
        color=(245, 250, 255),
        thickness=2,
        line_gap=6,
    )
    draw_text_block(
        hud,
        [detail_text],
        origin=(84, 224),
        font_scale=0.58,
        color=(225, 236, 245),
        thickness=1,
        line_gap=6,
    )

    chip_width = 262
    chip_height = 74
    chip_gap = 16
    chip_y = 274
    draw_hud_chip(hud, "session", "running" if START else "waiting", (60, chip_y), (chip_width, chip_height), (96, 176, 255), (26, 40, 52))
    if record_start_deadline is not None and not RECORD_RUNNING:
        record_value = f"countdown {max(0.0, record_start_deadline - time.time()):0.1f}s"
        record_accent = (255, 183, 84)
    else:
        record_value = "recording" if RECORD_RUNNING else "idle"
        record_accent = (70, 90, 255) if RECORD_RUNNING else (104, 154, 206)
    draw_hud_chip(hud, "record", record_value, (60 + chip_width + chip_gap, chip_y), (chip_width, chip_height), record_accent, (26, 40, 52))
    draw_hud_chip(hud, "tracking", "all ok" if all_valid else "partial", (60 + 2 * (chip_width + chip_gap), chip_y), (chip_width, chip_height), validity_color, (26, 40, 52))
    draw_hud_chip(hud, "input", args.input_mode, (60 + 3 * (chip_width + chip_gap), chip_y), (chip_width, chip_height), (255, 189, 89), (26, 40, 52))

    step_y = 374
    step_width = 278
    step_height = 116
    draw_step_card(hud, "1. Start session", "Press Left X in XR or keyboard R.", (60, step_y), (step_width, step_height), (96, 176, 255), not START)
    draw_step_card(hud, "2. Hold neutral pose", "Keep the controller still before recording begins.", (60 + step_width + 16, step_y), (step_width, step_height), (96, 222, 166), START and not RECORD_RUNNING and record_start_deadline is None)
    draw_step_card(hud, "3. Start recording", "Press Right B. There is a 3 second countdown.", (60 + 2 * (step_width + 16), step_y), (step_width, step_height), (255, 183, 84), record_start_deadline is not None and not RECORD_RUNNING)
    draw_step_card(hud, "4. Do action and save", "After recording starts, do the motion and press Right B again to save.", (60 + 3 * (step_width + 16), step_y), (step_width, step_height), (70, 90, 255), RECORD_RUNNING)

    cv2.rectangle(hud, (60, 520), (1220, 604), (18, 24, 31), thickness=-1)
    cv2.rectangle(hud, (60, 520), (1220, 604), (56, 72, 92), thickness=2)
    draw_text_block(
        hud,
        [validity_text, pose_text, shortcut_text, zero_hint],
        origin=(84, 552),
        font_scale=0.54,
        color=(163, 188, 210),
        thickness=1,
        line_gap=10,
    )
    return hud

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    parser.add_argument('--collector-only', action='store_true', help='XR data collection mode only: skip robot DDS, IK, and arm control')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")
    collector_only = args.collector_only
    arm_ctrl = None
    recorder = None
    bridge_writer = None
    sim_state_subscriber = None
    img_client = None
    tv_wrapper = None
    ipc_server = None
    listen_keyboard_thread = None
    record_start_deadline = None
    controller_button_state = {
        "start_button": False,
        "record_button": False,
    }

    try:
        # setup dds communication domains id
        if collector_only:
            logger_mp.info("Collector-only mode enabled: skip DDS initialization and robot control.")
            if args.motion:
                logger_mp.warning("--motion is ignored in collector-only mode.")
            if args.ee:
                logger_mp.warning("--ee is ignored in collector-only mode.")
            if args.sim:
                logger_mp.warning("--sim is ignored in collector-only mode. No simulator DDS is required.")
        else:
            if args.sim:
                ChannelFactoryInitialize(1, networkInterface=args.network_interface)
            else:
                ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            if sys.stdin.isatty():
                listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                          kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                          daemon=True)
                listen_keyboard_thread.start()
            else:
                logger_mp.warning("stdin is not a TTY, keyboard shortcuts are unavailable in this session.")

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        render_controller_panel = should_use_controller_record_panel(args, collector_only)
        xr_need_local_img = (not render_controller_panel) and not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])
        xr_binocular = camera_config['head_camera']['binocular']
        panel_height = 1040
        panel_width = 960
        xr_img_shape = (
            (panel_height, panel_width * 2 if xr_binocular else panel_width)
            if render_controller_panel
            else camera_config['head_camera']['image_shape']
        )

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=xr_binocular,
                                     img_shape=xr_img_shape,
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=True if render_controller_panel else camera_config['head_camera']['enable_zmq'],
                                     webrtc=False if render_controller_panel else camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     show_hud=not render_controller_panel,
                                     show_controller_models=False,
                                     hud_shape=(720, 1280),
                                     hud_height_m=0.62,
                                     hud_distance_m=0.95,
                                     hud_position=(0.0, 0.02, -0.92),
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if not collector_only:
            from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
            if args.motion:
                if args.input_mode == "controller":
                    loco_wrapper = LocoClientWrapper()
            else:
                motion_switcher = MotionSwitcher()
                status, result = motion_switcher.Enter_Debug_Mode()
                logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if not collector_only:
            from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
            if args.arm == "G1_29":
                arm_ik = G1_29_ArmIK()
                arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "G1_23":
                arm_ik = G1_23_ArmIK()
                arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "H1_2":
                arm_ik = H1_2_ArmIK()
                arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
            elif args.arm == "H1":
                arm_ik = H1_ArmIK()
                arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if not collector_only and args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif not collector_only and args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif not collector_only and args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif not collector_only and args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif not collector_only and args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim and not collector_only:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            try:
                recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                         task_goal = args.task_goal,
                                         task_desc = args.task_desc,
                                         task_steps = args.task_steps,
                                         frequency = args.frequency, 
                                         rerun_log = not args.headless)
            except Exception as exc:
                if args.headless:
                    raise
                logger_mp.warning(
                    f"EpisodeWriter failed with rerun enabled ({exc}). Falling back to headless recording."
                )
                recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                         task_goal = args.task_goal,
                                         task_desc = args.task_desc,
                                         task_steps = args.task_steps,
                                         frequency = args.frequency, 
                                         rerun_log = False)

        logger_mp.info("----------------------------------------------------------------")
        if collector_only:
            logger_mp.info("🟢  Press [r] to start XR data capture session.")
        else:
            logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.input_mode == "controller":
            logger_mp.info("🟢  XR controller shortcut: Left X starts the session.")
        if args.record:
            logger_mp.info(f"🟡  Press [s] to START recording after a {int(RECORD_START_DELAY_SEC)}s countdown, or SAVE recording when already recording.")
            if args.input_mode == "controller":
                logger_mp.info("🟡  XR controller shortcut: Right B starts the countdown or saves the current episode.")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            tele_data = tv_wrapper.get_tele_data()
            apply_controller_shortcuts(args, tele_data, controller_button_state)
            if render_controller_panel:
                tv_wrapper.render_to_xr(
                    make_xr_frame(
                        build_controller_record_panel(args, tele_data),
                        binocular=xr_binocular,
                    )
                )
            else:
                if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                    head_img = img_client.get_head_frame()
                    tv_wrapper.render_to_xr(head_img)
                tv_wrapper.render_hud_to_xr(build_status_hud(args, tele_data))
            time.sleep(0.033)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        if not collector_only and arm_ctrl is not None:
            arm_ctrl.speed_gradual_max()
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            head_img = None
            left_wrist_img = None
            right_wrist_img = None
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            apply_controller_shortcuts(args, tele_data, controller_button_state)

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if RECORD_RUNNING:
                    RECORD_RUNNING = False
                    if bridge_writer is not None:
                        bridge_writer.close()
                        bridge_writer = None
                    if render_controller_panel:
                        tv_wrapper.render_to_xr(
                            make_xr_frame(
                                build_controller_record_panel(
                                    args,
                                    tele_data,
                                    status_override="SAVING EPISODE",
                                    detail_override="Please keep the headset on. Saving current episode now.",
                                ),
                                binocular=xr_binocular,
                            )
                        )
                    else:
                        tv_wrapper.render_hud_to_xr(
                            build_status_hud(
                                args,
                                tele_data,
                                status_override="SAVING EPISODE",
                                detail_override="Please keep the headset on. Saving current episode now.",
                            )
                        )
                    recorder.save_episode()
                    if args.sim and not collector_only:
                        publish_reset_category(1, reset_pose_publisher)
                else:
                    if record_start_deadline is None:
                        record_start_deadline = time.time() + RECORD_START_DELAY_SEC
                        logger_mp.info(
                            f"⏳ Recording will start in {int(RECORD_START_DELAY_SEC)} seconds. "
                            "Please hold the neutral pose."
                        )
                    else:
                        record_start_deadline = None
                        logger_mp.info("🟤 Pending recording start canceled.")

            if args.record and not RECORD_RUNNING and record_start_deadline is not None:
                if time.time() >= record_start_deadline:
                    record_start_deadline = None
                    if recorder.create_episode():
                        if collector_only:
                            bridge_writer = XRBridgeWriter(recorder.episode_dir, args.input_mode)
                        RECORD_RUNNING = True
                        logger_mp.info("🔴 Recording started.")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")

            if collector_only:
                if args.record:
                    READY = recorder.is_ready()
                    if RECORD_RUNNING:
                        colors, depths = build_record_colors(camera_config, head_img, left_wrist_img, right_wrist_img)
                        recorder.add_item(colors=colors, depths=depths, states=build_collector_states(tele_data), actions={})
                        if bridge_writer is not None:
                            bridge_writer.add_frame(tele_data, int(start_time * 1000))
            else:
                if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_ctrl_triggerValue
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_ctrl_triggerValue
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = tele_data.left_hand_pinchValue
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = tele_data.right_hand_pinchValue
                else:
                    pass
                
                # high level control
                if args.input_mode == "controller" and args.motion:
                    # quit teleoperate
                    if tele_data.right_ctrl_aButton:
                        START = False
                        STOP = True
                    # command robot to enter damping mode. soft emergency stop function
                    if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                        loco_wrapper.Damp()
                    # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                    loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                      -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                      -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

                # get current robot state data.
                current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

                # record data
                if args.record:
                    READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                    # dex hand or gripper
                    if args.ee == "dex3" and args.input_mode == "hand":
                        with dual_hand_data_lock:
                            left_ee_state = dual_hand_state_array[:7]
                            right_ee_state = dual_hand_state_array[-7:]
                            left_hand_action = dual_hand_action_array[:7]
                            right_hand_action = dual_hand_action_array[-7:]
                            current_body_state = []
                            current_body_action = []
                    elif args.ee == "dex1" and args.input_mode == "hand":
                        with dual_gripper_data_lock:
                            left_ee_state = [dual_gripper_state_array[0]]
                            right_ee_state = [dual_gripper_state_array[1]]
                            left_hand_action = [dual_gripper_action_array[0]]
                            right_hand_action = [dual_gripper_action_array[1]]
                            current_body_state = []
                            current_body_action = []
                    elif args.ee == "dex1" and args.input_mode == "controller":
                        with dual_gripper_data_lock:
                            left_ee_state = [dual_gripper_state_array[0]]
                            right_ee_state = [dual_gripper_state_array[1]]
                            left_hand_action = [dual_gripper_action_array[0]]
                            right_hand_action = [dual_gripper_action_array[1]]
                            current_body_state = arm_ctrl.get_current_motor_q().tolist()
                            current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                                   -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                                   -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                    elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                        with dual_hand_data_lock:
                            left_ee_state = dual_hand_state_array[:6]
                            right_ee_state = dual_hand_state_array[-6:]
                            left_hand_action = dual_hand_action_array[:6]
                            right_hand_action = dual_hand_action_array[-6:]
                            current_body_state = []
                            current_body_action = []
                    else:
                        left_ee_state = []
                        right_ee_state = []
                        left_hand_action = []
                        right_hand_action = []
                        current_body_state = []
                        current_body_action = []

                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]
                    if RECORD_RUNNING:
                        colors, depths = build_record_colors(camera_config, head_img, left_wrist_img, right_wrist_img)
                        states = {
                            "left_arm": {                                                                    
                                "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                                "qvel":   [],                          
                                "torque": [],                        
                            }, 
                            "right_arm": {                                                                    
                                "qpos":   right_arm_state.tolist(),       
                                "qvel":   [],                          
                                "torque": [],                         
                            },                        
                            "left_ee": {                                                                    
                                "qpos":   left_ee_state,           
                                "qvel":   [],                           
                                "torque": [],                          
                            }, 
                            "right_ee": {                                                                    
                                "qpos":   right_ee_state,       
                                "qvel":   [],                           
                                "torque": [],  
                            }, 
                            "body": {
                                "qpos": current_body_state,
                            }, 
                        }
                        actions = {
                            "left_arm": {                                   
                                "qpos":   left_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],      
                            }, 
                            "right_arm": {                                   
                                "qpos":   right_arm_action.tolist(),       
                                "qvel":   [],       
                                "torque": [],       
                            },                         
                            "left_ee": {                                   
                                "qpos":   left_hand_action,       
                                "qvel":   [],       
                                "torque": [],       
                            }, 
                            "right_ee": {                                   
                                "qpos":   right_hand_action,       
                                "qvel":   [],       
                                "torque": [], 
                            }, 
                            "body": {
                                "qpos": current_body_action,
                            }, 
                        }
                        if args.sim:
                            sim_state = sim_state_subscriber.read_data()            
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                        else:
                            recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            if render_controller_panel:
                tv_wrapper.render_to_xr(
                    make_xr_frame(
                        build_controller_record_panel(args, tele_data, record_start_deadline=record_start_deadline),
                        binocular=xr_binocular,
                    )
                )
            else:
                tv_wrapper.render_hud_to_xr(build_status_hud(args, tele_data, record_start_deadline=record_start_deadline))

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                if ipc_server is not None:
                    ipc_server.stop()
            else:
                stop_listening()
                if listen_keyboard_thread is not None:
                    listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if bridge_writer is not None:
                bridge_writer.close()
        except Exception as e:
            logger_mp.error(f"Failed to close XR bridge writer: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim and sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record and recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
