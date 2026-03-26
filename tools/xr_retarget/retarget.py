from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .bridge import BridgeFrame, load_bridge_frames
from .csv_export import RetargetedFrame
from .math_utils import clamp_array, rpy_matrix
from .urdf_kinematics import UrdfChain, UrdfModel


@dataclass
class ArmConfig:
    name: str
    enabled: bool
    tip_link: str
    joints: List[str]
    home_joint_positions: np.ndarray
    tool_offset_xyz: np.ndarray
    task_space_scale_xyz: np.ndarray
    workspace_relative_min_xyz: Optional[np.ndarray]
    workspace_relative_max_xyz: Optional[np.ndarray]
    joint_blend_alpha: float
    joint_velocity_limit_rad_s: np.ndarray
    can_iface: str
    motor_ids: List[int]
    joint_signs: np.ndarray
    joint_offsets: np.ndarray

    @property
    def motor_order(self) -> List[Tuple[str, int]]:
        return [(self.can_iface, motor_id) for motor_id in self.motor_ids]


@dataclass
class CalibrationConfig:
    mode: str
    scale: float
    rotation_rpy_deg: np.ndarray
    use_auto_translation_from_first_frame: bool
    relative_input_zero_is_home: bool
    translation_xyz: np.ndarray
    default_frame_count: Optional[int]
    missing_arm_policy: str


@dataclass
class IKConfig:
    position_tolerance_m: float
    max_iterations: int
    damping: float
    max_step_rad: float
    smoothness_weight: float
    home_weight: float
    jacobian_delta: float
    multistart_seed_count: int
    random_seed: int
    coarse_search_samples: int
    coarse_search_top_k: int


@dataclass
class XRRetargetConfig:
    config_path: Path
    urdf_path: Path
    base_link: str
    arms: Dict[str, ArmConfig]
    calibration: CalibrationConfig
    ik: IKConfig
    csv_export: Dict[str, float]


def _resolve_path(config_path: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return candidate

    relative_to_config = (config_path.parent / candidate).resolve()
    if relative_to_config.exists():
        return relative_to_config

    relative_to_cwd = (Path.cwd() / candidate).resolve()
    return relative_to_cwd


def load_retarget_config(config_path: Path, urdf_override: Optional[str] = None) -> XRRetargetConfig:
    config_path = Path(config_path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    urdf_path = _resolve_path(config_path, urdf_override or payload["urdf_path"])
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF file not found: {urdf_path}. "
            "Please update config/xr_retarget.yaml or pass --urdf with the correct path."
        )
    robot_cfg = payload["robot"]
    arm_cfgs: Dict[str, ArmConfig] = {}

    for arm_name, arm_payload in robot_cfg["controlled_arms"].items():
        arm_cfgs[arm_name] = ArmConfig(
            name=arm_name,
            enabled=bool(arm_payload.get("enabled", True)),
            tip_link=str(arm_payload["tip_link"]),
            joints=list(arm_payload["joints"]),
            home_joint_positions=np.asarray(arm_payload["home_joint_positions"], dtype=float),
            tool_offset_xyz=np.asarray(arm_payload["tool_offset_xyz"], dtype=float),
            task_space_scale_xyz=np.asarray(arm_payload.get("task_space_scale_xyz", [1.0, 1.0, 1.0]), dtype=float),
            workspace_relative_min_xyz=(
                np.asarray(arm_payload["workspace_relative_min_xyz"], dtype=float)
                if arm_payload.get("workspace_relative_min_xyz") is not None
                else None
            ),
            workspace_relative_max_xyz=(
                np.asarray(arm_payload["workspace_relative_max_xyz"], dtype=float)
                if arm_payload.get("workspace_relative_max_xyz") is not None
                else None
            ),
            joint_blend_alpha=float(arm_payload.get("joint_blend_alpha", 1.0)),
            joint_velocity_limit_rad_s=np.asarray(
                arm_payload.get("joint_velocity_limit_rad_s", [999.0] * len(arm_payload["joints"])),
                dtype=float,
            ),
            can_iface=str(arm_payload["can_iface"]),
            motor_ids=[int(v) for v in arm_payload["motor_ids"]],
            joint_signs=np.asarray(arm_payload.get("joint_signs", [1.0] * len(arm_payload["joints"])), dtype=float),
            joint_offsets=np.asarray(arm_payload.get("joint_offsets", [0.0] * len(arm_payload["joints"])), dtype=float),
        )

    calibration_payload = payload["xr"]["calibration"]
    calibration_cfg = CalibrationConfig(
        mode=str(calibration_payload.get("mode", "home_average")),
        scale=float(calibration_payload.get("scale", 1.0)),
        rotation_rpy_deg=np.asarray(calibration_payload.get("rotation_rpy_deg", [0.0, 0.0, 0.0]), dtype=float),
        use_auto_translation_from_first_frame=bool(calibration_payload.get("use_auto_translation_from_first_frame", True)),
        relative_input_zero_is_home=bool(calibration_payload.get("relative_input_zero_is_home", False)),
        translation_xyz=np.asarray(calibration_payload.get("translation_xyz", [0.0, 0.0, 0.0]), dtype=float),
        default_frame_count=(
            int(calibration_payload["frame_count"])
            if calibration_payload.get("frame_count") is not None
            else None
        ),
        missing_arm_policy=str(payload["xr"].get("missing_arm_policy", "hold_last")),
    )

    ik_payload = payload["ik"]
    ik_cfg = IKConfig(
        position_tolerance_m=float(ik_payload["position_tolerance_m"]),
        max_iterations=int(ik_payload["max_iterations"]),
        damping=float(ik_payload["damping"]),
        max_step_rad=float(ik_payload["max_step_rad"]),
        smoothness_weight=float(ik_payload["smoothness_weight"]),
        home_weight=float(ik_payload["home_weight"]),
        jacobian_delta=float(ik_payload["jacobian_delta"]),
        multistart_seed_count=int(ik_payload.get("multistart_seed_count", 8)),
        random_seed=int(ik_payload.get("random_seed", 20260325)),
        coarse_search_samples=int(ik_payload.get("coarse_search_samples", 0)),
        coarse_search_top_k=int(ik_payload.get("coarse_search_top_k", 2)),
    )

    return XRRetargetConfig(
        config_path=config_path,
        urdf_path=urdf_path,
        base_link=str(robot_cfg["base_link"]),
        arms=arm_cfgs,
        calibration=calibration_cfg,
        ik=ik_cfg,
        csv_export=payload["csv_export"],
    )


@dataclass
class ArmState:
    prev_model_q: np.ndarray
    prev_output_q: np.ndarray
    chain: UrdfChain
    home_ee_position: np.ndarray
    prev_timestamp_ms: Optional[int] = None


@dataclass
class RetargetSummary:
    total_frames: int = 0
    left_failures: int = 0
    right_failures: int = 0
    left_clamped_frames: int = 0
    right_clamped_frames: int = 0
    used_auto_translation: bool = False
    frame_range: Optional[Tuple[int, int]] = None
    calibration_range: Optional[Tuple[int, int]] = None
    translation_xyz: Optional[np.ndarray] = None
    left_translation_xyz: Optional[np.ndarray] = None
    right_translation_xyz: Optional[np.ndarray] = None
    left_neutral_xr_position: Optional[np.ndarray] = None
    right_neutral_xr_position: Optional[np.ndarray] = None
    left_home_ee_position: Optional[np.ndarray] = None
    right_home_ee_position: Optional[np.ndarray] = None
    left_failure_samples: List[int] = field(default_factory=list)
    right_failure_samples: List[int] = field(default_factory=list)
    left_clamp_samples: List[int] = field(default_factory=list)
    right_clamp_samples: List[int] = field(default_factory=list)
    inactive_arms: List[str] = field(default_factory=list)


class XRRetargeter:
    def __init__(self, config: XRRetargetConfig):
        self.config = config
        self.model = UrdfModel.from_file(config.urdf_path)
        self.rotation = rpy_matrix(np.deg2rad(config.calibration.rotation_rpy_deg))
        self.translation = np.asarray(config.calibration.translation_xyz, dtype=float)
        self.summary = RetargetSummary()
        self.arm_translations: Dict[str, np.ndarray] = {}
        self.arm_rngs: Dict[str, np.random.Generator] = {}

        self.arm_states: Dict[str, ArmState] = {}
        for arm_index, (arm_name, arm_cfg) in enumerate(config.arms.items()):
            if not arm_cfg.enabled:
                continue
            chain = self.model.extract_chain(config.base_link, arm_cfg.tip_link, expected_joint_names=arm_cfg.joints)
            home_q = chain.clamp_joints(arm_cfg.home_joint_positions)
            home_ee = chain.end_effector_position(home_q, tool_offset_xyz=arm_cfg.tool_offset_xyz)
            self.arm_states[arm_name] = ArmState(
                prev_model_q=home_q.copy(),
                prev_output_q=arm_cfg.joint_signs * home_q + arm_cfg.joint_offsets,
                chain=chain,
                home_ee_position=home_ee,
            )
            if arm_name == "left":
                self.summary.left_home_ee_position = home_ee.copy()
            elif arm_name == "right":
                self.summary.right_home_ee_position = home_ee.copy()
            self.arm_translations[arm_name] = self.translation.copy()
            self.arm_rngs[arm_name] = np.random.default_rng(self.config.ik.random_seed + arm_index)

    def _frame_arm_position(self, frame: BridgeFrame, arm_name: str) -> Optional[np.ndarray]:
        if arm_name == "left":
            return frame.left_position
        if arm_name == "right":
            return frame.right_position
        raise KeyError(f"unknown arm {arm_name}")

    def calibrate(self, frames: Sequence[BridgeFrame], active_arms: Optional[Sequence[str]] = None) -> None:
        if not frames:
            raise ValueError("cannot calibrate with no frames")

        for arm_name in self.arm_translations:
            self.arm_translations[arm_name] = self.translation.copy()

        if self.config.calibration.relative_input_zero_is_home:
            selected_arms = set(active_arms or self.config.arms.keys())
            aligned_arms = []
            for arm_name, arm_cfg in self.config.arms.items():
                if not arm_cfg.enabled or arm_name not in selected_arms:
                    continue
                translation = self.arm_states[arm_name].home_ee_position.copy()
                self.arm_translations[arm_name] = translation
                aligned_arms.append(arm_name)
                if arm_name == "left":
                    self.summary.left_neutral_xr_position = np.zeros(3, dtype=float)
                    self.summary.left_translation_xyz = translation.copy()
                elif arm_name == "right":
                    self.summary.right_neutral_xr_position = np.zeros(3, dtype=float)
                    self.summary.right_translation_xyz = translation.copy()
            if aligned_arms:
                self.translation = np.mean(
                    np.asarray([self.arm_translations[arm_name] for arm_name in aligned_arms]),
                    axis=0,
                )
                self.summary.translation_xyz = self.translation.copy()
                return
            raise ValueError("no enabled arms available for relative-input zero-to-home calibration")

        if not self.config.calibration.use_auto_translation_from_first_frame:
            self.summary.translation_xyz = self.translation.copy()
            self.summary.left_translation_xyz = self.arm_translations.get("left")
            self.summary.right_translation_xyz = self.arm_translations.get("right")
            return

        selected_arms = set(active_arms or self.config.arms.keys())
        mode = self.config.calibration.mode

        if mode == "per_arm_average":
            calibrated_arms = []
            for arm_name, arm_cfg in self.config.arms.items():
                if not arm_cfg.enabled or arm_name not in selected_arms:
                    continue

                arm_positions = []
                for frame in frames:
                    xr_pos = self._frame_arm_position(frame, arm_name)
                    if xr_pos is not None:
                        arm_positions.append(xr_pos)

                if not arm_positions:
                    continue

                xr_mean_raw = np.mean(np.asarray(arm_positions), axis=0)
                xr_mean = self.rotation @ (xr_mean_raw * self.config.calibration.scale)
                translation = self.arm_states[arm_name].home_ee_position - xr_mean
                self.arm_translations[arm_name] = translation
                calibrated_arms.append(arm_name)

                if arm_name == "left":
                    self.summary.left_neutral_xr_position = xr_mean_raw.copy()
                    self.summary.left_translation_xyz = translation.copy()
                elif arm_name == "right":
                    self.summary.right_neutral_xr_position = xr_mean_raw.copy()
                    self.summary.right_translation_xyz = translation.copy()

            if calibrated_arms:
                self.translation = np.mean(
                    np.asarray([self.arm_translations[arm_name] for arm_name in calibrated_arms]),
                    axis=0,
                )
                self.summary.used_auto_translation = True
                self.summary.translation_xyz = self.translation.copy()
                return

            raise ValueError("no valid XR wrist positions found in frames for per-arm calibration")

        xr_points = []
        robot_points = []

        for arm_name, arm_cfg in self.config.arms.items():
            if not arm_cfg.enabled or arm_name not in selected_arms:
                continue

            arm_positions = []
            for frame in frames:
                xr_pos = self._frame_arm_position(frame, arm_name)
                if xr_pos is not None:
                    arm_positions.append(xr_pos)

            if not arm_positions:
                continue

            xr_mean_raw = np.mean(np.asarray(arm_positions), axis=0)
            xr_mean = self.rotation @ (xr_mean_raw * self.config.calibration.scale)
            xr_points.append(xr_mean)
            robot_points.append(self.arm_states[arm_name].home_ee_position)

            if arm_name == "left":
                self.summary.left_neutral_xr_position = xr_mean_raw.copy()
            elif arm_name == "right":
                self.summary.right_neutral_xr_position = xr_mean_raw.copy()

        if xr_points:
            xr_mean = np.mean(np.asarray(xr_points), axis=0)
            robot_mean = np.mean(np.asarray(robot_points), axis=0)
            self.translation = robot_mean - xr_mean
            for arm_name in self.arm_translations:
                self.arm_translations[arm_name] = self.translation.copy()
            self.summary.used_auto_translation = True
            self.summary.translation_xyz = self.translation.copy()
            self.summary.left_translation_xyz = self.arm_translations.get("left")
            self.summary.right_translation_xyz = self.arm_translations.get("right")
            return

        raise ValueError("no valid XR wrist positions found in frames for calibration")

    def _transform_xr_position(self, arm_name: str, position: np.ndarray) -> np.ndarray:
        arm_cfg = self.config.arms[arm_name]
        translation = self.arm_translations.get(arm_name, self.translation)
        scaled_position = position * self.config.calibration.scale * arm_cfg.task_space_scale_xyz
        return translation + self.rotation @ scaled_position

    def _clamp_target_position(self, arm_name: str, target_position: np.ndarray) -> Tuple[np.ndarray, bool]:
        arm_cfg = self.config.arms[arm_name]
        state = self.arm_states[arm_name]

        if arm_cfg.workspace_relative_min_xyz is None or arm_cfg.workspace_relative_max_xyz is None:
            return target_position, False

        relative_target = target_position - state.home_ee_position
        clamped_relative = np.clip(relative_target, arm_cfg.workspace_relative_min_xyz, arm_cfg.workspace_relative_max_xyz)
        was_clamped = not np.allclose(relative_target, clamped_relative, atol=1.0e-9)
        return state.home_ee_position + clamped_relative, was_clamped

    def _apply_missing_policy(self, arm_name: str) -> Tuple[np.ndarray, np.ndarray]:
        arm_cfg = self.config.arms[arm_name]
        state = self.arm_states[arm_name]
        if self.config.calibration.missing_arm_policy == "home":
            home_q = state.chain.clamp_joints(arm_cfg.home_joint_positions)
            output_q = arm_cfg.joint_signs * home_q + arm_cfg.joint_offsets
            state.prev_model_q = home_q
            state.prev_output_q = output_q
            return home_q, output_q
        return state.prev_model_q.copy(), state.prev_output_q.copy()

    def _apply_home_pose(self, arm_name: str) -> Tuple[np.ndarray, np.ndarray]:
        arm_cfg = self.config.arms[arm_name]
        state = self.arm_states[arm_name]
        home_q = state.chain.clamp_joints(arm_cfg.home_joint_positions)
        output_q = arm_cfg.joint_signs * home_q + arm_cfg.joint_offsets
        state.prev_model_q = home_q
        state.prev_output_q = output_q
        return home_q, output_q

    def _smooth_joint_output(self, arm_name: str, solved_q: np.ndarray, timestamp_ms: int) -> np.ndarray:
        arm_cfg = self.config.arms[arm_name]
        state = self.arm_states[arm_name]
        chain = state.chain

        prev_q = state.prev_model_q.copy()
        filtered_q = solved_q.copy()

        alpha = float(arm_cfg.joint_blend_alpha)
        if 0.0 < alpha < 1.0:
            filtered_q = prev_q + alpha * (filtered_q - prev_q)

        if state.prev_timestamp_ms is not None:
            dt_sec = max(1.0e-3, (timestamp_ms - state.prev_timestamp_ms) / 1000.0)
            max_delta = np.asarray(arm_cfg.joint_velocity_limit_rad_s, dtype=float) * dt_sec
            filtered_q = prev_q + np.clip(filtered_q - prev_q, -max_delta, max_delta)

        return chain.clamp_joints(filtered_q)

    def _solve_position_ik_once(self, arm_name: str, target_position: np.ndarray, seed_q: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        arm_cfg = self.config.arms[arm_name]
        state = self.arm_states[arm_name]
        chain = state.chain
        q = chain.clamp_joints(seed_q)
        q_home = chain.clamp_joints(arm_cfg.home_joint_positions)
        lower = chain.joint_lower_limits
        upper = chain.joint_upper_limits

        converged = False
        for _ in range(self.config.ik.max_iterations):
            current = chain.end_effector_position(q, tool_offset_xyz=arm_cfg.tool_offset_xyz)
            error = target_position - current
            if float(np.linalg.norm(error)) <= self.config.ik.position_tolerance_m:
                converged = True
                break

            jacobian = chain.numerical_position_jacobian(
                q,
                tool_offset_xyz=arm_cfg.tool_offset_xyz,
                delta=self.config.ik.jacobian_delta,
            )
            regularizer = (
                self.config.ik.smoothness_weight * (state.prev_model_q - q)
                + self.config.ik.home_weight * (q_home - q)
            )
            identity = np.eye(q.shape[0], dtype=float)
            lhs = jacobian.T @ jacobian + (
                self.config.ik.damping ** 2
                + self.config.ik.smoothness_weight
                + self.config.ik.home_weight
            ) * identity
            rhs = jacobian.T @ error + regularizer

            try:
                delta_q = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                delta_q = np.linalg.pinv(lhs) @ rhs

            delta_q = np.clip(delta_q, -self.config.ik.max_step_rad, self.config.ik.max_step_rad)
            q = clamp_array(q + delta_q, lower, upper)

        final_error = np.linalg.norm(
            target_position - chain.end_effector_position(q, tool_offset_xyz=arm_cfg.tool_offset_xyz)
        )
        if final_error <= self.config.ik.position_tolerance_m:
            converged = True

        return q, converged, float(final_error)

    def _ik_seed_candidates(self, arm_name: str) -> List[np.ndarray]:
        arm_cfg = self.config.arms[arm_name]
        state = self.arm_states[arm_name]
        chain = state.chain
        q_home = chain.clamp_joints(arm_cfg.home_joint_positions)
        lower = chain.joint_lower_limits
        upper = chain.joint_upper_limits

        unique: List[np.ndarray] = []
        for candidate in (
            state.prev_model_q.copy(),
            q_home.copy(),
            chain.clamp_joints(0.5 * (state.prev_model_q + q_home)),
        ):
            if not any(np.allclose(candidate, existing, atol=1.0e-6) for existing in unique):
                unique.append(candidate)

        rng = self.arm_rngs[arm_name]
        while len(unique) < self.config.ik.multistart_seed_count:
            candidate = rng.uniform(lower, upper)
            if not any(np.allclose(candidate, existing, atol=1.0e-6) for existing in unique):
                unique.append(candidate)
        return unique

    def _solve_position_ik(self, arm_name: str, target_position: np.ndarray) -> Tuple[np.ndarray, bool]:
        best_q = self.arm_states[arm_name].prev_model_q.copy()
        best_error = float("inf")
        converged = False

        for seed_q in self._ik_seed_candidates(arm_name):
            solved_q, solved, final_error = self._solve_position_ik_once(arm_name, target_position, seed_q)
            if final_error < best_error:
                best_error = final_error
                best_q = solved_q
            if solved:
                converged = True
                best_q = solved_q
                break

        if not converged and self.config.ik.coarse_search_samples > 0:
            arm_cfg = self.config.arms[arm_name]
            state = self.arm_states[arm_name]
            chain = state.chain
            lower = chain.joint_lower_limits
            upper = chain.joint_upper_limits
            rng = self.arm_rngs[arm_name]

            coarse_candidates: List[Tuple[float, np.ndarray]] = []
            for _ in range(self.config.ik.coarse_search_samples):
                q = rng.uniform(lower, upper)
                current = chain.end_effector_position(q, tool_offset_xyz=arm_cfg.tool_offset_xyz)
                error = float(np.linalg.norm(target_position - current))
                coarse_candidates.append((error, q))

            coarse_candidates.sort(key=lambda item: item[0])
            for _, seed_q in coarse_candidates[: self.config.ik.coarse_search_top_k]:
                solved_q, solved, final_error = self._solve_position_ik_once(arm_name, target_position, seed_q)
                if final_error < best_error:
                    best_error = final_error
                    best_q = solved_q
                if solved:
                    converged = True
                    best_q = solved_q
                    break

        return best_q, converged

    def _retarget_arm(
        self,
        arm_name: str,
        xr_position: Optional[np.ndarray],
        source_frame_index: int,
        timestamp_ms: int,
    ) -> np.ndarray:
        arm_cfg = self.config.arms[arm_name]
        state = self.arm_states[arm_name]

        if xr_position is None:
            _, output_q = self._apply_missing_policy(arm_name)
            return output_q.copy()

        target_position = self._transform_xr_position(arm_name, xr_position)
        target_position, was_clamped = self._clamp_target_position(arm_name, target_position)
        if was_clamped:
            if arm_name == "left":
                self.summary.left_clamped_frames += 1
                if len(self.summary.left_clamp_samples) < 10:
                    self.summary.left_clamp_samples.append(source_frame_index)
            else:
                self.summary.right_clamped_frames += 1
                if len(self.summary.right_clamp_samples) < 10:
                    self.summary.right_clamp_samples.append(source_frame_index)
        solved_q, converged = self._solve_position_ik(arm_name, target_position)
        if not converged:
            if arm_name == "left":
                self.summary.left_failures += 1
                if len(self.summary.left_failure_samples) < 10:
                    self.summary.left_failure_samples.append(source_frame_index)
            else:
                self.summary.right_failures += 1
                if len(self.summary.right_failure_samples) < 10:
                    self.summary.right_failure_samples.append(source_frame_index)
            solved_q = state.prev_model_q.copy()

        solved_q = self._smooth_joint_output(arm_name, solved_q, timestamp_ms)

        output_q = arm_cfg.joint_signs * solved_q + arm_cfg.joint_offsets
        state.prev_model_q = solved_q
        state.prev_output_q = output_q
        state.prev_timestamp_ms = timestamp_ms
        return output_q

    def retarget_frames(
        self,
        frames: Sequence[BridgeFrame],
        arms_mode: str = "both",
        calibration_frames: Optional[Sequence[BridgeFrame]] = None,
        source_frame_indices: Optional[Sequence[int]] = None,
        calibration_frame_indices: Optional[Sequence[int]] = None,
    ) -> List[RetargetedFrame]:
        self.summary.total_frames = len(frames)
        self.summary.left_failures = 0
        self.summary.right_failures = 0
        self.summary.left_clamped_frames = 0
        self.summary.right_clamped_frames = 0
        self.summary.used_auto_translation = False
        self.summary.left_failure_samples.clear()
        self.summary.right_failure_samples.clear()
        self.summary.left_clamp_samples.clear()
        self.summary.right_clamp_samples.clear()
        self.summary.inactive_arms = []
        self.summary.frame_range = None
        self.summary.calibration_range = None
        self.summary.translation_xyz = self.translation.copy()
        self.summary.left_translation_xyz = self.arm_translations.get("left")
        self.summary.right_translation_xyz = self.arm_translations.get("right")
        self.summary.left_neutral_xr_position = None
        self.summary.right_neutral_xr_position = None
        for state in self.arm_states.values():
            state.prev_timestamp_ms = None

        frame_indices = list(source_frame_indices) if source_frame_indices is not None else list(range(len(frames)))
        calibration_input = calibration_frames if calibration_frames is not None else frames
        calibration_indices = (
            list(calibration_frame_indices)
            if calibration_frame_indices is not None
            else frame_indices[: len(calibration_input)]
        )

        active_arms = {"left", "right"} if arms_mode == "both" else {arms_mode}
        self.summary.inactive_arms = sorted(
            [
                arm_name
                for arm_name, arm_cfg in self.config.arms.items()
                if arm_cfg.enabled and arm_name not in active_arms
            ]
        )
        if frame_indices:
            self.summary.frame_range = (frame_indices[0], frame_indices[-1])
        if calibration_indices:
            self.summary.calibration_range = (calibration_indices[0], calibration_indices[-1])

        self.calibrate(calibration_input, active_arms=sorted(active_arms))

        output_frames: List[RetargetedFrame] = []

        for local_index, frame in enumerate(frames):
            source_index = frame_indices[local_index]
            positions: Dict[Tuple[str, int], float] = {}
            for arm_name, arm_cfg in self.config.arms.items():
                if not arm_cfg.enabled:
                    continue
                if arm_name not in active_arms:
                    _, output_q = self._apply_home_pose(arm_name)
                else:
                    xr_pos = self._frame_arm_position(frame, arm_name)
                    output_q = self._retarget_arm(arm_name, xr_pos, source_index, frame.timestamp_ms)

                for index, motor_key in enumerate(arm_cfg.motor_order):
                    positions[motor_key] = float(output_q[index])

            output_frames.append(RetargetedFrame(timestamp_ms=frame.timestamp_ms, positions=positions))

        return output_frames

    @property
    def motor_order(self) -> List[Tuple[str, int]]:
        order: List[Tuple[str, int]] = []
        for arm_name in ("left", "right"):
            arm_cfg = self.config.arms.get(arm_name)
            if arm_cfg and arm_cfg.enabled:
                order.extend(arm_cfg.motor_order)
        return order


def load_frames_from_jsonl(path: Path) -> List[BridgeFrame]:
    return load_bridge_frames(path)
