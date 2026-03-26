import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .math_utils import apply_transform, clamp_array, homogeneous_rotation, transform_matrix, translation_matrix


@dataclass
class JointLimit:
    lower: float
    upper: float


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limit: Optional[JointLimit]


class UrdfModel:
    def __init__(self, joints: Sequence[JointSpec]):
        self.joints = list(joints)
        self.joints_by_name: Dict[str, JointSpec] = {joint.name: joint for joint in self.joints}
        self.child_to_joint: Dict[str, JointSpec] = {joint.child_link: joint for joint in self.joints}

    @classmethod
    def from_file(cls, path: Path) -> "UrdfModel":
        root = ET.parse(Path(path)).getroot()
        joints: List[JointSpec] = []

        for joint_node in root.findall("joint"):
            name = joint_node.attrib["name"]
            joint_type = joint_node.attrib.get("type", "fixed")
            parent_link = joint_node.find("parent").attrib["link"]
            child_link = joint_node.find("child").attrib["link"]

            origin_node = joint_node.find("origin")
            axis_node = joint_node.find("axis")
            limit_node = joint_node.find("limit")

            origin_xyz = np.fromstring(origin_node.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
            origin_rpy = np.fromstring(origin_node.attrib.get("rpy", "0 0 0"), sep=" ", dtype=float)
            axis = np.fromstring(axis_node.attrib.get("xyz", "0 0 1") if axis_node is not None else "0 0 1", sep=" ", dtype=float)

            limit = None
            if limit_node is not None and joint_type != "fixed":
                lower = float(limit_node.attrib.get("lower", "-3.141592653589793"))
                upper = float(limit_node.attrib.get("upper", "3.141592653589793"))
                limit = JointLimit(lower=lower, upper=upper)

            joints.append(
                JointSpec(
                    name=name,
                    joint_type=joint_type,
                    parent_link=parent_link,
                    child_link=child_link,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    axis=axis,
                    limit=limit,
                )
            )

        return cls(joints)

    def extract_chain(self, base_link: str, tip_link: str, expected_joint_names: Optional[Iterable[str]] = None) -> "UrdfChain":
        chain: List[JointSpec] = []
        current_link = tip_link

        while current_link != base_link:
            joint = self.child_to_joint.get(current_link)
            if joint is None:
                raise ValueError(f"no joint found while tracing from tip link {tip_link} to base link {base_link}")
            chain.append(joint)
            current_link = joint.parent_link

        chain.reverse()
        result = UrdfChain(base_link=base_link, tip_link=tip_link, joints=chain)

        if expected_joint_names is not None:
            expected = list(expected_joint_names)
            if result.active_joint_names != expected:
                raise ValueError(
                    f"URDF chain mismatch for {tip_link}: expected {expected}, got {result.active_joint_names}"
                )

        return result


@dataclass
class UrdfChain:
    base_link: str
    tip_link: str
    joints: Sequence[JointSpec]

    @property
    def active_joints(self) -> List[JointSpec]:
        return [joint for joint in self.joints if joint.joint_type != "fixed"]

    @property
    def active_joint_names(self) -> List[str]:
        return [joint.name for joint in self.active_joints]

    @property
    def joint_lower_limits(self) -> np.ndarray:
        values = []
        for joint in self.active_joints:
            if joint.limit is None:
                values.append(-3.141592653589793)
            else:
                values.append(joint.limit.lower)
        return np.asarray(values, dtype=float)

    @property
    def joint_upper_limits(self) -> np.ndarray:
        values = []
        for joint in self.active_joints:
            if joint.limit is None:
                values.append(3.141592653589793)
            else:
                values.append(joint.limit.upper)
        return np.asarray(values, dtype=float)

    def clamp_joints(self, joint_positions: np.ndarray) -> np.ndarray:
        return clamp_array(np.asarray(joint_positions, dtype=float), self.joint_lower_limits, self.joint_upper_limits)

    def forward_kinematics(self, joint_positions: Sequence[float], tool_offset_xyz: Optional[Iterable[float]] = None) -> np.ndarray:
        q = np.asarray(joint_positions, dtype=float)
        if q.shape != (len(self.active_joints),):
            raise ValueError(f"expected {len(self.active_joints)} joint values, got shape {q.shape}")

        transform = np.eye(4, dtype=float)
        active_index = 0

        for joint in self.joints:
            transform = transform @ transform_matrix(joint.origin_xyz, joint.origin_rpy)

            if joint.joint_type in ("revolute", "continuous"):
                transform = transform @ homogeneous_rotation(joint.axis, q[active_index])
                active_index += 1
            elif joint.joint_type == "prismatic":
                transform = transform @ translation_matrix(joint.axis * q[active_index])
                active_index += 1
            elif joint.joint_type == "fixed":
                continue
            else:
                raise ValueError(f"unsupported joint type: {joint.joint_type}")

        if tool_offset_xyz is not None:
            transform = transform @ translation_matrix(tool_offset_xyz)

        return transform

    def end_effector_position(self, joint_positions: Sequence[float], tool_offset_xyz: Optional[Iterable[float]] = None) -> np.ndarray:
        transform = self.forward_kinematics(joint_positions, tool_offset_xyz=tool_offset_xyz)
        return apply_transform(transform, (0.0, 0.0, 0.0))

    def numerical_position_jacobian(
        self,
        joint_positions: Sequence[float],
        tool_offset_xyz: Optional[Iterable[float]] = None,
        delta: float = 1.0e-4,
    ) -> np.ndarray:
        q = np.asarray(joint_positions, dtype=float)
        baseline = self.end_effector_position(q, tool_offset_xyz=tool_offset_xyz)
        jacobian = np.zeros((3, q.shape[0]), dtype=float)

        for index in range(q.shape[0]):
            q_shifted = q.copy()
            q_shifted[index] += delta
            shifted = self.end_effector_position(q_shifted, tool_offset_xyz=tool_offset_xyz)
            jacobian[:, index] = (shifted - baseline) / delta

        return jacobian
