"""Dataset-level configuration objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_DATA_DIRS = [
    "/home/num2/datasets/EXO/Phase1And2_Parsed/Parsed",
]
#/home/lenovo/code/CHT/datasets/EXO/Phase1And2_Parsed

DEFAULT_REAL_SENSOR_NAMES = [
    # "foot_imu_*_gyro_x",
    # "foot_imu_*_gyro_y",
    # "foot_imu_*_gyro_z",
    # "foot_imu_*_accel_x",
    # "foot_imu_*_accel_y",
    # "foot_imu_*_accel_z",
    "shank_imu_*_gyro_x",
    "shank_imu_*_gyro_y",
    "shank_imu_*_gyro_z",
    "shank_imu_*_accel_x",
    "shank_imu_*_accel_y",
    "shank_imu_*_accel_z",
    "thigh_imu_*_gyro_x",
    "thigh_imu_*_gyro_y",
    "thigh_imu_*_gyro_z",
    "thigh_imu_*_accel_x",
    "thigh_imu_*_accel_y",
    "thigh_imu_*_accel_z",
    # "insole_*_cop_x",
    # "insole_*_cop_z",
    # "insole_*_force_y",
    "hip_angle_*",
    "hip_angle_*_velocity_filt",
    "knee_angle_*",
    "knee_angle_*_velocity_filt",
]

DEFAULT_LABEL_NAMES = [
    "hip_flexion_*_moment",
    "knee_angle_*_moment",
]

DEFAULT_PARTICIPANT_MASSES = {
    "BT01": 80.59,
    "BT02": 72.24,
    "BT03": 95.29,
    "BT04": 98.23,
    "BT06": 79.33,
    "BT07": 64.49,
    "BT08": 69.13,
    "BT09": 82.31,
    "BT10": 93.45,
    "BT11": 50.39,
    "BT12": 78.15,
    "BT13": 89.85,
    "BT14": 67.30,
    "BT15": 58.40,
    "BT16": 64.33,
    "BT17": 60.03,
    "BT18": 67.96,
    "BT19": 69.95,
    "BT20": 55.44,
    "BT21": 58.85,
    "BT22": 76.79,
    "BT23": 67.23,
    "BT24": 77.79,
}


@dataclass
class DatasetConfig:
    """Container for dataset-related hyperparameters."""

    data_dirs: List[str] = field(default_factory=lambda: list(DEFAULT_DATA_DIRS))
    side: str = "r"
    real_sensor_names: List[str] = field(
        default_factory=lambda: list(DEFAULT_REAL_SENSOR_NAMES)
    )
    real_sensor_pick: List[str] = field(default_factory=list)
    label_names: List[str] = field(
        default_factory=lambda: list(DEFAULT_LABEL_NAMES)
    )
    participant_masses: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PARTICIPANT_MASSES)
    )
    action_patterns: Optional[List[str]] = None
    filter_nan_trials: bool = False

    model_path: str = "models/trained_tcn.tar"
    window_size: int = 280
    window_stride: int = 10

    def __post_init__(self):
        if self.side not in {"l", "r"}:
            raise ValueError("side 必须是 'l' 或 'r'")

    @property
    def input_names(self) -> List[str]:
        """DataManager 期望的真实传感器列表."""
        return self.real_sensor_pick or self.real_sensor_names
