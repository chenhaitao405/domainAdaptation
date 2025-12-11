"""快速检查 DataManager 是否能正确读取真实与模拟传感器数据。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain_adaptation.config import DatasetConfig
from domain_adaptation.data import DataManager


def _locate_sim_file(data_dirs: List[str], trial_name: str) -> str:
    """在给定 trial 目录下寻找 *_imu_sim.csv 文件。"""
    relative_path = trial_name.replace("\\", "/")
    for data_dir in data_dirs:
        candidate_dir = os.path.join(data_dir, relative_path)
        if not os.path.isdir(candidate_dir):
            continue
        for file_name in os.listdir(candidate_dir):
            if file_name.endswith("_imu_sim.csv"):
                return os.path.join(candidate_dir, file_name)
    raise FileNotFoundError(f"未在 {trial_name} 对应目录下找到 *_imu_sim.csv")


def run_smoke_test():
    """Load both real + simulated sensors to validate dataloader + schema."""
    config = DatasetConfig(
        data_dirs=["/home/num2/datasets/EXO/Phase1And2_Parsed/Parsed"],
        side="l",
        action_patterns=[r"normal_walk_1_1-2_on"],
    )

    dataset = DataManager.load_datasets(config, device=torch.device("cpu"))
    assert len(dataset) > 0, "数据集为空，请检查 data_dirs"

    inputs, labels, seq_lengths, trial_names = dataset[0]
    expected_channels = len(config.input_names)
    assert inputs.shape[1] == expected_channels, (
        f"真实传感器通道数不匹配: got {inputs.shape[1]}, expected {expected_channels}"
    )
    assert labels.shape[1] == len(config.label_names)

    trial_name = trial_names[0]
    sim_file = _locate_sim_file(config.data_dirs, trial_name)
    df_sim = pd.read_csv(sim_file)
    sim_columns = config.get_sim_sensor_columns(config.side)
    missing_columns = [col for col in sim_columns if col not in df_sim.columns]
    assert not missing_columns, (
        "模拟传感器字段缺失: " + ", ".join(missing_columns)
    )

    print("Smoke test passed!")
    print(f"  trials inspected: {trial_names}")
    print(f"  input tensor shape : {inputs.shape}")
    print(f"  label tensor shape : {labels.shape}")
    print(f"  seq lengths        : {seq_lengths}")
    print(f"  sim csv file       : {sim_file}")
    print(f"  sim columns check  : {len(sim_columns)} columns OK")


if __name__ == "__main__":
    run_smoke_test()
