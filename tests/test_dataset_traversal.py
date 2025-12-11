"""验证真实/模拟数据集遍历 DEFAULT_DATA_DIRS 是否成功。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain_adaptation.config import DatasetConfig
from domain_adaptation.data import DataManager


def _summarize(concat_dataset, config, label: str) -> None:
    sides = config.side if isinstance(config.side, list) else [config.side]
    idx = 0
    for data_dir in config.data_dirs:
        for side in sides:
            sub_dataset = concat_dataset.datasets[idx]
            assert len(sub_dataset) > 0, (
                f"{label}: 数据目录 {data_dir} (side={side}) 为空"
            )
            idx += 1
    print(f"[{label}] total chunks: {idx}, total trials: {len(concat_dataset)}")


def main():
    config = DatasetConfig()  # 使用 DEFAULT_DATA_DIRS
    device = torch.device("cpu")

    real_dataset = DataManager.load_datasets(config, device)
    sim_dataset = DataManager.load_sim_datasets(config, device)

    _summarize(real_dataset, config, label="real")
    _summarize(sim_dataset, config, label="sim")

    print("All dataset directories loaded successfully.")


if __name__ == "__main__":
    main()
