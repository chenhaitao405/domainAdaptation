"""轻量化通道方差排查：直接读取CSV指定列，输出统计和异常."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.config import DatasetConfig
from domain_adaptation.data.sensor_dataset import SensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV 通道方差 / 异常值排查")
    parser.add_argument("--channel", type=str, required=True, help="通道名，可含*占位符（如 thigh_imu_*_gyro_y）")
    parser.add_argument("--data-dirs", nargs="+", default=DatasetConfig().data_dirs, help="数据根目录列表")
    parser.add_argument("--side", type=str, default="r", choices=["l", "r"], help="侧别")
    parser.add_argument("--sim", action="store_true", help="是否分析 *_exo_sim.csv（默认真实域 *_exo.csv）")
    parser.add_argument("--threshold", type=float, default=1000.0, help="异常值阈值 |x|>threshold")
    parser.add_argument("--top-k", type=int, default=10, help="输出方差/异常的前K条记录")
    return parser.parse_args()


def list_trials(data_dirs: List[str]) -> List[Tuple[str, str]]:
    """返回 (data_dir, trial_rel_path) 列表."""
    trials: List[Tuple[str, str]] = []
    for root in data_dirs:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Data directory not found: {root}")
        participants = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
        for participant in participants:
            participant_dir = os.path.join(root, participant)
            for trial in os.listdir(participant_dir):
                trial_rel = os.path.join(participant, trial)
                trials.append((root, trial_rel))
    return trials


def find_input_file(trial_dir: str, suffix: str) -> str | None:
    for fname in os.listdir(trial_dir):
        lower = fname.lower()
        if not lower.endswith(suffix):
            continue
        if suffix == "_exo.csv" and lower.endswith("power_exo.csv"):
            continue
        return os.path.join(trial_dir, fname)
    return None


def collect_channel_stats(
    trials: List[Tuple[str, str]],
    channel_name: str,
    suffix: str,
    threshold: float,
) -> Tuple[float, float, float, float, float, List[Tuple[str, float]], List[Tuple[str, int, float]]]:
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0
    global_min = float("inf")
    global_max = float("-inf")
    per_trial_var: List[Tuple[str, float]] = []
    anomalies: List[Tuple[str, int, float]] = []

    for root, trial in tqdm(trials, desc="Scanning trials"):
        trial_dir = os.path.join(root, trial)
        data_file = find_input_file(trial_dir, suffix)
        if data_file is None:
            continue
        try:
            series = pd.read_csv(data_file, usecols=[channel_name])[channel_name].to_numpy()
        except ValueError:
            continue
        mask = np.isfinite(series)
        values = series[mask]
        if values.size == 0:
            continue

        per_trial_var.append((f"{root}/{trial}", float(values.var())))
        total_sum += float(values.sum())
        total_sumsq += float((values ** 2).sum())
        total_count += int(values.size)
        trial_min = float(values.min())
        trial_max = float(values.max())
        global_min = min(global_min, trial_min)
        global_max = max(global_max, trial_max)

        abs_values = np.abs(values)
        if threshold is not None and abs_values.max() > threshold:
            idx = int(abs_values.argmax())
            original_idx = int(np.nonzero(mask)[0][idx])
            anomalies.append((f"{root}/{trial}", original_idx, float(values[idx])))

    if total_count == 0:
        raise RuntimeError("指定通道没有有效数据。")
    mean = total_sum / total_count
    var = max(total_sumsq / total_count - mean ** 2, 0.0)
    std = var ** 0.5
    return mean, std, var, global_min, global_max, per_trial_var, anomalies, total_count


def main() -> None:
    args = parse_args()
    dataset_cfg = DatasetConfig(data_dirs=args.data_dirs, side=args.side)
    resolved_channel = args.channel.replace("*", args.side)

    trial_list = list_trials(args.data_dirs)
    suffix = "_exo_sim.csv" if args.sim else "_exo.csv"

    mean, std, var, vmin, vmax, per_trial_var, anomalies, total_count = collect_channel_stats(
        trial_list, resolved_channel, suffix, args.threshold
    )

    print(f"Channel: {resolved_channel} ({'sim' if args.sim else 'real'})")
    print(f"Total samples: {total_count}")
    print(f"Global mean: {mean:.6f}")
    print(f"Global std: {std:.6f}")
    print(f"Global var: {var:.6f}")
    print(f"Min value: {vmin:.6f}")
    print(f"Max value: {vmax:.6f}\n")

    per_trial_var.sort(key=lambda x: x[1], reverse=True)
    print("Top trials by variance:")
    for trial, value in per_trial_var[: args.top_k]:
        print(f"  {trial}: var={value:.6f}")

    if anomalies:
        print(f"\nAnomalies | abs(value) > {args.threshold}:")
        anomalies.sort(key=lambda x: abs(x[2]), reverse=True)
        for trial, index, value in anomalies[: args.top_k]:
            print(f"  {trial} @ {index}: value={value:.6f}")
    else:
        print(f"\nNo anomalies exceeding threshold {args.threshold}.")


if __name__ == "__main__":
    main()
