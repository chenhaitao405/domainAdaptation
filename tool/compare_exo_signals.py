#!/usr/bin/env python3
"""Compare real EXO signals against OpenSim simulated signals.

The script loads both CSV files, keeps a configurable time window, and plots
selected columns to highlight how the real sensors differ from the simulated
ones (e.g., motor joint states vs. IMU readings).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

REAL_DEFAULT = Path(
    "/home/num2/datasets/EXO/Phase3_Parsed/Parsed/BT01/"
    "ball_toss_1_1_center_on/BT01_ball_toss_1_1_center_on_exo.csv"
)
SIM_DEFAULT = Path(
    "/home/num2/datasets/EXO/Phase3_Parsed/Parsed/BT01/"
    "ball_toss_1_1_center_on/BT01_ball_toss_1_1_center_on_exo_sim.csv"
)

HIP_COLUMNS = [
    "hip_angle_l",
    "hip_angle_l_velocity",
    "hip_angle_l_velocity_filt",
]

THIGH_IMU_COLUMNS = [
    "thigh_imu_l_accel_x",
    "thigh_imu_l_accel_y",
    "thigh_imu_l_accel_z",
    "thigh_imu_l_gyro_x",
    "thigh_imu_l_gyro_y",
    "thigh_imu_l_gyro_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取真实与模拟EXO数据，截取部分样本，并绘制两者在部分关节/IMU"
            "信号上的差异。"
        )
    )
    parser.add_argument(
        "--real",
        type=Path,
        default=REAL_DEFAULT,
        help="Path to the real-device CSV file.",
    )
    parser.add_argument(
        "--sim",
        type=Path,
        default=SIM_DEFAULT,
        help="Path to the OpenSim simulated CSV file.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Keep samples with time >= this value (seconds).",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Keep samples with time <= this value (seconds).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=800,
        help="Limit the plot to the first N aligned samples after filtering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "If provided, save the hip/IMU comparison plots into this directory"
            " instead of opening an interactive window."
        ),
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Plot every Nth sample (default: 1, i.e. no decimation).",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {dataset_name}: {missing}")


def subset_by_time(
    df: pd.DataFrame, start_time: float | None, end_time: float | None
) -> pd.DataFrame:
    subset = df
    if start_time is not None:
        subset = subset[subset["time"] >= start_time]
    if end_time is not None:
        subset = subset[subset["time"] <= end_time]
    return subset


def align_on_time(
    real_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    unique_columns = list(dict.fromkeys(columns))

    ensure_columns(real_df, unique_columns, "real dataset")
    ensure_columns(sim_df, unique_columns, "sim dataset")

    keep_cols = ["time"] + unique_columns
    merged = pd.merge(
        real_df[keep_cols],
        sim_df[keep_cols],
        on="time",
        suffixes=("_real", "_sim"),
        how="inner",
    ).sort_values("time")
    return merged


def plot_group(
    merged: pd.DataFrame,
    columns: Sequence[str],
    title: str,
    output_path: Path | None,
) -> None:
    n_rows = len(columns)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(12, max(3, 2.8 * n_rows)),
        sharex=True,
        squeeze=False,
    )

    time_axis = merged["time"]
    for idx, col in enumerate(columns):
        ax_left = axes[idx, 0]
        ax_right = axes[idx, 1]

        ax_left.plot(time_axis, merged[f"{col}_real"], label="Real", linewidth=1.2)
        ax_left.plot(time_axis, merged[f"{col}_sim"], label="Sim", linewidth=1.2)
        ax_left.set_ylabel(col)
        if idx == 0:
            ax_left.legend(loc="upper right")
        ax_left.grid(True, alpha=0.3)

        diff = merged[f"{col}_real"] - merged[f"{col}_sim"]
        ax_right.plot(time_axis, diff, color="tab:red", linewidth=1)
        ax_right.axhline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax_right.set_ylabel(f"Δ {col}")
        ax_right.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")
        plt.close(fig)


def main() -> None:
    args = parse_args()

    real_df = pd.read_csv(args.real)
    sim_df = pd.read_csv(args.sim)

    if "time" not in real_df.columns or "time" not in sim_df.columns:
        raise ValueError("Both CSV files must contain a 'time' column.")

    real_df = subset_by_time(real_df, args.start_time, args.end_time)
    sim_df = subset_by_time(sim_df, args.start_time, args.end_time)

    merged = align_on_time(real_df, sim_df, HIP_COLUMNS + THIGH_IMU_COLUMNS)

    if args.max_samples is not None and args.max_samples > 0:
        merged = merged.iloc[: args.max_samples]

    if args.downsample > 1:
        merged = merged.iloc[:: args.downsample, :]

    if merged.empty:
        raise RuntimeError("No overlapping samples were found after filtering.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        hip_output = args.output_dir / "hip_comparison.png"
        imu_output = args.output_dir / "thigh_imu_comparison.png"
    else:
        hip_output = imu_output = None

    plot_group(merged, HIP_COLUMNS, "Hip Joint (motor) comparison", hip_output)
    plot_group(merged, THIGH_IMU_COLUMNS, "Thigh IMU comparison", imu_output)

    if not args.output_dir:
        plt.show()


if __name__ == "__main__":
    main()
