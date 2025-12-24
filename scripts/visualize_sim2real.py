"""Qt交互窗口：选择通道对比真实 vs sim2real 序列，支持横轴拖动与纵轴自适应."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.config import DatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sim->Real 序列可视化")
    parser.add_argument("--trial", type=str, required=True, help="目标trial（例如 BT01/ball_toss_1_1_center_on）")
    parser.add_argument("--data-dirs", nargs="+", default=DatasetConfig().data_dirs, help="数据根目录列表")
    parser.add_argument("--side", type=str, default="r", choices=["l", "r"], help="选择的侧别")
    parser.add_argument("--window", type=int, default=1000, help="初始可视窗口长度")
    return parser.parse_args()


def _resolve_trial_dir(data_dirs: List[str], trial_name: str) -> str:
    for root in data_dirs:
        path = os.path.join(root, trial_name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Trial '{trial_name}' not found in provided data directories.")


def _find_file(trial_dir: str, suffix: str) -> str:
    for fname in os.listdir(trial_dir):
        lower = fname.lower()
        if lower.endswith(suffix) and (suffix != "_exo.csv" or not lower.endswith("power_exo.csv")):
            return os.path.join(trial_dir, fname)
    raise FileNotFoundError(f"Cannot find file ending with {suffix} under {trial_dir}")


def _load_sequences(real_path: str, s2r_path: str, sim_path: str, channel_order: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    real_df = pd.read_csv(real_path)
    s2r_df = pd.read_csv(s2r_path)
    sim_df = pd.read_csv(sim_path)
    columns = [
        col
        for col in channel_order
        if col in real_df.columns and col in s2r_df.columns and col in sim_df.columns
    ]
    if not columns:
        raise ValueError("No overlapping columns between real/sim/sim2real CSV.")
    real_arr = real_df[columns].to_numpy()
    s2r_arr = s2r_df[columns].to_numpy()
    sim_arr = sim_df[columns].to_numpy()
    min_len = min(len(real_arr), len(s2r_arr), len(sim_arr))
    return real_arr[:min_len], s2r_arr[:min_len], sim_arr[:min_len], columns


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(10, 5))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Sensor Value")
        self.line_real, = self.ax.plot([], [], label="real")
        self.line_s2r, = self.ax.plot([], [], label="sim2real")
        self.line_sim, = self.ax.plot([], [], linestyle="--", label="sim")
        self.ax.legend(loc="upper right")

    def update_data(self, x: np.ndarray, real: np.ndarray, s2r: np.ndarray, sim: np.ndarray, channel_name: str):
        self.line_real.set_data(x, real)
        self.line_s2r.set_data(x, s2r)
        self.line_sim.set_data(x, sim)
        self.ax.set_title(channel_name)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_idle()

    def set_xlim(self, start: int, end: int):
        self.ax.set_xlim(start, end)
        self.ax.figure.canvas.draw_idle()


class VisualWindow(QtWidgets.QMainWindow):
    def __init__(self, trial: str, real_seq: np.ndarray, s2r_seq: np.ndarray, sim_seq: np.ndarray, columns: List[str], window_len: int):
        super().__init__()
        self.trial = trial
        self.real_seq = real_seq
        self.s2r_seq = s2r_seq
        self.sim_seq = sim_seq
        self.columns = columns
        self.seq_len = real_seq.shape[0]
        self.window_len = window_len

        self.setWindowTitle(f"Sim2Real Visualization - {trial}")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.canvas = PlotCanvas()
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        control_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(control_layout)

        control_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(columns)
        control_layout.addWidget(self.channel_combo)

        control_layout.addWidget(QtWidgets.QLabel("Window length:"))
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(10, max(10, self.seq_len))
        self.window_spin.setValue(min(window_len, self.seq_len))
        control_layout.addWidget(self.window_spin)

        control_layout.addWidget(QtWidgets.QLabel("Start:"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, max(0, self.seq_len - 1))
        control_layout.addWidget(self.slider)

        self.channel_combo.currentTextChanged.connect(self.update_plot)
        self.window_spin.valueChanged.connect(self.update_slider_range)
        self.slider.valueChanged.connect(self.update_xlim)

        self._init_plot()

    def _init_plot(self):
        self.slider.setValue(0)
        self.update_plot(self.channel_combo.currentText())
        self.update_slider_range(self.window_spin.value())

    def update_plot(self, channel: str):
        idx = self.columns.index(channel)
        x = np.arange(self.seq_len)
        self.canvas.update_data(x, self.real_seq[:, idx], self.s2r_seq[:, idx], self.sim_seq[:, idx], channel)
        self.update_xlim(self.slider.value())

    def update_slider_range(self, value: int):
        self.window_len = value
        max_start = max(0, self.seq_len - self.window_len)
        self.slider.setRange(0, max_start if max_start > 0 else 0)
        self.update_xlim(self.slider.value())

    def update_xlim(self, start: int):
        end = min(start + self.window_len, self.seq_len - 1 if self.seq_len > 0 else 0)
        if start >= end:
            end = min(start + 1, self.seq_len - 1)
        self.canvas.set_xlim(start, end)


def main() -> None:
    args = parse_args()
    dataset_cfg = DatasetConfig(data_dirs=args.data_dirs, side=args.side)
    channel_order = [name.replace("*", args.side) for name in dataset_cfg.input_names]

    trial_dir = _resolve_trial_dir(args.data_dirs, args.trial)
    real_path = _find_file(trial_dir, "_exo.csv")
    s2r_path = _find_file(trial_dir, "_exo_s2r.csv")
    sim_path = _find_file(trial_dir, "_exo_sim.csv")
    real_seq, s2r_seq, sim_seq, columns = _load_sequences(real_path, s2r_path, sim_path, channel_order)

    app = QtWidgets.QApplication(sys.argv)
    window = VisualWindow(args.trial, real_seq, s2r_seq, sim_seq, columns, args.window)
    window.resize(1200, 700)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
