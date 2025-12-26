"""无监督生成对抗训练入口."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import mlflow
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain_adaptation.config import DatasetConfig
from domain_adaptation.data import DataManager
from domain_adaptation.data.sensor_dataset import SensorDataset
from domain_adaptation.models.gan import DomainAdaptationGAN, GanConfig
from domain_adaptation.utils.eval import (
    load_trial_tensor,
    get_channel_stats_tensors,
    denormalize_sequence,
    translate_sim_to_real,
    compute_channel_mse,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _replace_nan(tensor: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))


def _find_valid_segments(valid_mask: torch.Tensor) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start_idx = None
    mask_list = valid_mask.tolist()
    for idx, flag in enumerate(mask_list):
        if flag:
            if start_idx is None:
                start_idx = idx
        else:
            if start_idx is not None:
                segments.append((start_idx, idx))
                start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(mask_list)))
    return segments


def _random_window(sample: torch.Tensor, valid_len: int, target_len: int) -> torch.Tensor:
    """从序列中随机裁剪定长窗口，尽量避开NaN（遇到NaN则换段）。"""
    if valid_len <= 0:
        return torch.zeros(sample.shape[0], target_len, device=sample.device, dtype=sample.dtype)

    sample = sample[:, :valid_len]
    valid_mask = torch.isfinite(sample).all(dim=0)
    segments = _find_valid_segments(valid_mask)

    window: torch.Tensor
    if segments:
        long_segments = [seg for seg in segments if (seg[1] - seg[0]) >= target_len]
        if long_segments:
            seg_start, seg_end = random.choice(long_segments)
            max_start = seg_end - target_len
            start = random.randint(seg_start, max_start)
            window = sample[:, start:start + target_len]
        else:
            seg_start, seg_end = max(segments, key=lambda seg: seg[1] - seg[0])
            window = sample[:, seg_start:seg_end]
    else:
        window = sample

    window = _replace_nan(window)

    if window.shape[-1] < target_len:
        pad = target_len - window.shape[-1]
        window = F.pad(window, (0, pad))
    elif window.shape[-1] > target_len:
        start = random.randint(0, window.shape[-1] - target_len)
        window = window[:, start:start + target_len]
    return window


def _random_window_pair(
    sample: torch.Tensor,
    paired: Optional[torch.Tensor],
    valid_len: int,
    target_len: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """在保证同一起止位置的前提下，同时裁剪输入与标签窗口."""
    if valid_len <= 0:
        base = torch.zeros(sample.shape[0], target_len, device=sample.device, dtype=sample.dtype)
        if paired is None:
            return base, None
        pair_base = torch.zeros(paired.shape[0], target_len, device=paired.device, dtype=paired.dtype)
        return base, pair_base

    sample = sample[:, :valid_len]
    paired_tensor = paired[:, :valid_len] if paired is not None else None
    valid_mask = torch.isfinite(sample).all(dim=0)
    if paired_tensor is not None:
        valid_mask = valid_mask & torch.isfinite(paired_tensor).all(dim=0)
    segments = _find_valid_segments(valid_mask)

    def _slice_tensor(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return tensor[:, start:end]

    if segments:
        long_segments = [seg for seg in segments if (seg[1] - seg[0]) >= target_len]
        if long_segments:
            seg_start, seg_end = random.choice(long_segments)
            max_start = seg_end - target_len
            start = random.randint(seg_start, max_start)
            end = start + target_len
        else:
            seg_start, seg_end = max(segments, key=lambda seg: seg[1] - seg[0])
            start, end = seg_start, seg_end
    else:
        start, end = 0, sample.shape[-1]

    window = _slice_tensor(sample, start, min(end, sample.shape[-1]))
    pair_window = (
        _slice_tensor(paired_tensor, start, min(end, paired_tensor.shape[-1]))
        if paired_tensor is not None
        else None
    )

    window = _replace_nan(window)
    if pair_window is not None:
        pair_window = _replace_nan(pair_window)

    def _pad_if_needed(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] < target_len:
            pad = target_len - tensor.shape[-1]
            tensor = F.pad(tensor, (0, pad))
        elif tensor.shape[-1] > target_len:
            start_idx = random.randint(0, tensor.shape[-1] - target_len)
            tensor = tensor[:, start_idx:start_idx + target_len]
        return tensor

    window = _pad_if_needed(window)
    if pair_window is not None:
        pair_window = _pad_if_needed(pair_window)

    return window, pair_window


def _collate_windows(
    batch,
    target_len: int,
    return_labels: bool = False,
    return_names: bool = False,
):
    input_windows = []
    label_windows: List[torch.Tensor] = []
    trial_names_batch: List[List[str]] = []
    for inputs, labels, seq_lengths, trial_names in batch:
        seq_len = int(seq_lengths[0]) if isinstance(seq_lengths, list) else int(seq_lengths)
        input_tensor = inputs.squeeze(0)
        label_tensor = labels.squeeze(0) if return_labels else None
        window, label_window = _random_window_pair(input_tensor, label_tensor, seq_len, target_len)
        input_windows.append(window)
        if return_labels:
            if label_window is None:
                raise RuntimeError("期望标签窗口但未生成，请检查数据加载流程")
            label_windows.append(label_window)
        if return_names:
            names = trial_names if isinstance(trial_names, list) else [trial_names]
            trial_names_batch.append(names)
    input_batch = torch.stack(input_windows, dim=0)
    outputs: List[Any] = [input_batch]
    if return_labels:
        outputs.append(torch.stack(label_windows, dim=0))
    if return_names:
        outputs.append(trial_names_batch)
    if len(outputs) == 1:
        return input_batch
    return tuple(outputs)


def _build_loader(
    dataset,
    batch_size: int,
    target_len: int,
    num_workers: int,
    shuffle: bool = True,
    return_labels: bool = False,
    return_names: bool = False,
) -> DataLoader:
    collate_fn = partial(
        _collate_windows,
        target_len=target_len,
        return_labels=return_labels,
        return_names=return_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )


def _filter_trials_by_participant(dataset: SensorDataset, participant: str) -> List[str]:
    trials = [name for name in dataset.trial_names if name.startswith(participant)]
    dataset.trial_names = trials
    return trials


def _build_validation_dataset(
    dataset_cfg: DatasetConfig,
    device: torch.device,
    input_suffix: str,
    label_suffix: str,
    participant: str,
) -> SensorDataset:
    for data_dir in dataset_cfg.data_dirs:
        dataset = SensorDataset(
            data_dir=data_dir,
            input_names=[name.replace("*", dataset_cfg.side) for name in dataset_cfg.input_names],
            label_names=[name.replace("*", dataset_cfg.side) for name in dataset_cfg.label_names],
            side=dataset_cfg.side,
            participant_masses=dataset_cfg.participant_masses,
            device=device,
            input_file_suffix=input_suffix,
            label_file_suffix=label_suffix,
        )
        filtered = _filter_trials_by_participant(dataset, participant)
        if filtered:
            return dataset
    raise RuntimeError(f"未找到 {participant} 的校验trial")


def _run_validation(
    model: DomainAdaptationGAN,
    dataset_cfg: DatasetConfig,
    device: torch.device,
    participant: str = "BT17",
) -> float:
    model.eval()
    val_real_dataset = _build_validation_dataset(
        dataset_cfg, device, "_exo.csv", "_moment_filt.csv", participant
    )
    val_sim_dataset = _build_validation_dataset(
        dataset_cfg, device, "_exo_sim.csv", "_moment_filt_bio.csv", participant
    )
    trial_set = sorted(set(val_real_dataset.trial_names) & set(val_sim_dataset.trial_names))
    if not trial_set:
        raise RuntimeError(f"{participant} 缺少成对trial用于校验")
    mean_tensor, std_tensor = get_channel_stats_tensors(val_real_dataset, device)
    total_mse = 0.0
    count = 0
    with torch.no_grad():
        for trial in trial_set:
            real_tensor, real_len = load_trial_tensor(val_real_dataset, trial, device)
            sim_tensor, sim_len = load_trial_tensor(val_sim_dataset, trial, device)
            valid_len = min(real_len, sim_len)
            sim_slice = sim_tensor[..., :valid_len]
            real_slice = real_tensor[..., :valid_len]
            generated = translate_sim_to_real(model, sim_slice)
            gen_denorm = denormalize_sequence(generated, mean_tensor, std_tensor)
            real_denorm = denormalize_sequence(real_slice, mean_tensor, std_tensor)
            _, mse = compute_channel_mse(gen_denorm, real_denorm, valid_len)
            total_mse += mse
            count += 1
    model.train()
    return total_mse / max(count, 1)


def _identify_modality(channel_name: str) -> Optional[str]:
    name = channel_name.lower()
    if "thigh_imu" in name and "_accel_" in name:
        return "thigh_accel"
    if "thigh_imu" in name and "_gyro_" in name:
        return "thigh_gyro"
    if "shank_imu" in name and "_accel_" in name:
        return "shank_accel"
    if "shank_imu" in name and "_gyro_" in name:
        return "shank_gyro"
    return None


def _identify_modality(channel_name: str) -> Optional[str]:
    name = channel_name.lower()
    if "thigh_imu" in name and "_accel_" in name:
        return "thigh_accel"
    if "thigh_imu" in name and "_gyro_" in name:
        return "thigh_gyro"
    if "shank_imu" in name and "_accel_" in name:
        return "shank_accel"
    if "shank_imu" in name and "_gyro_" in name:
        return "shank_gyro"
    return None


def _compute_modality_weights(channel_names: Optional[List[str]], variances: Optional[List[float]]) -> Optional[List[float]]:
    if not channel_names or not variances:
        return None
    name_to_var = {name: max(var, 1e-6) for name, var in zip(channel_names, variances)}
    group_max: Dict[str, float] = {}
    for name, var in zip(channel_names, variances):
        group = _identify_modality(name)
        if group is None:
            continue
        value = max(var, 1e-6)
        group_max[group] = max(group_max.get(group, 0.0), value)
    if not group_max:
        return None
    weights: List[float] = []
    for name in channel_names:
        group = _identify_modality(name)
        if group and group in group_max:
            max_var = max(group_max[group], 1e-6)
            channel_var = name_to_var[name]
            weights.append(min(channel_var / max_var, 1.0))
        else:
            weights.append(1.0)
    return weights


def _aggregate_channel_stats(dataset: Any, use_norm: bool = True) -> Tuple[Optional[List[str]], Optional[List[float]]]:
    components = getattr(dataset, "datasets", None)
    if components is None:
        components = [dataset]
    names: Optional[List[str]] = None
    total_var: Optional[torch.Tensor] = None
    total_weight = 0
    for ds in components:
        stats = getattr(ds, "channel_stats", None)
        ds_names = getattr(ds, "input_names", None)
        if stats is None or ds_names is None:
            continue
        if names is None:
            names = list(ds_names)
        weight = len(ds)
        key = "norm_variance" if use_norm else "variance"
        var_values = stats.get(key)
        if var_values is None:
            continue
        var_tensor = torch.tensor(var_values, dtype=torch.float64)
        total_var = var_tensor * weight if total_var is None else total_var + var_tensor * weight
        total_weight += weight
    if names is None or total_var is None or total_weight == 0:
        return None, None
    avg_var = (total_var / total_weight).tolist()
    return names, avg_var


def _prepare_real_dataset(config: DatasetConfig, device: torch.device, max_trials: int | None):
    dataset = DataManager.load_datasets(config, device=device)
    channel_names, variances = _aggregate_channel_stats(dataset, use_norm=False)
    channel_weights = _compute_modality_weights(channel_names, variances)
    valid_indices = DataManager.get_or_compute_valid_indices(dataset, config)
    if max_trials is not None:
        valid_indices = valid_indices[:max_trials]
    if len(valid_indices) == 0:
        raise RuntimeError("真实域数据过滤后为空，请检查数据配置")
    return Subset(dataset, valid_indices), channel_weights


def _prepare_sim_dataset(config: DatasetConfig, device: torch.device, max_trials: int | None):
    dataset = DataManager.load_sim_datasets(config, device=device)
    channel_names, variances = _aggregate_channel_stats(dataset, use_norm=False)
    channel_weights = _compute_modality_weights(channel_names, variances)
    if max_trials is not None:
        dataset = Subset(dataset, list(range(min(max_trials, len(dataset)))))
    if len(dataset) == 0:
        raise RuntimeError("模拟域数据为空，请检查数据配置")
    return dataset, channel_weights


def _next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无监督域适配GAN训练")
    parser.add_argument("--data-dirs", nargs="+", default=DatasetConfig().data_dirs, help="数据根目录列表")
    parser.add_argument("--side", type=str, default="r", choices=["l", "r"], help="训练侧别")
    parser.add_argument("--action-patterns", nargs="*", default=None, help="可选的trial过滤正则")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="每个epoch迭代次数；默认=min(len loaders))")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256, help="裁剪窗口长度")
    parser.add_argument("--base-channels", type=int, default=128, help="U-Net初始通道数")
    parser.add_argument("--unet-depth", type=int, default=4, help="U-Net下采样深度")
    parser.add_argument("--gen-lr", type=float, default=2e-3)
    parser.add_argument("--disc-lr", type=float, default=5e-4)
    parser.add_argument("--lambda-cycle", type=float, default=0.5)
    parser.add_argument("--lambda-identity", type=float, default=1.0)
    parser.add_argument("--lambda-gan", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", help="训练设备，例如 cuda 或 cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="runs", help="训练输出根目录")
    parser.add_argument("--run-name", type=str, default=None, help="当前训练run名称（也用作MLflow run name）")
    parser.add_argument("--resume", type=str, default=None, help="可选checkpoint路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-real-trials", type=int, default=None, help="调试用，限制真实trial数")
    parser.add_argument("--max-sim-trials", type=int, default=None, help="调试用，限制模拟trial数")
    parser.add_argument("--mlflow", action="store_true", help="启用MLflow记录")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="可选MLflow Tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="domain_adaptation", help="MLflow实验名")
    parser.add_argument("--lambda-moment", type=float, default=2.0, help="力矩估计损失权重")
    parser.add_argument("--moment-start-epoch", type=int, default=10, help="力矩估计损失的启动epoch")
    parser.add_argument("--tcn-num-channels", type=str, default="80,80,80,80,80", help="TCN每层通道数, 逗号分隔")
    parser.add_argument("--tcn-kernel-size", type=int, default=5, help="TCN卷积核大小")
    parser.add_argument("--tcn-dropout", type=float, default=0.15, help="TCN dropout")
    parser.add_argument("--tcn-learning-rate", type=float, default=5e-4, help="TCN优化器学习率")
    parser.add_argument("--tcn-eff-hist", type=int, default=248, help="TCN有效历史长度")
    parser.add_argument("--tcn-load-path", type=str, default=None, help="可选TCN checkpoint路径")
    parser.add_argument("--tcn-freeze", action="store_true", help="固定TCN参数，仅用于计算moment loss")
    parser.add_argument("--disc-real-label", type=float, default=0.9, help="判别器label smoothing的真实标签值")
    parser.add_argument("--disc-fake-label", type=float, default=0.0, help="判别器label smoothing的假标签值")
    parser.add_argument("--replay-buffer-size", type=int, default=50, help="判别器fake buffer大小（0表示关闭）")
    parser.add_argument("--val-participant", type=str, default="BT17", help="验证集受试者ID")
    parser.add_argument("--val-interval", type=int, default=3, help="验证间隔（单位：epoch）")
    parser.add_argument("--val-patience", type=int, default=3, help="验证连续不提升次数触发早停")
    parser.add_argument("--no-validation", action="store_true", help="禁用验证与早停逻辑")
    return parser.parse_args()


def _stringify(value: Any) -> str:
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _log_params_to_mlflow(args: argparse.Namespace, dataset_cfg: DatasetConfig) -> None:
    loggable_args = {
        k: v for k, v in vars(args).items()
        if k not in {"resume"}
    }
    mlflow.log_params({k: _stringify(v) for k, v in loggable_args.items()})

    # ds_params = {
    #     "dataset.data_dirs": dataset_cfg.data_dirs,
    #     "dataset.side": dataset_cfg.side,
    #     "dataset.real_sensor_names": dataset_cfg.real_sensor_names,
    #     "dataset.label_names": dataset_cfg.label_names,
    #     "dataset.action_patterns": dataset_cfg.action_patterns,
    #     "dataset.participant_masses": dataset_cfg.participant_masses,
    #     "dataset.window_size": dataset_cfg.window_size,
    #     "dataset.window_stride": dataset_cfg.window_stride,
    #     "dataset.filter_nan_trials": getattr(dataset_cfg, "filter_nan_trials", False),
    # }
    # mlflow.log_params({k: _stringify(v) for k, v in ds_params.items()})


def _prepare_run_directory(base_dir: str, run_name: Optional[str]) -> Tuple[str, str]:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    default_name = datetime.now().strftime("%m%d-%H%M%S")
    final_name = run_name or default_name
    run_path = base_path / final_name
    if run_path.exists():
        suffix = datetime.now().strftime("%H%M%S")
        final_name = f"{final_name}-{suffix}"
        run_path = base_path / final_name
    run_path.mkdir(parents=True, exist_ok=True)
    return final_name, str(run_path)


_EPOCH_METRIC_FIELDS = [
    "epoch",
    "gen_total",
    "adv_loss",
    "cycle_loss",
    "identity_loss",
    "moment_loss",
    "disc_loss",
]



def _append_epoch_metrics(csv_path: str, epoch: int, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    row = {"epoch": epoch}
    for field in _EPOCH_METRIC_FIELDS[1:]:
        value = metrics.get(field)
        row[field] = f"{value:.6f}" if isinstance(value, (float, int)) else ""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EPOCH_METRIC_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    try:
        tcn_channels = tuple(
            int(x.strip())
            for x in args.tcn_num_channels.split(",")
            if x.strip()
        )
    except ValueError as exc:
        raise ValueError(f"无法解析 --tcn-num-channels: {args.tcn_num_channels}") from exc
    if len(tcn_channels) == 0:
        raise ValueError("至少需要一个TCN通道配置")

    dataset_cfg = DatasetConfig(
        data_dirs=args.data_dirs,
        side=args.side,
        action_patterns=args.action_patterns,
    )

    run_name, run_dir = _prepare_run_directory(args.output_dir, args.run_name)
    args.run_name = run_name
    print(f"Run directory: {run_dir}")
    metrics_csv_path = os.path.join(run_dir, "epoch_metrics.csv")

    if args.mlflow:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    mlflow_run = mlflow.start_run(run_name=run_name) if args.mlflow else None

    try:
        cpu_device = torch.device("cpu")
        real_dataset, real_weights = _prepare_real_dataset(dataset_cfg, cpu_device, args.max_real_trials)
        sim_dataset, sim_weights = _prepare_sim_dataset(dataset_cfg, cpu_device, args.max_sim_trials)

        real_loader = _build_loader(
            real_dataset,
            batch_size=args.batch_size,
            target_len=args.seq_len,
            num_workers=args.num_workers,
            return_labels=False,
        )
        sim_loader = _build_loader(
            sim_dataset,
            batch_size=args.batch_size,
            target_len=args.seq_len,
            num_workers=args.num_workers,
            return_labels=args.lambda_moment > 0,
        )

        if len(real_loader) == 0 or len(sim_loader) == 0:
            raise RuntimeError("DataLoader为空，请适当减小batch_size或检查数据")

        channel_count = len(dataset_cfg.input_names)
        gan_config = GanConfig(
            sim_channels=channel_count,
            real_channels=channel_count,
            sequence_length=args.seq_len,
            base_channels=args.base_channels,
            depth=args.unet_depth,
            gen_learning_rate=args.gen_lr,
            disc_learning_rate=args.disc_lr,
            cycle_loss_weight=args.lambda_cycle,
            identity_loss_weight=args.lambda_identity,
            gan_loss_weight=args.lambda_gan,
            device=args.device,
            sim_modal_weights=sim_weights,
        real_modal_weights=real_weights,
        label_channels=len(dataset_cfg.label_names),
        lambda_moment=args.lambda_moment,
        moment_start_epoch=args.moment_start_epoch,
        tcn_num_channels=tcn_channels,
        tcn_kernel_size=args.tcn_kernel_size,
            tcn_dropout=args.tcn_dropout,
            tcn_learning_rate=args.tcn_learning_rate,
            tcn_eff_hist=args.tcn_eff_hist,
            tcn_load_path=args.tcn_load_path,
            tcn_freeze=args.tcn_freeze,
        )
        model = DomainAdaptationGAN(gan_config)
        model.fake_real_buffer.max_size = max(0, args.replay_buffer_size)
        model.fake_sim_buffer.max_size = max(0, args.replay_buffer_size)
        model.real_label = args.disc_real_label
        model.fake_label = args.disc_fake_label

        if args.resume:
            print(f"加载checkpoint: {args.resume}")
            model.load_checkpoint(args.resume)

        if args.mlflow:
            _log_params_to_mlflow(args, dataset_cfg)
            mlflow.log_params({
                "dataset.num_real_trials": len(real_dataset),
                "dataset.num_sim_trials": len(sim_dataset),
            })

        steps_per_epoch = (
            args.steps_per_epoch
            if args.steps_per_epoch is not None
            else min(len(real_loader), len(sim_loader))
        )
        if steps_per_epoch <= 0:
            raise RuntimeError("steps_per_epoch <= 0，请检查数据或参数")

        real_iter = iter(real_loader)
        sim_iter = iter(sim_loader)
        best_metric = float("inf")
        best_epoch = None
        best_val_metric = float("inf")
        val_patience = 0
        use_validation = not args.no_validation
        best_path = os.path.join(run_dir, "best.pt")
        last_path = os.path.join(run_dir, "last.pt")
        best_tcn_path = os.path.join(run_dir, "best_tcn.tar") if args.lambda_moment > 0 else None
        last_tcn_path = os.path.join(run_dir, "last_tcn.tar") if args.lambda_moment > 0 else None

        for epoch in range(1, args.epochs + 1):
            model.set_current_epoch(epoch)
            epoch_metrics: Dict[str, float] = {}
            progress = tqdm(
                range(1, steps_per_epoch + 1),
                desc=f"Epoch {epoch}/{args.epochs}",
                leave=False,
            )
            for step in progress:
                real_batch, real_iter = _next_batch(real_iter, real_loader)
                sim_batch, sim_iter = _next_batch(sim_iter, sim_loader)
                real_inputs = real_batch[0] if isinstance(real_batch, tuple) else real_batch
                if isinstance(sim_batch, tuple):
                    sim_inputs, sim_labels = sim_batch
                else:
                    sim_inputs, sim_labels = sim_batch, None
                if args.lambda_moment > 0 and sim_labels is None:
                    raise RuntimeError("需要力矩标签但未从sim_loader返回，请检查DataLoader设置")
                metrics = model.training_step(
                    {"real": real_inputs, "sim": sim_inputs, "sim_labels": sim_labels}
                )
                for key, value in metrics.items():
                    epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

                if step % args.log_interval == 0 or step == steps_per_epoch:
                    avg_metrics = {k: v / step for k, v in epoch_metrics.items()}
                    metric_str = " | ".join(f"{k}: {val:.4f}" for k, val in avg_metrics.items())
                    print(f"[Epoch {epoch:03d} Step {step:04d}/{steps_per_epoch}] {metric_str}")
                    display = {
                        k: f"{avg_metrics[k]:.4f}"
                        for k in ["gen_total", "disc_loss", "cycle_loss"]
                        if k in avg_metrics
                    }
                    if display:
                        progress.set_postfix(display)
                    if args.mlflow:
                        global_step = (epoch - 1) * steps_per_epoch + step
                        mlflow.log_metrics({k: v for k, v in avg_metrics.items()}, step=global_step)

            epoch_avg = {k: v / steps_per_epoch for k, v in epoch_metrics.items()}
            _append_epoch_metrics(metrics_csv_path, epoch, epoch_avg)
            metric = epoch_avg.get("gen_total")
            if metric is not None and metric < best_metric:
                best_metric = metric

            model.save_checkpoint(last_path)
            if last_tcn_path:
                model.save_tcn_checkpoint(last_tcn_path, epoch, metric)
            print(f"Latest checkpoint saved to: {last_path}")

            if use_validation and epoch % args.val_interval == 0:
                val_mse = _run_validation(model, dataset_cfg, model.device, participant=args.val_participant)
                print(f"[Validation] Epoch {epoch}: Mean MSE={val_mse:.6f}")
                if val_mse < best_val_metric:
                    best_val_metric = val_mse
                    best_epoch = epoch
                    val_patience = 0
                    model.save_checkpoint(best_path)
                    if best_tcn_path:
                        model.save_tcn_checkpoint(best_tcn_path, epoch, val_mse)
                    print(f"Validation checkpoint updated at epoch {epoch} (MSE={val_mse:.6f}) -> {best_path}")
                else:
                    val_patience += 1
                    print(f"Validation did not improve for {val_patience} consecutive evaluations.")
                    if val_patience >= args.val_patience:
                        print("Early stopping triggered due to lack of validation improvement.")
                        break
        if best_epoch is not None:
            print(f"Best epoch: {best_epoch} (val MSE={best_val_metric:.6f}) -> {best_path}")
        else:
            print("No validation improvement recorded; only last checkpoint available.")

        if args.mlflow:
            if os.path.exists(best_path):
                mlflow.log_artifact(best_path, artifact_path="checkpoints")
            if os.path.exists(last_path):
                mlflow.log_artifact(last_path, artifact_path="checkpoints")
    finally:
        if mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
