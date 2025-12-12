"""无监督生成对抗训练入口."""
from __future__ import annotations

import argparse
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
from domain_adaptation.models.gan import DomainAdaptationGAN, GanConfig


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


def _collate_windows(batch, target_len: int) -> torch.Tensor:
    windows = []
    for inputs, _, seq_lengths, _ in batch:
        seq_len = int(seq_lengths[0]) if isinstance(seq_lengths, list) else int(seq_lengths)
        tensor = inputs.squeeze(0)
        windows.append(_random_window(tensor, seq_len, target_len))
    return torch.stack(windows, dim=0)


def _build_loader(
    dataset,
    batch_size: int,
    target_len: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    collate_fn = partial(_collate_windows, target_len=target_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )


def _prepare_real_dataset(config: DatasetConfig, device: torch.device, max_trials: int | None):
    dataset = DataManager.load_datasets(config, device=device)
    valid_indices = DataManager.get_or_compute_valid_indices(dataset, config)
    if max_trials is not None:
        valid_indices = valid_indices[:max_trials]
    if len(valid_indices) == 0:
        raise RuntimeError("真实域数据过滤后为空，请检查数据配置")
    return Subset(dataset, valid_indices)


def _prepare_sim_dataset(config: DatasetConfig, device: torch.device, max_trials: int | None):
    dataset = DataManager.load_sim_datasets(config, device=device)
    if max_trials is not None:
        dataset = Subset(dataset, list(range(min(max_trials, len(dataset)))))
    if len(dataset) == 0:
        raise RuntimeError("模拟域数据为空，请检查数据配置")
    return dataset


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
    parser.add_argument("--base-channels", type=int, default=64, help="U-Net初始通道数")
    parser.add_argument("--unet-depth", type=int, default=4, help="U-Net下采样深度")
    parser.add_argument("--gen-lr", type=float, default=5e-4)
    parser.add_argument("--disc-lr", type=float, default=5e-4)
    parser.add_argument("--lambda-cycle", type=float, default=0.9)
    parser.add_argument("--lambda-identity", type=float, default=0.3)
    parser.add_argument("--lambda-gan", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", help="训练设备，例如 cuda 或 cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="runs", help="训练输出根目录")
    parser.add_argument("--run-name", type=str, default=None, help="当前训练run名称")
    parser.add_argument("--resume", type=str, default=None, help="可选checkpoint路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-real-trials", type=int, default=None, help="调试用，限制真实trial数")
    parser.add_argument("--max-sim-trials", type=int, default=None, help="调试用，限制模拟trial数")
    parser.add_argument("--mlflow", action="store_true", help="启用MLflow记录")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="可选MLflow Tracking URI")
    parser.add_argument("--mlflow-experiment", type=str, default="domain_adaptation", help="MLflow实验名")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="MLflow run名称")
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


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    dataset_cfg = DatasetConfig(
        data_dirs=args.data_dirs,
        side=args.side,
        action_patterns=args.action_patterns,
    )

    run_name, run_dir = _prepare_run_directory(args.output_dir, args.run_name)
    args.run_name = run_name
    if args.mlflow_run_name is None:
        args.mlflow_run_name = run_name
    print(f"Run directory: {run_dir}")

    if args.mlflow:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    mlflow_run = mlflow.start_run(run_name=args.mlflow_run_name) if args.mlflow else None

    try:
        cpu_device = torch.device("cpu")
        real_dataset = _prepare_real_dataset(dataset_cfg, cpu_device, args.max_real_trials)
        sim_dataset = _prepare_sim_dataset(dataset_cfg, cpu_device, args.max_sim_trials)

        real_loader = _build_loader(
            real_dataset,
            batch_size=args.batch_size,
            target_len=args.seq_len,
            num_workers=args.num_workers,
        )
        sim_loader = _build_loader(
            sim_dataset,
            batch_size=args.batch_size,
            target_len=args.seq_len,
            num_workers=args.num_workers,
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
        )
        model = DomainAdaptationGAN(gan_config)

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
        best_path = os.path.join(run_dir, "best.pt")
        last_path = os.path.join(run_dir, "last.pt")

        for epoch in range(1, args.epochs + 1):
            epoch_metrics: Dict[str, float] = {}
            progress = tqdm(
                range(1, steps_per_epoch + 1),
                desc=f"Epoch {epoch}/{args.epochs}",
                leave=False,
            )
            for step in progress:
                real_batch, real_iter = _next_batch(real_iter, real_loader)
                sim_batch, sim_iter = _next_batch(sim_iter, sim_loader)
                metrics = model.training_step({"real": real_batch, "sim": sim_batch})
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
            metric = epoch_avg.get("gen_total")
            if metric is not None and metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                model.save_checkpoint(best_path)
                print(f"Best checkpoint updated at epoch {epoch} (gen_total={metric:.4f}) -> {best_path}")

            model.save_checkpoint(last_path)
            print(f"Latest checkpoint saved to: {last_path}")

        if best_epoch is not None:
            print(f"Best epoch: {best_epoch} (gen_total={best_metric:.4f}) -> {best_path}")
        else:
            print("No valid best checkpoint (gen_total missing); only last checkpoint available.")

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
