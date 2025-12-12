"""无监督生成对抗训练入口."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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


def _random_window(sample: torch.Tensor, valid_len: int, target_len: int) -> torch.Tensor:
    """从序列中随机裁剪定长窗口，不足部分补零。"""
    total_len = sample.shape[-1]
    valid_len = max(1, min(valid_len, total_len))
    if valid_len > target_len:
        start = random.randint(0, valid_len - target_len)
        end = start + target_len
    else:
        start = 0
        end = valid_len
    window = sample[:, start:end]
    if window.shape[-1] < target_len:
        pad = target_len - window.shape[-1]
        window = F.pad(window, (0, pad))
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
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="可选checkpoint路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-real-trials", type=int, default=None, help="调试用，限制真实trial数")
    parser.add_argument("--max-sim-trials", type=int, default=None, help="调试用，限制模拟trial数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset_cfg = DatasetConfig(
        data_dirs=args.data_dirs,
        side=args.side,
        action_patterns=args.action_patterns,
    )

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

    steps_per_epoch = (
        args.steps_per_epoch
        if args.steps_per_epoch is not None
        else min(len(real_loader), len(sim_loader))
    )
    if steps_per_epoch <= 0:
        raise RuntimeError("steps_per_epoch <= 0，请检查数据或参数")

    real_iter = iter(real_loader)
    sim_iter = iter(sim_loader)

    for epoch in range(1, args.epochs + 1):
        epoch_metrics: Dict[str, float] = {}
        for step in range(1, steps_per_epoch + 1):
            real_batch, real_iter = _next_batch(real_iter, real_loader)
            sim_batch, sim_iter = _next_batch(sim_iter, sim_loader)
            metrics = model.training_step({"real": real_batch, "sim": sim_batch})
            for key, value in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

            if step % args.log_interval == 0 or step == steps_per_epoch:
                avg_metrics = {k: v / step for k, v in epoch_metrics.items()}
                metric_str = " | ".join(f"{k}: {val:.4f}" for k, val in avg_metrics.items())
                print(f"[Epoch {epoch:03d} Step {step:04d}/{steps_per_epoch}] {metric_str}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"gan_epoch_{epoch:03d}.pt")
        model.save_checkpoint(ckpt_path)
        print(f"已保存checkpoint至: {ckpt_path}")


if __name__ == "__main__":
    main()
