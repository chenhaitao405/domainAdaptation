"""通用的推理与评估辅助函数."""
from __future__ import annotations

from typing import Tuple, Sequence
import torch
import pandas as pd


def load_trial_tensor(dataset, trial_name: str, device: torch.device) -> Tuple[torch.Tensor, int]:
    """从SensorDataset加载指定trial的输入序列（保持batch维），返回张量和有效长度."""
    if trial_name not in dataset.trial_names:
        raise ValueError(f"Trial '{trial_name}' not found in dataset.")
    idx = dataset.trial_names.index(trial_name)
    inputs, _, seq_lengths, _ = dataset[idx]
    if isinstance(seq_lengths, list):
        valid_len = int(seq_lengths[0])
    else:
        valid_len = int(seq_lengths)
    tensor = inputs.to(device)
    return tensor, valid_len


def get_channel_stats_tensors(dataset, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回通道均值/标准差张量（形状1×C×1）."""
    stats = dataset.channel_stats
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, -1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device)
    std = torch.clamp(std, min=1e-6).view(1, -1, 1)
    return mean, std


def denormalize_sequence(sequence: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """按通道均值/方差反归一化，并在归一化域内做±3σ限幅."""
    clamped = torch.clamp(sequence, min=-3.0, max=3.0)
    return clamped * std + mean


def translate_sim_to_real(model, sim_sequence: torch.Tensor) -> torch.Tensor:
    """调用生成器进行sim->real推理（输入shape: B×C×L）."""
    model.eval()
    model.sim2real.eval()
    with torch.no_grad():
        return model.sim2real(sim_sequence)


def compute_channel_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_length: int,
) -> Tuple[torch.Tensor, float]:
    """在有效长度上计算每个通道的MSE及其均值."""
    pred_slice = prediction[..., :valid_length]
    target_slice = target[..., :valid_length]
    diff = pred_slice - target_slice
    mse_per_channel = (diff.pow(2).mean(dim=-1)).squeeze(0)
    overall = float(mse_per_channel.mean().item())
    return mse_per_channel, overall


def save_sequence_to_csv(
    sequence: torch.Tensor,
    valid_length: int,
    channel_names: Sequence[str],
    path: str,
) -> None:
    """将形状(1,C,L)的序列写入CSV."""
    slice_tensor = sequence[..., :valid_length].squeeze(0).transpose(0, 1).cpu().numpy()
    df = pd.DataFrame(slice_tensor, columns=channel_names)
    df.to_csv(path, index=False)
