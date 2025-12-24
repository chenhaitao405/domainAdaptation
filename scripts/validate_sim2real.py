"""针对单个trial的sim->real生成结果进行校验与导出."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from domain_adaptation.config import DatasetConfig
from domain_adaptation.data.sensor_dataset import SensorDataset
from domain_adaptation.models.gan import DomainAdaptationGAN, GanConfig
from domain_adaptation.utils import (
    load_trial_tensor,
    get_channel_stats_tensors,
    denormalize_sequence,
    translate_sim_to_real,
    compute_channel_mse,
    save_sequence_to_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sim->Real 生成结果校验")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的GAN checkpoint路径")
    parser.add_argument("--trial", type=str, required=True, help="目标trial（例如 BT01/ball_toss_1_1_center_on）")
    parser.add_argument("--data-dirs", nargs="+", default=DatasetConfig().data_dirs, help="数据根目录列表")
    parser.add_argument("--side", type=str, default="r", choices=["l", "r"], help="选择的侧别")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--save", action="store_true", help="是否保存生成序列为 *_exo_s2r.csv")
    return parser.parse_args()


def _resolve_trial_dir(data_dirs, trial_name: str) -> Tuple[str, str]:
    for root in data_dirs:
        trial_dir = os.path.join(root, trial_name)
        if os.path.isdir(trial_dir):
            return root, trial_dir
    raise FileNotFoundError(f"Trial '{trial_name}' not found under any provided data_dir.")


def _build_dataset(
    data_dir: str,
    dataset_cfg: DatasetConfig,
    side: str,
    device: torch.device,
    input_suffix: str,
    label_suffix: str,
) -> SensorDataset:
    return SensorDataset(
        data_dir=data_dir,
        input_names=[name.replace("*", side) for name in dataset_cfg.input_names],
        label_names=[name.replace("*", side) for name in dataset_cfg.label_names],
        side=side,
        participant_masses=dataset_cfg.participant_masses,
        action_patterns=None,
        device=device,
        input_file_suffix=input_suffix,
        label_file_suffix=label_suffix,
    )


def _load_model(checkpoint_path: str, device: torch.device) -> DomainAdaptationGAN:
    state = torch.load(checkpoint_path, map_location=device)
    cfg_dict = dict(state["config"])
    cfg_dict["device"] = device.type if device.type != "cuda" else "cuda"
    config = GanConfig(**cfg_dict)
    model = DomainAdaptationGAN(config)
    model.sim2real.load_state_dict(state["sim2real"])
    return model


def _find_real_file(trial_dir: str) -> str:
    for fname in os.listdir(trial_dir):
        lower = fname.lower()
        if lower.endswith("_exo.csv") and not lower.endswith("power_exo.csv"):
            return os.path.join(trial_dir, fname)
    raise FileNotFoundError(f"Cannot find *_exo.csv in {trial_dir}")


def main() -> None:
    args = parse_args()
    dataset_cfg = DatasetConfig(data_dirs=args.data_dirs, side=args.side)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cpu = torch.device("cpu")

    data_root, trial_dir = _resolve_trial_dir(args.data_dirs, args.trial)
    channel_names = [name.replace("*", args.side) for name in dataset_cfg.input_names]

    real_dataset = _build_dataset(data_root, dataset_cfg, args.side, cpu, "_exo.csv", "_moment_filt.csv")
    sim_dataset = _build_dataset(data_root, dataset_cfg, args.side, cpu, "_exo_sim.csv", "_moment_filt_bio.csv")

    real_tensor, real_len = load_trial_tensor(real_dataset, args.trial, device)
    sim_tensor, sim_len = load_trial_tensor(sim_dataset, args.trial, device)
    valid_len = min(real_len, sim_len)

    mean_tensor, std_tensor = get_channel_stats_tensors(real_dataset, device)

    model = _load_model(args.checkpoint, device)

    generated = translate_sim_to_real(model, sim_tensor)

    gen_denorm = denormalize_sequence(generated, mean_tensor, std_tensor)
    real_denorm = denormalize_sequence(real_tensor, mean_tensor, std_tensor)

    channel_mse, overall = compute_channel_mse(gen_denorm, real_denorm, valid_len)

    print(f"Trial: {args.trial}")
    for name, mse in zip(channel_names, channel_mse.tolist()):
        print(f"  {name}: {mse:.6f}")
    print(f"Mean MSE: {overall:.6f}")

    if args.save:
        real_file = _find_real_file(trial_dir)
        if real_file.endswith("_exo.csv"):
            save_path = real_file[:-8] + "_exo_s2r.csv"
        else:
            save_path = real_file + "_s2r.csv"
        save_sequence_to_csv(gen_denorm, valid_len, channel_names, save_path)
        print(f"Saved generated sequence to: {save_path}")


if __name__ == "__main__":
    main()
