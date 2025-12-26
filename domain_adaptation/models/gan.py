"""无监督域翻译GAN（CycleGAN变体）实现."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parents[2]
TCN_ROOT = PROJECT_DIR.parent / "TCN"
if TCN_ROOT.exists() and str(TCN_ROOT) not in sys.path:
    sys.path.append(str(TCN_ROOT))

try:
    from utils.tcn import TCN
except ImportError:
    TCN = None


def _make_padding(kernel_size: int) -> int:
    return (kernel_size - 1) // 2


class ConvBlock1D(nn.Module):
    """一维卷积 + ReLU 的双卷积块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = _make_padding(kernel_size)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x


class UpBlock1D(nn.Module):
    """线性上采样 + 拼接跳连后的双卷积块。"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = _make_padding(kernel_size)
        self.conv1 = nn.Conv1d(
            in_channels + skip_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x


class UNet1D(nn.Module):
    """用于传感器序列翻译的1D U-Net。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.depth = depth
        downs: list[nn.Module] = []
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            downs.append(ConvBlock1D(in_ch, out_ch, kernel_size))
            in_ch = out_ch
        self.down_blocks = nn.ModuleList(downs)

        self.bottleneck = ConvBlock1D(
            in_ch,
            in_ch * 2,
            kernel_size=kernel_size,
        )
        bottleneck_out = in_ch * 2

        ups: list[UpBlock1D] = []
        for i in reversed(range(depth)):
            skip_ch = base_channels * (2**i)
            ups.append(UpBlock1D(bottleneck_out, skip_ch, skip_ch, kernel_size))
            bottleneck_out = skip_ch
        self.up_blocks = nn.ModuleList(ups)
        self.final_dropout = nn.Dropout(p=0.5)
        self.final_conv = nn.Conv1d(bottleneck_out, out_channels, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            skips.append(x)
            if i < self.depth - 1:
                x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bottleneck(x)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)
        x = self.final_dropout(x)
        x = self.final_conv(x)
        return 3.0 * self.final_activation(x)


class AdaptNetDiscriminator1D(nn.Module):
    """AdaptNet风格的一维判别器."""

    def __init__(
        self,
        in_channels: int,
        feature_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        num_domains: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        feature_dims = feature_dims or [32, 64, 128, 256, 512, 1024]
        convs: list[nn.Module] = []
        current = in_channels
        for out_channels in feature_dims:
            convs.append(
                nn.Sequential(
                    nn.Conv1d(current, out_channels, kernel_size=kernel_size, stride=2, padding=0),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout),
                )
            )
            current = out_channels
        self.feature_extractor = nn.Sequential(*convs)
        self.num_domains = num_domains
        self.fc = nn.Linear(current + num_domains, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, domain_onehot: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = torch.mean(features, dim=-1)
        concat = torch.cat([features, domain_onehot], dim=1)
        logits = self.fc(concat)
        return self.activation(logits)


@dataclass
class GanConfig:
    """无监督CycleGAN相关的配置."""

    sim_channels: int
    real_channels: int
    label_channels: int = 0
    sequence_length: int = 256
    base_channels: int = 32
    depth: int = 4
    gen_learning_rate: float = 1e-3
    disc_learning_rate: float = 1e-3
    cycle_loss_weight: float = 0.9
    identity_loss_weight: float = 1.86
    gan_loss_weight: float = 1.0
    betas: Tuple[float, float] = (0.5, 0.999)
    device: str = "cuda"
    sim_modal_weights: Optional[List[float]] = None
    real_modal_weights: Optional[List[float]] = None
    lambda_moment: float = 0.0
    moment_start_epoch: int = 20
    tcn_num_channels: Tuple[int, ...] = (64, 64, 64, 64)
    tcn_kernel_size: int = 5
    tcn_dropout: float = 0.2
    tcn_learning_rate: float = 1e-3
    tcn_eff_hist: int = 248
    tcn_load_path: Optional[str] = None
    tcn_freeze: bool = False


class Sim2RealTranslator(UNet1D):
    """模拟 -> 真实传感器的生成器."""

    def __init__(self, config: GanConfig) -> None:
        super().__init__(
            in_channels=config.sim_channels,
            out_channels=config.real_channels,
            base_channels=config.base_channels,
            depth=config.depth,
        )


class Real2SimTranslator(UNet1D):
    """真实 -> 模拟传感器的生成器."""

    def __init__(self, config: GanConfig) -> None:
        super().__init__(
            in_channels=config.real_channels,
            out_channels=config.sim_channels,
            base_channels=config.base_channels,
            depth=config.depth,
        )


class DomainAdaptationGAN(nn.Module):
    """封装CycleGAN式训练流程."""

    def __init__(self, config: GanConfig) -> None:
        super().__init__()
        self.config = config
        device = (
            torch.device(config.device)
            if torch.cuda.is_available() or config.device == "cpu"
            else torch.device("cpu")
        )
        self.device = device

        self.sim2real = Sim2RealTranslator(config)
        self.real2sim = Real2SimTranslator(config)
        self.disc_real = AdaptNetDiscriminator1D(config.real_channels)
        self.disc_sim = AdaptNetDiscriminator1D(config.sim_channels)

        self.to(device)
        self.real_weights = self._prepare_weight_tensor(config.real_modal_weights, config.real_channels)
        self.sim_weights = self._prepare_weight_tensor(config.sim_modal_weights, config.sim_channels)
        self.current_epoch = 0
        self.tcn_freeze = config.tcn_freeze
        self.moment_estimator: Optional[TCN] = None
        self.moment_optimizer: Optional[torch.optim.Optimizer] = None
        self._init_moment_estimator()

        gen_params = list(self.sim2real.parameters()) + list(self.real2sim.parameters())
        disc_params = list(self.disc_real.parameters()) + list(self.disc_sim.parameters())
        self.gen_optimizer = torch.optim.Adam(
            gen_params,
            lr=config.gen_learning_rate,
            betas=config.betas,
        )
        self.disc_optimizer = torch.optim.Adam(
            disc_params,
            lr=config.disc_learning_rate,
            betas=config.betas,
        )
        if self.moment_estimator is not None and not config.tcn_freeze:
            self.moment_optimizer = torch.optim.Adam(
                self.moment_estimator.parameters(),
                lr=config.tcn_learning_rate,
                betas=config.betas,
            )

    def _prepare_weight_tensor(self, values: Optional[List[float]], channels: int) -> Optional[torch.Tensor]:
        if values is None:
            return None
        if len(values) != channels:
            raise ValueError("modal weight长度与通道数不匹配")
        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        tensor = torch.clamp(tensor, min=1e-3)
        return tensor.view(1, channels, 1)

    def _normalized_mse(self, prediction: torch.Tensor, target: torch.Tensor, domain: str) -> torch.Tensor:
        diff = prediction - target
        weights = self.real_weights if domain == "real" else self.sim_weights
        loss = diff ** 2
        if weights is not None:
            loss = loss * weights
        return loss.mean()

    def _init_moment_estimator(self) -> None:
        if self.config.lambda_moment <= 0 or self.config.label_channels == 0:
            return
        if TCN is None:
            raise ImportError("未找到TCN模块，无法构建力矩估计器")
        estimator = TCN(
            input_size=self.config.real_channels,
            output_size=self.config.label_channels,
            num_channels=list(self.config.tcn_num_channels),
            ksize=self.config.tcn_kernel_size,
            dropout=self.config.tcn_dropout,
            eff_hist=self.config.tcn_eff_hist,
            spatial_dropout=False,
            activation="ReLU",
            norm="weight_norm",
            center=0.0,
            scale=1.0,
        )
        if self.config.tcn_load_path:
            state = torch.load(self.config.tcn_load_path, map_location=self.device)
            state_dict = state.get("state_dict", state)
            estimator.load_state_dict(state_dict, strict=False)
        estimator = estimator.to(self.device)
        if self.config.tcn_freeze:
            for param in estimator.parameters():
                param.requires_grad = False
        self.moment_estimator = estimator

    def set_current_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def _should_use_moment_loss(self, sim_labels: Optional[torch.Tensor]) -> bool:
        return (
            self.config.lambda_moment > 0
            and self.moment_estimator is not None
            and sim_labels is not None
            and self.current_epoch >= self.config.moment_start_epoch
        )

    def _tcn_model_info(self) -> Optional[Dict[str, Any]]:
        if self.moment_estimator is None:
            return None
        center = torch.zeros(1, self.config.real_channels, 1, dtype=torch.float32)
        scale = torch.ones(1, self.config.real_channels, 1, dtype=torch.float32)
        return {
            "input_size": self.config.real_channels,
            "output_size": self.config.label_channels,
            "num_channels": list(self.config.tcn_num_channels),
            "ksize": self.config.tcn_kernel_size,
            "dropout": self.config.tcn_dropout,
            "eff_hist": self.config.tcn_eff_hist,
            "spatial_dropout": False,
            "activation": "ReLU",
            "norm": "weight_norm",
            "center": center,
            "scale": scale,
        }

    def _build_tcn_checkpoint(self, epoch: int, loss: Optional[float]):
        if self.moment_estimator is None:
            return None
        model_info = self._tcn_model_info()
        if model_info is None:
            return None
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.moment_estimator.state_dict(),
            "optimizer_state_dict": (
                self.moment_optimizer.state_dict() if self.moment_optimizer is not None else None
            ),
            "loss": loss,
        }
        checkpoint.update(model_info)
        return checkpoint

    def save_tcn_checkpoint(self, path: str, epoch: int, loss: Optional[float]) -> None:
        checkpoint = self._build_tcn_checkpoint(epoch, loss)
        if checkpoint is None:
            return
        torch.save(checkpoint, path)

    def set_requires_grad(self, modules: Iterable[nn.Module], flag: bool) -> None:
        for module in modules:
            for param in module.parameters():
                param.requires_grad = flag

    def _domain_onehot(self, batch_size: int, domain_id: int) -> torch.Tensor:
        num_classes = getattr(self.disc_real, "num_domains", 2)
        onehot = torch.zeros(batch_size, num_classes, device=self.device, dtype=torch.float32)
        idx = max(0, min(num_classes - 1, domain_id))
        onehot[:, idx] = 1.0
        return onehot

    def _generator_losses(
        self,
        real_inputs: torch.Tensor,
        sim_inputs: torch.Tensor,
        sim_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_real = self.sim2real(sim_inputs)
        fake_sim = self.real2sim(real_inputs)

        real_domain_onehot = self._domain_onehot(fake_real.size(0), 1)
        sim_domain_onehot = self._domain_onehot(fake_sim.size(0), 0)
        adv_real = torch.mean((self.disc_real(fake_real, real_domain_onehot) - 1) ** 2)
        adv_sim = torch.mean((self.disc_sim(fake_sim, sim_domain_onehot) - 1) ** 2)
        adv_loss = (adv_real + adv_sim)*0.5

        cycle_sim = self.real2sim(fake_real)
        cycle_real = self.sim2real(fake_sim)
        cycle_loss = self._normalized_mse(cycle_sim, sim_inputs, "sim") + self._normalized_mse(
            cycle_real, real_inputs, "real"
        )

        identity_loss = torch.tensor(0.0, device=self.device)
        if real_inputs.shape[1] == self.config.sim_channels:
            identity_loss = identity_loss + self._normalized_mse(
                self.sim2real(real_inputs), real_inputs, "real"
            )
        if sim_inputs.shape[1] == self.config.real_channels:
            identity_loss = identity_loss + self._normalized_mse(
                self.real2sim(sim_inputs), sim_inputs, "sim"
            )
        identity_loss = identity_loss*0.5
        moment_loss = torch.tensor(0.0, device=self.device)
        if self._should_use_moment_loss(sim_labels):
            if not self.tcn_freeze:
                self.moment_estimator.train()
            else:
                self.moment_estimator.eval()
            target = sim_labels.to(self.device)
            moment_pred = self.moment_estimator(fake_real)
            moment_loss = F.mse_loss(moment_pred, target)
        total_loss = (
            self.config.gan_loss_weight * adv_loss
            + self.config.cycle_loss_weight * cycle_loss
            + self.config.identity_loss_weight * identity_loss
            + self.config.lambda_moment * moment_loss
        )
        metrics = {
            "adv_loss": adv_loss.detach(),
            "cycle_loss": cycle_loss.detach(),
            "identity_loss": identity_loss.detach(),
            "moment_loss": moment_loss.detach(),
            "gen_total": total_loss.detach(),
        }
        return total_loss, metrics

    def _discriminator_loss(
        self,
        disc: AdaptNetDiscriminator1D,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        real_domain_id: int,
        fake_domain_id: int,
    ) -> torch.Tensor:
        real_labels = self._domain_onehot(real_data.size(0), real_domain_id)
        fake_labels = self._domain_onehot(fake_data.size(0), fake_domain_id)
        pred_real = disc(real_data, real_labels)
        pred_fake = disc(fake_data.detach(), fake_labels)
        return 0.5 * (
            torch.mean((pred_real - 1) ** 2) + torch.mean(pred_fake**2)
        )

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一次生成器 + 判别器的优化."""
        real_inputs: torch.Tensor = batch["real"].to(self.device)
        sim_inputs: torch.Tensor = batch["sim"].to(self.device)
        sim_labels: Optional[torch.Tensor] = batch.get("sim_labels")
        use_moment = self._should_use_moment_loss(sim_labels)
        self.set_requires_grad([self.disc_real, self.disc_sim], False)
        if use_moment and self.moment_optimizer is not None:
            self.moment_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        gen_loss, gen_metrics = self._generator_losses(real_inputs, sim_inputs, sim_labels)
        gen_loss.backward()
        self.gen_optimizer.step()
        if use_moment and self.moment_optimizer is not None:
            self.moment_optimizer.step()

        self.set_requires_grad([self.disc_real, self.disc_sim], True)
        self.disc_optimizer.zero_grad()
        fake_real = self.sim2real(sim_inputs).detach()
        fake_sim = self.real2sim(real_inputs).detach()
        disc_real_loss = self._discriminator_loss(self.disc_real, real_inputs, fake_real, 1, 0)
        disc_sim_loss = self._discriminator_loss(self.disc_sim, sim_inputs, fake_sim, 0, 1)
        disc_loss = disc_real_loss + disc_sim_loss
        disc_loss.backward()
        self.disc_optimizer.step()

        metrics = {
            "gen_total": gen_metrics["gen_total"].item(),
            "adv_loss": gen_metrics["adv_loss"].item(),
            "cycle_loss": gen_metrics["cycle_loss"].item(),
            "identity_loss": gen_metrics["identity_loss"].item(),
            "moment_loss": gen_metrics["moment_loss"].item(),
            "disc_loss": disc_loss.detach().item(),
        }
        return metrics

    def save_checkpoint(self, path: str) -> None:
        state = {
            "config": self.config.__dict__,
            "sim2real": self.sim2real.state_dict(),
            "real2sim": self.real2sim.state_dict(),
            "disc_real": self.disc_real.state_dict(),
            "disc_sim": self.disc_sim.state_dict(),
            "gen_opt": self.gen_optimizer.state_dict(),
            "disc_opt": self.disc_optimizer.state_dict(),
            "current_epoch": self.current_epoch,
        }
        if self.moment_estimator is not None:
            state["moment_estimator"] = self.moment_estimator.state_dict()
        if self.moment_optimizer is not None:
            state["moment_opt"] = self.moment_optimizer.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.sim2real.load_state_dict(state["sim2real"])
        self.real2sim.load_state_dict(state["real2sim"])
        self.disc_real.load_state_dict(state["disc_real"])
        self.disc_sim.load_state_dict(state["disc_sim"])
        self.gen_optimizer.load_state_dict(state["gen_opt"])
        self.disc_optimizer.load_state_dict(state["disc_opt"])
        self.current_epoch = state.get("current_epoch", 0)
        if self.moment_estimator is not None and "moment_estimator" in state:
            self.moment_estimator.load_state_dict(state["moment_estimator"])
        if self.moment_optimizer is not None and "moment_opt" in state:
            self.moment_optimizer.load_state_dict(state["moment_opt"])
