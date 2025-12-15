"""无监督域翻译GAN（CycleGAN变体）实现."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_padding(kernel_size: int) -> int:
    return (kernel_size - 1) // 2


class ConvBlock1D(nn.Module):
    """一维卷积 + InstanceNorm + 激活的组合。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        padding = _make_padding(kernel_size)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm1d(out_channels),
            activation or nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock1D(nn.Module):
    """上采样 + 拼接跳连后的卷积块。"""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        padding = _make_padding(kernel_size)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels + skip_channels,
                out_channels,
                kernel_size,
                padding=padding,
            ),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


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
        pools: list[nn.Module] = []
        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            downs.append(ConvBlock1D(in_ch, out_ch, kernel_size))
            if i != depth - 1:
                pools.append(nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True))
            in_ch = out_ch
        self.down_blocks = nn.ModuleList(downs)
        self.pool_layers = nn.ModuleList(pools)

        self.bottleneck = ConvBlock1D(
            in_ch,
            in_ch * 2,
            kernel_size=kernel_size,
            activation=nn.ReLU(inplace=True),
        )
        bottleneck_out = in_ch * 2

        ups: list[UpBlock1D] = []
        for i in reversed(range(depth)):
            skip_ch = base_channels * (2**i)
            ups.append(UpBlock1D(bottleneck_out, skip_ch, skip_ch, kernel_size))
            bottleneck_out = skip_ch
        self.up_blocks = nn.ModuleList(ups)

        self.final_conv = nn.Conv1d(bottleneck_out, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)
        x = self.bottleneck(x)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)
        return torch.tanh(self.final_conv(x))


class PatchDiscriminator1D(nn.Module):
    """PatchGAN 风格的一维判别器。"""

    def __init__(self, in_channels: int, base_channels: int = 64) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = base_channels
        layers.append(
            nn.Sequential(
                nn.Conv1d(in_channels, ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )
        in_ch = ch
        for mult in [2, 4]:
            out_ch = base_channels * mult
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm1d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_ch = out_ch
        layers.append(
            nn.Conv1d(in_ch, 1, kernel_size=4, stride=1, padding=1)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class GanConfig:
    """无监督CycleGAN相关的配置."""

    sim_channels: int
    real_channels: int
    sequence_length: int = 256
    base_channels: int = 64
    depth: int = 4
    gen_learning_rate: float = 1e-3
    disc_learning_rate: float = 1e-3
    cycle_loss_weight: float = 0.9
    identity_loss_weight: float = 1.86
    gan_loss_weight: float = 1.0
    betas: Tuple[float, float] = (0.5, 0.999)
    device: str = "cuda"
    sim_channel_scale: Optional[List[float]] = None
    real_channel_scale: Optional[List[float]] = None


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
        self.disc_real = PatchDiscriminator1D(config.real_channels, config.base_channels)
        self.disc_sim = PatchDiscriminator1D(config.sim_channels, config.base_channels)

        self.to(device)
        self.real_scale = self._prepare_scale_tensor(config.real_channel_scale, config.real_channels)
        self.sim_scale = self._prepare_scale_tensor(config.sim_channel_scale, config.sim_channels)

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

    def _prepare_scale_tensor(self, values: Optional[List[float]], channels: int) -> Optional[torch.Tensor]:
        if values is None:
            return None
        if len(values) != channels:
            raise ValueError("channel scale长度与通道数不匹配")
        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        tensor = torch.clamp(tensor, min=1e-6)
        return tensor.view(1, channels, 1)

    def _normalized_mse(self, prediction: torch.Tensor, target: torch.Tensor, domain: str) -> torch.Tensor:
        diff = prediction - target
        scale = self.real_scale if domain == "real" else self.sim_scale
        if scale is None:
            return (diff ** 2).mean()
        return (((diff ** 2) / scale).mean())*0.5

    def set_requires_grad(self, modules: Iterable[nn.Module], flag: bool) -> None:
        for module in modules:
            for param in module.parameters():
                param.requires_grad = flag

    def _generator_losses(
        self, real_inputs: torch.Tensor, sim_inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_real = self.sim2real(sim_inputs)
        fake_sim = self.real2sim(real_inputs)

        adv_real = torch.mean((self.disc_real(fake_real) - 1) ** 2)
        adv_sim = torch.mean((self.disc_sim(fake_sim) - 1) ** 2)
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
        total_loss = (
            self.config.gan_loss_weight * adv_loss
            + self.config.cycle_loss_weight * cycle_loss
            + self.config.identity_loss_weight * identity_loss
        )
        metrics = {
            "adv_loss": adv_loss.detach(),
            "cycle_loss": cycle_loss.detach(),
            "identity_loss": identity_loss.detach(),
            "gen_total": total_loss.detach(),
        }
        return total_loss, metrics

    def _discriminator_loss(
        self, disc: PatchDiscriminator1D, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        pred_real = disc(real_data)
        pred_fake = disc(fake_data.detach())
        return 0.5 * (
            torch.mean((pred_real - 1) ** 2) + torch.mean(pred_fake**2)
        )

    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """执行一次生成器 + 判别器的优化."""
        real_inputs: torch.Tensor = batch["real"].to(self.device)
        sim_inputs: torch.Tensor = batch["sim"].to(self.device)

        self.set_requires_grad([self.disc_real, self.disc_sim], False)
        self.gen_optimizer.zero_grad()
        gen_loss, gen_metrics = self._generator_losses(real_inputs, sim_inputs)
        gen_loss.backward()
        self.gen_optimizer.step()

        self.set_requires_grad([self.disc_real, self.disc_sim], True)
        self.disc_optimizer.zero_grad()
        fake_real = self.sim2real(sim_inputs).detach()
        fake_sim = self.real2sim(real_inputs).detach()
        disc_real_loss = self._discriminator_loss(self.disc_real, real_inputs, fake_real)
        disc_sim_loss = self._discriminator_loss(self.disc_sim, sim_inputs, fake_sim)
        disc_loss = disc_real_loss + disc_sim_loss
        disc_loss.backward()
        self.disc_optimizer.step()

        metrics = {
            "gen_total": gen_metrics["gen_total"].item(),
            "adv_loss": gen_metrics["adv_loss"].item(),
            "cycle_loss": gen_metrics["cycle_loss"].item(),
            "identity_loss": gen_metrics["identity_loss"].item(),
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
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.sim2real.load_state_dict(state["sim2real"])
        self.real2sim.load_state_dict(state["real2sim"])
        self.disc_real.load_state_dict(state["disc_real"])
        self.disc_sim.load_state_dict(state["disc_sim"])
        self.gen_optimizer.load_state_dict(state["gen_opt"])
        self.disc_optimizer.load_state_dict(state["disc_opt"])
