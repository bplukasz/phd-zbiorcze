from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
ModuleT = TypeVar("ModuleT", nn.Conv2d, nn.Linear)


def setup_nvidia_performance(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _fan_in(weight: Tensor) -> int:
    if weight.ndim == 4:
        return weight.size(1) * weight[0][0].numel()
    if weight.ndim == 2:
        return weight.size(1)
    raise ValueError("Unsupported weight dimensionality.")


def msr_init_(module: ModuleT, activation_gain: float = 1.0) -> ModuleT:
    fan_in = _fan_in(module.weight)
    nn.init.normal_(module.weight, mean=0.0, std=activation_gain / math.sqrt(fan_in))
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


def zero_last_conv_(conv: nn.Conv2d) -> None:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _num_groups(channels: int, group_size: int) -> int:
    if group_size <= 0 or group_size >= channels:
        return 1
    groups = max(1, channels // group_size)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class BiasAct2d(nn.Module):
    negative_slope: float = 0.2
    gain: float = math.sqrt(2.0 / (1.0 + negative_slope ** 2))

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x + self.bias.to(dtype=x.dtype, device=x.device).view(1, -1)
        else:
            x = x + self.bias.to(dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return F.leaky_relu(x, negative_slope=self.negative_slope, inplace=False)


class Conv2dNoBias(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1, activation_gain: float = 1.0):
        super().__init__()
        self.conv = msr_init_(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            activation_gain=activation_gain,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion_factor: int, group_size: int, total_residual_blocks: int, kernel_size: int = 3) -> None:
        super().__init__()
        expanded_channels = channels * expansion_factor
        groups = _num_groups(expanded_channels, group_size)
        n_linear_layers = 3
        fixup_scale = total_residual_blocks ** (-1.0 / (2 * n_linear_layers - 2))
        init_gain = BiasAct2d.gain * fixup_scale

        self.conv1 = Conv2dNoBias(channels, expanded_channels, 1, activation_gain=init_gain)
        self.act1 = BiasAct2d(expanded_channels)
        self.conv2 = Conv2dNoBias(expanded_channels, expanded_channels, kernel_size, groups=groups, activation_gain=init_gain)
        self.act2 = BiasAct2d(expanded_channels)
        self.conv3 = Conv2dNoBias(expanded_channels, channels, 1, activation_gain=1.0)
        zero_last_conv_(self.conv3.conv)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.conv2(self.act1(y))
        y = self.conv3(self.act2(y))
        return x + y


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "bilinear") -> None:
        super().__init__()
        self.mode = mode
        self.proj = Conv2dNoBias(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        if self.mode == "bilinear":
            return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return F.interpolate(x, scale_factor=2.0, mode=self.mode)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "bilinear") -> None:
        super().__init__()
        self.mode = mode
        self.proj = Conv2dNoBias(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "bilinear":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.proj(x)


class GenerativeBasis(nn.Module):
    def __init__(self, in_dim: int, out_channels: int) -> None:
        super().__init__()
        self.basis = nn.Parameter(torch.empty(out_channels, 4, 4))
        nn.init.normal_(self.basis, mean=0.0, std=1.0)
        self.linear = msr_init_(nn.Linear(in_dim, out_channels, bias=False))

    def forward(self, z: Tensor) -> Tensor:
        return self.basis.unsqueeze(0) * self.linear(z).view(z.size(0), -1, 1, 1)


class DiscriminativeBasis(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.depthwise = msr_init_(nn.Conv2d(in_channels, in_channels, 4, 1, 0, groups=in_channels, bias=False))
        self.linear = msr_init_(nn.Linear(in_channels, out_dim, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.depthwise(x).flatten(1))


class GeneratorStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, expansion_factor: int, group_size: int, total_residual_blocks: int, is_first: bool, resample_mode: str = "bilinear") -> None:
        super().__init__()
        transition = GenerativeBasis(in_channels, out_channels) if is_first else UpsampleLayer(in_channels, out_channels, mode=resample_mode)
        blocks = [ResidualBlock(out_channels, expansion_factor, group_size, total_residual_blocks) for _ in range(num_blocks)]
        self.layers = nn.ModuleList([transition, *blocks])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DiscriminatorStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, expansion_factor: int, group_size: int, total_residual_blocks: int, is_last: bool, resample_mode: str = "bilinear") -> None:
        super().__init__()
        blocks = [ResidualBlock(in_channels, expansion_factor, group_size, total_residual_blocks) for _ in range(num_blocks)]
        transition = DiscriminativeBasis(in_channels, out_channels) if is_last else DownsampleLayer(in_channels, out_channels, mode=resample_mode)
        self.layers = nn.ModuleList([*blocks, transition])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _assert_power_of_two_resolution(img_resolution: int) -> None:
    if img_resolution < 4 or (img_resolution & (img_resolution - 1)) != 0:
        raise ValueError("img_resolution must be a power of two and >= 4")


def build_stage_channels(img_resolution: int, base_channels: int = 96, channel_max: int = 1024) -> List[int]:
    _assert_power_of_two_resolution(img_resolution)
    n_stages = int(math.log2(img_resolution)) - 1
    channels: List[int] = []
    for i in range(n_stages):
        mul = 2 ** max(n_stages - i - 2, 0)
        channels.append(min(channel_max, base_channels * mul))
    if len(channels) >= 2:
        channels[0] = min(channel_max, channels[0] * 2)
    return channels


class R3GANGenerator(nn.Module):
    def __init__(self, z_dim: int, img_resolution: int, stage_channels: Sequence[int], blocks_per_stage: Union[int, Sequence[int]] = 2, expansion_factor: int = 2, group_size: int = 16, cond_dim: int = 0, cond_embed_dim: int = 0, resample_mode: str = "bilinear", out_channels: int = 3) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.img_resolution = img_resolution
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_embed_dim

        if isinstance(blocks_per_stage, int):
            blocks_per_stage_list = [blocks_per_stage] * len(stage_channels)
        else:
            blocks_per_stage_list = list(blocks_per_stage)
        total_blocks = sum(blocks_per_stage_list)

        in_dim = z_dim + (cond_embed_dim if cond_dim > 0 else 0)
        self.cond_embed = msr_init_(nn.Linear(cond_dim, cond_embed_dim, bias=False)) if cond_dim > 0 else None

        self.stages = nn.ModuleList()
        prev_channels = in_dim
        for i, (out_ch, n_blocks) in enumerate(zip(stage_channels, blocks_per_stage_list)):
            self.stages.append(GeneratorStage(prev_channels, out_ch, n_blocks, expansion_factor, group_size, total_blocks, i == 0, resample_mode))
            prev_channels = out_ch

        self.to_rgb = Conv2dNoBias(stage_channels[-1], out_channels, 1)

    def forward(self, z: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        if self.cond_embed is not None:
            if cond is None:
                raise ValueError("Conditional generator expects cond tensor.")
            z = torch.cat([z, self.cond_embed(cond)], dim=1)
        x = z
        for stage in self.stages:
            x = stage(x)
        return self.to_rgb(x)


class R3GANDiscriminator(nn.Module):
    def __init__(self, img_resolution: int, stage_channels: Sequence[int], blocks_per_stage: Union[int, Sequence[int]] = 2, expansion_factor: int = 2, group_size: int = 16, cond_dim: int = 0, cond_embed_dim: int = 128, in_channels: int = 3, resample_mode: str = "bilinear") -> None:
        super().__init__()
        self.img_resolution = img_resolution
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_embed_dim

        if isinstance(blocks_per_stage, int):
            blocks_per_stage_list = [blocks_per_stage] * len(stage_channels)
        else:
            blocks_per_stage_list = list(blocks_per_stage)
        total_blocks = sum(blocks_per_stage_list)

        self.from_rgb = Conv2dNoBias(in_channels, stage_channels[0], 1)
        self.stages = nn.ModuleList()
        for i in range(len(stage_channels) - 1):
            self.stages.append(DiscriminatorStage(stage_channels[i], stage_channels[i + 1], blocks_per_stage_list[i], expansion_factor, group_size, total_blocks, False, resample_mode))
        final_out_dim = cond_embed_dim if cond_dim > 0 else 1
        self.stages.append(DiscriminatorStage(stage_channels[-1], final_out_dim, blocks_per_stage_list[-1], expansion_factor, group_size, total_blocks, True, resample_mode))
        self.cond_embed = msr_init_(nn.Linear(cond_dim, cond_embed_dim, bias=False), activation_gain=1.0 / math.sqrt(cond_embed_dim)) if cond_dim > 0 else None

    def _apply_conditional_projection(self, x: Tensor, cond: Optional[Tensor]) -> Tensor:
        if self.cond_embed is not None:
            if cond is None:
                raise ValueError("Conditional discriminator expects cond tensor.")
            x = (x * self.cond_embed(cond)).sum(dim=1, keepdim=True)
        return x

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        x = self.from_rgb(x)
        for stage in self.stages:
            x = stage(x)
        x = self._apply_conditional_projection(x, cond)
        return x.view(x.size(0))


@dataclass
class R3GANPreset:
    img_resolution: int = 64
    z_dim: int = 256
    base_channels: int = 96
    channel_max: int = 768
    blocks_per_stage: int = 2
    expansion_factor: int = 2
    group_size: int = 16
    cond_dim: int = 0
    cond_embed_dim: int = 128

    def build(self):
        g_channels = build_stage_channels(self.img_resolution, self.base_channels, self.channel_max)
        d_channels = list(reversed(g_channels))
        G = R3GANGenerator(self.z_dim, self.img_resolution, g_channels, self.blocks_per_stage, self.expansion_factor, self.group_size, self.cond_dim, self.cond_embed_dim)
        D = R3GANDiscriminator(self.img_resolution, d_channels, self.blocks_per_stage, self.expansion_factor, self.group_size, self.cond_dim, self.cond_embed_dim)
        return G, D


def zero_centered_gradient_penalty(samples: Tensor, critics: Tensor) -> Tensor:
    (grad,) = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True, retain_graph=True, only_inputs=True)
    return grad.square().sum(dim=(1, 2, 3))


class R3GANLoss:
    def __init__(self, generator: nn.Module, discriminator: nn.Module, gamma: float = 10.0, augment_fn: Optional[Callable[[Tensor], Tensor]] = None) -> None:
        self.G = generator
        self.D = discriminator
        self.gamma = gamma
        self.augment_fn = augment_fn

    def _prep(self, x: Tensor) -> Tensor:
        return self.augment_fn(x) if self.augment_fn is not None else x

    def generator_loss(self, z: Tensor, real: Tensor, cond: Optional[Tensor] = None):
        fake = self.G(z, cond)
        fake_logits = self.D(self._prep(fake), cond)
        real_logits = self.D(self._prep(real.detach()), cond)
        rel = fake_logits - real_logits
        adv = F.softplus(-rel)
        return adv.mean(), {
            "g_loss": float(adv.mean().detach().cpu()),
            "fake_score_mean": float(fake_logits.detach().mean().cpu()),
            "real_score_mean_for_g": float(real_logits.detach().mean().cpu()),
        }

    def discriminator_loss(self, z: Tensor, real: Tensor, cond: Optional[Tensor] = None):
        real = real.detach().requires_grad_(True)
        fake = self.G(z, cond).detach().requires_grad_(True)
        real_logits = self.D(self._prep(real), cond)
        fake_logits = self.D(self._prep(fake), cond)
        r1 = zero_centered_gradient_penalty(real, real_logits)
        r2 = zero_centered_gradient_penalty(fake, fake_logits)
        rel = real_logits - fake_logits
        adv = F.softplus(-rel)
        loss = (adv + (self.gamma / 2.0) * (r1 + r2)).mean()
        return loss, {
            "d_loss": float(loss.detach().cpu()),
            "d_adv": float(adv.mean().detach().cpu()),
            "r1": float(r1.mean().detach().cpu()),
            "r2": float(r2.mean().detach().cpu()),
            "real_score_mean": float(real_logits.detach().mean().cpu()),
            "fake_score_mean": float(fake_logits.detach().mean().cpu()),
        }


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, beta: float = 0.999) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(beta).add_(p, alpha=1.0 - beta)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)


def prepare_condition(labels: Optional[Tensor], num_classes: int, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Optional[Tensor]:
    if num_classes <= 0:
        return None
    if labels is None:
        raise ValueError("Conditional model requires labels.")
    labels = labels.to(device)
    if labels.ndim == 1:
        cond = F.one_hot(labels.long(), num_classes=num_classes).to(dtype=dtype)
    elif labels.ndim == 2 and labels.shape[1] == num_classes:
        cond = labels.to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported label shape: {tuple(labels.shape)}")
    if cond.shape[0] != batch_size:
        raise ValueError("Condition batch size mismatch.")
    return cond


@dataclass
class TrainerConfig:
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.0, 0.99)
    gamma: float = 10.0
    ema_beta: float = 0.999
    use_amp_for_g: bool = True
    use_amp_for_d: bool = False
    amp_dtype: torch.dtype = torch.bfloat16
    channels_last: bool = True
    grad_clip: Optional[float] = None


class R3GANTrainer:
    def __init__(self, G: R3GANGenerator, D: R3GANDiscriminator, num_classes: int = 0, device: Optional[torch.device] = None, train_cfg: Optional[TrainerConfig] = None, augment_fn: Optional[Callable[[Tensor], Tensor]] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = train_cfg or TrainerConfig()
        self.num_classes = num_classes

        setup_nvidia_performance(self.device)

        self.G = G.to(self.device)
        self.D = D.to(self.device)
        if self.device.type == "cuda" and self.cfg.channels_last:
            self.G = self.G.to(memory_format=torch.channels_last)  # type: ignore[call-arg]
            self.D = self.D.to(memory_format=torch.channels_last)  # type: ignore[call-arg]

        self.G_ema = copy.deepcopy(self.G).eval()
        for p in self.G_ema.parameters():
            p.requires_grad_(False)

        self.g_opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr_g, betas=self.cfg.betas)
        self.d_opt = torch.optim.Adam(self.D.parameters(), lr=self.cfg.lr_d, betas=self.cfg.betas)
        self.loss = R3GANLoss(self.G, self.D, gamma=self.cfg.gamma, augment_fn=augment_fn)

    @staticmethod
    def set_requires_grad(module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad_(flag)

    def _move_images(self, x: Tensor) -> Tensor:
        x = x.to(self.device, non_blocking=True)
        if self.device.type == "cuda" and self.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _discriminator_step(self, real_images: Tensor, cond: Optional[Tensor]) -> Dict[str, float]:
        self.set_requires_grad(self.D, True)
        self.set_requires_grad(self.G, False)
        self.d_opt.zero_grad(set_to_none=True)

        batch_size = real_images.shape[0]
        z = torch.randn(batch_size, self.G.z_dim, device=self.device)

        if self.cfg.use_amp_for_d and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.cfg.amp_dtype):
                d_loss, d_metrics = self.loss.discriminator_loss(z, real_images, cond)
        else:
            d_loss, d_metrics = self.loss.discriminator_loss(z, real_images.float(), cond)

        d_loss.backward()
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.cfg.grad_clip)
        self.d_opt.step()

        return d_metrics

    def _generator_step(self, real_images: Tensor, cond: Optional[Tensor]) -> Dict[str, float]:
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, True)
        self.g_opt.zero_grad(set_to_none=True)

        batch_size = real_images.shape[0]
        z = torch.randn(batch_size, self.G.z_dim, device=self.device)
        real_for_g = real_images if (self.cfg.use_amp_for_g and self.device.type == "cuda") else real_images.float()

        if self.cfg.use_amp_for_g and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.cfg.amp_dtype):
                fake = self.G(z, cond)
                fake_logits = self.D(self.loss._prep(fake), cond)
                real_logits = self.D(self.loss._prep(real_for_g.detach()), cond)
                rel = fake_logits - real_logits
                g_adv = F.softplus(-rel).mean()
        else:
            fake = self.G(z, cond)
            fake_logits = self.D(self.loss._prep(fake), cond)
            real_logits = self.D(self.loss._prep(real_for_g.detach()), cond)
            rel = fake_logits - real_logits
            g_adv = F.softplus(-rel).mean()

        g_adv.backward()
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.cfg.grad_clip)
        self.g_opt.step()

        return {
            "g_loss": float(g_adv.detach().cpu()),
            "g_adv": float(g_adv.detach().cpu()),
            "g_reg": 0.0,
            "fake_score_mean": float(fake_logits.detach().mean().cpu()),
            "real_score_mean_for_g": float(real_logits.detach().mean().cpu()),
        }

    def train_step(self, real_images: Tensor, labels: Optional[Tensor] = None) -> Dict[str, float]:
        self.G.train()
        self.D.train()
        real_images = self._move_images(real_images)
        batch_size = real_images.shape[0]
        cond = prepare_condition(labels, self.num_classes, batch_size, self.device)

        d_metrics = self._discriminator_step(real_images, cond)
        g_metrics = self._generator_step(real_images, cond)

        update_ema(self.G_ema, self.G, beta=self.cfg.ema_beta)
        return {**d_metrics, **g_metrics}

    @torch.no_grad()
    def sample(self, num_samples: int, labels: Optional[Tensor] = None, use_ema: bool = True) -> Tensor:
        model = self.G_ema if use_ema else self.G
        model.eval()
        z = torch.randn(num_samples, model.z_dim, device=self.device)
        cond = prepare_condition(labels, self.num_classes, num_samples, self.device) if self.num_classes > 0 else None
        return model(z, cond)


def parse_batch(batch):
    if isinstance(batch, torch.Tensor):
        return batch, None
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    raise ValueError("Batch must be either images tensor or (images, labels).")


def fit(trainer: R3GANTrainer, dataloader: torch.utils.data.DataLoader, epochs: int, log_every: int = 100) -> None:
    step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            images, labels = parse_batch(batch)
            metrics = trainer.train_step(images, labels)
            if step % log_every == 0:
                print(
                    f"epoch={epoch:03d} step={step:06d} "
                    f"d_loss={metrics['d_loss']:.4f} "
                    f"g_loss={metrics['g_loss']:.4f} "
                    f"r1={metrics['r1']:.4f} r2={metrics['r2']:.4f}"
                )
            step += 1


