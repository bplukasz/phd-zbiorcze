from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
BatchType = Union[Tensor, Sequence[Any], Dict[str, Any]]
SampleFn = Callable[[int], Tensor]


class DependencyError(RuntimeError):
    """Raised when an optional dependency required by the metric suite is missing."""


@dataclass
class GANMetricsConfig:
    """Configuration for GAN evaluation.

    The defaults are aimed at a single modern NVIDIA GPU such as GB10 / GB20:
    - FID/KID on up to 50k samples
    - PR on up to 10k features (to keep pairwise kNN tractable)
    - LPIPS diversity on a sampled pool of fake images
    """

    device: str = "cuda"
    input_range: Literal["minus_one_to_one", "zero_one", "uint8"] = "minus_one_to_one"

    # FID / KID
    fid_feature: int = 2048
    kid_feature: int = 2048
    kid_subsets: int = 100
    kid_subset_size: int = 1000
    max_real_images_fid_kid: Optional[int] = 50_000
    reset_real_features: bool = False

    # Precision / Recall (k-NN manifold metric)
    pr_num_samples: int = 10_000
    pr_k: int = 3
    pr_chunk_size: int = 1024
    pr_feature_device: Optional[str] = None

    # LPIPS diversity
    lpips_net_type: Literal["alex", "vgg", "squeeze"] = "alex"
    lpips_num_pairs: int = 2048
    lpips_pool_size: int = 4096
    lpips_pair_batch_size: int = 64

    # Performance
    use_amp_for_feature_extractor: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    use_channels_last: bool = True

    # Misc
    seed: int = 42
    verbose: bool = True


class TorchvisionInceptionPool3(nn.Module):
    """Pool3 feature extractor for PR metric.

    Note:
        This uses torchvision's ImageNet-pretrained InceptionV3 pool features.
        FID/KID are still delegated to torchmetrics, which internally use the
        default extractor expected by those metrics. PR therefore uses a feature
        space that is *very standard*, but not guaranteed to be byte-identical to
        torch-fidelity / TensorFlow FID features.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import inception_v3
            from torchvision.models.inception import Inception_V3_Weights
        except Exception as exc:  # pragma: no cover - depends on env
            raise DependencyError(
                "torchvision is required for the PR feature extractor. "
                "Install a torchvision build matching your PyTorch/CUDA stack."
            ) from exc

        # Newer torchvision requires aux_logits=True when pretrained weights are used.
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.fc = nn.Identity()
        model.dropout = nn.Identity()
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, images_01: Tensor) -> Tensor:
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError(f"Expected images in shape (N, 3, H, W), got {tuple(images_01.shape)}")
        x = F.interpolate(images_01.float(), size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        x = self.model(x)
        if hasattr(x, "logits"):
            x = x.logits
        elif isinstance(x, tuple):
            x = x[0]
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return x.float()


class GANMetricsSuite:
    """GAN evaluation suite for FID, KID, Precision, Recall and LPIPS diversity.

    Typical usage:
        suite = GANMetricsSuite(config)
        suite.prepare_real(real_loader)
        metrics = suite.evaluate_generator(sample_fn, num_fake_images=50_000, fake_batch_size=256)

    Integration assumptions:
        - real_loader yields either images or (images, labels)
        - sample_fn(batch_size) returns fake images with the configured input range
        - the same suite instance is reused across evaluations so real stats stay cached
    """

    def __init__(self, config: GANMetricsConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self.pr_device = torch.device(config.pr_feature_device or config.device)
        self._rng = random.Random(config.seed)

        self.fid, self.kid = self._build_torchmetrics_metrics()
        self.pr_extractor = TorchvisionInceptionPool3().to(self.pr_device)
        self.pr_extractor.eval()
        self.lpips_metric = self._build_lpips_metric()

        self._real_prepared = False
        self._real_feature_cache: Optional[Tensor] = None
        self._real_image_count_fid_kid = 0
        self._last_fake_feature_cache: Optional[Tensor] = None
        self._last_fake_lpips_pool: Optional[Tensor] = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def prepare_real(self, real_loader: Iterable[BatchType], *, force_recompute: bool = False) -> None:
        """Populate and cache real statistics / features.

        This should normally be called once per dataset. FID/KID real features are
        cached inside torchmetrics as long as `reset_real_features=False`.
        """
        if self._real_prepared and not force_recompute:
            return

        if force_recompute:
            self.fid, self.kid = self._build_torchmetrics_metrics()
            self._real_feature_cache = None
            self._real_image_count_fid_kid = 0
            self._real_prepared = False

        pr_features: List[Tensor] = []
        collected_pr = 0
        total_for_fid_kid = 0

        for batch in real_loader:
            images = self._extract_images(batch)
            if not torch.is_tensor(images):
                raise TypeError("Could not extract image tensor from real_loader batch.")

            remaining_fid = None
            if self.cfg.max_real_images_fid_kid is not None:
                remaining_fid = self.cfg.max_real_images_fid_kid - total_for_fid_kid
                if remaining_fid <= 0 and collected_pr >= self.cfg.pr_num_samples:
                    break

            if remaining_fid is None:
                images_for_fid = images
            else:
                images_for_fid = images[:remaining_fid]

            if images_for_fid.numel() > 0:
                imgs_01 = self._to_zero_one(images_for_fid).to(self.device, non_blocking=True)
                if self.cfg.use_channels_last and imgs_01.ndim == 4:
                    imgs_01 = imgs_01.contiguous(memory_format=torch.channels_last)
                self.fid.update(imgs_01, real=True)
                self.kid.update(imgs_01, real=True)
                total_for_fid_kid += int(imgs_01.shape[0])

            if collected_pr < self.cfg.pr_num_samples:
                take = min(self.cfg.pr_num_samples - collected_pr, int(images.shape[0]))
                if take > 0:
                    feats = self._extract_pr_features(images[:take])
                    pr_features.append(feats.cpu())
                    collected_pr += take

            if (
                (self.cfg.max_real_images_fid_kid is not None and total_for_fid_kid >= self.cfg.max_real_images_fid_kid)
                and collected_pr >= self.cfg.pr_num_samples
            ):
                break

        if total_for_fid_kid == 0:
            raise ValueError("No real images were seen while preparing real metrics.")
        if collected_pr == 0:
            raise ValueError("No real features were collected for the PR metric.")

        self._real_feature_cache = torch.cat(pr_features, dim=0).contiguous()
        self._real_image_count_fid_kid = total_for_fid_kid
        self._real_prepared = True

    @torch.no_grad()
    def evaluate_generator(
        self,
        sample_fn: SampleFn,
        *,
        num_fake_images: int = 50_000,
        fake_batch_size: int = 256,
    ) -> Dict[str, float]:
        """Evaluate a generator via a `sample_fn(batch_size) -> images` callback."""
        self._require_real_prepared()
        self._reset_fake_states()

        fake_features: List[Tensor] = []
        fake_images_for_lpips: List[Tensor] = []
        generated = 0

        while generated < num_fake_images:
            cur_bs = min(fake_batch_size, num_fake_images - generated)
            fake = sample_fn(cur_bs)
            if not torch.is_tensor(fake):
                raise TypeError("sample_fn must return a torch.Tensor of images.")
            if fake.shape[0] != cur_bs:
                raise ValueError(f"sample_fn returned batch of size {fake.shape[0]}, expected {cur_bs}.")

            self._update_fake_metrics(fake)

            if self._current_feature_count(fake_features) < self.cfg.pr_num_samples:
                take = min(self.cfg.pr_num_samples - self._current_feature_count(fake_features), int(fake.shape[0]))
                feats = self._extract_pr_features(fake[:take])
                fake_features.append(feats.cpu())

            if self._current_image_count(fake_images_for_lpips) < self.cfg.lpips_pool_size:
                take = min(self.cfg.lpips_pool_size - self._current_image_count(fake_images_for_lpips), int(fake.shape[0]))
                fake_images_for_lpips.append(fake[:take].detach().cpu())

            generated += cur_bs

        return self._finalize_metrics(fake_features, fake_images_for_lpips)

    @torch.no_grad()
    def evaluate_fake_loader(
        self,
        fake_loader: Iterable[BatchType],
        *,
        max_fake_images: int = 50_000,
    ) -> Dict[str, float]:
        """Evaluate metrics from a loader of already-generated fake images."""
        self._require_real_prepared()
        self._reset_fake_states()

        fake_features: List[Tensor] = []
        fake_images_for_lpips: List[Tensor] = []
        seen = 0

        for batch in fake_loader:
            images = self._extract_images(batch)
            if not torch.is_tensor(images):
                raise TypeError("Could not extract image tensor from fake_loader batch.")
            if seen >= max_fake_images:
                break
            images = images[: max_fake_images - seen]
            if images.numel() == 0:
                continue

            self._update_fake_metrics(images)

            if self._current_feature_count(fake_features) < self.cfg.pr_num_samples:
                take = min(self.cfg.pr_num_samples - self._current_feature_count(fake_features), int(images.shape[0]))
                fake_features.append(self._extract_pr_features(images[:take]).cpu())

            if self._current_image_count(fake_images_for_lpips) < self.cfg.lpips_pool_size:
                take = min(self.cfg.lpips_pool_size - self._current_image_count(fake_images_for_lpips), int(images.shape[0]))
                fake_images_for_lpips.append(images[:take].detach().cpu())

            seen += int(images.shape[0])
            if seen >= max_fake_images:
                break

        if seen == 0:
            raise ValueError("No fake images were seen while evaluating fake_loader.")

        return self._finalize_metrics(fake_features, fake_images_for_lpips)

    # ---------------------------------------------------------------------
    # Builders
    # ---------------------------------------------------------------------
    def _build_torchmetrics_metrics(self):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance
        except Exception as exc:  # pragma: no cover - depends on env
            raise DependencyError(
                "torchmetrics is required for FID/KID. Install e.g. "
                "`pip install torchmetrics[image] torch-fidelity torchvision`."
            ) from exc

        fid = FrechetInceptionDistance(
            feature=self.cfg.fid_feature,
            reset_real_features=self.cfg.reset_real_features,
            normalize=True,
        ).to(self.device)

        kid = KernelInceptionDistance(
            feature=self.cfg.kid_feature,
            subsets=self.cfg.kid_subsets,
            subset_size=self.cfg.kid_subset_size,
            reset_real_features=self.cfg.reset_real_features,
            normalize=True,
        ).to(self.device)
        return fid, kid

    def _build_lpips_metric(self):
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except Exception as exc:  # pragma: no cover - depends on env
            raise DependencyError(
                "torchmetrics[image] is required for LPIPS. Install e.g. "
                "`pip install torchmetrics[image] torchvision`."
            ) from exc

        metric = LearnedPerceptualImagePatchSimilarity(
            net_type=self.cfg.lpips_net_type,
            reduction="mean",
            normalize=False,
        ).to(self.device)
        metric.eval()
        return metric

    # ---------------------------------------------------------------------
    # Finalization
    # ---------------------------------------------------------------------
    def _finalize_metrics(self, fake_features: List[Tensor], fake_images_for_lpips: List[Tensor]) -> Dict[str, float]:
        if not fake_features:
            raise ValueError("No fake features were collected for PR.")
        if not fake_images_for_lpips:
            raise ValueError("No fake images were collected for LPIPS diversity.")

        fake_feature_cache = torch.cat(fake_features, dim=0).contiguous()
        fake_lpips_pool = torch.cat(fake_images_for_lpips, dim=0).contiguous()

        real_feature_cache = self._real_feature_cache
        if real_feature_cache is None:
            raise RuntimeError("Real metrics are not prepared. Call prepare_real(real_loader) first.")

        fid_val = float(self.fid.compute().item())
        kid_mean, kid_std = self.kid.compute()
        precision, recall = self._compute_precision_recall(real_feature_cache, fake_feature_cache)
        lpips_diversity = self._compute_lpips_diversity(fake_lpips_pool)

        self._last_fake_feature_cache = fake_feature_cache
        self._last_fake_lpips_pool = fake_lpips_pool

        return {
            "fid": fid_val,
            "kid_mean": float(kid_mean.item()),
            "kid_std": float(kid_std.item()),
            "precision": float(precision),
            "recall": float(recall),
            "lpips_diversity": float(lpips_diversity),
            "num_real_fid_kid": float(self._real_image_count_fid_kid),
            "num_real_pr": float(self._real_feature_cache.shape[0]),
            "num_fake_pr": float(fake_feature_cache.shape[0]),
            "num_fake_lpips_pool": float(fake_lpips_pool.shape[0]),
        }

    # ---------------------------------------------------------------------
    # Fake metric accumulation
    # ---------------------------------------------------------------------
    def _update_fake_metrics(self, fake_images: Tensor) -> None:
        imgs_01 = self._to_zero_one(fake_images).to(self.device, non_blocking=True)
        if self.cfg.use_channels_last and imgs_01.ndim == 4:
            imgs_01 = imgs_01.contiguous(memory_format=torch.channels_last)
        self.fid.update(imgs_01, real=False)
        self.kid.update(imgs_01, real=False)

    # ---------------------------------------------------------------------
    # PR metric
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _extract_pr_features(self, images: Tensor) -> Tensor:
        imgs_01 = self._to_zero_one(images).to(self.pr_device, non_blocking=True)
        if self.cfg.use_channels_last and imgs_01.ndim == 4:
            imgs_01 = imgs_01.contiguous(memory_format=torch.channels_last)
        amp_context = self._autocast_context(self.pr_device)
        with amp_context:
            feats = self.pr_extractor(imgs_01)
        return feats.float().detach()

    @torch.no_grad()
    def _compute_precision_recall(self, real_feats_cpu: Tensor, fake_feats_cpu: Tensor) -> Tuple[float, float]:
        real_feats = real_feats_cpu.to(self.pr_device, dtype=torch.float32, non_blocking=True)
        fake_feats = fake_feats_cpu.to(self.pr_device, dtype=torch.float32, non_blocking=True)

        real_radii = self._compute_knn_radii(real_feats, k=self.cfg.pr_k, chunk_size=self.cfg.pr_chunk_size)
        fake_radii = self._compute_knn_radii(fake_feats, k=self.cfg.pr_k, chunk_size=self.cfg.pr_chunk_size)

        precision_mask = self._membership_mask(fake_feats, real_feats, real_radii, chunk_size=self.cfg.pr_chunk_size)
        recall_mask = self._membership_mask(real_feats, fake_feats, fake_radii, chunk_size=self.cfg.pr_chunk_size)

        precision = precision_mask.float().mean().item()
        recall = recall_mask.float().mean().item()
        return precision, recall

    @staticmethod
    def _compute_knn_radii(features: Tensor, k: int, chunk_size: int) -> Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected feature matrix of shape (N, D), got {tuple(features.shape)}")
        n = features.shape[0]
        if n <= k:
            raise ValueError(f"Need more than k samples for PR. Got n={n}, k={k}.")

        norms = (features * features).sum(dim=1)
        radii = torch.empty(n, device=features.device, dtype=features.dtype)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            d2 = GANMetricsSuite._pairwise_squared_distance(features[start:end], features, query_norms=norms[start:end], ref_norms=norms)
            diag_cols = torch.arange(start, end, device=features.device)
            diag_rows = torch.arange(end - start, device=features.device)
            d2[diag_rows, diag_cols] = torch.inf
            kth = torch.topk(d2, k=k, dim=1, largest=False).values[:, -1]
            radii[start:end] = kth
        return radii

    @staticmethod
    def _membership_mask(query: Tensor, centers: Tensor, center_radii: Tensor, chunk_size: int) -> Tensor:
        if query.ndim != 2 or centers.ndim != 2:
            raise ValueError("query and centers must be 2D feature matrices.")
        q_norms = (query * query).sum(dim=1)
        c_norms = (centers * centers).sum(dim=1)
        out = torch.empty(query.shape[0], device=query.device, dtype=torch.bool)
        radii = center_radii.view(1, -1)

        for start in range(0, query.shape[0], chunk_size):
            end = min(start + chunk_size, query.shape[0])
            d2 = GANMetricsSuite._pairwise_squared_distance(query[start:end], centers, query_norms=q_norms[start:end], ref_norms=c_norms)
            inside = (d2 <= radii).any(dim=1)
            out[start:end] = inside
        return out

    @staticmethod
    def _pairwise_squared_distance(
        query: Tensor,
        ref: Tensor,
        *,
        query_norms: Optional[Tensor] = None,
        ref_norms: Optional[Tensor] = None,
    ) -> Tensor:
        query = query.float()
        ref = ref.float()
        if query_norms is None:
            query_norms = (query * query).sum(dim=1)
        if ref_norms is None:
            ref_norms = (ref * ref).sum(dim=1)
        d2 = query_norms[:, None] + ref_norms[None, :] - 2.0 * (query @ ref.t())
        return d2.clamp_min_(0.0)

    # ---------------------------------------------------------------------
    # LPIPS diversity
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _compute_lpips_diversity(self, fake_pool: Tensor) -> float:
        n = int(fake_pool.shape[0])
        if n < 2:
            raise ValueError("Need at least two fake images to compute LPIPS diversity.")

        num_pairs = max(1, min(self.cfg.lpips_num_pairs, n // 2))

        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.cfg.seed))
        idx1 = perm[:num_pairs]
        idx2 = perm[num_pairs : 2 * num_pairs]

        total = 0.0
        seen = 0
        for start in range(0, num_pairs, self.cfg.lpips_pair_batch_size):
            end = min(start + self.cfg.lpips_pair_batch_size, num_pairs)
            img1 = self._to_minus_one_one(fake_pool[idx1[start:end]]).to(self.device, non_blocking=True)
            img2 = self._to_minus_one_one(fake_pool[idx2[start:end]]).to(self.device, non_blocking=True)
            if self.cfg.use_channels_last:
                img1 = img1.contiguous(memory_format=torch.channels_last)
                img2 = img2.contiguous(memory_format=torch.channels_last)
            val = self.lpips_metric(img1, img2)
            batch_size = end - start
            total += float(val.item()) * batch_size
            seen += batch_size
        return total / max(seen, 1)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def _reset_fake_states(self) -> None:
        self.fid.reset()
        self.kid.reset()
        self._last_fake_feature_cache = None
        self._last_fake_lpips_pool = None

    def _require_real_prepared(self) -> None:
        if not self._real_prepared or self._real_feature_cache is None:
            raise RuntimeError("Real metrics are not prepared. Call prepare_real(real_loader) first.")

    @staticmethod
    def _extract_images(batch: BatchType) -> Tensor:
        if torch.is_tensor(batch):
            return batch
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            for item in batch:
                if torch.is_tensor(item):
                    return item
        if isinstance(batch, dict):
            for key in ("images", "image", "img", "pixel_values", "x"):
                val = batch.get(key)
                if torch.is_tensor(val):
                    return val
        raise TypeError("Unsupported batch format. Expected Tensor, tuple/list with Tensor, or dict with image tensor.")

    def _to_zero_one(self, images: Tensor) -> Tensor:
        if self.cfg.input_range == "minus_one_to_one":
            return ((images.float() + 1.0) * 0.5).clamp_(0.0, 1.0)
        if self.cfg.input_range == "zero_one":
            return images.float().clamp_(0.0, 1.0)
        if self.cfg.input_range == "uint8":
            if images.dtype == torch.uint8:
                return images.float().div_(255.0)
            return images.float().clamp_(0.0, 255.0).div_(255.0)
        raise ValueError(f"Unsupported input_range: {self.cfg.input_range}")

    def _to_minus_one_one(self, images: Tensor) -> Tensor:
        if self.cfg.input_range == "minus_one_to_one":
            return images.float().clamp_(-1.0, 1.0)
        if self.cfg.input_range == "zero_one":
            return images.float().mul(2.0).sub_(1.0).clamp_(-1.0, 1.0)
        if self.cfg.input_range == "uint8":
            if images.dtype == torch.uint8:
                return images.float().div(255.0).mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)
            return images.float().clamp_(0.0, 255.0).div_(255.0).mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)
        raise ValueError(f"Unsupported input_range: {self.cfg.input_range}")

    def _autocast_context(self, device: torch.device):
        if device.type != "cuda" or not self.cfg.use_amp_for_feature_extractor:
            return nullcontext()
        if self.cfg.amp_dtype == "bf16":
            dtype = torch.bfloat16
        elif self.cfg.amp_dtype == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"Unsupported amp_dtype: {self.cfg.amp_dtype}")
        return torch.autocast(device_type="cuda", dtype=dtype)

    @staticmethod
    def _current_feature_count(chunks: List[Tensor]) -> int:
        return int(sum(chunk.shape[0] for chunk in chunks))

    @staticmethod
    def _current_image_count(chunks: List[Tensor]) -> int:
        return int(sum(chunk.shape[0] for chunk in chunks))


def format_metrics(metrics: Dict[str, float]) -> str:
    """Pretty formatter for console logging."""
    ordered = [
        "fid",
        "kid_mean",
        "kid_std",
        "precision",
        "recall",
        "lpips_diversity",
        "num_real_fid_kid",
        "num_real_pr",
        "num_fake_pr",
        "num_fake_lpips_pool",
    ]
    parts = []
    for key in ordered:
        if key in metrics:
            val = metrics[key]
            if key.startswith("num_"):
                parts.append(f"{key}={int(val)}")
            else:
                parts.append(f"{key}={val:.6f}")
    for key, val in metrics.items():
        if key not in ordered:
            if isinstance(val, (int, float)):
                parts.append(f"{key}={val}")
            else:
                parts.append(f"{key}={val}")
    return " | ".join(parts)


__all__ = [
    "DependencyError",
    "GANMetricsConfig",
    "GANMetricsSuite",
    "TorchvisionInceptionPool3",
    "format_metrics",
]
