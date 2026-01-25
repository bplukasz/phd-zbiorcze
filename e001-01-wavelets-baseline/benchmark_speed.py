#!/usr/bin/env python3
"""
Benchmark różnych profili konfiguracji.
Mierzy rzeczywisty czas per iterację dla różnych kombinacji.
"""

import time
import torch
from src.experiment import Generator, Discriminator, DiffAugment, hinge_loss_d, hinge_loss_g
from src.config_loader import get_config


def benchmark_profile(profile_name: str, n_iters: int = 100):
    """
    Benchmarkuje dany profil przez n iteracji.

    Args:
        profile_name: Nazwa profilu (train, fast, fast-small-batch, etc.)
        n_iters: Liczba iteracji do pomiaru

    Returns:
        Dict z metrykami
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {profile_name}")
    print(f"{'='*60}")

    cfg = get_config(profile_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("⚠️  UWAGA: Benchmark bez GPU może być niereprezentywny!")

    # Setup
    G = Generator(
        z_dim=cfg.z_dim,
        ch=cfg.g_ch,
        img_channels=cfg.img_channels,
        img_size=cfg.img_size
    ).to(device)

    D = Discriminator(
        ch=cfg.d_ch,
        img_channels=cfg.img_channels,
        img_size=cfg.img_size
    ).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_G, betas=cfg.betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_D, betas=cfg.betas)

    # Warmup
    print("Warmup (5 iteracji)...")
    for _ in range(5):
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        real_imgs = torch.randn(cfg.batch_size, cfg.img_channels, cfg.img_size, cfg.img_size, device=device)

        # D step
        D.zero_grad()
        real_aug = DiffAugment(real_imgs, cfg.diffaug_policy)
        real_logits = D(real_aug)
        fake_imgs = G(z).detach()
        fake_aug = DiffAugment(fake_imgs, cfg.diffaug_policy)
        fake_logits = D(fake_aug)
        loss_D = hinge_loss_d(real_logits, fake_logits)
        loss_D.backward()
        opt_D.step()

        # G step
        G.zero_grad()
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake_imgs = G(z)
        fake_aug = DiffAugment(fake_imgs, cfg.diffaug_policy)
        fake_logits = D(fake_aug)
        loss_G = hinge_loss_g(fake_logits)
        loss_G.backward()
        opt_G.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Pomiar ({n_iters} iteracji)...")
    times = []

    for i in range(n_iters):
        iter_start = time.time()

        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        real_imgs = torch.randn(cfg.batch_size, cfg.img_channels, cfg.img_size, cfg.img_size, device=device)

        # D step
        D.zero_grad()
        real_aug = DiffAugment(real_imgs, cfg.diffaug_policy)
        real_logits = D(real_aug)
        fake_imgs = G(z).detach()
        fake_aug = DiffAugment(fake_imgs, cfg.diffaug_policy)
        fake_logits = D(fake_aug)
        loss_D = hinge_loss_d(real_logits, fake_logits)
        loss_D.backward()
        opt_D.step()

        # G step
        G.zero_grad()
        z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
        fake_imgs = G(z)
        fake_aug = DiffAugment(fake_imgs, cfg.diffaug_policy)
        fake_logits = D(fake_aug)
        loss_G = hinge_loss_g(fake_logits)
        loss_G.backward()
        opt_G.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        iter_time = time.time() - iter_start
        times.append(iter_time)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_iters} iteracji...")

    # Stats
    import statistics
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)

    # Memory
    if device.type == "cuda":
        vram_peak = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
    else:
        vram_peak = 0

    # Pikseli
    pixels_per_batch = cfg.batch_size * cfg.img_size * cfg.img_size * cfg.img_channels

    print(f"\n{'='*60}")
    print(f"Wyniki: {profile_name}")
    print(f"{'='*60}")
    print(f"Config:")
    print(f"  - Batch size: {cfg.batch_size}")
    print(f"  - Image size: {cfg.img_size}x{cfg.img_size}")
    print(f"  - Channels: {cfg.img_channels}")
    print(f"  - Dataset: {getattr(cfg, 'dataset_name', 'celeba')}")
    print(f"  - Pikseli/batch: {pixels_per_batch:,}")
    print(f"\nCzasy:")
    print(f"  - Średnia: {mean_time*1000:.1f} ms/iter")
    print(f"  - Mediana: {median_time*1000:.1f} ms/iter")
    print(f"  - Std dev: {std_time*1000:.1f} ms")
    print(f"  - Min: {min_time*1000:.1f} ms")
    print(f"  - Max: {max_time*1000:.1f} ms")
    print(f"\nPamięć:")
    print(f"  - VRAM peak: {vram_peak:.0f} MB")
    print(f"\nWydajność:")
    print(f"  - Pikseli/ms: {pixels_per_batch/mean_time/1000:.0f}")

    return {
        'profile': profile_name,
        'batch_size': cfg.batch_size,
        'img_size': cfg.img_size,
        'mean_ms': mean_time * 1000,
        'median_ms': median_time * 1000,
        'std_ms': std_time * 1000,
        'vram_mb': vram_peak,
        'pixels_per_batch': pixels_per_batch,
    }


def main():
    """Benchmarkuje wszystkie profile."""
    profiles = ['train', 'fast', 'fast-small-batch', 'fast64']
    results = []

    for profile in profiles:
        try:
            result = benchmark_profile(profile, n_iters=50)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Błąd dla {profile}: {e}")

    # Podsumowanie
    if results:
        print(f"\n\n{'='*80}")
        print("PODSUMOWANIE")
        print(f"{'='*80}")
        print(f"{'Profile':<20} {'Batch':<8} {'Size':<8} {'ms/iter':<10} {'VRAM(MB)':<10} {'Speedup':<10}")
        print(f"{'-'*80}")

        baseline = next((r for r in results if r['profile'] == 'train'), results[0])
        baseline_time = baseline['mean_ms']

        for r in results:
            speedup = baseline_time / r['mean_ms']
            print(f"{r['profile']:<20} {r['batch_size']:<8} {r['img_size']:<8} "
                  f"{r['mean_ms']:<10.1f} {r['vram_mb']:<10.0f} {speedup:<10.2f}x")

    print(f"\n✅ Benchmark zakończony!")


if __name__ == "__main__":
    main()
