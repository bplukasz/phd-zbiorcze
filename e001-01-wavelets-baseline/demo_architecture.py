#!/usr/bin/env python3
"""
Demo pokazujące jak architektura modelu zmienia się dla różnych rozmiarów obrazów.
"""

import torch
from src.experiment import Generator, Discriminator


def count_parameters(model):
    """Liczy parametry modelu."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_architecture(img_size: int, ch: int = 64):
    """Analizuje architekturę dla danego rozmiaru obrazu."""
    print(f"\n{'='*70}")
    print(f"Architektura dla obrazów {img_size}x{img_size}")
    print(f"{'='*70}")

    G = Generator(z_dim=128, ch=ch, img_channels=3, img_size=img_size)
    D = Discriminator(ch=ch, img_channels=3, img_size=img_size)

    g_params = count_parameters(G)
    d_params = count_parameters(D)
    total_params = g_params + d_params

    print(f"\n📊 Parametry modelu:")
    print(f"  Generator:     {g_params:,} parametrów ({g_params/1e6:.2f}M)")
    print(f"  Discriminator: {d_params:,} parametrów ({d_params/1e6:.2f}M)")
    print(f"  RAZEM:         {total_params:,} parametrów ({total_params/1e6:.2f}M)")

    print(f"\n🏗️  Struktura Generatora:")
    print(f"  Liczba bloków upsampling: {G.n_blocks}")
    print(f"  Przepływ: 4x4 ", end="")
    current_size = 4
    for i in range(G.n_blocks):
        current_size *= 2
        print(f"→ {current_size}x{current_size} ", end="")
    print()

    print(f"\n🏗️  Struktura Discriminatora:")
    print(f"  Liczba bloków downsampling: {D.n_blocks}")
    print(f"  Przepływ: {img_size}x{img_size} ", end="")
    current_size = img_size
    for i in range(D.n_blocks):
        current_size //= 2
        print(f"→ {current_size}x{current_size} ", end="")
    print("→ 4x4")

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = G.to(device)
    D = D.to(device)

    z = torch.randn(1, 128, device=device)
    x = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        g_out = G(z)
        d_out = D(x)

    print(f"\n✅ Test forward pass:")
    print(f"  Generator: z({list(z.shape)}) → img({list(g_out.shape)})")
    print(f"  Discriminator: img({list(x.shape)}) → logit({list(d_out.shape)})")

    # Memory estimate
    batch_size = 64
    img_memory = batch_size * 3 * img_size * img_size * 4  # 4 bytes per float32
    print(f"\n💾 Pamięć dla batch={batch_size}:")
    print(f"  Obrazy (input): {img_memory/1024**2:.1f} MB")
    print(f"  Parametry G+D: {(g_params+d_params)*4/1024**2:.1f} MB")
    print(f"  Gradient G+D: {(g_params+d_params)*4/1024**2:.1f} MB")
    print(f"  Activations (szacunek): ~{img_memory*10/1024**2:.0f} MB")
    total_mem = img_memory/1024**2 + (g_params+d_params)*8/1024**2 + img_memory*10/1024**2
    print(f"  SZACUNEK TOTAL: ~{total_mem:.0f} MB VRAM")


def main():
    """Porównuje różne rozmiary obrazów."""
    print("\n" + "="*70)
    print("PORÓWNANIE ARCHITEKTUR DLA RÓŻNYCH ROZMIARÓW OBRAZÓW")
    print("="*70)

    sizes = [32, 64, 128]

    for size in sizes:
        analyze_architecture(size)

    # Porównanie
    print(f"\n\n{'='*70}")
    print("PODSUMOWANIE")
    print(f"{'='*70}")

    results = []
    for size in sizes:
        G = Generator(z_dim=128, ch=64, img_channels=3, img_size=size)
        D = Discriminator(ch=64, img_channels=3, img_size=size)
        g_params = count_parameters(G)
        d_params = count_parameters(D)
        results.append({
            'size': size,
            'blocks': G.n_blocks,
            'params': g_params + d_params,
        })

    print(f"\n{'Rozmiar':<10} {'Bloki':<8} {'Parametry':<15} {'vs 32x32':<10}")
    print(f"{'-'*70}")

    baseline = results[0]['params']
    for r in results:
        ratio = r['params'] / baseline
        print(f"{r['size']}x{r['size']:<6} {r['blocks']:<8} {r['params']:,} ({r['params']/1e6:.2f}M)  {ratio:.2f}x")

    print(f"\n💡 Wnioski:")
    print(f"  • 32x32: {results[0]['blocks']} bloków, najszybszy, najmniej parametrów")
    print(f"  • 64x64: {results[1]['blocks']} bloków, kompromis")
    print(f"  • 128x128: {results[2]['blocks']} bloków, najwolniejszy, najwięcej parametrów")
    print(f"\n  ⚠️  Liczba parametrów NIE skaluje się proporcjonalnie do rozmiaru obrazu!")
    print(f"     To dlatego mniejsze obrazy nie dają 16x przyspieszenia.")


if __name__ == "__main__":
    main()
