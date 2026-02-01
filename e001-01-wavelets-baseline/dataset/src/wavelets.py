"""
Moduł transformaty falkowej 2D (DWT2D/IDWT2D) dla tensorów PyTorch.
Obsługuje falki haar i db2, padding reflect, output jako concat kanałów [LL,LH,HL,HH].

UWAGA: Rozmiar współczynników NIE musi być dokładnie H//2 x W//2.
Dla db2 przy reflect będzie typowo (H//2 + 1, W//2 + 1).
Rekonstrukcja używa conv_transpose2d i docina do output_size na końcu.
"""

import math
from typing import Tuple, Dict, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Filtry falkowe - wartości z PyWavelets
# Można zweryfikować: import pywt; pywt.Wavelet('haar').dec_lo, pywt.Wavelet('db2').dec_lo
WAVELET_FILTERS: Dict[str, Dict[str, List[float]]] = {
    'haar': {
        # Haar wavelet: [1/sqrt(2), 1/sqrt(2)]
        'dec_lo': [0.7071067811865476, 0.7071067811865476],
        'dec_hi': [-0.7071067811865476, 0.7071067811865476],
        'rec_lo': [0.7071067811865476, 0.7071067811865476],
        'rec_hi': [0.7071067811865476, -0.7071067811865476],
    },
    'db2': {
        # Daubechies-2 wavelet
        'dec_lo': [-0.12940952255126037, 0.2241438680420134,
                   0.8365163037378079, 0.48296291314453416],
        'dec_hi': [-0.48296291314453416, 0.8365163037378079,
                   -0.2241438680420134, -0.12940952255126037],
        'rec_lo': [0.48296291314453416, 0.8365163037378079,
                   0.2241438680420134, -0.12940952255126037],
        'rec_hi': [-0.12940952255126037, -0.2241438680420134,
                   0.8365163037378079, -0.48296291314453416],
    },
}

WaveletName = Literal['haar', 'db2']


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Iloczyn zewnętrzny dwóch wektorów 1D -> macierz 2D."""
    return torch.einsum("i,j->ij", a, b)


def _get_filter_tensors(wavelet: WaveletName, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zwraca tensory filtrów dla danej falki.

    Returns:
        dec_lo, dec_hi, rec_lo, rec_hi - filtry jako 1D tensory
    """
    if wavelet not in WAVELET_FILTERS:
        raise ValueError(f"Nieznana falka: {wavelet}. Dostępne: {list(WAVELET_FILTERS.keys())}")

    filters = WAVELET_FILTERS[wavelet]
    dec_lo = torch.tensor(filters['dec_lo'], device=device, dtype=dtype)
    dec_hi = torch.tensor(filters['dec_hi'], device=device, dtype=dtype)
    rec_lo = torch.tensor(filters['rec_lo'], device=device, dtype=dtype)
    rec_hi = torch.tensor(filters['rec_hi'], device=device, dtype=dtype)

    return dec_lo, dec_hi, rec_lo, rec_hi


class DWT2D(nn.Module):
    """
    Discrete Wavelet Transform 2D dla obrazów BxCxHxW.

    Używa konwolucji 2D z kernelami separowalnymi (iloczyn zewnętrzny filtrów 1D).
    Padding reflect z pad = K-2 (haar: 0, db2: 2).

    Output: tensor BxC*4xH'xW' z kanałami [LL, LH, HL, HH] dla każdego kanału wejściowego.
    UWAGA: H', W' NIE muszą być dokładnie H//2, W//2 - dla db2 będą większe.
    """

    def __init__(self, wavelet: WaveletName = 'haar', pad_mode: str = 'reflect'):
        super().__init__()
        self.wavelet = wavelet
        self.pad_mode = pad_mode

        # Pobierz filtry i zarejestruj jako bufory (nie parametry)
        filters = WAVELET_FILTERS[wavelet]

        # Filtry dekompozycji
        self.register_buffer('dec_lo', torch.tensor(filters['dec_lo'], dtype=torch.float32))
        self.register_buffer('dec_hi', torch.tensor(filters['dec_hi'], dtype=torch.float32))

        self.filter_len = len(filters['dec_lo'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wykonuje DWT2D na wejściowym tensorze.

        Args:
            x: Tensor BxCxHxW

        Returns:
            Tensor BxC*4xH'xW' z kanałami [LL, LH, HL, HH] skonkatenowanymi
            (H', W' mogą być większe niż H//2, W//2 dla db2)
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Upewnij się że filtry są na tym samym urządzeniu i dtype
        dec_lo = self.dec_lo.to(device=device, dtype=dtype)
        dec_hi = self.dec_hi.to(device=device, dtype=dtype)

        # conv2d = korelacja, więc dla DWT flipujemy filtry analizy
        h0 = dec_lo.flip(0)
        h1 = dec_hi.flip(0)
        K = h0.numel()

        # padding, który daje poprawną rekonstrukcję z conv_transpose dla reflect:
        # haar: K=2 -> pad=0, db2: K=4 -> pad=2
        pad = K - 2
        if pad > 0:
            x = F.pad(x, (pad, pad, pad, pad), mode=self.pad_mode)

        # 2D separable kernels: LL, LH, HL, HH
        LL = _outer(h0, h0)
        LH = _outer(h0, h1)
        HL = _outer(h1, h0)
        HH = _outer(h1, h1)
        base = torch.stack([LL, LH, HL, HH], dim=0)  # (4, K, K)

        # grupy per kanał: (4C, 1, K, K)
        w = torch.zeros((4 * C, 1, K, K), device=device, dtype=dtype)
        for c in range(C):
            w[4*c:4*c+4, 0] = base

        y = F.conv2d(x, w, stride=2, groups=C)

        return y


class IDWT2D(nn.Module):
    """
    Inverse Discrete Wavelet Transform 2D.

    Rekonstruuje obraz BxCxHxW z tensorów falkowych BxC*4xH'xW'.
    Używa conv_transpose2d dla poprawnej rekonstrukcji.
    """

    def __init__(self, wavelet: WaveletName = 'haar', pad_mode: str = 'reflect'):
        super().__init__()
        self.wavelet = wavelet
        self.pad_mode = pad_mode

        # Pobierz filtry rekonstrukcji
        filters = WAVELET_FILTERS[wavelet]

        self.register_buffer('rec_lo', torch.tensor(filters['rec_lo'], dtype=torch.float32))
        self.register_buffer('rec_hi', torch.tensor(filters['rec_hi'], dtype=torch.float32))

        self.filter_len = len(filters['rec_lo'])

    def forward(self, coeffs: torch.Tensor, output_size: Tuple[int, int] = None) -> torch.Tensor:
        """
        Wykonuje IDWT2D na współczynnikach falkowych.

        Args:
            coeffs: Tensor BxC*4xH'xW' z kanałami [LL, LH, HL, HH]
            output_size: Opcjonalny (H, W) dla wyjścia - przydatne gdy oryginał miał nieparzyste wymiary

        Returns:
            Tensor BxCxHxW (rekonstruowany obraz)
        """
        B, C4, Hc, Wc = coeffs.shape
        C = C4 // 4
        device, dtype = coeffs.device, coeffs.dtype

        # Upewnij się że filtry są na tym samym urządzeniu i dtype
        rec_lo = self.rec_lo.to(device=device, dtype=dtype)
        rec_hi = self.rec_hi.to(device=device, dtype=dtype)

        # spójnie z DWT: dla syntezy NIE flipujemy (działa z conv_transpose)
        g0 = rec_lo
        g1 = rec_hi
        K = g0.numel()

        # padding = K - 2 (spójnie z DWT)
        pad = K - 2

        # 2D separable kernels: LL, LH, HL, HH
        LL = _outer(g0, g0)
        LH = _outer(g0, g1)
        HL = _outer(g1, g0)
        HH = _outer(g1, g1)
        base = torch.stack([LL, LH, HL, HH], dim=0)  # (4, K, K)

        # conv_transpose2d: weight ma kształt (in_ch, out_ch/groups, K, K) => (4C, 1, K, K)
        w = torch.zeros((4 * C, 1, K, K), device=device, dtype=dtype)
        for c in range(C):
            w[4*c:4*c+4, 0] = base

        x = F.conv_transpose2d(coeffs, w, stride=2, groups=C, padding=pad)

        # docięcie do dokładnego rozmiaru wejścia (ważne dla parzystych/nieparzystych)
        if output_size is not None:
            H, W = output_size
            x = x[:, :, :H, :W]

        return x


# ============================================================================
# Convenience functions
# ============================================================================

def dwt2d(x: torch.Tensor, wavelet: WaveletName = 'haar') -> torch.Tensor:
    """
    Funkcjonalna wersja DWT2D.

    Args:
        x: Tensor BxCxHxW
        wavelet: 'haar' lub 'db2'

    Returns:
        Tensor BxC*4xH'xW' z kanałami [LL, LH, HL, HH]
        (H', W' mogą być większe niż H//2, W//2 dla db2)
    """
    return DWT2D(wavelet).to(x.device)(x)


def idwt2d(coeffs: torch.Tensor, wavelet: WaveletName = 'haar',
           output_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    Funkcjonalna wersja IDWT2D.

    Args:
        coeffs: Tensor BxC*4xH'xW' z kanałami [LL, LH, HL, HH]
        wavelet: 'haar' lub 'db2'
        output_size: Opcjonalny (H, W) dla wyjścia

    Returns:
        Tensor BxCxHxW (rekonstruowany obraz)
    """
    return IDWT2D(wavelet).to(coeffs.device)(coeffs, output_size)


def split_subbands(coeffs: torch.Tensor, C: int):
    """
    Rozdziela tensor współczynników na osobne podpasma.

    UWAGA: DWT2D zwraca kanały w kolejności per-kanał:
      [LL0, LH0, HL0, HH0,  LL1, LH1, HL1, HH1, ...]
    """
    B, C4, Hc, Wc = coeffs.shape
    if C4 != 4 * C:
        raise ValueError(f"split_subbands: expected {4*C} channels, got {C4}")

    # (B, 4C, Hc, Wc) -> (B, C, 4, Hc, Wc)
    coeffs = coeffs.reshape(B, C, 4, Hc, Wc)

    LL = coeffs[:, :, 0, :, :]
    LH = coeffs[:, :, 1, :, :]
    HL = coeffs[:, :, 2, :, :]
    HH = coeffs[:, :, 3, :, :]
    return LL, LH, HL, HH


# ============================================================================
# Test functions
# ============================================================================

def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor, max_val: float = 1.0) -> float:
    """Oblicza PSNR między dwoma tensorami."""
    mse = F.mse_loss(original, reconstructed).item()
    if mse < 1e-10:
        return float('inf')
    psnr = 10 * math.log10(max_val ** 2 / mse)
    return psnr


def compute_mae(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Oblicza MAE (Mean Absolute Error) między dwoma tensorami."""
    return F.l1_loss(original, reconstructed).item()


def test_dwt_reconstruction(
    wavelet: WaveletName = 'haar',
    batch_size: int = 4,
    channels: int = 3,
    height: int = 64,
    width: int = 64,
    device: str = 'cpu',
    save_diff_image: bool = True,
    output_dir: str = '.',
) -> Dict[str, float]:
    """
    Testuje rekonstrukcję DWT2D -> IDWT2D dla losowych obrazów.

    Args:
        wavelet: Nazwa falki ('haar' lub 'db2')
        batch_size: Rozmiar batcha
        channels: Liczba kanałów
        height: Wysokość obrazu
        width: Szerokość obrazu
        device: Urządzenie ('cpu' lub 'cuda')
        save_diff_image: Czy zapisywać obraz różnicy
        output_dir: Katalog wyjściowy dla obrazów

    Returns:
        Dict z metrykami (psnr, mae)
    """
    import os
    from torchvision.utils import save_image

    print(f"\n{'='*60}")
    print(f"Test DWT2D/IDWT2D - {wavelet.upper()}")
    print(f"{'='*60}")
    print(f"Parametry: B={batch_size}, C={channels}, H={height}, W={width}")
    print(f"Device: {device}")

    # Losowe obrazy w zakresie [0, 1]
    torch.manual_seed(42)
    x = torch.rand(batch_size, channels, height, width, device=device)

    print(f"\nWejście: shape={x.shape}, min={x.min():.4f}, max={x.max():.4f}")

    # Forward DWT
    dwt = DWT2D(wavelet).to(device)
    coeffs = dwt(x)

    print(f"Współczynniki DWT: shape={coeffs.shape}")

    # Podpasma
    LL, LH, HL, HH = split_subbands(coeffs, channels)
    print(f"  LL: shape={LL.shape}, range=[{LL.min():.4f}, {LL.max():.4f}]")
    print(f"  LH: shape={LH.shape}, range=[{LH.min():.4f}, {LH.max():.4f}]")
    print(f"  HL: shape={HL.shape}, range=[{HL.min():.4f}, {HL.max():.4f}]")
    print(f"  HH: shape={HH.shape}, range=[{HH.min():.4f}, {HH.max():.4f}]")

    # Inverse DWT
    idwt = IDWT2D(wavelet).to(device)
    x_rec = idwt(coeffs, output_size=(height, width))

    print(f"\nRekonstrukcja: shape={x_rec.shape}")
    print(f"  range=[{x_rec.min():.4f}, {x_rec.max():.4f}]")

    # Metryki
    psnr = compute_psnr(x, x_rec)
    mae = compute_mae(x, x_rec)

    print(f"\n--- Metryki rekonstrukcji ---")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE:  {mae:.6f}")

    # Zapisz obraz różnicy
    if save_diff_image:
        os.makedirs(output_dir, exist_ok=True)

        # Różnica (przeskalowana dla widoczności)
        diff = (x - x_rec).abs()
        diff_scaled = diff / diff.max().clamp(min=1e-6)  # Normalize to [0, 1]

        # Zapisz oryginał, rekonstrukcję i różnicę
        save_image(x[0], os.path.join(output_dir, f'test_{wavelet}_original.png'))
        save_image(x_rec[0].clamp(0, 1), os.path.join(output_dir, f'test_{wavelet}_reconstructed.png'))
        save_image(diff_scaled[0], os.path.join(output_dir, f'test_{wavelet}_diff.png'))

        # Wizualizacja podpasm (pierwszy obrazek)
        vis_coeffs = torch.cat([
            torch.cat([LL[0], LH[0]], dim=2),  # Górny wiersz: LL, LH
            torch.cat([HL[0], HH[0]], dim=2),  # Dolny wiersz: HL, HH
        ], dim=1)
        # Normalizuj do [0, 1]
        vis_min = vis_coeffs.min()
        vis_max = vis_coeffs.max()
        vis_coeffs = (vis_coeffs - vis_min) / (vis_max - vis_min + 1e-6)
        save_image(vis_coeffs, os.path.join(output_dir, f'test_{wavelet}_subbands.png'))

        print(f"\nZapisano obrazy do: {output_dir}")
        print(f"  - test_{wavelet}_original.png")
        print(f"  - test_{wavelet}_reconstructed.png")
        print(f"  - test_{wavelet}_diff.png")
        print(f"  - test_{wavelet}_subbands.png")

    return {'psnr': psnr, 'mae': mae}


def test_with_real_image(
    image_path: str,
    wavelet: WaveletName = 'haar',
    device: str = 'cpu',
    output_dir: str = '.',
) -> Dict[str, float]:
    """
    Testuje rekonstrukcję DWT2D -> IDWT2D na prawdziwym obrazie.

    Args:
        image_path: Ścieżka do obrazu
        wavelet: Nazwa falki ('haar' lub 'db2')
        device: Urządzenie
        output_dir: Katalog wyjściowy

    Returns:
        Dict z metrykami
    """
    import os
    from PIL import Image
    from torchvision import transforms
    from torchvision.utils import save_image

    print(f"\n{'='*60}")
    print(f"Test z prawdziwym obrazem - {wavelet.upper()}")
    print(f"{'='*60}")
    print(f"Obraz: {image_path}")

    # Wczytaj obraz
    img = Image.open(image_path).convert('RGB')

    # Upewnij się że wymiary są parzyste
    w, h = img.size
    new_w = w - (w % 2)
    new_h = h - (h % 2)

    transform = transforms.Compose([
        transforms.CenterCrop((new_h, new_w)),
        transforms.ToTensor(),
    ])

    x = transform(img).unsqueeze(0).to(device)  # (1, 3, H, W)

    print(f"Wejście: shape={x.shape}")

    # DWT
    dwt = DWT2D(wavelet).to(device)
    coeffs = dwt(x)

    # IDWT
    idwt = IDWT2D(wavelet).to(device)
    x_rec = idwt(coeffs, output_size=(x.shape[2], x.shape[3]))

    # Metryki
    psnr = compute_psnr(x, x_rec)
    mae = compute_mae(x, x_rec)

    print(f"\n--- Metryki rekonstrukcji ---")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE:  {mae:.6f}")

    # Zapisz obrazy
    os.makedirs(output_dir, exist_ok=True)

    diff = (x - x_rec).abs()
    diff_scaled = diff / diff.max().clamp(min=1e-6)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_image(x[0], os.path.join(output_dir, f'{base_name}_{wavelet}_original.png'))
    save_image(x_rec[0].clamp(0, 1), os.path.join(output_dir, f'{base_name}_{wavelet}_reconstructed.png'))
    save_image(diff_scaled[0], os.path.join(output_dir, f'{base_name}_{wavelet}_diff.png'))

    # Podpasma
    C = x.shape[1]
    LL, LH, HL, HH = split_subbands(coeffs, C)
    vis_coeffs = torch.cat([
        torch.cat([LL[0], LH[0]], dim=2),
        torch.cat([HL[0], HH[0]], dim=2),
    ], dim=1)
    vis_min = vis_coeffs.min()
    vis_max = vis_coeffs.max()
    vis_coeffs = (vis_coeffs - vis_min) / (vis_max - vis_min + 1e-6)
    save_image(vis_coeffs, os.path.join(output_dir, f'{base_name}_{wavelet}_subbands.png'))

    print(f"\nZapisano obrazy do: {output_dir}")

    return {'psnr': psnr, 'mae': mae}


def run_all_tests(output_dir: str = './dwt_test_output', real_image_path: str = None):
    """
    Uruchamia wszystkie testy DWT2D/IDWT2D.

    Args:
        output_dir: Katalog wyjściowy
        real_image_path: Opcjonalna ścieżka do prawdziwego obrazu
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"To jest nowa wersja testów DWT2D/IDWT2D z poprawkami.")

    results = {}

    # Testy dla różnych falek
    for wavelet in ['haar', 'db2']:
        print(f"\n{'#'*60}")
        print(f"# FALKA: {wavelet.upper()}")
        print(f"{'#'*60}")

        # Test z losowymi obrazami
        results[f'{wavelet}_random'] = test_dwt_reconstruction(
            wavelet=wavelet,
            batch_size=4,
            channels=3,
            height=128,
            width=128,
            device=device,
            save_diff_image=True,
            output_dir=output_dir,
        )

        # Test z różnymi rozmiarami
        for size in [32, 64, 256]:
            print(f"\n--- Test rozmiar {size}x{size} ---")
            r = test_dwt_reconstruction(
                wavelet=wavelet,
                batch_size=2,
                channels=3,
                height=size,
                width=size,
                device=device,
                save_diff_image=False,
            )
            print(f"  PSNR: {r['psnr']:.2f} dB, MAE: {r['mae']:.6f}")

        # Test z prawdziwym obrazem
        if real_image_path and os.path.exists(real_image_path):
            results[f'{wavelet}_real'] = test_with_real_image(
                image_path=real_image_path,
                wavelet=wavelet,
                device=device,
                output_dir=output_dir,
            )

    # Podsumowanie
    print(f"\n{'='*60}")
    print("PODSUMOWANIE")
    print(f"{'='*60}")
    for key, metrics in results.items():
        print(f"  {key}: PSNR={metrics['psnr']:.2f} dB, MAE={metrics['mae']:.6f}")

    return results


if __name__ == '__main__':
    # Uruchom testy
    run_all_tests(output_dir='./dwt_test_output')
