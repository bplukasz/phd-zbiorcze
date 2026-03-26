#!/usr/bin/env python3
"""Dataset downloader for GAN experiments.

Supported datasets:
- MNIST
- CIFAR-10
- CIFAR-100
- CelebA
- FFHQ
- CelebA-HQ

The script can download a single dataset or all of them into one root folder.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


Downloader = Callable[["DatasetSpec", "DatasetVariantSpec", Path, argparse.Namespace], Path]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetVariantSpec:
    key: str
    resolution: str
    approx_size: str
    notes: str
    kaggle_slugs: tuple[str, ...] = ()
    hf_repo: str | None = None
    download_fn_name: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    image_count: str
    download_fn_name: str
    variants: tuple[DatasetVariantSpec, ...]
    default_variant: str
    notes: str = ""


@dataclass(frozen=True)
class DatasetSelection:
    dataset_key: str
    variant_key: str


def _target_root_for_variant(spec: DatasetSpec, variant: DatasetVariantSpec, root: Path) -> Path:
    if len(spec.variants) == 1:
        return root / spec.key
    return root / spec.key / variant.key


def _effective_download_fn_name(spec: DatasetSpec, variant: DatasetVariantSpec) -> str:
    return variant.download_fn_name or spec.download_fn_name


def _parse_image_count(value: str) -> int:
    return int(value.replace(",", "").strip())


def _parse_resolution(value: str) -> tuple[int, int]:
    left, right = value.lower().split("x", 1)
    return int(left.strip()), int(right.strip())


def _collect_image_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for current_root, dirnames, names in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for name in names:
            path = Path(current_root) / name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(path)
    files.sort()
    return files


def _validate_torchvision_dataset(spec: DatasetSpec, variant: DatasetVariantSpec, target_root: Path) -> None:
    try:
        import torchvision.datasets as tv_datasets
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("Walidacja wymaga torchvision.") from exc

    expected_count = _parse_image_count(spec.image_count)
    expected_resolution = _parse_resolution(variant.resolution)

    if spec.key == "mnist":
        train = tv_datasets.MNIST(root=str(target_root), train=True, download=False)
        test = tv_datasets.MNIST(root=str(target_root), train=False, download=False)
        count = len(train) + len(test)
        sample_size = train[0][0].size
    elif spec.key == "cifar10":
        train = tv_datasets.CIFAR10(root=str(target_root), train=True, download=False)
        test = tv_datasets.CIFAR10(root=str(target_root), train=False, download=False)
        count = len(train) + len(test)
        sample_size = train[0][0].size
    elif spec.key == "cifar100":
        train = tv_datasets.CIFAR100(root=str(target_root), train=True, download=False)
        test = tv_datasets.CIFAR100(root=str(target_root), train=False, download=False)
        count = len(train) + len(test)
        sample_size = train[0][0].size
    elif spec.key == "celeba":
        all_split = tv_datasets.CelebA(root=str(target_root), split="all", download=False)
        count = len(all_split)
        sample_size = all_split[0][0].size
    else:  # pragma: no cover - safeguarded by registry
        raise RuntimeError(f"Brak walidatora torchvision dla {spec.key}")

    if count != expected_count:
        raise RuntimeError(
            f"Walidacja nie przeszla: oczekiwano {expected_count} obrazow, znaleziono {count}."
        )
    if sample_size != expected_resolution:
        raise RuntimeError(
            "Walidacja nie przeszla: rozdzielczosc niezgodna. "
            f"Oczekiwano {expected_resolution[0]}x{expected_resolution[1]}, "
            f"znaleziono {sample_size[0]}x{sample_size[1]}."
        )


def _validate_image_folder_dataset(
    spec: DatasetSpec,
    variant: DatasetVariantSpec,
    target_root: Path,
) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("Walidacja obrazow wymaga pillow.") from exc

    expected_count = _parse_image_count(spec.image_count)
    expected_resolution = _parse_resolution(variant.resolution)
    image_files = _collect_image_files(target_root)
    count = len(image_files)

    if count != expected_count:
        raise RuntimeError(
            f"Walidacja nie przeszla: oczekiwano {expected_count} obrazow, znaleziono {count}."
        )

    sample = image_files[: min(1024, count)]
    mismatches: list[tuple[str, tuple[int, int]]] = []
    for path in sample:
        with Image.open(path) as image:
            size = image.size
        if size != expected_resolution:
            mismatches.append((str(path.relative_to(target_root)), size))
            if len(mismatches) >= 5:
                break

    if mismatches:
        first = "; ".join(
            f"{name} -> {size[0]}x{size[1]}" for name, size in mismatches
        )
        raise RuntimeError(
            "Walidacja nie przeszla: rozdzielczosc niezgodna z wybranym wariantem. "
            f"Oczekiwano {expected_resolution[0]}x{expected_resolution[1]}. Przyklady: {first}"
        )


def _validate_download(
    spec: DatasetSpec,
    variant: DatasetVariantSpec,
    target_root: Path,
) -> None:
    backend = _effective_download_fn_name(spec, variant)
    if backend == "torchvision":
        _validate_torchvision_dataset(spec, variant, target_root)
        return
    _validate_image_folder_dataset(spec, variant, target_root)


def _prepare_target_root(target_root: Path, args: argparse.Namespace) -> None:
    if args.force_clean and target_root.exists():
        print(f"  -> usuwam istniejacy katalog: {target_root}")
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)


def _download_torchvision(
    spec: DatasetSpec,
    variant: DatasetVariantSpec,
    root: Path,
    args: argparse.Namespace,
) -> Path:
    try:
        import torchvision.datasets as tv_datasets
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "Brak torchvision. Zainstaluj zaleznosci i uruchom ponownie."
        ) from exc

    target_root = _target_root_for_variant(spec, variant, root)
    _prepare_target_root(target_root, args)

    if args.dry_run:
        print(f"[dry-run] torchvision download: {spec.key} -> {target_root}")
        return target_root

    if spec.key == "mnist":
        tv_datasets.MNIST(root=str(target_root), train=True, download=True)
        tv_datasets.MNIST(root=str(target_root), train=False, download=True)
    elif spec.key == "cifar10":
        tv_datasets.CIFAR10(root=str(target_root), train=True, download=True)
        tv_datasets.CIFAR10(root=str(target_root), train=False, download=True)
    elif spec.key == "cifar100":
        tv_datasets.CIFAR100(root=str(target_root), train=True, download=True)
        tv_datasets.CIFAR100(root=str(target_root), train=False, download=True)
    elif spec.key == "celeba":
        tv_datasets.CelebA(root=str(target_root), split="all", download=True)
    else:  # pragma: no cover - safeguarded by registry
        raise RuntimeError(f"Nieznany dataset torchvision: {spec.key}")

    return target_root


def _download_kaggle(
    spec: DatasetSpec,
    variant: DatasetVariantSpec,
    root: Path,
    args: argparse.Namespace,
) -> Path:
    if not variant.kaggle_slugs:
        raise RuntimeError(
            f"Brak skonfigurowanych slugow Kaggle dla {spec.key}@{variant.key}"
        )

    kaggle_cmd = "kaggle"
    if not args.dry_run:
        probe = subprocess.run(
            [kaggle_cmd, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "Brak CLI `kaggle` w PATH. Zainstaluj: pip install kaggle, "
                "skonfiguruj API token i uruchom ponownie."
            )

    target_root = _target_root_for_variant(spec, variant, root)
    _prepare_target_root(target_root, args)

    errors: list[str] = []
    for slug in variant.kaggle_slugs:
        if args.dry_run:
            print(f"[dry-run] kaggle datasets download -d {slug} -p {target_root} --unzip")
            return target_root

        cmd = [
            kaggle_cmd,
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(target_root),
            "--unzip",
        ]

        print(f"  -> probuje Kaggle slug: {slug}")
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode == 0:
            return target_root

        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        errors.append(f"{slug}: {stderr or stdout or 'unknown error'}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "Nie udalo sie pobrac z Kaggle zadnym slugiem. "
        "Sprawdz uprawnienia i token Kaggle.\n"
        f"Szczegoly:\n{joined}"
    )


def _download_huggingface(
    spec: DatasetSpec,
    variant: DatasetVariantSpec,
    root: Path,
    args: argparse.Namespace,
) -> Path:
    if not variant.hf_repo:
        raise RuntimeError(
            f"Brak skonfigurowanego repo Hugging Face dla {spec.key}@{variant.key}"
        )

    target_root = _target_root_for_variant(spec, variant, root)
    _prepare_target_root(target_root, args)

    if args.dry_run:
        print(
            "[dry-run] huggingface snapshot_download "
            f"repo_id={variant.hf_repo} repo_type=dataset local_dir={target_root}"
        )
        return target_root

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "Brak huggingface_hub. Zainstaluj: pip install huggingface_hub"
        ) from exc

    print(f"  -> probuje Hugging Face dataset: {variant.hf_repo}")
    snapshot_download(
        repo_id=variant.hf_repo,
        repo_type="dataset",
        local_dir=str(target_root),
        allow_patterns=[
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.webp",
            "*.txt",
            "*.json",
            "*.csv",
            "*.md",
            "LICENSE*",
        ],
        resume_download=True,
    )
    return target_root


def _download_manual(
    spec: DatasetSpec,
    variant: DatasetVariantSpec,
    root: Path,
    args: argparse.Namespace,
) -> Path:
    target_root = _target_root_for_variant(spec, variant, root)
    _prepare_target_root(target_root, args)
    raise RuntimeError(
        f"{spec.display_name} wymaga recznego pobrania. "
        f"Umiesc pliki w: {target_root}"
    )


DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        key="mnist",
        display_name="MNIST",
        image_count="70,000",
        download_fn_name="torchvision",
        variants=(
            DatasetVariantSpec(
                key="native",
                resolution="28x28",
                approx_size="~70 MB",
                notes="10 klas cyfr, grayscale.",
            ),
        ),
        default_variant="native",
    ),
    "cifar10": DatasetSpec(
        key="cifar10",
        display_name="CIFAR-10",
        image_count="60,000",
        download_fn_name="torchvision",
        variants=(
            DatasetVariantSpec(
                key="native",
                resolution="32x32",
                approx_size="~170 MB",
                notes="10 klas, RGB.",
            ),
        ),
        default_variant="native",
    ),
    "cifar100": DatasetSpec(
        key="cifar100",
        display_name="CIFAR-100",
        image_count="60,000",
        download_fn_name="torchvision",
        variants=(
            DatasetVariantSpec(
                key="native",
                resolution="32x32",
                approx_size="~180 MB",
                notes="100 klas, RGB.",
            ),
        ),
        default_variant="native",
    ),
    "celeba": DatasetSpec(
        key="celeba",
        display_name="CelebA",
        image_count="202,599",
        download_fn_name="torchvision",
        variants=(
            DatasetVariantSpec(
                key="aligned",
                resolution="178x218",
                approx_size="~1.4 GB",
                notes="img_align_celeba, split all.",
            ),
        ),
        default_variant="aligned",
    ),
    "ffhq": DatasetSpec(
        key="ffhq",
        display_name="FFHQ",
        image_count="70,000",
        download_fn_name="kaggle",
        variants=(
            DatasetVariantSpec(
                key="256",
                resolution="256x256",
                approx_size="~2-3 GB",
                notes="Wersja resize, zwykle szybsza do eksperymentow.",
                kaggle_slugs=(
                    "xhlulu/flickrfaceshq-dataset-nvidia-resized-256px",
                ),
            ),
            DatasetVariantSpec(
                key="1024",
                resolution="1024x1024",
                approx_size="~13-15 GB",
                notes="Pelna jakosc FFHQ. Mirror na Hugging Face zamiast blednego Kaggle thumbnails128x128.",
                hf_repo="marcosv/ffhq-dataset",
                download_fn_name="huggingface",
            ),
        ),
        default_variant="1024",
        notes="Pobieranie przez Kaggle CLI lub Hugging Face.",
    ),
    "celebahq": DatasetSpec(
        key="celebahq",
        display_name="CelebA-HQ",
        image_count="30,000",
        download_fn_name="kaggle",
        variants=(
            DatasetVariantSpec(
                key="256",
                resolution="256x256",
                approx_size="~300-500 MB",
                notes="Wariant resize 256x256.",
                kaggle_slugs=(
                    "badasstechie/celebahq-resized-256x256",
                ),
            ),
            DatasetVariantSpec(
                key="512",
                resolution="512x512",
                approx_size="~1-3 GB",
                notes="Mirror 512x512; walidacja wymusza 30k obrazow 512x512.",
                kaggle_slugs=(
                    "vincenttamml/celebamaskhq512",
                ),
            ),
        ),
        default_variant="256",
        notes="Pobieranie przez Kaggle CLI.",
    ),
}

DOWNLOADERS: dict[str, Downloader] = {
    "torchvision": _download_torchvision,
    "kaggle": _download_kaggle,
    "huggingface": _download_huggingface,
    "manual": _download_manual,
}


def _print_dataset_table() -> None:
    print("\nDostepne datasety:\n")
    print(f"{'key':<10} {'wariant':<10} {'rozdz.':<11} {'obrazy':<12} {'rozmiar':<16} opis")
    print("-" * 120)
    for spec in DATASETS.values():
        for variant in spec.variants:
            variant_key = variant.key
            if variant.key == spec.default_variant:
                variant_key = f"{variant.key}*"
            notes = variant.notes
            if spec.notes:
                notes = f"{notes} {spec.notes}".strip()
            print(
                f"{spec.key:<10} {variant_key:<10} {variant.resolution:<11} "
                f"{spec.image_count:<12} {variant.approx_size:<16} {notes}"
            )
    print("\n* wariant domyslny")
    print()


def _variant_map(spec: DatasetSpec) -> dict[str, DatasetVariantSpec]:
    return {variant.key: variant for variant in spec.variants}


def _parse_dataset_token(token: str) -> DatasetSelection:
    if "@" not in token:
        spec = DATASETS.get(token)
        if spec is None:
            raise ValueError(f"Nieznany dataset: {token}")
        return DatasetSelection(dataset_key=token, variant_key=spec.default_variant)

    dataset_key, variant_key = token.split("@", 1)
    spec = DATASETS.get(dataset_key)
    if spec is None:
        raise ValueError(f"Nieznany dataset: {dataset_key}")
    if variant_key not in _variant_map(spec):
        known = ", ".join(_variant_map(spec).keys())
        raise ValueError(
            f"Nieznany wariant `{variant_key}` dla `{dataset_key}`. Dostepne: {known}"
        )
    return DatasetSelection(dataset_key=dataset_key, variant_key=variant_key)


def _expand_dataset_selection(selected: list[str], all_variants: bool) -> list[DatasetSelection]:
    result: list[DatasetSelection] = []
    seen: set[tuple[str, str]] = set()

    def _append(selection: DatasetSelection) -> None:
        key = (selection.dataset_key, selection.variant_key)
        if key not in seen:
            seen.add(key)
            result.append(selection)

    for item in selected:
        if item == "all":
            for spec in DATASETS.values():
                variant_keys = [v.key for v in spec.variants]
                if not all_variants:
                    variant_keys = [spec.default_variant]
                for variant_key in variant_keys:
                    _append(DatasetSelection(dataset_key=spec.key, variant_key=variant_key))
            continue

        _append(_parse_dataset_token(item))

    return result


def _dataset_choices_help() -> str:
    lines = [
        "Dozwolone formaty: <dataset>, <dataset>@<wariant>, all",
        "Przyklady: mnist, ffhq@256, ffhq@1024, all",
        "Dostepne warianty:",
    ]
    for spec in DATASETS.values():
        variants = ", ".join(v.key for v in spec.variants)
        lines.append(f"- {spec.key}: {variants} (default: {spec.default_variant})")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pobieranie datasetow GAN do wspolnego katalogu danych.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        default="~/edx/datasets/raw",
        help="Katalog docelowy dla datasetow.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        metavar="NAME",
        help=(
            "Dataset do pobrania; wspiera <dataset> lub <dataset>@<wariant>. "
            "Uzyj wielokrotnie dla wielu pozycji."
        ),
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Dla --dataset all pobierz wszystkie warianty rozdzielczosci.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Wypisz tabele datasetow (rozdzielczosc, rozmiar i liczba obrazow).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pokaz co zostanie uruchomione, bez pobierania.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Pomin walidacje liczby obrazow i rozdzielczosci po pobraniu.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Nie pobieraj, tylko zwaliduj istniejace dane.",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Przed pobraniem usun istniejacy katalog wariantu. Przydatne po blednym poprzednim pobraniu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        _print_dataset_table()
        if not args.dataset:
            return 0

    if not args.dataset:
        print("Podaj co najmniej jeden --dataset <name> albo --dataset all.", file=sys.stderr)
        print(_dataset_choices_help(), file=sys.stderr)
        return 2

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        selected = _expand_dataset_selection(args.dataset, all_variants=args.all_variants)
    except ValueError as exc:
        print(f"BLAD: {exc}", file=sys.stderr)
        print(_dataset_choices_help(), file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []

    print(f"Root danych: {root}")
    print(
        "Wybrane datasety: "
        + ", ".join(f"{item.dataset_key}@{item.variant_key}" for item in selected)
    )

    for item in selected:
        spec = DATASETS[item.dataset_key]
        variant = _variant_map(spec)[item.variant_key]
        try:
            target_path = _target_root_for_variant(spec, variant, root)
            if args.validate_only:
                print(
                    f"\n[{spec.display_name} | {variant.resolution} | wariant {variant.key}] "
                    "walidacja (bez pobierania)..."
                )
                if not target_path.exists():
                    raise RuntimeError(f"Brak katalogu do walidacji: {target_path}")
            else:
                print(
                    f"\n[{spec.display_name} | {variant.resolution} | wariant {variant.key}] "
                    "pobieranie..."
                )
                downloader_name = _effective_download_fn_name(spec, variant)
                downloader = DOWNLOADERS[downloader_name]
                target_path = downloader(spec, variant, root, args)

            if not args.dry_run and not args.skip_validate:
                _validate_download(spec, variant, target_path)
                print(
                    f"[OK] Walidacja: {spec.display_name}@{variant.key} "
                    f"({variant.resolution}, {spec.image_count} obrazow)"
                )

            print(f"[OK] {spec.display_name}@{variant.key} -> {target_path}")
        except Exception as exc:  # noqa: BLE001
            failures.append((f"{spec.display_name}@{variant.key}", str(exc)))
            print(f"[BLAD] {spec.display_name}@{variant.key}: {exc}")

    print("\n--- Raport ---")
    for item in selected:
        spec = DATASETS[item.dataset_key]
        variant = _variant_map(spec)[item.variant_key]
        print(
            f"- {spec.display_name}@{variant.key}: {spec.image_count} obrazow, "
            f"{variant.resolution}, {variant.approx_size}"
        )

    if failures:
        print("\nNieudane pobrania:")
        for name, err in failures:
            print(f"- {name}: {err}")
        return 1

    print("\nGotowe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
