# Plan: porządne zarządzanie zbiorami danych dla GAN

## Cel
Uprościć i ustandaryzować warstwę danych w projekcie (obecnie monolityczne `get_dataloader` w `src/data.py`) tak, aby:
- łatwo dodawać nowe datasety bez rozrastających się `if/elif`,
- korzystać z jednego globalnego katalogu danych na dużym dysku,
- mieć automatyczne pobieranie przez biblioteki (`torchvision`, `datasets`),
- zachować powtarzalność eksperymentów.

## Zakres (co zmieniamy)
- Refaktor `src/data.py` na modularny system:
  - `DatasetConfig` (konfiguracja wejściowa),
  - `DatasetSpec` (metadane i reguły per dataset),
  - registry/factory (`register_dataset`, `create_dataset`),
  - buildery transformacji i dataloadera.
- Wspólny standard ścieżek i cache dla wszystkich datasetów.
- Lista rekomendowanych datasetów GAN (od lekkich do ciężkich) i kryteria wyboru.

## Architektura docelowa

### 1) Konfiguracja
Wprowadzić `DatasetConfig` (np. dataclass):
- `name`: nazwa datasetu (`cifar10`, `celeba`, `ffhq`, ...),
- `resolution`: docelowa rozdzielczość,
- `channels`: liczba kanałów,
- `split`: `train`/`val`/`test`,
- `root`: opcjonalny root datasetu,
- `download`: `True/False`,
- `source`: `torchvision|huggingface|custom`,
- `num_workers`, `batch_size`, `seed`, `pin_memory`, itp.

### 2) Registry datasetów
Zamiast `if/elif` użyć rejestru:
- `DATASET_REGISTRY: dict[str, DatasetSpec]`,
- każdy `DatasetSpec` definiuje:
  - loader źródłowy (torchvision/HF/custom),
  - domyślne transformacje,
  - wymagania (licencja, split, kanały),
  - czy wspiera auto-download.

### 3) Podział odpowiedzialności
- `build_dataset(config)` — tylko obiekt datasetu.
- `build_transforms(config, spec)` — tylko transformacje.
- `build_dataloader(dataset, config)` — tylko `DataLoader`.
- `seed_worker` i generator PRNG jako współdzielone utility.

## Dobre praktyki: jeden duży magazyn danych

## Struktura katalogów (globalna)
Przykład (Linux):
- `/mnt/datasets/raw` — surowe dane (niezmienne),
- `/mnt/datasets/processed` — dane przetworzone/rekodowane,
- `/mnt/datasets/cache` — cache bibliotek (można czyścić).

## Zasady operacyjne
- Jeden wspólny root danych na host (`DATA_ROOT=/mnt/datasets`).
- Brak duplikacji datasetów per projekt/branch.
- W projekcie używać symlinków lub konfiguracji ENV, nie kopiowania.
- `raw` traktować jako immutable; czyścić tylko `cache`.
- Trzymać manifest datasetu: źródło, wersja/snapshot, checksum.

## Auto-download (bez ręcznego pobierania)

### Biblioteki
- `torchvision.datasets` — MNIST/FashionMNIST/CIFAR/STL10/CelebA (zależnie od warunków źródła).
- `datasets` (Hugging Face) — szeroki katalog, plus opcja streaming.
- Opcjonalnie `kagglehub`/API tylko tam, gdzie licencja i regulamin wymagają autoryzacji.

### Ujednolicenie cache przez ENV
- `DATA_ROOT=/mnt/datasets`
- `TORCH_HOME=/mnt/datasets/cache/torch`
- `HF_HOME=/mnt/datasets/cache/hf`
- `HUGGINGFACE_HUB_CACHE=/mnt/datasets/cache/hf/hub`

### Polityka działania
- `download=True` domyślnie dla wspieranych datasetów.
- Gdy auto-download niemożliwy: jasny błąd z instrukcją następnego kroku.
- Tryb offline: jeśli brak plików lokalnie, fail-fast z komunikatem.

## Top datasety do GAN (praktyczny shortlist)

### Lekkie (szybkie iteracje)
- **MNIST / FashionMNIST** — szybki smoke test i debug pętli treningowej.
- **CIFAR-10 / CIFAR-100 (32x32)** — tani benchmark porównawczy.
- **STL-10 (96x96)** — krok pośredni przed cięższymi zbiorami.

### Średnie
- **CelebA (64-128)** — klasyczny benchmark twarzy.
- **LSUN (Bedroom/Church, 128-256)** — bardziej złożone sceny.

### Ciężkie / SOTA-like
- **FFHQ (256+)** — wysoka jakość twarzy, dobry test jakości generacji.
- **CelebA-HQ (256+)** — alternatywa dla FFHQ.
- **ImageNet 64/128** — mocny benchmark skali i stabilności treningu.

## Kryteria wyboru datasetu do eksperymentu
- Cel eksperymentu: debug / porównanie metod / wynik jakościowy.
- Budżet GPU i czas treningu.
- Ryzyko etyczne i licencyjne (zwłaszcza twarze).
- Łatwość automatycznego pobierania i reprodukowalność.

## Plan wdrożenia (2-3 sprinty)

## Sprint 1: architektura i kompatybilność
- [ ] Dodać `DatasetConfig` i `DatasetSpec`.
- [ ] Rozbić `get_dataloader` na 3 buildery.
- [ ] Wprowadzić registry dla obecnych datasetów (`mnist`, `fashion_mnist`, `cifar10`, `cifar100`, `celeba`).
- [ ] Zachować kompatybilny wrapper (`get_dataloader(...)`) na czas migracji.

**Kryteria akceptacji:**
- obecne eksperymenty działają bez zmiany zachowania,
- dodanie nowego datasetu nie wymaga modyfikacji centralnego `if/elif`.

## Sprint 2: storage i auto-download
- [ ] Ustawić globalny `DATA_ROOT` + cache ENV.
- [ ] Dodać logowanie źródła danych, splitu i ścieżki cache.
- [ ] Dodać czytelne komunikaty dla trybu offline i brakujących danych.

**Kryteria akceptacji:**
- świeże środowisko pobiera CIFAR/MNIST bez manualnych kroków,
- ścieżki danych nie zależą od lokalnych hacków typu `/tmp`.

## Sprint 3: rozszerzenie i governance
- [ ] Dodać 1-2 cięższe datasety (np. LSUN/FFHQ) przez adapter.
- [ ] Spisać politykę licencji i publikacji próbek.
- [ ] Dodać checklistę „nowy dataset” do README projektu.

**Kryteria akceptacji:**
- dodanie datasetu = `DatasetSpec` + ewentualny transform builder,
- zespół ma jeden playbook danych i jednolite zasady pracy.

## Ryzyka i kontrola
- Licencje i ograniczenia użycia (research/commercial).
- Prywatność i kwestie PII dla datasetów twarzy.
- Drift reprodukowalności przy niepinowanych wersjach datasetów.
- Rosnące koszty I/O i zajętości dysku przy ciężkich zbiorach.

## Definition of Done
- Refaktor warstwy danych wdrożony i używany domyślnie.
- Dokumentacja konfiguracji `DATA_ROOT` i cache gotowa.
- Co najmniej 3 datasety uruchamiane automatycznie przez bibliotekę.
- Jasna procedura dodawania nowego datasetu bez zmian architektury.
