# Plan: Wydzielenie czystego baseline'u `e000-01-r3gan-baseline`

**Data:** 2026-03-16  
**Źródło:** `e001-02-r3gan-baseline`  
**Cel:** Nowy folder `e000-01-r3gan-baseline` — najbardziej podstawowa wersja R3GAN bez żadnych eksperymentalnych modyfikacji. Służy jako zamrożony punkt startowy dla kolejnych eksperymentów.

---

## Kontekst i uzasadnienie

`e001-02-r3gan-baseline` zawiera zarówno czysty rdzeń R3GAN, jak i całą warstwę badawczą dodaną w trakcie eksperymentów faz A–E:

- wavelet-aware dyskryminator (`WaveletR3GANDiscriminator`)
- matched-capacity control (`MatchedCapacityR3GANDiscriminator`)
- regularyzatory częstotliwościowe po stronie generatora (`WaveletStatRegularizer` / `WaveReg`, `FFTStatRegularizer` / `FFTReg`)
- dynamiczne planowanie wag regularyzatorów (piecewise schedules)
- FID-gated activation dla regularyzatorów
- aux branch gate warmup
- metryki spektralne (RPSE + WBED)
- kilkanaście konfiguracji eksperymentalnych (`phase_*`, `ablation_*`, `sweep_*`)
- 5 plików testów dla w/w rozszerzeń

Celem `e000-01-r3gan-baseline` jest zachowanie **tylko rdzenia** — tak żeby każdy kolejny eksperyment mógł go wziąć jako punkt startowy z gwarancją, że nie ma w nim żadnych niezamierzonych modyfikacji.

---

## Struktura docelowa `e000-01-r3gan-baseline`

```
e000-01-r3gan-baseline/
    __init__.py
    conftest.py
    pytest.ini
    r3gan-source.py          ← okrojony (patrz niżej)
    requirements.txt
    run.py
    src/
        __init__.py
        config_loader.py     ← okrojony (patrz niżej)
        data.py              ← bez zmian
        experiment.py        ← okrojony (patrz niżej)
        gan_metrics.py       ← okrojony (patrz niżej)
        configs/
            base.yaml        ← okrojony (patrz niżej)
            smoke.yaml
            fast.yaml
            overnight.yaml
            README.md        ← nowy, uproszczony
    tests/
        test_baseline.py     ← nowy plik (patrz niżej)
```

---

## Szczegółowa tabela decyzji: co przenieść, co uprościć, co pominąć

### 1. `r3gan-source.py`

| Symbol / sekcja | Decyzja | Uzasadnienie |
|---|---|---|
| `setup_nvidia_performance` | **przenieść bez zmian** | infrastruktura, potrzebna zawsze |
| `_fan_in`, `msr_init_`, `zero_last_conv_`, `_num_groups` | **przenieść bez zmian** | prymitywy inicjalizacyjne używane przez wszystkie klasy |
| `BiasAct2d`, `Conv2dNoBias` | **przenieść bez zmian** | budulec wszystkich bloków modelu |
| `ResidualBlock` | **przenieść bez zmian** | core architektura G i D |
| `UpsampleLayer`, `DownsampleLayer` | **przenieść bez zmian** | core architektura G i D |
| `GenerativeBasis`, `DiscriminativeBasis` | **przenieść bez zmian** | core architektura G i D |
| `GeneratorStage`, `DiscriminatorStage` | **przenieść bez zmian** | core architektura G i D |
| `_assert_power_of_two_resolution` | **przenieść bez zmian** | walidacja parametrów |
| `build_stage_channels` | **przenieść bez zmian** | używane wszędzie do budowy modeli |
| `R3GANGenerator` | **przenieść bez zmian** | core |
| `R3GANDiscriminator` | **przenieść bez zmian** | core |
| `R3GANPreset` | **przenieść bez zmian** | convenience factory, czysta |
| `R3GANLoss` (+ `zero_centered_gradient_penalty`, `prepare_condition`, `update_ema`) | **przenieść bez zmian** | core training logic |
| `TrainerConfig` | **przenieść bez zmian** | core |
| `R3GANTrainer` | **przenieść, ale okroić** | usunąć pola `wave_reg`, `fft_reg`, `wave_reg_active`, `fft_reg_active`; usunąć metody `set_aux_branch_gate`, `set_wave_reg_weight`, `set_wave_reg_active`, `set_fft_reg_weight`, `set_fft_reg_active`; uprosić `_generator_step` — usunąć `_apply_regs` i rejestrację `extra_reg_metrics`; zachować `_discriminator_step`, `train_step`, `sample` |
| `parse_batch`, `fit` | **przenieść bez zmian** | utility |
| **`WaveletHFBranch`** | **POMINĄĆ** | wavelet extension |
| **`MatchedCapacityBranch`** | **POMINĄĆ** | matched-capacity extension |
| **`WaveletR3GANDiscriminator`** | **POMINĄĆ** | wavelet extension |
| **`MatchedCapacityR3GANDiscriminator`** | **POMINĄĆ** | matched-capacity extension |
| **`WaveletStatRegularizer` / `WaveReg`** | **POMINĄĆ** | frequency regularizer |
| **`FFTStatRegularizer` / `FFTReg`** | **POMINĄĆ** | frequency regularizer |
| **import `src/wavelets.py` (FixedHaarDWT2d)** | **POMINĄĆ** | używany tylko przez wavelet branch |

> **Konsekwencja:** plik `src/wavelets.py` w ogóle nie jest potrzebny w `e000`. Nie przenosić.

---

### 2. `src/config_loader.py`

| Pole `RunConfig` | Decyzja | Uzasadnienie |
|---|---|---|
| `name`, `seed`, `deterministic` | **zachować** | reprodukowalność |
| `steps`, `batch_size`, `lr_g`, `lr_d`, `betas`, `gamma`, `ema_beta` | **zachować** | core training |
| `use_amp_for_g`, `use_amp_for_d`, `channels_last`, `grad_clip` | **zachować** | performance |
| `z_dim`, `img_resolution`, `base_channels`, `channel_max`, `blocks_per_stage`, `expansion_factor`, `group_size`, `resample_mode`, `out_channels`, `in_channels` | **zachować** | architektura |
| `dataset_name`, `img_channels`, `img_size` | **zachować** | dane |
| `log_every`, `grid_every`, `ckpt_every`, `save_n_samples`, `real_grid_samples` | **zachować** | logowanie |
| `metrics_every`, `metrics_num_fake`, `metrics_fake_batch_size`, `metrics_fid_feature`, `metrics_kid_feature`, `metrics_kid_subsets`, `metrics_kid_subset_size`, `metrics_max_real`, `metrics_pr_num_samples`, `metrics_pr_k`, `metrics_lpips_num_pairs`, `metrics_lpips_pool_size`, `metrics_amp_dtype` | **zachować** | ewaluacja FID/KID/PR/LPIPS |
| `out_dir`, `data_dir` | **zachować** | ścieżki |
| **`wavelet_enabled`** | **USUNĄĆ** | |
| **`matched_capacity_enabled`** | **USUNĄĆ** | |
| **`wavelet_type`** | **USUNĄĆ** | |
| **`wavelet_level`** | **USUNĄĆ** | |
| **`wavelet_hf_only`** | **USUNĄĆ** | |
| **`wavelet_fuse_after_stage`** | **USUNĄĆ** | |
| **`wavelet_branch_mid_scale`** | **USUNĄĆ** | |
| **`wavelet_init_gate`** | **USUNĄĆ** | |
| **`aux_branch_gate_warmup_enabled`** | **USUNĄĆ** | |
| **`aux_branch_gate_warmup_start_step`** | **USUNĄĆ** | |
| **`aux_branch_gate_warmup_end_step`** | **USUNĄĆ** | |
| **`aux_branch_gate_warmup_start_value`** | **USUNĄĆ** | |
| **`aux_branch_gate_warmup_end_value`** | **USUNĄĆ** | |
| **`wave_reg_enabled`** ... wszystkie `wave_reg_*` (12 pól) | **USUNĄĆ** | |
| **`fft_reg_enabled`** ... wszystkie `fft_reg_*` (12 pól) | **USUNĄĆ** | |
| **`metrics_spectral`** | **USUNĄĆ** | metryki spektralne są rozszerzeniem |
| **`metrics_spectral_num_images`** | **USUNĄĆ** | |
| **`metrics_spectral_rpse_num_bins`** | **USUNĄĆ** | |

> **Razem:** usunąć ~40 pól z `RunConfig`. Zachować ~35 pól.

Funkcja `_auto_out_dir` — **zachować bez zmian** (automatyczne nazewnictwo katalogów wynikowych).  
Klasa `ConfigLoader` — **zachować bez zmian**.

---

### 3. `src/experiment.py`

| Funkcja / sekcja | Decyzja | Co zmienić |
|---|---|---|
| Importy modułu `r3gan-source.py` | **uprościć** | usunąć `WaveletR3GANDiscriminator`, `MatchedCapacityR3GANDiscriminator`, `WaveReg`, `FFTReg`; zostawić `R3GANGenerator`, `R3GANDiscriminator`, `R3GANTrainer`, `TrainerConfig`, `build_stage_channels`, `setup_nvidia_performance`, `parse_batch` |
| `_build_models` | **uprościć** | usunąć obsługę `wavelet_enabled` i `matched_capacity_enabled`; zawsze budować `R3GANDiscriminator` |
| `_step_to_kimg` | **zachować** | |
| `_update_ema` | **zachować** | |
| `_format_eta`, `_format_eta_finish_ts` | **zachować** | |
| `_count_remaining_metric_evals` | **zachować** | |
| `_estimate_remaining_seconds` | **zachować** | |
| `_load_metrics_elapsed_average` | **zachować** | |
| `_update_fid_auc` | **zachować** | |
| `_lerp` | **USUNĄĆ** | używana tylko przez schedule/gate, które są usuwane |
| **`_compute_aux_branch_gate`** | **USUNĄĆ** | aux branch warmup — extension |
| **`_compute_piecewise_weight`** | **USUNĄĆ** | regularizer schedule — extension |
| **`_resolve_fid_gated_activation`** | **USUNĄĆ** | FID gate — extension |
| `_make_csv_logger` | **uprościć** | usunąć kolumny: `aux_branch_gate`, `last_fid_for_gates`, `wave_reg_weight_eff`, `wave_reg_active`, `fft_reg_weight_eff`, `fft_reg_active`, wszystkie `wave_*`, `fft_*`, `rpse`, `wbed` |
| `_save_grid` | **zachować** | |
| `_save_real_grid` | **zachować** | |
| `_export_real_samples` | **zachować** | |
| `_export_samples` | **zachować** | |
| `_validate_metrics_config` | **uprościć** | usunąć walidacje `wave_reg_fid_gate_enabled`, `fft_reg_fid_gate_enabled`, `aux_branch_gate_warmup_*`, `wave_reg_schedule_*`, `fft_reg_schedule_*`, `metrics_spectral_*` |
| `_build_metrics_suite` | **uprościć** | usunąć `spectral_enabled`, `spectral_num_images`, `spectral_rpse_num_bins` |
| `_count_params`, `_save_model_info` | **zachować** | |
| `train()` — główna pętla | **uprościć** | usunąć całą logikę `aux_branch_gate`, `wave_reg_*`, `fft_reg_*` w pętli; usunąć `last_fid_for_gates`, `wave_reg_latched_active`, `fft_reg_latched_active`; uprościć `row` w CSV loggerze; usunąć `rpse`/`wbed` z metrics row |

---

### 4. `src/gan_metrics.py`

| Sekcja | Decyzja | Uzasadnienie |
|---|---|---|
| `DependencyError`, `GANMetricsConfig` | **zachować** | usunąć tylko `spectral_enabled`, `spectral_num_images`, `spectral_rpse_num_bins` z dataclass |
| `TorchvisionInceptionPool3` | **zachować** | używane przez PR |
| **`_haar_dwt2d`** | **USUNĄĆ** | spektralne extension |
| **`compute_radial_power_spectrum`** | **USUNĄĆ** | RPSE |
| **`compute_rpse`** | **USUNĄĆ** | RPSE |
| **`compute_wavelet_band_energies`** | **USUNĄĆ** | WBED |
| **`compute_wbed`** | **USUNĄĆ** | WBED |
| `GANMetricsSuite` | **uprościć** | usunąć `_real_spectral_images`, obsługę `spectral_enabled` w `prepare_real` i `evaluate_generator`; zachować FID/KID/PR/LPIPS |
| `format_metrics` | **zachować** | |

> **Uwaga:** Usunięcie `_haar_dwt2d` itp. z `gan_metrics.py` oznacza, że `e000` nie ma żadnej metryki spektralnej. Jest to zamierzone — metryki spektralne (RPSE, WBED) są rozszerzeniem eksperymentalnym, nie częścią podstawowego baseline'u.

---

### 5. `src/configs/` — pliki YAML

| Plik | Decyzja | Uzasadnienie |
|---|---|---|
| `base.yaml` | **przenieść i uprościć** | usunąć wszystkie `wavelet_*`, `matched_capacity_*`, `wave_reg_*`, `fft_reg_*`, `aux_branch_gate_*`, `metrics_spectral*` |
| `smoke.yaml` | **przenieść bez zmian** | bazowy smoke test R0 |
| `fast.yaml` | **przenieść bez zmian** | szybki test |
| `overnight.yaml` | **przenieść bez zmian** | pełny run |
| **`matched_capacity.yaml`** | **POMINĄĆ** | extension |
| **`smoke_matched_capacity.yaml`** | **POMINĄĆ** | extension |
| **`smoke_waved_wavereg.yaml`** | **POMINĄĆ** | extension |
| **`smoke_waved_fftreg.yaml`** | **POMINĄĆ** | extension |
| **`phase_b_r0_baseline_32.yaml`** | **POMINĄĆ** | historyczny eksperyment |
| **`phase_b_r0_baseline_64.yaml`** | **POMINĄĆ** | historyczny eksperyment |
| **`phase_b_r1_matched_capacity_32.yaml`** | **POMINĄĆ** | extension |
| **`phase_b_r1_matched_capacity_64.yaml`** | **POMINĄĆ** | extension |
| **`phase_b_r2_waved_32.yaml`** | **POMINĄĆ** | extension |
| **`phase_b_r2_waved_64.yaml`** | **POMINĄĆ** | extension |
| **Wszystkie `phase_c_*`** (9 plików) | **POMINĄĆ** | historyczne eksperymenty |
| **Wszystkie `ablation_*`** (3 pliki) | **POMINĄĆ** | historyczne eksperymenty |
| **Wszystkie `phase_d_*`** (6 plików) | **POMINĄĆ** | historyczne eksperymenty |
| **Wszystkie `phase_e_*`** (4 pliki) | **POMINĄĆ** | historyczne eksperymenty |
| `README.md` | **zastąpić nowym** | opisać tylko 4 profile: `base`, `smoke`, `fast`, `overnight` |

---

### 6. `tests/`

| Plik | Decyzja | Uzasadnienie |
|---|---|---|
| **`test_wavelets.py`** | **POMINĄĆ** | testuje `FixedHaarDWT2d`/`FixedHaarIDWT2d` z `src/wavelets.py`, która w `e000` nie istnieje |
| **`test_wavelet_branch.py`** | **POMINĄĆ** | testuje `WaveletHFBranch`, `MatchedCapacityBranch`, `WaveletR3GANDiscriminator`, `MatchedCapacityR3GANDiscriminator` — wszystkie usunięte |
| **`test_stat_regularizers.py`** | **POMINĄĆ** | testuje `WaveletStatRegularizer`, `FFTStatRegularizer` — usunięte |
| **`test_spectral_metrics.py`** | **POMINĄĆ** | testuje RPSE/WBED z `gan_metrics.py` — usunięte |
| **`test_training_controls.py`** | **POMINĄĆ** | testuje `_compute_aux_branch_gate`, `_compute_piecewise_weight`, `_resolve_fid_gated_activation` — usunięte |
| **`test_baseline.py`** | **STWORZYĆ NOWY** | patrz niżej |

**Zawartość nowego `tests/test_baseline.py`** (minimalne testy sanity-check rdzenia):

1. `test_build_stage_channels_returns_correct_length` — sprawdza liczbę stadiów dla różnych rozdzielczości
2. `test_generator_forward_shape` — G produkuje tensor `(B, out_ch, H, W)`
3. `test_discriminator_forward_shape` — D produkuje wektor `(B,)`
4. `test_trainer_train_step_returns_expected_keys` — krok treningowy zwraca `d_loss`, `g_loss`, `d_adv`, `g_adv`, `r1`, `r2`, `real_score_mean`, `fake_score_mean`
5. `test_trainer_ema_diverges_from_live_after_training` — po kilku krokach G i G_ema różnią się
6. `test_config_loader_roundtrip` — zapis i odczyt YAML daje ten sam `RunConfig`
7. `test_get_config_smoke_profile_overrides_steps` — profil smoke ma mniej kroków niż base

---

### 7. Pliki korzenia eksperymentu

| Plik | Decyzja | Co zmienić |
|---|---|---|
| `run.py` | **przenieść, drobna edycja** | zmienić stringi `e001-02-r3gan-baseline` → `e000-01-r3gan-baseline` w docstringu i printach |
| `conftest.py` | **przenieść bez zmian** | |
| `pytest.ini` | **przenieść bez zmian** | |
| `requirements.txt` | **przenieść bez zmian** | te same zależności (torch, torchvision, torchmetrics, torch-fidelity, pyyaml) |
| `__init__.py` | **przenieść bez zmian** | |
| **`r3gan-source.py`** | **przenieść i okroić** | jak opisano w sekcji 1 |
| **`roadmap.md`** | **POMINĄĆ** | dokument historyczny e001 |
| **`roadmapa-2.md`** | **POMINĄĆ** | dokument historyczny e001 |
| **`plan.md`** | **POMINĄĆ** | dokument historyczny e001 |
| **`artifacts-*`** (wszystkie katalogi wynikowe) | **POMINĄĆ** | artefakty eksperymentów e001 |
| **`summary/`** | **POMINĄĆ** | wyniki eksperymentów e001 |

---

### 8. Katalogi i pliki spoza `e001-02-r3gan-baseline`

| Plik/folder | Decyzja | Uzasadnienie |
|---|---|---|
| `shared/utils/` (set_seed, CSVLogger) | **pozostawić współdzielone** | `e000` importuje z `shared.utils` tak samo jak `e001`; nie kopiować |
| `runs/` (katalogi z runs/ na poziomie repo) | **POMINĄĆ** | artefakty e001 |
| `scripts/run_e001_02.sh` itp. | **POMINĄĆ** | skrypty e001; ewentualnie dodać nowe `scripts/run_e000_01.sh` osobno |

---

## Podsumowanie rozmiaru zmiany

| Kategoria | e001-02 (źródło) | e000-01 (cel) |
|---|---|---|
| Pola `RunConfig` | ~75 pól | ~35 pól |
| Klasy w `r3gan-source.py` | 20 klas | 14 klas |
| Pliki `src/configs/*.yaml` | 30+ | 4 (`base`, `smoke`, `fast`, `overnight`) |
| Pliki testów | 5 | 1 (nowy) |
| Pliki źródłowe w `src/` | 6 (`config_loader`, `data`, `experiment`, `gan_metrics`, `wavelets`, `__init__`) | 5 (bez `wavelets.py`) |
| Linie kodu `experiment.py` | 838 | ~380 (szacunkowo po usunięciu gate/schedule/reg) |
| Linie kodu `r3gan-source.py` | 944 | ~580 (szacunkowo po usunięciu 6 klas i reg. kodu) |

---

## Otwarte decyzje do podjęcia przed realizacją

1. **`src/gan_metrics.py` — zakres okrojenia:**
   - **Wariant A (rekomendowany):** zachować FID/KID/PR/LPIPS, usunąć RPSE/WBED i `_haar_dwt2d`. Plik zostaje w e000, jest pełnoprawnym narzędziem ewaluacyjnym.
   - **Wariant B:** zachować cały plik bez zmian, ale wyłączyć spektralne przez domyślnie `spectral_enabled=False` (już jest domyślnie False). Prościej w realizacji, ale plik zawiera nieużywany martwy kod.
   - **Wariant C:** pominąć `gan_metrics.py` całkowicie i wyłączyć metryki przez `metrics_every=0` w `base.yaml`. Najprostsza wersja, ale traci środowisko ewaluacyjne.

2. **Głębokość okrojenia pętli treningowej:** Czy w `e000` pętla `train()` ma zawierać pełne logowanie ETA, `fid_auc_vs_kimg`, `model_info.json`, eksport `real_samples`? Rekomendacja: **tak** — to infrastruktura bazowa, nie eksperymentalna.

3. **`R3GANTrainer` — usunięcie `augment_fn`:** parametr `augment_fn` w `__init__` nigdy nie jest używany w eksperymentach e001/e000. Można go usunąć lub zostawić dla przyszłych rozszerzeń. Rekomendacja: **zostawić** (nie jest kosztowne, zwiększa elastyczność).

---

## Uwagi implementacyjne dla wykonawcy

- Wszystkie ścieżki względne w `r3gan-source.py` odwołują się do `src/wavelets.py` przez `Path(__file__).resolve().parent / "src" / "wavelets.py"`. Po usunięciu tego importu cały blok `_WAVELETS_PATH` / `_WAVELETS_SPEC` należy usunąć z nowego `r3gan-source.py`.
- W `src/experiment.py` import `r3gan-source.py` odbywa się przez `importlib.util` z dynamiczną ścieżką `Path(__file__).resolve().parents[1] / "r3gan-source.py"`. Mechanizm pozostaje bez zmian — tylko lista importowanych symboli ulega skróceniu.
- Testy w `e001` używają ścieżki `EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]` i wczytują `r3gan-source.py` przez `importlib.util`. Ten mechanizm pozostaje w `test_baseline.py`.
- Nowe `tests/test_baseline.py` powinno importować z `src.config_loader` i `src.experiment` przez `sys.path` — tak samo jak obecne testy w `e001`.
- `conftest.py` dodaje katalog eksperymentu do `sys.path` — **zachować bez zmian**.
- Katalog `e000-01-r3gan-baseline` powinien być samodzielny: nie zależy od niczego w `e001-02-r3gan-baseline`. Jedyna zależność zewnętrzna to `shared/utils` na poziomie repo.

