Jasne. Poniżej masz **nową roadmapę**, napisaną już pod Twój obecny kontekst: **Twój własny baseline R3GAN, PyTorch, MSI EdgeXpert z GB10, nacisk na uczciwy protokół eksperymentalny i wdrożenie przez „głupszy” model**.

Będę traktował to jako **roadmapę badawczo-implementacyjną**, nie jako szkic paperu pod szybki wynik.

---

# Roadmapa v2: Frequency-aware rozszerzenie baseline’u R3GAN

## 0. Założenie główne

Ta gałąź badawcza ma odpowiadać na pytanie:

> **Czy można poprawić stabilność i/lub jakość generacji baseline’u R3GAN przez kontrolowane wprowadzenie biasu częstotliwościowego, bez rozwalania uczciwości porównania?**

W Twoim kodzie punkt wyjścia jest już sensowny i spójny: masz `R3GANPreset`, `R3GANGenerator`, `R3GANDiscriminator`, `R3GANLoss`, `TrainerConfig` i `R3GANTrainer`, a trening jest rozdzielony na osobny krok D i G, z EMA, Adamem, AMP dla G oraz gradient penalty liczonym osobno dla real i fake. To jest bardzo dobry fundament pod modularne rozszerzenie, zamiast przepisywania wszystkiego od nowa.

---

# 1. Co dokładnie robimy

## Wersja minimalna

Rozszerzamy tylko dyskryminator o **małą gałąź waveletową HF**, bez zmiany generatora.

To jest pierwszy, najczystszy eksperyment.

## Wersja pełna

Dodajemy drugi składnik:

* **Wavelet-aware discriminator**
* **Generator-only wavelet statistic regularization**
* **FFT control baseline**
* **matched-capacity control**

To jest wersja, która ma sens jako rozdział/sekcja doktoratu.

## Czego nie robimy na starcie

Na początku **nie** robimy:

* wavelet-output generatora,
* learnable wavelets,
* poziomu `L=2`,
* `db2`,
* mieszania tego z CLIP/DINO/I-JEPA,
* zmiany lossu adversarialnego,
* zmiany bazowej procedury treningowej.

To byłoby metodologicznie za szerokie.

---

# 2. Architektura docelowego eksperymentu

## Warianty, które będą porównywane

### R0 — Baseline

Czysty Twój R3GAN, bez zmian.

### R1 — Matched-capacity control

Dodatkowa gałąź w D o podobnym koszcie i liczbie parametrów, ale **bez DWT**.

Cel:
sprawdzić, czy zysk daje sama dodatkowa ścieżka, a nie wavelety.

### R2 — WaveD

Dyskryminator z gałęzią waveletową, tylko HF: `LH/HL/HH`.

### R3 — WaveD + WaveReg

Jak wyżej, plus lekka regularizacja statystyk HF po stronie generatora.

### R4 — WaveD + FFTReg

Jak wyżej, ale zamiast `WaveReg` dajesz regularyzację FFT jako kontrolę.

To jest główny zestaw.

---

# 3. Jak to wpiąć w Twój obecny kod

## 3.1. Co już masz i czego nie ruszać

Nie ruszałbym na starcie:

* `R3GANGenerator`
* logiki `R3GANPreset.build()`
* głównego lossu relatywistycznego
* `gamma`
* EMA
* `Adam(lr=2e-4, betas=(0.0, 0.99))`
* `channels_last`
* AMP dla generatora

Te elementy są już częścią Twojego baseline’u i powinny zostać zamrożone na potrzeby tej gałęzi badań. W pliku są one ustawione wprost w `TrainerConfig` i `R3GANTrainer`.

---

## 3.2. Najważniejsza decyzja konstrukcyjna

Najmniej inwazyjna i najbardziej sensowna wersja to:

* nie przepisywać całego `R3GANDiscriminator`,
* tylko zrobić jego rozszerzenie, np.

  * `WaveletR3GANDiscriminator`
  * albo refaktor obecnego `R3GANDiscriminator` o opcjonalny branch.

W Twoim kodzie D wygląda tak:

* `from_rgb`
* potem `self.stages`
* potem opcjonalna projekcja warunkowa przez `cond_embed`.

To daje bardzo naturalny punkt wpięcia.

---

# 4. Nowa struktura plików

Polecam taki podział:

```text
r3gan_source.py                 # Twoja obecna baza (lub lekko odchudzona)
wavelets.py                     # DWT / IDWT
wavelet_branches.py             # branch do D i matched-capacity control
freq_regularizers.py            # WaveReg i FFTReg
metrics_freq.py                 # RPSE, WBED, helpery diagnostyczne
trainer_wavelet.py              # rozszerzony trainer / eksperymentalny trainer
experiment_configs.py           # nowe dataclasses konfiguracyjne
run_experiment.py               # runner
analyze_results.py              # składanie tabel, wykresów, agregacja seedów
tests/
    test_wavelets.py
    test_wavelet_branch.py
    test_regularizers.py
```

Jeśli chcesz minimalizować liczbę plików, możesz to złożyć do 4:

* `wavelets.py`
* `freq_regularizers.py`
* `trainer_wavelet.py`
* `metrics_freq.py`

---

# 5. Nowe klasy i konfiguracje

## 5.1. WaveletConfig

```python
@dataclass
class WaveletConfig:
    enabled: bool = False
    wavelet_type: str = "haar"       # na start tylko haar
    level: int = 1                   # na start tylko 1
    hf_only: bool = True             # tylko LH, HL, HH
    fuse_after_stage: int = 0        # fuzja po pierwszym stage D
    branch_mid_scale: float = 0.5    # ile kanałów w gałęzi pomocniczej
    init_gate: float = 0.0           # krytyczne
```

## 5.2. FrequencyRegularizerConfig

```python
@dataclass
class FrequencyRegularizerConfig:
    wave_reg_enabled: bool = False
    fft_reg_enabled: bool = False
    lambda_wave: float = 0.02
    lambda_fft: float = 0.02
    use_log_energy: bool = True
    match_mean: bool = True
    match_std: bool = True
    ema_stats_beta: float = 0.99
    eps: float = 1e-8
    fft_num_bins: int = 16
```

## 5.3. ExperimentControlConfig

```python
@dataclass
class ExperimentControlConfig:
    matched_capacity_enabled: bool = False
    eval_every_kimg: int = 20
    total_kimg: int = 1000
    num_eval_samples: int = 50000
    num_seeds_final: int = 3
```

---

# 6. Implementacja krok po kroku

## Etap 1. DWT / IDWT

## Cel

Najpierw budujesz poprawne, banalne, testowalne wavelety.

## Implementacja

Plik: `wavelets.py`

### Klasa 1

```python
class FixedHaarDWT2d(nn.Module):
    def __init__(self, in_channels: int):
        ...
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        ...
```

### Klasa 2

```python
class FixedHaarIDWT2d(nn.Module):
    def __init__(self, in_channels: int):
        ...
    def forward(self, bands: dict[str, Tensor]) -> Tensor:
        ...
```

## Jak to zrobić

Zrób to przez stałe filtry 2x2 jako `register_buffer` i `groups=in_channels`.

Filtry Haar:

```python
LL = [[ 0.5,  0.5],
      [ 0.5,  0.5]]

LH = [[ 0.5,  0.5],
      [-0.5, -0.5]]

HL = [[ 0.5, -0.5],
      [ 0.5, -0.5]]

HH = [[ 0.5, -0.5],
      [-0.5,  0.5]]
```

## Ważne

Na start:

* zakładaj parzyste `H, W`,
* bez paddingu,
* bez `db2`,
* bez multi-level.

## Testy obowiązkowe

Plik: `tests/test_wavelets.py`

### Test 1

`IDWT(DWT(x)) ~= x`

### Test 2

Na losowym tensorze:

* `max_abs_err`
* `mae`
* `psnr`

### Test 3

Na realnym batchu obrazów.

## Kryterium zaliczenia

Przechodzisz dalej dopiero, gdy rekonstrukcja jest praktycznie idealna.

---

## Etap 2. Gałąź waveletowa do D

## Cel

Dodać częstotliwościowy sygnał do D, ale bez przebudowy całej architektury.

## Punkt wpięcia

W Twoim D jest:

* `from_rgb`
* potem `self.stages` jako lista kolejnych etapów.

Najlepszy punkt fuzji:

* po `self.stages[0]`

Czyli:

1. obraz RGB trafia normalnie do `from_rgb`
2. przechodzi przez pierwszy stage D
3. równolegle obraz trafia do DWT
4. HF branch daje mapę cech zgodną z wyjściem pierwszego stage’u
5. dodajesz tę mapę przez residual fusion

## Dlaczego tak

Bo:

* rozdzielczość się zgadza (`H/2 x W/2`),
* liczba kanałów da się wygodnie dopasować,
* ingerencja w D jest mała.

## Nowa klasa

```python
class WaveletHFBranch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        ...
    def forward(self, x_rgb: Tensor) -> Tensor:
        ...
```

## Wnętrze brancha

1. DWT
2. wybór `LH/HL/HH`
3. concat po kanałach
4. 2 małe convolucje
5. zero-init ostatniej konwolucji

Przykład:

```python
self.dwt = FixedHaarDWT2d(in_channels=3)

hf_ch = 3 * 3  # 3 kanały RGB * 3 pasma HF
self.conv1 = Conv2dNoBias(hf_ch, mid_channels, 3)
self.act1 = BiasAct2d(mid_channels)
self.conv2 = Conv2dNoBias(mid_channels, out_channels, 3)
zero_last_conv_(self.conv2.conv)

self.gate = nn.Parameter(torch.tensor(0.0))
```

## Forward

```python
bands = self.dwt(x_rgb)
hf = torch.cat([bands["LH"], bands["HL"], bands["HH"]], dim=1)
y = self.conv2(self.act1(self.conv1(hf)))
return self.gate * y
```

---

## Etap 3. Refaktor dyskryminatora

## Wariant rekomendowany

Zamiast zmieniać starą klasę w miejscu, zrób nową:

```python
class WaveletR3GANDiscriminator(R3GANDiscriminator):
    ...
```

albo skopiuj logikę `R3GANDiscriminator` i dodaj jawnie branch.

To będzie czytelniejsze dla „głupszego” implementującego.

## Pseudokod forward

```python
def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
    rgb = x
    x = self.from_rgb(x)

    # pierwszy stage główny
    x = self.stages[0](x)

    # dodatkowa gałąź
    if self.wavelet_branch is not None:
        x = x + self.wavelet_branch(rgb)

    # reszta stages
    for stage in self.stages[1:]:
        x = stage(x)

    if self.cond_embed is not None:
        ...
    return x.view(x.size(0))
```

## Uwaga krytyczna

`wavelet_branch` ma brać **dokładnie to samo wejście**, które trafia do D.
Jeżeli augmentacja jest aplikowana przed D przez `R3GANLoss._prep()`, to branch ma widzieć już obraz po augmentacji, bo jest częścią D. W Twoim kodzie `_prep()` działa właśnie w lossie przed wywołaniem `self.D(...)`. 

---

## Etap 4. Matched-capacity control

## Cel

Odpowiedzieć na zarzut:

> poprawa wynika z większej pojemności modelu, a nie z waveletów.

## Implementacja

Tworzysz branch o bardzo podobnej strukturze:

```python
class MatchedCapacityBranch(nn.Module):
    def __init__(self, out_channels: int, mid_channels: int):
        ...
    def forward(self, x_rgb: Tensor) -> Tensor:
        x = F.avg_pool2d(x_rgb, kernel_size=2, stride=2)
        x = self.conv2(self.act1(self.conv1(x)))
        return self.gate * x
```

Tu wejście to zwykły RGB po `avg_pool2d`, bez DWT.

Branch:

* ma podobną liczbę parametrów,
* podobny koszt,
* tę samą fuzję,
* ten sam `gate=0`,
* ten sam zero-init końca.

To jest bardzo ważny eksperyment kontrolny.

---

## Etap 5. Refaktor trainera

Twój obecny `train_step()` robi:

* osobny step D,
* osobny step G,
* opiera się na `self.loss.discriminator_loss(...)` i `self.loss.generator_loss(...)`. 

To jest sensowne, ale pod regularizer częstotliwościowy trzeba zrobić mały refaktor.

## Co zmienić

Podziel `train_step()` na:

* `_discriminator_step(...)`
* `_generator_step(...)`
* `train_step(...)` jako wrapper

## Dlaczego

Bo w kroku G potrzebujesz:

* mieć `fake = G(z, cond)`
* policzyć `g_adv`
* policzyć `wave_reg(fake, real)`
* zsumować

A obecny `generator_loss()` liczy tylko adversarial część. 

## Nowa logika generator step

```python
fake = self.G(z, cond)
fake_logits = self.D(self.loss._prep(fake), cond)
real_logits = self.D(self.loss._prep(real.detach()), cond)

rel = fake_logits - real_logits
g_adv = F.softplus(-rel).mean()

g_reg = 0.0
if self.wave_reg is not None:
    g_reg = self.wave_reg(fake, real)

if self.fft_reg is not None:
    g_reg = self.fft_reg(fake, real)

g_total = g_adv + g_reg
```

## Ważna decyzja

`WaveReg` i `FFTReg` dajesz **tylko do generatora**.

Nie dodajesz tego do D-loss.

---

## Etap 6. Wavelet statistic regularizer

## Cel

Wymusić zgodność rozkładu energii HF bez narzucania paired loss.

## Klasa

```python
class WaveletStatRegularizer(nn.Module):
    def __init__(self, lambda_weight: float = 0.02, ema_beta: float = 0.99, eps: float = 1e-8):
        ...
    def forward(self, fake: Tensor, real: Tensor) -> tuple[Tensor, dict[str, float]]:
        ...
```

## Jak liczyć statystyki

Dla każdego obrazu:

1. DWT
2. weź `LH, HL, HH`
3. policz:

```python
E = mean(coeff ** 2)
```

dla każdego pasma

4. najlepiej:

```python
logE = torch.log(E + eps)
```

## Potem

Po batchu:

* `mu_fake`
* `std_fake`
* `mu_real`
* `std_real`

## Stabilniejsza wersja

Dla real używaj EMA bufferów:

* `mu_real_ema`
* `std_real_ema`

## Loss

```python
loss_mu = F.l1_loss(mu_fake, mu_real_ema)
loss_std = F.l1_loss(std_fake, std_real_ema)
loss = lambda_weight * (loss_mu + 0.5 * loss_std)
```

## Dlaczego nie L2

L1 jest trochę bardziej odporne na pojedyncze wyskoki.

## Co logować

* `wave_reg_total`
* `wave_mu_loss`
* `wave_std_loss`
* `wave_fake_mu_lh`, `wave_fake_mu_hl`, `wave_fake_mu_hh`
* `wave_real_mu_lh`, ...

---

## Etap 7. FFT control regularizer

## Klasa

```python
class FFTStatRegularizer(nn.Module):
    ...
```

## Jak liczyć

1. FFT2 na obrazie
2. moc:

```python
power = real^2 + imag^2
```

3. radial binning
4. pomijasz DC
5. bierzesz np. 16 binów
6. liczysz mean/std po batchu
7. loss analogiczny jak wyżej

To musi być możliwie równoległe do `WaveletStatRegularizer`, żeby porównanie było uczciwe.

---

# 7. Plan eksperymentów

Teraz najważniejsze: **kolejność**, żeby nie utopić się w nadmiarze kombinacji.

---

## Faza A — infrastruktura i sanity

## Cel

Najpierw upewniasz się, że eksperyment da się prowadzić rzetelnie.

### Zadania

1. Zamrozić baseline config
2. Dodać runner eksperymentów
3. Dodać logging do CSV / W&B
4. Dodać metryki:

   * FID
   * KID
   * P/R
   * LPIPS-diversity
   * RPSE
   * WBED
5. Dodać testy waveletów

## Stop condition

Nie idziesz dalej, dopóki:

* baseline nie działa stabilnie,
* metryki nie liczą się poprawnie,
* DWT/IDWT nie przechodzi testów.

---

## Faza B — szybki test naukowego sensu

## Cel

Sprawdzić, czy sama gałąź waveletowa daje sygnał.

## Porównanie

* `R0`: baseline
* `R1`: matched-capacity
* `R2`: WaveD

## Ustawienia

* 1 dataset
* 1 rozdzielczość
* 1 seed
* umiarkowany budżet kimg

## Decyzja

Jeżeli:

* `R2` nie jest lepszy od `R1`,
* albo poprawa jest tylko kosmetyczna,
* albo koszt rośnie wyraźnie bez sensownego zysku,

to ta gałąź wymaga rewizji zanim przejdziesz dalej.

---

## Faza C — pełna metoda

## Porównanie

* `R0`: baseline
* `R1`: matched-capacity
* `R2`: WaveD
* `R3`: WaveD + WaveReg
* `R4`: WaveD + FFTReg

## Ustawienia

* 1 dataset
* 1 rozdzielczość
* 2 seedy

## Cel

Wyłonić finalny wariant.

---

## Faza D — eksperyment potwierdzający

## Porównanie końcowe

* `R0`: baseline
* `R_best`

## Ustawienia

* 2 datasety albo 1 dataset + 2 rozdzielczości
* 3 seedy

## Rekomendacja

Na początek lepiej:

* 2 datasety przy tej samej rozdzielczości

niż

* 1 dataset i wiele rozdzielczości.

To czyściej pokazuje generalizację.

---

# 8. Jakie dane i ustawienia są krytyczne

To są elementy, które muszą być identyczne między wariantami:

* ten sam `R3GANPreset`
* ten sam `TrainerConfig`
* ten sam `gamma`
* ten sam `ema_beta`
* ten sam optimizer
* ten sam preprocessing
* ta sama augmentacja
* ten sam batch size w obrębie porównania
* ten sam budżet w **kimg**
* ten sam harmonogram ewaluacji
* ten sam licznik próbek do FID/KID
* te same seedy dla eksperymentów finalnych
* te same latent vectors dla sample grids

W Twoim kodzie parametry bazowe są jawne, więc to bardzo łatwo zamrozić konfiguracyjnie.

---

# 9. Konkretne hiperparametry startowe

To nie są prawdy objawione, tylko **punkt wejścia**.

## Baseline

Zostaw:

* `lr_g = 2e-4`
* `lr_d = 2e-4`
* `betas = (0.0, 0.99)`
* `gamma = 10.0`
* `ema_beta = 0.999`
* `use_amp_for_g = True`
* `use_amp_for_d = False`
* `amp_dtype = torch.bfloat16`
* `channels_last = True`

To jest zgodne z Twoim obecnym trainerem.

## WaveD

* `wavelet_type = haar`
* `level = 1`
* `hf_only = True`
* `fuse_after_stage = 0`
* `gate = 0.0`
* `branch_mid_scale = 0.5`

## WaveReg

* `lambda_wave = 0.02` start
* sweep: `0.005, 0.02, 0.05`

## FFTReg

* `lambda_fft = 0.02`
* `fft_num_bins = 16`

## Ablacje dopiero później

* `all_bands = True`
* `db2`
* `level = 2`

---

# 10. Jak ocenić wynik

## Metryki główne

* FID
* KID
* Precision / Recall
* LPIPS-diversity

## Metryki pomocnicze

* RPSE
* WBED

## Dodatkowe wskaźniki bardzo warte raportowania

* liczba parametrów
* czas kroku
* peak VRAM
* AUC FID-vs-kimg
* success rate treningu

## Jak rozpoznać realną poprawę

Realna poprawa jest wtedy, gdy:

1. poprawia się FID/KID,
2. nie pogarsza się wyraźnie P/R,
3. LPIPS-diversity nie zapada się,
4. RPSE/WBED poprawiają się zgodnie z mechanizmem,
5. matched-capacity control nie robi tego samego,
6. wynik utrzymuje się między seedami.

Jeżeli poprawia się tylko RPSE/WBED, a FID/KID stoją w miejscu, to nie masz jeszcze poprawy modelu — masz raczej poprawę zgodności częstotliwościowej.

---

# 11. Sanity-checki obowiązkowe

1. `IDWT(DWT(x)) ≈ x`
2. `gate=0` daje baseline behavior
3. zero-init ostatniej conv naprawdę nie zaburza startu
4. `WaveReg` nie prowadzi do sztucznego szumu HF
5. nearest-neighbor check dla próbek
6. te same latent vectors w sample grids
7. porównanie po `kimg`, nie po samych iteracjach

---

# 12. Ablacje — ale dopiero po wyborze wariantu

Nie rób ablacjii od początku.
Najpierw wybierz sensowny wariant, potem dopiero:

### Ablacja 1

HF-only vs all-bands

### Ablacja 2

Haar vs db2

### Ablacja 3

Branch fusion:

* add
* concat + 1x1

### Ablacja 4

WaveReg:

* mean only
* mean + std

### Ablacja 5

Pozycja fuzji:

* po `stage[0]`
* po `stage[1]`

---

# 13. Co dokładnie ma zrobić „głupszy model” — task list

Poniżej masz wersję bardzo operacyjną.

## Task 1

Dodaj `wavelets.py` z:

* `FixedHaarDWT2d`
* `FixedHaarIDWT2d`

## Task 2

Dodaj testy:

* `test_dwt_idwt_reconstruction()`
* `test_dwt_band_shapes()`

## Task 3

Dodaj `wavelet_branches.py` z:

* `WaveletHFBranch`
* `MatchedCapacityBranch`

## Task 4

Dodaj nową klasę:

* `WaveletR3GANDiscriminator`

która ma:

* `from_rgb`
* `stages`
* opcjonalny `aux_branch`
* opcjonalny `cond_embed`

i forward:

* bierze `rgb = x`
* robi `x = from_rgb(x)`
* `x = stages[0](x)`
* `if aux_branch: x = x + aux_branch(rgb)`
* reszta stages
* cond head
* flatten

## Task 5

Dodaj `freq_regularizers.py` z:

* `WaveletStatRegularizer`
* `FFTStatRegularizer`

## Task 6

Refaktor `R3GANTrainer`:

* wyodrębnij `_discriminator_step`
* wyodrębnij `_generator_step`
* w `_generator_step` dodaj `g_adv + g_reg`

## Task 7

Dodaj `metrics_freq.py`:

* `compute_rpse_from_folders(...)`
* `compute_wbed_from_folders(...)`
* helpery do batchowego liczenia

## Task 8

Dodaj `run_experiment.py`:

* przyjmuje config YAML/JSON
* zapisuje `metrics.csv`
* zapisuje `config.json`
* zapisuje checkpointy
* zapisuje grids

## Task 9

Dodaj `analyze_results.py`

* agregacja seedów
* mean/std
* wykres FID-vs-kimg
* tabela wyników

---

# 14. Szkic opisu do rozprawy

## Sens badawczy

Badanie ma zweryfikować, czy kontrolowane rozszerzenie baseline’u R3GAN o pomocniczy sygnał częstotliwościowy może poprawić jakość i stabilność generacji, bez zmiany podstawowego celu adversarialnego.

## Hipoteza

Wprowadzenie gałęzi dyskryminatora operującej na pasmach wysokoczęstotliwościowych oraz lekkiej regularizacji statystyk HF po stronie generatora poprawi zgodność generowanych obrazów z rozkładem danych rzeczywistych, zwłaszcza w zakresie struktur drobnoskalowych, przy niewielkim narzucie obliczeniowym.

## Eksperyment

Porównano baseline R3GAN z wariantami rozszerzonymi o gałąź waveletową, odpowiedni wariant kontrolny o zbliżonej pojemności oraz regularizacje oparte na statystykach waveletowych i FFT.

## Interpretacja

Jeżeli poprawa utrzymuje się w FID/KID i nie prowadzi do pogorszenia pokrycia rozkładu, można interpretować ją jako dowód użyteczności biasu częstotliwościowego w nowoczesnym baseline’ie GAN.

## Ograniczenia

Efekt może zależeć od typu danych, rozdzielczości i charakteru tekstur, a poprawa zgodności częstotliwościowej nie musi zawsze przekładać się na poprawę semantyczną.

---

# 15. Końcowy werdykt

Ta roadmapa ma sens i jest **dużo lepiej dopasowana** do Twojego doktoratu niż stara wersja, bo:

* nie zmienia baseline’u,
* jest modularna,
* ma uczciwe kontrole,
* rozdziela efekt architektury od efektu samego biasu częstotliwościowego,
* jest wdrażalna na Twoim kodzie,
* daje szansę zarówno na wynik pozytywny, jak i na sensowny wynik negatywny.

Najważniejsze jest to, żebyś **nie próbował od razu robić wszystkiego**.
Kolejność powinna być sztywna:

1. baseline i metryki,
2. DWT i testy,
3. WaveD,
4. matched-capacity,
5. WaveReg,
6. FFT control,
7. dopiero potem ablacjki.

W następnej wiadomości mogę Ci z tego zrobić jeszcze **wersję ultra-operacyjną w formie checklisty / promptów dla drugiego modelu**, np. „Commit 1”, „Commit 2”, „Commit 3”, z dokładnymi klasami, sygnaturami metod i pseudokodem do wklejania.
