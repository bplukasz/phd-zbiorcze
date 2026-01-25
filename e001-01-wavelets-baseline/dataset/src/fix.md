Widzę tu **dwa “killery”, które bardzo często dają efekt: “2h i dalej szum”**, nawet jeśli sam trening *w tle* coś tam jednak rusza.

---

## 1) Najbardziej podejrzane: w ogóle nie trenujesz na danych (lecisz na `dummy data`)

Masz to:

```python
try:
    dataloader = get_dataloader(...)
    data_iter = iter(dataloader)
except Exception as e:
    print(f"Błąd ładowania danych: {e}")
    print("Używam dummy data dla testów...")
    dataloader = None
    data_iter = None
```

A później:

```python
else:
    real_imgs = torch.randn(...)
```

Jeśli CIFAR-10 nie pobierze się (np. brak internetu / zły path), to **G uczy się generować losowy Gaussian noise**, więc w gridach *nigdy* nie zobaczysz twarzy czy sensownych struktur.

**Zrób tak, żeby to nie przechodziło “po cichu”:**

```python
try:
    dataloader = get_dataloader(...)
    data_iter = iter(dataloader)
except Exception as e:
    raise RuntimeError(f"Data loading failed, stop training (otherwise you'll train on dummy noise). Error: {e}")
```

I dorzuć szybki sanity check (raz na starcie):

```python
real_imgs, _ = next(iter(dataloader))
print("REAL stats:", real_imgs.min().item(), real_imgs.max().item(), real_imgs.mean().item())
save_image((real_imgs[:64] + 1)/2, os.path.join(cfg.out_dir, "real_grid.png"), nrow=8)
```

Jeśli `real_grid.png` wygląda sensownie – dane są OK. Jeśli nie powstaje / jest czarne / śmieci – to *tu* jest problem.

---

## 2) Drugi killer (mega ważne u Ciebie): **EMA + BatchNorm** (EMA nie aktualizuje BN running stats)

Twój EMA robi:

```python
for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
    ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
```

Ale **BatchNorm ma kluczowe bufory** (`running_mean`, `running_var`, `num_batches_tracked`) które **nie są parametrami**, tylko **buffers**. One w EMA-shadow zostają “zamrożone” na init (0/1), a Ty **gridy zapisujesz z `G_ema`**:

```python
fake_grid = G_ema(fixed_z)
```

Efekt: nawet jeśli “prawdziwy” `G` zaczyna się uczyć, **EMA-generator w eval-mode używa złych BN statystyk i potrafi pluć szumem bardzo długo**.

### Naprawa EMA (kopiuj bufory z modelu, a parametry EMA-uj)

Podmień klasę EMA na taką:

```python
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

        # cache maps for fast updates
        self._shadow_params = dict(self.shadow.named_parameters())
        self._shadow_bufs = dict(self.shadow.named_buffers())

    @torch.no_grad()
    def update(self, model: nn.Module):
        # EMA for parameters
        for name, p in model.named_parameters():
            self._shadow_params[name].data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

        # COPY buffers (BatchNorm running stats etc.)
        for name, b in model.named_buffers():
            if name in self._shadow_bufs:
                self._shadow_bufs[name].copy_(b)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
```

### Dodatkowo: Twoje `generate_samples()` psuje tryb modelu

Masz na końcu:

```python
G.train()
```

Jeśli przekażesz `G_ema.shadow`, to włącza mu train-mode (BN zaczyna aktualizować statystyki na wygenerowanych obrazkach…).

Zmień `generate_samples()` na wersję, która **przywraca tryb**:

```python
def generate_samples(...):
    was_training = G.training
    G.eval()
    ...
    if was_training:
        G.train()
    else:
        G.eval()
    return out_dir
```

---

## 3) Rzeczy, które nie zabiją treningu, ale potrafią sprawić “nic nie widać”

### (a) Logowanie co krok = dramatyczne spowolnienie

W “fast” masz `log_every: 1`, a `CSVLogger.log()` otwiera plik **za każdym krokiem**. To potrafi zabić throughput i po 2h możesz mieć śmiesznie mało iteracji.

Na start daj:

* `log_every = 50` albo `100`
* a print tylko co `log_every`

albo zrób buforowanie (otwórz plik raz i flush co jakiś czas).

### (b) DiffAugment może być za mocny na 32×32

Na CIFAR-10 cutout 0.5 bywa agresywny. Na debug daj policy np. `"color,translation"` albo nawet `"color"`.

Możesz też po augmentacji trzymać zakres:

```python
def DiffAugment(x, policy=''):
    ...
    return x.clamp(-1, 1)
```

### (c) FID/KID dla CIFAR-10 i tak masz ustawione “średnio”

`compute_fid_kid(cfg.data_dir, ...)` – a `cfg.data_dir` dla CIFAR to cache z batchami, nie folder PNGów. To będzie sypać błędami / nan. Na debug ustaw `eval_every=0`.

---

## Co bym zrobił najszybciej (minimalny plan)

1. **Zablokuj dummy-data** (zamiast fallbacku → `raise`) i zapisz `real_grid.png`.
2. **Napraw EMA pod BatchNorm** (kopiuj bufory) i napraw `generate_samples()` (restore mode).
3. Na 1. test: `diffaug_policy="color"`, `log_every=50`, `eval_every=0`.
4. Porównaj grid z **G (bez EMA)** i z **EMA** przez pierwsze 2–3k kroków (żeby zobaczyć, czy EMA nie kłamie).

Jeśli chcesz, wklej mi jeszcze `config_loader.py` / `get_config()` + realny `cfg` (albo log z początku runa), to od razu powiem czy **na pewno idziesz na prawdziwych danych** i czy bety/LR nie są skopane pod hinge+SN.
