# Konfiguracje `e000-01-r3gan-baseline`

Ten katalog zawiera tylko cztery podstawowe profile dla czystego baseline'u R3GAN:

| Profil | Cel |
|---|---|
| `base` | domyślna konfiguracja treningu 64×64 |
| `smoke` | bardzo szybki sanity-check pipeline'u |
| `fast` | krótszy eksperyment na CIFAR-10 32×32 |
| `overnight` | dłuższy trening 64×64 |

## Zasady

- Każdy profil nadpisuje tylko potrzebny podzbiór pól z `base.yaml`.

## Przykłady

```bash
python run.py --profile smoke
python run.py --profile fast
python run.py --profile overnight --data-dir /data/celeba/img_align_celeba
python run.py --profile fast --override steps=5000 batch_size=64
```

