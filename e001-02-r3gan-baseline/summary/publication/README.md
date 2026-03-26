# Publication Tables

Ten katalog zawiera automatycznie generowane tabele porownawcze dla runow z `artifacts/*/logs.csv`.

## Co jest liczone

- filtr runow: `max(step) >= 30000`
- punkt porownania: metryki z `step <= 30000` (preferowany dokladnie 30000)
- tabele per-run, ranking wielokryterialny i ocena `claim_readiness` per recipe

## Generowanie

```bash
python scripts/build_publication_tables.py
```

## Artefakty wyjsciowe

- `run_metrics_step_ge_30000.csv`
- `run_rankings_step_ge_30000.csv`
- `claim_readiness_step_ge_30000.csv`
- `report_step_ge_30000.md`

## Uwagi metodologiczne

- `FID@30000` jest glownym punktem porownania miedzy runami o roznej dlugosci.
- `claim_readiness` oznacza gotowosc do claimu paperowego i wymaga m.in. powtarzalnosci (w praktyce >=3 seedy per recipe).

