<<<<<<< HEAD
# Calculation code

Code repository for *Epidemic spreading on multilayer networks with asymmetric layer-preferential coupling*.

Calculation-only reproduction package for the revised manuscript. It writes CSV/JSON data; plotting and generated figures are not included.

## Contents

- `dmp_core.py`: DMP operator and spectral threshold routines.
- `metroflow_dmp.py`: MetroFlow DMP scans and data audit.
- `metroflow_mc.py`: MetroFlow early-growth MC estimates.
- `calculation_code/`: importable experiment interface; Figure 8 is provided as the example.

## Setup and MetroFlow outputs

The processed MetroFlow directory must contain `stationInfo.csv`, `metroData_InOutFlow.csv`, and `MetaData/workday_calendar.csv`.

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv sync

uv run python metroflow_dmp.py `
  --metroflow-dir D:\path\to\MetroFlow `
  --output-dir results\metroflow_dmp

uv run python metroflow_mc.py `
  --metroflow-dir D:\path\to\MetroFlow `
  --output-dir results\metroflow_mc
```

The scripts read the released processed data without changing it and write machine-readable CSV/JSON outputs only.

## Figure 8 data

Use the common interface rather than a separate plotting script:

```powershell
uv run python -m calculation_code figure8 `
  --output-dir results\figure8 `
  --seed 13000
```

This writes `figure8_instances.csv`, `figure8_grid.csv`, and `figure8_settings.json`. For a short installation check, append `--quick`.

```python
from calculation_code import Figure8Config, generate_figure8_data

paths = generate_figure8_data("results/figure8", Figure8Config(seed=13000))
print(paths["grid"])
```

## Random seeds

The full figure-by-figure seed list is in [RANDOM_SEEDS.md](RANDOM_SEEDS.md). MetroFlow MC uses base seed `20260729`; DMP calculations are deterministic.
=======
# Bilayer
Code repository for Epidemic spreading on multilayer networks with asymmetric layer-preferential coupling.
>>>>>>> b9ce31f04e4c011957c6c248edf8bcb7f9069b88
