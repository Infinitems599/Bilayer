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

uv run python metroflow_dmp.py `
  --metroflow-dir D:\path\to\MetroFlow `
  --output-dir results\metroflow_hourly `
  --hourly-perron-only

uv run python metroflow_mc.py `
  --metroflow-dir D:\path\to\MetroFlow `
  --output-dir results\metroflow_mc
```

The scripts read the released processed data without changing it and write
machine-readable CSV/JSON outputs only. In particular,
`hourly_perron_roots.csv` contains the 17 hourly roots used in Figure 13;
`primary_dmp_scan.csv` contains the Figure 14 DMP curve and Theorem 4.2
certificate fields; `primary_scan_mc.csv`, `allocation_rule_mc.csv`, and
`omega_sensitivity_mc.csv` contain the corresponding MC point estimates. Any
preliminary DMP/MC point-sign mismatch is automatically recomputed with 5000
runs using the same deterministic seed.

## Figure 8 data

Use the common interface rather than a separate plotting script:

```powershell
uv run python -m calculation_code figure8 `
  --output-dir results\figure8 `
  --seed 13000
```

This writes `figure8_instances.csv`, `figure8_grid.csv`, and `figure8_settings.json`. For a short installation check, append `--quick`.
The default configuration is the manuscript configuration:
$N=80$, 25 degree values from 4 to 22, $r=4,8,12$,
$\omega=0.62$, 10 realizations, and base seed `13000`.

```python
from calculation_code import Figure8Config, generate_figure8_data

paths = generate_figure8_data("results/figure8", Figure8Config(seed=13000))
print(paths["grid"])
```

## Random seeds

The full figure-by-figure seed list is in [RANDOM_SEEDS.md](RANDOM_SEEDS.md). MetroFlow MC uses base seed `20260729`; DMP calculations are deterministic.
