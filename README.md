# Humboldt Ocean Watch

Humboldt Ocean Watch is an **experimental thermal monitoring product** for
AI-assisted daily sea surface temperature (SST) monitoring in the Niño 1+2
region (90–80°W, 10°S–0°). It is not an official operational product and does
not classify the official magnitude of El Niño Costero.

The minimal application works with local NetCDF input and does not require
Copernicus credentials. If the configured SST file is missing, it creates a
small synthetic daily dataset automatically. Recognized SST variables are
`analysed_sst`, `sst`, `thetao`, and `sea_surface_temperature`; Kelvin input is
normalized to degrees Celsius.

## Run locally

Use Python 3.12, install `requirements.txt`, and then run:

```powershell
python scripts/generate_demo_data.py
python scripts/build_daily_snapshot.py
streamlit run app.py
pytest -q
```

The current scope is SST and thermal anomalies only. Chlorophyll, currents,
forecasts, and climate-change attribution are intentionally excluded.

## Daily climatology

The primary operational baseline is the 1991–2020 daily smoothed climatology.
Each calendar day pools observations within ±5 calendar days across all
reference years; the mean, P10, and P90 are then smoothed with a circular
31-day window. A canonical 366-day calendar gives February 29 its own day-60
bin. Non-leap years contribute no February 29 observation, and March 1 always
maps to day 61. Both sampling and smoothing wrap across December–January.

Build it explicitly (it is never downloaded during application startup):

```powershell
python scripts/build_daily_climatology.py
```

The builder stores resumable raw yearly checkpoints so median and percentile
fields remain exact. If exact percentiles cannot be derived, it reports
`percentile_calculation_state=not_calculated` and does not substitute an
approximation. The existing monthly climatology remains an explicitly labelled
fallback and sensitivity baseline; daily and monthly baselines are never mixed.
