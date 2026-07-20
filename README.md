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
