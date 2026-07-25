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

## Scientific briefs with Gemini, Ollama, or OpenAI

The recommended operational provider is native Google Gemini through the
maintained `google-genai` Python SDK. The configured model is
`gemini-3.5-flash`. Ollama remains available for optional local development,
and OpenAI remains an optional paid provider. The generator never silently
falls back between providers.

Install all repository dependencies, including the native Gemini SDK:

```powershell
uv pip install -r requirements.txt
```

Create a Gemini API key through an authorized Google AI project, keep it out
of version control, and expose it only through the environment:

```powershell
$env:GEMINI_API_KEY="your-key"
$env:LLM_PROVIDER="gemini"
$env:GEMINI_MODEL="gemini-3.5-flash"
```

Never commit API keys. Free-tier provider submissions must not contain
unauthorized, confidential, or restricted institutional information. Provider
quotas and terms apply; this project does not claim unlimited free usage.

Run the explicit full Gemini access preflight (it sends only two small test
requests and no scientific context):

```powershell
uv run python scripts/check_llm_access.py `
  --provider gemini `
  --model gemini-3.5-flash
```

Inspect the complete local request and schema diagnostics without an API key
or provider call:

```powershell
uv run python scripts/generate_scientific_brief.py `
  --provider gemini `
  --model gemini-3.5-flash `
  --context outputs/briefs/brief_context_latest.json `
  --language es `
  --dry-run
```

Generate Spanish or English manually after preflight succeeds:

```powershell
uv run python scripts/generate_scientific_brief.py `
  --provider gemini `
  --model gemini-3.5-flash `
  --context outputs/briefs/brief_context_latest.json `
  --language es

uv run python scripts/generate_scientific_brief.py `
  --provider gemini `
  --model gemini-3.5-flash `
  --context outputs/briefs/brief_context_latest.json `
  --language en
```

Gemini receives no tools, search grounding, URL context, or conversation
state. Automatic function calling is explicitly disabled. Operational Gemini
generation uses JSON mode (`response_mime_type="application/json"`) and does
not send the complete JSON Schema; its JSON output must pass the same
authoritative local schema and scientific validation used for every provider
before any artifact is published.

`scripts/check_llm_access.py` owns the full model GET, basic-generation, and
structured-generation preflight. `scripts/generate_scientific_brief.py` does
not repeat those test requests: it performs only local package, API-key,
configuration, and output-path checks before the scientific request. Gemini
transport errors 429, 500, 502, 503, 504, network interruption, and timeout
are retried at most three total attempts with bounded exponential delays. A
transport retry is independent of the single permitted scientific repair and
never switches providers.

### Optional local scientific briefs with Ollama

The native local Ollama provider uses JSON mode directly and applies the
complete strict scientific schema during authoritative local Python
validation. It requires no provider API key and uses no tools or web search.

1. Install Ollama and this repository's Python requirements.
2. Pull and confirm the model:

   ```powershell
   ollama pull qwen3:4b
   ollama list
   ```

3. Optionally configure environment overrides:

   ```powershell
   $env:LLM_PROVIDER="ollama"
   $env:OLLAMA_MODEL="qwen3:4b"
   $env:OLLAMA_BASE_URL="http://localhost:11434"
   $env:OLLAMA_STRUCTURED_OUTPUT_MODE="json"
   $env:OLLAMA_REQUEST_TIMEOUT_SECONDS="600"
   ```

4. Run the independent preflight:

   ```powershell
   uv run python scripts/check_llm_access.py `
     --provider ollama `
     --model qwen3:4b
   ```

5. Generate Spanish manually:

   ```powershell
   uv run python scripts/generate_scientific_brief.py `
     --provider ollama `
     --model qwen3:4b `
     --context outputs/briefs/brief_context_latest.json `
     --language es
   ```

Generation speed depends on local hardware. The compact `qwen3:4b` model may have lower writing quality than larger models. Bounded repair and all validators remain mandatory, and the application never silently falls back between providers.
