# `run_with_real_api.py`

## Responsibility

Compatibility wrapper for a real Gemini run. It checks for a provider key, limits iterations, enables logging, and delegates to `main.main()`.

## Use

Prefer configuring `.env` directly and running `python main.py` for normal operation. Use this wrapper when the explicit real-API safety defaults are desired.
