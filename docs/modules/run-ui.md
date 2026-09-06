# `run_ui.py`

## Responsibility

Launcher for the Control Deck. It detects and stops an existing listener on the configured port, then starts `web.app`.

## Platform behavior

Windows uses `netstat` and `taskkill`; Unix-like systems prefer `lsof` and fall back to `fuser`.
