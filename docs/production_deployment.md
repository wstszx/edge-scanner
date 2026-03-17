# Production Deployment

This project can run in production, but the safest setup is to split HTTP serving from background auto-scan work.

## Recommended Topology

- `gunicorn` serves the Flask app on `127.0.0.1:5000`
- `auto_scan_worker.py` runs as a separate single-process service
- `nginx` reverse-proxies public traffic to Gunicorn and terminates TLS

## Web Service

Example:

```bash
SERVER_AUTO_SCAN_ENABLED=0 gunicorn -w 2 --threads 8 -b 127.0.0.1:5000 wsgi:app
```

Why:

- avoids Flask development server in production
- keeps the app port private
- allows multiple web workers without multiplying background auto-scan threads

## Background Scanner

Example:

```bash
SERVER_AUTO_SCAN_ENABLED=1 python auto_scan_worker.py
```

Notes:

- the worker reads the same `SERVER_AUTO_SCAN_CONFIG_PATH` JSON config as the web app
- if multiple worker copies start, only one process acquires the auto-scan lease file and runs the thread
- for cheapest unattended mode, prefer `SCAN_CUSTOM_PROVIDERS_ONLY=1`

## Nginx

Suggested proxy pattern:

- listen on `80/443`
- redirect `80` to `443`
- proxy `https://your-domain` to `http://127.0.0.1:5000`
- use Let's Encrypt for certificates

## Storage Guardrails

- `SCAN_SAVE_ENABLED=1` keeps only the newest `scan_*.json`
- provider snapshots overwrite fixed files in `data/provider_snapshots/`
- history is trimmed by `HISTORY_MAX_RECORDS`
- request logs are trimmed by `SCAN_REQUEST_LOG_RETENTION_FILES`
- rotate Gunicorn/systemd/Nginx logs with `logrotate`

## Cost Guardrails

- `SERVER_AUTO_SCAN_ENABLED` is opt-in
- `ODDS_API_KEYS=key1,key2,...` can rotate across multiple Odds API keys
- scans that use only custom providers do not require The Odds API
- if you enable Odds API polling, choose the interval based on your actual quota, sports count, and market count
