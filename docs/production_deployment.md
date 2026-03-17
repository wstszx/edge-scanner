# Production Deployment

This project can run in production, but the safest setup is to split HTTP serving from background auto-scan work.

## Recommended Topology

- `gunicorn` serves the Flask app on `127.0.0.1:5000`
- `auto_scan_worker.py` runs as a separate single-process service
- `nginx` reverse-proxies public traffic to Gunicorn and terminates TLS
- a non-root deploy user updates the repo and restarts only the required services

Repo templates included in this directory tree:

- `deploy/systemd/edge-scanner.service`
- `deploy/systemd/edge-scanner-worker.service`
- `deploy/systemd/root/edge-scanner.service`
- `deploy/systemd/root/edge-scanner-worker.service`
- `deploy/systemd/edge-scanner.env.example`
- `deploy/nginx/edge-scanner.conf`
- `deploy/sudoers/edge-scanner-deploy`

## Recommended Paths

- app repo: `/srv/edge-scanner/app`
- shared writable data: `/srv/edge-scanner/shared`
- system env file: `/etc/edge-scanner/edge-scanner.env`

## Deploy User

Suggested first-time setup:

```bash
sudo useradd --create-home --shell /bin/bash deploy
sudo mkdir -p /srv/edge-scanner/app /srv/edge-scanner/shared /etc/edge-scanner
sudo chown -R deploy:deploy /srv/edge-scanner
sudo chmod 755 /srv/edge-scanner
```

Clone and create the virtualenv as the deploy user:

```bash
sudo -u deploy git clone <YOUR_REPO_URL> /srv/edge-scanner/app
sudo -u deploy python3 -m venv /srv/edge-scanner/app/venv
sudo -u deploy /srv/edge-scanner/app/venv/bin/python -m pip install -r /srv/edge-scanner/app/requirements.txt
```

Install your SSH public key for the deploy user and keep GitHub Actions on SSH-key auth only.

## Root Layout

If you want to keep your current layout and continue deploying as `root`, the repository defaults already support it.

Expected layout:

- repo: `/root/edge-scanner`
- virtualenv: `/root/edge-scanner/venv`
- app env file: `/root/edge-scanner/.env`

Use the root-specific systemd templates:

```bash
sudo cp deploy/systemd/root/edge-scanner.service /etc/systemd/system/edge-scanner.service
sudo cp deploy/systemd/root/edge-scanner-worker.service /etc/systemd/system/edge-scanner-worker.service
sudo systemctl daemon-reload
sudo systemctl enable edge-scanner edge-scanner-worker
sudo systemctl restart edge-scanner edge-scanner-worker
```

For GitHub Actions, the current workflow can stay on the default values:

- SSH user: `root`
- repo dir: `/root/edge-scanner`
- venv Python: `/root/edge-scanner/venv/bin/python`

If you keep the root layout, you do not need to add any repository variables unless you rename the services.

## Systemd

Copy the service templates, then adjust paths, usernames, and worker names if you use different values:

```bash
sudo cp deploy/systemd/edge-scanner.service /etc/systemd/system/
sudo cp deploy/systemd/edge-scanner-worker.service /etc/systemd/system/
sudo cp deploy/systemd/edge-scanner.env.example /etc/edge-scanner/edge-scanner.env
sudo systemctl daemon-reload
sudo systemctl enable edge-scanner edge-scanner-worker
sudo systemctl start edge-scanner edge-scanner-worker
```

The templates intentionally keep background scanning out of the web service:

- web service sets `SERVER_AUTO_SCAN_ENABLED=0`
- worker service sets `SERVER_AUTO_SCAN_ENABLED=1`

If you use the root templates, the same split still applies.

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
- if you do not want automatic background scans, disable the worker service instead of running the web service with scans enabled

## Nginx

Suggested proxy pattern:

- listen on `80/443`
- proxy `https://your-domain` to `http://127.0.0.1:5000`
- use Let's Encrypt for certificates

Template:

```bash
sudo cp deploy/nginx/edge-scanner.conf /etc/nginx/sites-available/edge-scanner.conf
sudo ln -s /etc/nginx/sites-available/edge-scanner.conf /etc/nginx/sites-enabled/edge-scanner.conf
sudo nginx -t
sudo systemctl restart nginx
```

Replace these placeholders first:

- `example.com`
- no certificate paths are required in the bootstrap config; `certbot --nginx` will add them

## GitHub Actions Deployment

The repository workflow `.github/workflows/deploy.yml` can deploy over SSH.

Required GitHub secrets:

- `VPS_HOST`
- `VPS_SSH_KEY`

Optional GitHub secrets:

- `VPS_SSH_USER`
  Default fallback is `root`, but a dedicated deploy user is preferred.
- `VPS_HOST_FINGERPRINT`
  Strongly recommended for host key pinning.

Optional GitHub repository variables:

- `DEPLOY_REPO_DIR`
- `DEPLOY_VENV_PYTHON`
- `DEPLOY_WEB_SERVICE`
- `DEPLOY_WORKER_SERVICE`

Recommended values for the non-root layout above:

- `DEPLOY_REPO_DIR=/srv/edge-scanner/app`
- `DEPLOY_VENV_PYTHON=/srv/edge-scanner/app/venv/bin/python`
- `DEPLOY_WEB_SERVICE=edge-scanner`
- `DEPLOY_WORKER_SERVICE=edge-scanner-worker`

Root layout values, if you want to set them explicitly:

- `DEPLOY_REPO_DIR=/root/edge-scanner`
- `DEPLOY_VENV_PYTHON=/root/edge-scanner/venv/bin/python`
- `DEPLOY_WEB_SERVICE=edge-scanner`
- `DEPLOY_WORKER_SERVICE=edge-scanner-worker`

## Sudoers For Deploy User

If your GitHub Actions SSH user is not `root`, allow only the exact `systemctl` commands required by deployment.

Example:

```bash
sudo cp deploy/sudoers/edge-scanner-deploy /etc/sudoers.d/edge-scanner-deploy
sudo visudo -cf /etc/sudoers.d/edge-scanner-deploy
```

Before enabling it:

- replace `deploy` with your real SSH deploy username
- replace service names if you renamed them

## TLS

Typical Let's Encrypt flow on Ubuntu:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d example.com
```

For first-time setup, keep Nginx on plain HTTP first, verify the reverse proxy works, and only then run `certbot --nginx`.

## Important Differences From Legacy Flask-Only Setup

Do not use these older production shortcuts anymore:

- do not patch `app.py` to bind Flask directly on `0.0.0.0`
- do not expose port `5000` publicly when Nginx is available
- do not run `python app.py` under systemd for production traffic

Use:

- Gunicorn for the web app
- `auto_scan_worker.py` for background scans
- Nginx on `80/443`

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

## Verification

Useful checks after first setup:

```bash
sudo systemctl status edge-scanner --no-pager
sudo systemctl status edge-scanner-worker --no-pager
curl -I http://127.0.0.1:5000/
sudo nginx -t
```
