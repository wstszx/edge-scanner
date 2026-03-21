# Edge Scanner

**Arbitrage • Middles • +EV**

Open-source sports betting scanner that finds arbitrage opportunities, middle bets, and +EV plays across US and EU bookmakers.

## Features

| Tab | What it finds | Risk profile |
|-----|---------------|--------------|
| **Arbitrage** | Guaranteed profit by betting both sides across different books | Zero risk |
| **Middles** | Line gaps where both bets can win if the result lands in between | High variance, +EV over time |
| **+EV** | Soft book odds better than sharp book fair value | Medium variance, profit over volume |

## How It Works

The scanner pulls odds from The Odds API across multiple regions (US, EU, UK, AU) and:

1. **Arbitrage** — finds where the sum of inverse odds across books is less than 1, guaranteeing profit regardless of outcome
2. **Middles** — finds where books disagree on spreads/totals, creating a gap where both sides can win
3. **+EV** — compares soft book odds to sharp reference (Pinnacle), identifying bets with positive expected value

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Get a free API key from [The Odds API](https://the-odds-api.com/)
4. Run locally: `python app.py`

By default the development server binds to `127.0.0.1` and opens your browser automatically. Use `python app.py --host 0.0.0.0 --no-browser` only when you explicitly want LAN access.

## Production Deployment

Do not use Flask's built-in development server for production traffic.

Recommended Linux split:

1. Web service: `SERVER_AUTO_SCAN_ENABLED=0 gunicorn -w 2 --threads 8 -b 127.0.0.1:5000 wsgi:app`
2. Background scanner: `SERVER_AUTO_SCAN_ENABLED=1 python auto_scan_worker.py`
3. Reverse proxy: put Nginx in front of `127.0.0.1:5000` and terminate HTTPS there

Notes:

- `SERVER_AUTO_SCAN_ENABLED` now defaults to `0`, so background scans are opt-in
- `SCAN_CUSTOM_PROVIDERS_ONLY=1` remains the safest low-cost mode for unattended scans
- `auto_scan_worker.py` is single-process by design; if you accidentally start multiple copies, only one instance will acquire the auto-scan lease
- keep the Flask/Gunicorn port private and expose only Nginx on `80/443`
- rotate system logs with `logrotate` or your process manager
- if public `443` is already used by x-ui/xray or another TLS service, use the shared-443 stream templates instead of forcing a second TLS listener onto the same socket

More detail: `docs/production_deployment.md`
Current root/VPS runbook: `docs/current_vps_deployment_zh.md`
Template files: `deploy/systemd/`, `deploy/nginx/`, `deploy/sudoers/`
Root layout templates: `deploy/systemd/root/`
Shared-443 Nginx templates: `deploy/nginx/edge-scanner-shared-443-site.conf`, `deploy/nginx/edge-scanner-shared-443-stream.conf`

## Project Docs

Structured project documentation lives in `docs/project_docs/` and follows the required workflow:

- `00_WORKFLOW.md` - workflow and prompt template
- `01_PRD.md` - product requirements
- `02_ARCH.md` - architecture document
- `03_API_DOC.md` - API contract
- `04_SPEC.md` - implementation specification
- `05_TEST_CASE.md` - test cases
- `06_CODE_GENERATION_PROMPT.md` - code generation prompt
- `07_TEST_GENERATION_PROMPT.md` - test generation prompt
- `08_PROJECT_SUMMARY.md` - project summary
- `09_PROVIDER_VERIFICATION.md` - provider verification workflow and latest findings

## Provider Verification

Run the repeatable provider verification script to check provider-focused tests, execute a provider-only scan, and write a JSON + Markdown report:

```bash
python provider_verification.py --sport basketball_nba
```

Windows shortcuts:

```powershell
.\run_provider_verification.ps1
```

```bat
run_provider_verification.bat
```

Local automation helpers:

```powershell
.\scheduled_provider_verification.ps1 --sport basketball_nba --fail-on-alert
.\register_provider_verification_task.ps1 -DailyAt 09:00 -Sport basketball_nba -FailOnAlert
```

GitHub Actions automation is available at `.github/workflows/provider-verification.yml`. Scheduled runs upload the generated reports as workflow artifacts. Manual runs can enable strict mode to fail on alerts.

Useful flags:

- `--skip-tests` - skip the provider-focused pytest subset
- `--providers betdex bookmaker_xyz sx_bet polymarket` - verify only selected providers
- `--out-dir data/provider_verification` - choose a custom report directory
- `--summary-only` - print only failed providers and suspicious result highlights
- `--json-stdout` - print the full JSON payload to stdout
- `--fail-on-alert` - return exit code `2` when provider or result alerts are detected

The Windows wrapper scripts preserve the Python exit code, so `--fail-on-alert` can be used directly in scheduled tasks or CI.

Each run writes both timestamped history files and fixed latest files:

- `data/provider_verification/provider_verification_<timestamp>.json`
- `data/provider_verification/provider_verification_<timestamp>.md`
- `data/provider_verification/provider_verification_latest.json`
- `data/provider_verification/provider_verification_latest.md`

Optional: configure via `settings.json` (loaded automatically if present). Keys mirror the
environment variables used in this README. You can override with real env vars.
Common defaults: `DEFAULT_REGION_KEYS`, `DEFAULT_SPORT_KEYS`, `DEFAULT_STAKE_AMOUNT`,
`DEFAULT_COMMISSION`, `DEFAULT_SHARP_BOOK`, `DEFAULT_BANKROLL`, `DEFAULT_KELLY_FRACTION`.
Use `APP_CONFIG_PATH` to point at a different config file.
UI defaults can be set too (e.g., `DEFAULT_MIN_ROI`, `DEFAULT_MIDDLE_SORT`, `DEFAULT_PLUS_EV_SORT`,
`DEFAULT_AUTO_SCAN_MINUTES`, `DEFAULT_NOTIFY_SOUND_ENABLED`, `DEFAULT_ODDS_FORMAT`, `DEFAULT_THEME`).

**Optional:** Create a `.env` file with your API key to skip manual entry:
```
ODDS_API_KEY=your_api_key_here
```
Or provide a key pool (comma/space separated) to rotate when a key is exhausted:
```
ODDS_API_KEYS=key_one,key_two,key_three
```
If you run provider-only scans (`includeProviders`/provider bookmakers without Odds API books), API keys are optional.

Custom provider routing (scanner-level, provider modules in `providers/`):
```
BETDEX_ENABLED=0
BOOKMAKER_XYZ_ENABLED=0
SX_BET_ENABLED=0
POLYMARKET_ENABLED=0
```
Provider rate-limit reference: `docs/provider_rate_limits.md`

bookmaker.xyz dictionary settings:
```
BOOKMAKER_XYZ_DICTIONARY_SOURCE=auto
BOOKMAKER_XYZ_GAMES_PER_PAGE=100
BOOKMAKER_XYZ_MAX_GAME_PAGES=10
BOOKMAKER_XYZ_CONDITION_BATCH_SIZE=50
```
`BOOKMAKER_XYZ_DICTIONARY_SOURCE=auto` prefers the official `@azuro-org/dictionaries` package and falls back to bookmaker.xyz frontend assets only if needed. `bookmaker.xyz` now reads live events from Azuro's official public `market-manager` API instead of the old GraphQL snapshot path.

Supported sport keys now include:
- Existing project keys such as `basketball_nba`, `soccer_epl`, `soccer_spain_la_liga`, `soccer_usa_mls`
- Expanded curated keys such as `basketball_euroleague`, `soccer_england_championship`, `soccer_portugal_primeira_liga`, `icehockey_khl`, `mma_ufc`, `boxing_professional`
- Generic Azuro keys in the form `azuro__<sport_slug>__<league_slug>__<country_slug>`, for example:
  - `azuro__basketball__euroleague__international-tournaments`
  - `azuro__football__premier-league__england`
  - `azuro__mma__ufc__international-tournaments`

SX Bet provider settings (API mode):
```
SX_BET_SOURCE=api
SX_BET_API_BASE=https://api.sx.bet
SX_BET_BASE_TOKEN=0x6629Ce1Cf35Cc1329ebB4F63202F3f197b3F050B
SX_BET_TIMEOUT_SECONDS=20
SX_BET_RETRIES=2
SX_BET_RETRY_BACKOFF=0.5
SX_BET_PAGE_CACHE_TTL=45
SX_BET_ODDS_CACHE_TTL=30
```
Optional auth/session fields (if your environment requires them):
```
SX_BET_BEARER_TOKEN=
SX_BET_API_KEY=
SX_BET_COOKIE=
```
Optional league pinning (JSON):
```
SX_BET_LEAGUE_MAP={"basketball_nba":[1],"basketball_ncaab":[2]}
```

BetDEX provider settings (API mode):
```
BETDEX_SOURCE=api
BETDEX_MONACO_API_BASE=https://production.api.monacoprotocol.xyz
BETDEX_PUBLIC_BASE=https://www.betdex.com
BETDEX_APP_ID=
BETDEX_API_KEY=
BETDEX_SESSION_EXPIRY_SKEW_SECONDS=30
# Legacy public-site fallback only. This endpoint may return 429 from Vercel Security Checkpoint.
BETDEX_SESSION_URL=https://www.betdex.com/api/session
BETDEX_TIMEOUT_SECONDS=20
BETDEX_RETRIES=2
BETDEX_RETRY_BACKOFF=0.5
BETDEX_EVENTS_PAGE_SIZE=250
BETDEX_EVENTS_MAX_PAGES=8
BETDEX_MARKETS_PAGE_SIZE=500
BETDEX_MARKETS_MAX_PAGES=8
BETDEX_EVENT_BATCH_SIZE=60
BETDEX_PRICE_BATCH_SIZE=120
BETDEX_MARKET_STATUSES=Open
BETDEX_BACK_PRICE_SIDE=against
```
If `BETDEX_APP_ID` and `BETDEX_API_KEY` are set, the provider uses the official Monaco `POST /sessions` flow before calling `/events`, `/markets`, and `/market-prices`. If they are unset, the code falls back to the public BetDEX website session endpoint, which is best-effort and may be blocked by Vercel bot protection.

Optional file fallback for local fixtures:
```
BETDEX_SOURCE=file
BETDEX_SAMPLE_PATH=data/betdex_sample.json
```
BetDEX is treated as an exchange in scanner settings, so exchange commission applies.
You can also override per request with `includeProviders`:
```json
{
  "includeProviders": ["BetDEX", "bookmaker.xyz", "SX Bet", "polymarket"]
}
```

Live arbitrage mode:
```json
{
  "scanMode": "live"
}
```
`scanMode` supports `prematch` (default) and `live`.
In `live` mode the scanner:
- skips The Odds API fetch path
- uses custom providers only
- keeps only in-play / already-started events inside the configured recent-event window
- prefers provider live feeds where the provider module supports a runtime live override

Optional: scan all available markets for arbitrage (per event data returned by providers):
```
ARBITRAGE_ALL_MARKETS=1
ODDS_API_MARKET_BATCH_SIZE=8
```
When enabled, scanner expands API requests beyond base `h2h/spreads/totals` and batches
market calls automatically. Unsupported market keys are skipped per sport.

Optional custom market list override (comma/space separated or JSON array):
```
ODDS_API_ALL_MARKETS=h2h_3_way,alternate_spreads,alternate_totals,player_points,player_assists
```
If omitted, scanner uses built-in sport-specific expanded market presets.

Optional: save only the latest scan payload + result to disk:
```
SCAN_SAVE_ENABLED=1
SCAN_SAVE_DIR=data/scans
```
You can also pass `saveScan=true` in the `/scan` request JSON to save per-request.
When enabled, the scanner keeps only the newest `scan_*.json` file in `SCAN_SAVE_DIR`.
Saved request payloads automatically redact `apiKey`/`apiKeys`.

Optional: write per-scan HTTP request logs (URL, params, status, response preview, duration):
```
SCAN_REQUEST_LOG_ENABLED=1
SCAN_REQUEST_LOG_DIR=data/request_logs
SCAN_REQUEST_LOG_MAX_BODY_CHARS=2000
SCAN_REQUEST_LOG_RETENTION_FILES=20
```
When enabled, each scan writes a `requests_*.jsonl` file to `SCAN_REQUEST_LOG_DIR`.
Older request logs are trimmed automatically to the newest `SCAN_REQUEST_LOG_RETENTION_FILES` files.
The scan response includes:
- `request_log.path`
- `request_log.requests_logged`

Optional: save latest custom-provider fetch results per provider (overwrites each scan):
```
CUSTOM_PROVIDER_SNAPSHOT_ENABLED=1
CUSTOM_PROVIDER_SNAPSHOT_DIR=data/provider_snapshots
```
Each provider is written to its own file, e.g.:
- `data/provider_snapshots/polymarket.json`
- `data/provider_snapshots/betdex.json`

Optional: drop stale fixtures before opportunity calculation (applies to merged API + custom provider events):
```
EVENT_MAX_PAST_MINUTES=30
```
Events older than this threshold are excluded from arbitrage/middle/+EV detection.

## Markets Scanned

Default mode (`allMarkets=false`):

| Sport | Default Markets |
|-------|------------------|
| NFL, NCAAF, NBA, NCAAB, MLB, NHL | `h2h`, `spreads`, `totals` |
| Soccer (EPL, La Liga, etc.) | `spreads`, `totals` |

Extended mode (`allMarkets=true` or `ARBITRAGE_ALL_MARKETS=1`):
- Odds API requests sport-specific extra market keys in batches; unsupported keys are auto-skipped per sport.
- Providers attempt dynamic mapping for additional two-way markets where source data includes them.
- `polymarket` supports `h2h`, `h2h_3_way`, and BTTS (`both_teams_to_score`) when detectable from market questions.

Note: arbitrage and +EV calculations still require compatible two-way pricing on both sides of the same line; not every returned market will produce opportunities.

## API Usage

Each scan uses one API call per sport. Free tier: 500 requests/month.

Default sports use ~6-10 calls per scan depending on what's in season.

## Understanding the Results

### Arbitrage

ROI = guaranteed return on total stake. A 2% ROI means $2 profit on $100 staked.

Stakes are split so you get the same payout regardless of outcome.

### Middles

EV = expected value based on historical probability of landing in the gap.

NFL key numbers (3, 7) significantly boost probability for spread middles — these margins occur more often due to field goals and touchdowns.

Middles lose small amounts most of the time and win big occasionally — positive EV over many bets, but high variance on any single bet.

### +EV

Edge = how much better the soft book odds are compared to sharp book fair value.

Sharp books (Pinnacle) have tight lines with minimal vig (~2%). When soft books offer better odds than the sharp-implied fair price, that's a +EV bet.

Kelly staking helps size bets based on edge — quarter Kelly recommended to reduce variance.

## Configuration

Configurable via the UI:
- Regions (US, US2, UK, EU, AU)
- Sports selection
- Exchange commission rate (for Betfair, etc.)
- Minimum gap for middles
- Minimum edge for +EV
- Sharp reference book (Pinnacle, Betfair, Matchbook)
- Kelly fraction (full, half, quarter, tenth)
- Stake amount / bankroll

## Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Data:** The Odds API
```
Each provider is written to its own file, e.g.:
- `data/provider_snapshots/polymarket.json`
- `data/provider_snapshots/betdex.json`

Optional: drop stale fixtures before opportunity calculation (applies to merged API + custom provider events):
```
EVENT_MAX_PAST_MINUTES=30
```
Events older than this threshold are excluded from arbitrage/middle/+EV detection.

## Markets Scanned

Default mode (`allMarkets=false`):

| Sport | Default Markets |
|-------|------------------|
| NFL, NCAAF, NBA, NCAAB, MLB, NHL | `h2h`, `spreads`, `totals` |
| Soccer (EPL, La Liga, etc.) | `spreads`, `totals` |

Extended mode (`allMarkets=true` or `ARBITRAGE_ALL_MARKETS=1`):
- Odds API requests sport-specific extra market keys in batches; unsupported keys are auto-skipped per sport.
- Providers attempt dynamic mapping for additional two-way markets where source data includes them.
- `polymarket` supports `h2h`, `h2h_3_way`, and BTTS (`both_teams_to_score`) when detectable from market questions.

Note: arbitrage and +EV calculations still require compatible two-way pricing on both sides of the same line; not every returned market will produce opportunities.

## API Usage

Each scan uses one API call per sport. Free tier: 500 requests/month.

Default sports use ~6-10 calls per scan depending on what's in season.

## Understanding the Results

### Arbitrage

ROI = guaranteed return on total stake. A 2% ROI means $2 profit on $100 staked.

Stakes are split so you get the same payout regardless of outcome.

### Middles

EV = expected value based on historical probability of landing in the gap.

NFL key numbers (3, 7) significantly boost probability for spread middles — these margins occur more often due to field goals and touchdowns.

Middles lose small amounts most of the time and win big occasionally — positive EV over many bets, but high variance on any single bet.

### +EV

Edge = how much better the soft book odds are compared to sharp book fair value.

Sharp books (Pinnacle) have tight lines with minimal vig (~2%). When soft books offer better odds than the sharp-implied fair price, that's a +EV bet.

Kelly staking helps size bets based on edge — quarter Kelly recommended to reduce variance.

## Configuration

Configurable via the UI:
- Regions (US, US2, UK, EU, AU)
- Sports selection
- Exchange commission rate (for Betfair, etc.)
- Minimum gap for middles
- Minimum edge for +EV
- Sharp reference book (Pinnacle, Betfair, Matchbook)
- Kelly fraction (full, half, quarter, tenth)
- Stake amount / bankroll

## Features
- Alerts/notifications
- Historical tracking

## Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Data:** The Odds API

## License

MIT

## Contributing

PRs welcome. Ideas for future versions:
- Three-way arbitrage (soccer moneylines)
- Player props

## Support

If this tool saves you time or helps you find an edge, consider buying me a coffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow.svg)](https://buymeacoffee.com/deliciouspipe1326)
