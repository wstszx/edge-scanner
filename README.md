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
4. Run: `python app.py`

Your browser opens automatically. Enter your API key and scan.

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

Optional Purebet (Solana) integration (read-only for now):
```
PUREBET_ENABLED=1
PUREBET_SOURCE=file
PUREBET_SAMPLE_PATH=data/purebet_sample.json
```
`PUREBET_SAMPLE_PATH` should be a JSON array of events normalized to the Odds API event schema
(bookmakers -> markets -> outcomes).
V3 API mode (default):
```
PUREBET_ENABLED=1
PUREBET_SOURCE=api
PUREBET_API_BASE=https://v3api.purebet.io
PUREBET_LIVE=0
PUREBET_LEAGUE_MAP={"487":"basketball_nba","1980":"soccer_epl"}
PUREBET_MARKETS_ENABLED=1
PUREBET_MIN_STAKE=50
PUREBET_MAX_AGE_SECONDS=60
PUREBET_MARKET_WORKERS=8
PUREBET_MARKET_RETRIES=2
PUREBET_RETRY_BACKOFF=0.4
PUREBET_LEAGUE_SYNC_ENABLED=1
PUREBET_LEAGUE_SYNC_TTL=600
PUREBET_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...
PUREBET_ORIGIN=https://purebet.io
PUREBET_REFERER=https://purebet.io/
PUREBET_FUZZY_MATCH_THRESHOLD=0.85
EVENT_TIME_TOLERANCE_MINUTES=15
```
`PUREBET_LEAGUE_MAP` lets you pin league IDs manually.
When `PUREBET_LEAGUE_SYNC_ENABLED=1`, scanner also pulls `/activeLeagues` and auto-infers league mapping
for supported sports, cached for `PUREBET_LEAGUE_SYNC_TTL` seconds.
Purebet is treated as an exchange, so the commission rate applies to its prices.

Custom provider routing (scanner-level, provider modules in `providers/`):
```
BOOKMAKER_XYZ_ENABLED=0
SX_BET_ENABLED=0
OVERTIMEMARKETS_XYZ_ENABLED=0
POLYMARKET_ENABLED=0
```
`PUREBET_ENABLED` still controls Purebet default.

SX Bet provider settings (API mode):
```
SX_BET_SOURCE=api
SX_BET_API_BASE=https://api.sx.bet
SX_BET_BASE_TOKEN=0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
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

Overtime Markets provider settings (API mode):
```
OVERTIMEMARKETS_SOURCE=api
OVERTIMEMARKETS_API_BASE=https://api.overtime.io/overtime-v2
OVERTIMEMARKETS_API_KEY=your_overtime_api_key
OVERTIMEMARKETS_NETWORK=10
OVERTIMEMARKETS_TIMEOUT_SECONDS=20
OVERTIMEMARKETS_RETRIES=2
OVERTIMEMARKETS_RETRY_BACKOFF=0.5
OVERTIMEMARKETS_CACHE_TTL=45
OVERTIMEMARKETS_INCLUDE_LIVE=0
OVERTIMEMARKETS_ONLY_BASIC_PROPERTIES=1
OVERTIMEMARKETS_ONLY_MAIN_MARKETS=1
OVERTIMEMARKETS_UNGROUP=1
OVERTIMEMARKETS_STATUS=open
```
Optional league name pinning (JSON, value supports substrings):
```
OVERTIMEMARKETS_LEAGUE_MAP={
  "basketball_nba":["nba"],
  "soccer_epl":["premier league"]
}
```
You can also override per request with `includeProviders`:
```json
{
  "includeProviders": ["bookmaker.xyz", "SX Bet", "overtimemarkets.xyz", "polymarket"]
}
```
`includePurebet` remains supported and overrides Purebet on/off for that request.

Optional: scan all available markets for arbitrage (per event data returned by providers):
```
ARBITRAGE_ALL_MARKETS=1
```
Note: The Odds API still only returns the markets you request, so this mainly affects
providers that already supply multiple markets in their payloads.

Optional: save only the latest scan payload + result to disk:
```
SCAN_SAVE_ENABLED=1
SCAN_SAVE_DIR=data/scans
```
You can also pass `saveScan=true` in the `/scan` request JSON to save per-request.
When enabled, the scanner keeps only the newest `scan_*.json` file in `SCAN_SAVE_DIR`.
Saved request payloads automatically redact `apiKey`/`apiKeys`.

## Markets Scanned

| Sport | Markets |
|-------|---------|
| NFL, NBA, MLB, NHL | Moneyline, Spreads, Totals |
| Soccer (EPL, La Liga, etc.) | Spreads, Totals only* |

*Soccer moneyline is three-way (win/draw/lose) — excluded from two-way scanning.

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

## License

MIT

## Contributing

PRs welcome. Ideas for future versions:
- Alerts/notifications
- Historical tracking
- Three-way arbitrage (soccer moneylines)
- Player props

## Support

If this tool saves you time or helps you find an edge, consider buying me a coffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy
## Support

If this tool saves you time or helps you find an edge, consider buying me a coffee:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-yellow.svg)](https://buymeacoffee.com/deliciouspipe1326)
