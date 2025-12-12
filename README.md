# Edge Scanner

**Arbitrage • Middles**

Open-source sports betting scanner that finds arbitrage opportunities and middle bets across US and EU bookmakers.

## Features

| Tab | What it finds | Risk profile |
|-----|---------------|--------------|
| **Arbitrage** | Guaranteed profit by betting both sides across different books | Zero risk |
| **Middles** | Line gaps where both bets can win if the result lands in between | High variance, +EV over time |

## How It Works

The scanner pulls odds from The Odds API across multiple regions (US, EU, UK, AU) and:

1. **Arbitrage** — finds where the sum of inverse odds across books is less than 1, guaranteeing profit regardless of outcome
2. **Middles** — finds where books disagree on spreads/totals, creating a gap where both sides can win

## Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Get a free API key from [The Odds API](https://the-odds-api.com/)
4. Run: `python app.py`

Your browser opens automatically. Enter your API key and scan.

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

NFL key numbers (3, 7) significantly boost probability for spread middles.

Middles lose small amounts most of the time and win big occasionally — positive EV over many bets, but high variance on any single bet.

## Configuration

Configurable via the UI:
- Regions (US, US2, UK, EU, AU)
- Sports selection
- Exchange commission rate (for Betfair, etc.)
- Minimum gap for middles
- Stake amount

## Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Data:** The Odds API

## License

MIT

## Contributing

PRs welcome. Ideas for future versions:
- +EV scanner (compare to sharp lines)
- Three-way arbitrage (soccer moneylines)
- Alerts/notifications
- Historical tracking
