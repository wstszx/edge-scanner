# Sports Arbitrage Scanner

Scan for cross-book sports betting arbitrage using The Odds API.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Get a free API key from [The Odds API](https://the-odds-api.com/)
3. Run the app: `python app.py` (add `--port 5050` if port 5000 is busy)

Your browser will open automatically. Enter your API key, pick sports (or use the default set), and click **Scan**. Each sport selected consumes one API request.  
If you see "Access to 127.0.0.1 was denied", another process (often macOS AirPlay) is already on port 5000 — rerun with `python app.py --port 5050`.

## How It Works

- The backend pulls odds from US (retail) and EU (sharp) regions across moneyline, spreads, and totals markets.
- For each event and market, it collects the best price per outcome and checks if the inverse-odds sum is below 1 (classic two-way arbitrage).
- Results are grouped with ROI bands and sport summaries, and displayed in an interactive table.

## Notes

- Default sports: NFL, NBA, MLB, NHL, top European soccer leagues, MLS.
- Use the **All sports** toggle to scan every active non-futures sport (uses many API credits).
- Handle API rate limits gracefully; errors appear in the UI if the key is invalid or quota exceeded.

MIT License
