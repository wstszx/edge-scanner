# Paper Settlement And DEX Consensus Plan

## Goal

Make the scanner more operationally honest by showing which DEX providers can actually be paper-executed, settling paper trades once results are supplied, and allowing +EV to use a multi-DEX consensus reference instead of a single sharp book only.

## Steps

1. Add tests for paper-trade settlement outcomes, ledger update ordering, DEX consensus +EV, capability summary redaction, and frontend settlement rendering.
2. Implement settlement helpers in `paper_trading.py` without external result fetching.
3. Add DEX liquidity-weighted consensus +EV fallback in `scanner.py`.
4. Surface execution capability summary in `app.py` and the Providers tab.
5. Run focused Python and Node tests, restart local Flask, and smoke-check the page.
