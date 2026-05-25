# SX Bet Centrifugo Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate SX Bet realtime best-odds streaming from the deprecated Ably channel to SX Bet's Centrifugo websocket API.

**Architecture:** Keep the existing REST prematch scanner unchanged. Replace only `SXBetRealtimeManager`'s websocket client factory/session layer so it gets a realtime token from `/user/realtime-token/api-key`, connects to `wss://ws.sx.bet/connection/websocket`, and subscribes to `best_odds:global`. Reuse the existing best-odds message normalizer and cache merge path.

**Tech Stack:** Python, `centrifuge-python`, existing `providers/sx_bet.py`, pytest/unittest.

---

### Task 1: Add Centrifugo Realtime Tests

**Files:**
- Modify: `tests/test_sx_bet_realtime.py`

- [x] **Step 1: Write failing tests**

Add fake Centrifugo client/subscription classes and update the thread-start test to expect `best_odds:global` instead of the old `best_odds:{baseToken}` Ably channel.

- [x] **Step 2: Verify red**

Run: `pytest tests/test_sx_bet_realtime.py::SXBetRealtimeTests::test_ensure_started_runs_async_client_in_thread -q`

Expected before implementation: fail because `SXBetRealtimeManager` still calls `_create_ably_client`.

### Task 2: Migrate SXBetRealtimeManager

**Files:**
- Modify: `providers/sx_bet.py`
- Modify: `requirements.txt`

- [x] **Step 1: Replace Ably dependency**

Change `requirements.txt` from `ably==3.1.1` to `centrifuge-python==0.4.4`.

- [x] **Step 2: Add Centrifugo config**

Add env-backed constants for websocket URL, realtime token path, and channel:
`SX_BET_WS_URL`, `SX_BET_REALTIME_TOKEN_PATH`, and `SX_BET_REALTIME_CHANNEL`.

- [x] **Step 3: Implement Centrifugo session**

Fetch token from `/user/realtime-token/api-key`, create a Centrifugo client, subscribe to `best_odds:global`, translate publication contexts into existing `_handle_best_odds_message`.

- [x] **Step 4: Verify green**

Run:
`pytest tests/test_sx_bet_realtime.py tests/test_provider_sx_bet.py tests/test_provider_market_segmentation.py -q`

### Task 3: Validate Scanner Still Works

**Files:**
- No additional files.

- [x] **Step 1: Compile**

Run: `python -m py_compile providers/sx_bet.py tools/hunt_real_arbitrage.py`

- [x] **Step 2: Live REST smoke**

Run a short DEX-only scan with the configured SX Bet API key:
`python tools/hunt_real_arbitrage.py --sports basketball_nba baseball_mlb --provider-sets all --allow-quality high medium low --stop-on-first --out data/provider_verification/hunt_dex_sxbet_centrifugo_latest.json`

Expected: no SX Bet 403; report file written; arbitrage candidates may still be zero depending on live markets.
