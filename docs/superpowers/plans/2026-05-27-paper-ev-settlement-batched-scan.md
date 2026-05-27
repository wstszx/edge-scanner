# Paper EV Settlement Batched Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the paper-trading loop for +EV bets, expose a settlement API, and make DEX scans report per-job timeouts instead of hanging large all-markets runs.

**Architecture:** Keep provider execution rules in `paper_trading.py` and `tools/hunt_dex_opportunities.py`, with quote-only providers blocked before any paper-ready ticket can be created. Add app-level settlement through a POST endpoint that calls the existing append-only ledger update path.

**Tech Stack:** Python 3.10, Flask, pytest, Node built-in test runner for frontend helpers.

---

### Task 1: +EV Paper Trade Tickets

**Files:**
- Modify: `paper_trading.py`
- Test: `tests/test_paper_trading.py`
- Test: `tests/test_app_scan_inputs.py`
- Test: `tests/frontend_paper_trades.test.js`

- [ ] **Step 1: Write failing tests**

Add tests that assert a Polymarket +EV opportunity creates an `execution_type == "plus_ev"` paper record, and that an Artline/bookmaker.xyz +EV bet is blocked by quote-only/missing adapter reasons.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
pytest tests/test_paper_trading.py tests/test_app_scan_inputs.py::ScanInputValidationTests::test_execute_scan_payload_records_plus_ev_paper_trades -q
node --test tests/frontend_paper_trades.test.js
```

Expected: Python tests fail because plus-ev tickets are not implemented.

- [ ] **Step 3: Implement minimal +EV ticket builder**

Add `build_plus_ev_execution_ticket()` and include `plus_ev_opportunities` in `record_scan_paper_trades()`. Reuse existing adapter, quote-time, liquidity, and identifier checks.

- [ ] **Step 4: Verify GREEN**

Run the same tests and confirm they pass.

### Task 2: Settlement API

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_scan_inputs.py`

- [ ] **Step 1: Write failing endpoint test**

Add a POST `/paper-trades/settle` test that submits a final result keyed by `paper_trade_key`, expects `settled_count == 1`, and verifies `/paper-trades` returns the settled record.

- [ ] **Step 2: Run endpoint test and verify RED**

Run:

```powershell
pytest tests/test_app_scan_inputs.py::ScanInputValidationTests::test_paper_trades_settle_endpoint_appends_settlement -q
```

Expected: 404 or missing endpoint.

- [ ] **Step 3: Implement endpoint**

Import `settle_paper_trades`, parse JSON body with `results`, call the ledger helper, and return updated records. Keep auth behavior aligned with `/paper-trades`.

- [ ] **Step 4: Verify GREEN**

Run the endpoint test again.

### Task 3: DEX Scan Timeout Reporting

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [ ] **Step 1: Write failing timeout tests**

Add tests for a `--per-scan-timeout-seconds` option. A fake `_scan_once` that exceeds timeout should produce a scan row with `success == False`, `timed_out == True`, `reason_code == "scan_timeout"`, and the runner should still write JSON/markdown.

- [ ] **Step 2: Run timeout test and verify RED**

Run:

```powershell
pytest tests/test_hunt_dex_opportunities.py::test_runner_records_per_scan_timeout_and_continues -q
```

Expected: CLI option/function missing.

- [ ] **Step 3: Implement minimal per-job timeout wrapper**

Run `_scan_once` in a one-worker `ThreadPoolExecutor`, catch `TimeoutError`, and return a synthetic timeout scan row. Shut down executor without waiting for the stuck worker so the report can complete.

- [ ] **Step 4: Verify GREEN**

Run the timeout test and then the full DEX tool tests.

### Task 4: Final Verification

**Files:**
- All touched files

- [ ] **Step 1: Run full focused test suite**

```powershell
pytest tests/test_app_scan_inputs.py tests/test_browser_scan_flow.py tests/test_hunt_dex_opportunities.py tests/test_scanner_regressions.py tests/test_paper_trading.py -q
node --test tests/frontend_paper_trades.test.js tests/frontend_quote_meta.test.js tests/frontend_arb_calculator_roi.test.js tests/frontend_market_line_formatting.test.js tests/frontend_scan_form_helpers.test.js
git diff --check
```

- [ ] **Step 2: Restart Flask and smoke check**

Restart `app.py --host 127.0.0.1 --port 5057 --no-browser`, then verify the home page contains paper trade settlement and execution capability elements.
