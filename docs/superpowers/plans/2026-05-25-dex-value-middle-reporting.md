# DEX Value Middle Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DEX-only hunt report include compact middle and +EV evidence, plus per-scan diagnostics that explain why no actionable opportunity exists.

**Architecture:** Keep scanner core unchanged and extend `tools/hunt_real_arbitrage.py` as a reporting/orchestration layer. Reuse the existing `run_scan()` result shape and add compact summaries for arbitrage, middle, and +EV results.

**Tech Stack:** Python, pytest, existing scanner/provider modules.

---

### Task 1: Compact Value Opportunity Helpers

**Files:**
- Modify: `tools/hunt_real_arbitrage.py`
- Test: `tests/test_hunt_real_arbitrage.py`

- [ ] **Step 1: Add tests for compact middle and +EV helpers**

Add tests that assert compact middle entries include event, market, EV, probability, books, stakes, and quality flags; compact +EV entries include soft book, sharp book, effective odds, gross/net edge, fee/risk notes, and Kelly stake.

- [ ] **Step 2: Run focused tests and verify they fail before implementation**

Run: `pytest -q tests/test_hunt_real_arbitrage.py`

Expected: helper tests fail because the helper functions do not exist yet.

- [ ] **Step 3: Implement compact helper functions**

Add `_compact_middle()`, `_compact_plus_ev()`, `_leg_quality_flags()`, and sorting helpers in `tools/hunt_real_arbitrage.py`.

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `pytest -q tests/test_hunt_real_arbitrage.py`

Expected: all tests pass.

### Task 2: Include Value Results In Live Hunt Report

**Files:**
- Modify: `tools/hunt_real_arbitrage.py`
- Test: `tests/test_hunt_real_arbitrage.py`

- [ ] **Step 1: Add tests for `_scan_once()` result shape**

Mock `run_scan()` and assert `_scan_once()` returns `middle_count`, `positive_middle_count`, `plus_ev_count`, `top_middles`, and `top_plus_ev`.

- [ ] **Step 2: Implement result extraction**

Read `result["middles"]["opportunities"]` and `result["plus_ev"]["opportunities"]`, compact and sort each list, and include them in each scan row.

- [ ] **Step 3: Add aggregate report fields**

Add `top_middles` and `top_plus_ev` to the final JSON payload across all scans.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_hunt_real_arbitrage.py`

Expected: all tests pass.

### Task 3: Verify With A Real DEX Scan

**Files:**
- Runtime output under `data/provider_verification/`

- [ ] **Step 1: Run a focused DEX scan**

Run: `python tools\hunt_real_arbitrage.py --sports basketball_nba baseball_mlb --provider-sets all --out data\provider_verification\dex_value_middle_latest.json`

- [ ] **Step 2: Inspect the JSON report**

Confirm the report has arbitrage candidates plus `top_middles` and `top_plus_ev` aggregate fields.

- [ ] **Step 3: Summarize live result**

Report whether there is a positive actionable arbitrage, positive middle, or +EV candidate, and cite the JSON report path.
