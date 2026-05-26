# Actionable Middles Bookmaker Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate executable middle opportunities from model-only positives, surface bookmaker.xyz timestamp/liquidity provenance, and cleanly shut down async resources after CLI scans.

**Architecture:** Keep scanner math unchanged. Add actionable filtering in the hunt report layer, enrich bookmaker.xyz normalized outcomes with observed/liquidity metadata from source conditions, and make the hunt CLI explicitly shut down shared async resources before process exit.

**Tech Stack:** Python, pytest, existing scanner/provider modules.

---

### Task 1: Actionable Middle Filtering

**Files:**
- Modify: `tools/hunt_real_arbitrage.py`
- Test: `tests/test_hunt_real_arbitrage.py`

- [ ] **Step 1: Add failing tests**

Add tests for `_is_actionable_middle()` and `_scan_once()` asserting a positive middle with no risk flags is counted as actionable, while missing liquidity or quote time excludes it.

- [ ] **Step 2: Implement filtering**

Add `_is_actionable_middle()` and include `actionable_middle_count` plus aggregate `actionable_middles` in the JSON payload.

- [ ] **Step 3: Verify focused tests**

Run: `pytest -q tests/test_hunt_real_arbitrage.py`

Expected: all tests pass.

### Task 2: bookmaker.xyz Metadata Tracking

**Files:**
- Modify: `providers/bookmaker_xyz.py`
- Test: `tests/test_bookmaker_xyz_cache.py` or `tests/test_provider_bookmaker_xyz.py` if present

- [ ] **Step 1: Add tests around normalized condition markets**

Call `_normalize_condition_market()` with `__observed_at`, `turnover`, and/or liquidity-like fields and assert normalized outcomes include `observed_at`, `max_stake`, and `liquidity`.

- [ ] **Step 2: Implement metadata extraction**

Add helper functions to derive observed time and liquidity from condition/outcome fields, and attach those fields to all normalized bookmaker.xyz outcome paths.

- [ ] **Step 3: Verify focused provider tests**

Run: `pytest -q tests/test_bookmaker_xyz_cache.py tests/test_hunt_real_arbitrage.py`

Expected: all tests pass.

### Task 3: Clean CLI Async Shutdown

**Files:**
- Modify: `tools/hunt_real_arbitrage.py`
- Test: `tests/test_hunt_real_arbitrage.py`

- [ ] **Step 1: Add a shutdown hook test**

Patch `scanner.shutdown_scan_runtime` and `providers._async_http.shutdown_shared_clients`, call `main()`, and assert both shutdown hooks are called.

- [ ] **Step 2: Implement cleanup**

Import modules rather than direct functions where shutdown is needed, then call cleanup in `main()` after writing the report.

- [ ] **Step 3: Verify and run a real scan**

Run focused tests and a small DEX scan. Confirm no `Event loop is closed` warning appears in command output.
