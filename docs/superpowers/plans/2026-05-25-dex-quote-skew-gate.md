# DEX Quote Skew Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent stale or time-skewed two-leg DEX middle opportunities from being reported as directly actionable.

**Architecture:** Keep scanner math unchanged and add a report-layer execution gate in `tools/hunt_dex_opportunities.py`. The runner will compute quote time skew from compact opportunity books, keep fresh middle rows in `actionable_middles`, and demote stale/skewed rows to `execution_risky_middles`.

**Tech Stack:** Python, pytest, existing DEX hunt runner.

---

### Task 1: Quote Time Skew Helpers

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [x] **Step 1: Add skew helper tests**

Add tests for parsing ISO quote timestamps and calculating seconds between the oldest and newest quote on a two-leg opportunity.

- [x] **Step 2: Implement helper functions**

Add `_parse_quote_time()`, `_quote_time_skew_seconds()`, and `_with_quote_time_skew()`.

### Task 2: Actionable Middle Demotion

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [x] **Step 1: Add demotion test**

Assert a middle with clean liquidity but quote skew greater than the configured threshold moves from `actionable_middles` to `execution_risky_middles`.

- [x] **Step 2: Implement CLI flag and report fields**

Add `--max-quote-skew-seconds` defaulting to `120`, report `execution_risky_middles`, `execution_risky_middle_count`, and `execution_risk_counts`.

- [ ] **Step 3: Verify with execution-only smoke scan**

Run unit tests and rerun the explicit liquidity smoke scan to confirm stale opportunities are demoted.
