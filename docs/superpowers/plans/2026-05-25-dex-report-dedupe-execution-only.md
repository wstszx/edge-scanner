# DEX Report Dedupe And Execution-Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make DEX hunt reports less noisy by deduplicating repeated opportunities, summarizing blocked reasons, and adding an execution-only scan mode that excludes quote-only liquidity sources.

**Architecture:** Keep scanner math unchanged and enhance `tools/hunt_dex_opportunities.py` as the report/orchestration layer. Reuse provider capability metadata to filter provider sets and to explain why model-only opportunities are blocked.

**Tech Stack:** Python, pytest, existing `tools/hunt_real_arbitrage.py` helpers and provider capability registry.

---

### Task 1: Report Deduplication And Blocked Reasons

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [x] **Step 1: Add report behavior tests**

Add tests that feed duplicate model-only middles from multiple scan jobs and assert the JSON report keeps one copy with `blocked_reasons` and aggregate blocked reason counts.

- [x] **Step 2: Implement dedupe helpers**

Add stable opportunity keys for middles, arbitrage, and +EV rows. Keep the highest metric row and merge provider/job provenance.

- [x] **Step 3: Implement blocked reason annotation**

Convert risk flags plus provider capability data into `blocked_reasons`, including `bookmaker_xyz_quote_only` when a quote-only leg has no executable stake.

### Task 2: Execution-Only Provider Filtering

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [x] **Step 1: Add provider filtering tests**

Assert `--require-explicit-liquidity` removes `bookmaker_xyz` from all/pair/all-markets scan jobs while retaining `betdex`, `sx_bet`, and `polymarket`.

- [x] **Step 2: Implement CLI flag**

Add `--require-explicit-liquidity`, filter generated jobs to providers whose capability `liquidity_confidence` is `explicit`, and skip jobs with fewer than two providers.

- [x] **Step 3: Verify with smoke scan**

Run unit tests and a small real scan comparing normal vs execution-only report shape.
