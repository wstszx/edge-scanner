# DEX Actionable Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add executable arbitrage filtering, provider liquidity capability metadata, and a unified DEX hunt runner that scans all/pair/all-markets combinations into one report.

**Architecture:** Extend the existing `tools/hunt_real_arbitrage.py` report layer instead of changing scanner math. Add provider capability metadata to the provider registry, and build `tools/hunt_dex_opportunities.py` as a thin orchestrator over `hunt_real_arbitrage` helpers.

**Tech Stack:** Python, pytest, existing provider/scanner modules.

---

### Task 1: Actionable Arbitrage

**Files:**
- Modify: `tools/hunt_real_arbitrage.py`
- Test: `tests/test_hunt_real_arbitrage.py`

- [x] **Step 1: Add tests for actionable arbitrage**

Assert high-quality ROI-positive arbitrage is actionable, while missing liquidity, low quality, or ROI <= 0 is not.

- [x] **Step 2: Implement filtering**

Add `_is_actionable_arbitrage()`, `actionable_arbitrage_count`, and aggregate `actionable_arbitrage` fields.

- [x] **Step 3: Verify**

Run: `pytest -q tests/test_hunt_real_arbitrage.py`

### Task 2: Provider Liquidity Capability Metadata

**Files:**
- Modify: `providers/capabilities.py`
- Modify: provider modules under `providers/`
- Test: `tests/test_provider_capabilities_registry.py`

- [x] **Step 1: Add registry tests**

Assert every provider declares `liquidity_confidence`, and bookmaker.xyz is `quote_only`.

- [x] **Step 2: Add dataclass fields and provider values**

Add `liquidity_confidence` and `notes` to `ProviderCapability`.

- [x] **Step 3: Include capabilities in reports**

Expose selected provider capability summaries in hunt JSON payloads.

### Task 3: Unified DEX Hunt Runner

**Files:**
- Create: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [x] **Step 1: Add tests for runner plan**

Assert the runner builds baseline, pair, and all-markets scan jobs, and aggregates actionable/model-only results.

- [x] **Step 2: Implement runner**

Reuse `hunt_real_arbitrage._scan_once()` and write a combined JSON report with job summaries and aggregate top lists.

- [x] **Step 3: Run real scan**

Run a small smoke scan and inspect generated JSON.
