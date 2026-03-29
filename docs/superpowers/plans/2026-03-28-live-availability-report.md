# Live Availability Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate JSON and Markdown reports that explain real live-provider availability, overlap, and market alignment for custom providers.

**Architecture:** Add a small `live_availability.py` module that runs a live-only provider sample, aggregates provider-level and overlap-level diagnostics, and writes JSON/Markdown outputs. Keep the report generator separate from scanner internals, then add a lightweight manual CLI entry first and optionally call it after live scans complete.

**Tech Stack:** Python, existing provider fetchers, existing scan diagnostics, pytest, Markdown/JSON file output

---

### Task 1: Add Failing Aggregation Tests

**Files:**
- Create: `D:/pythonProject/edge-scanner/tests/test_live_availability.py`
- Modify: `D:/pythonProject/edge-scanner/live_availability.py`

- [ ] Step 1: Write failing tests for provider event classification and overlap aggregation
- [ ] Step 2: Run `pytest tests/test_live_availability.py -q` and verify failure

### Task 2: Implement Aggregation Module

**Files:**
- Create: `D:/pythonProject/edge-scanner/live_availability.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_live_availability.py`

- [ ] Step 1: Implement provider sample normalization, live-state classification, and overlap helpers
- [ ] Step 2: Run `pytest tests/test_live_availability.py -q` and verify pass

### Task 3: Add JSON/Markdown Output Tests And Writers

**Files:**
- Modify: `D:/pythonProject/edge-scanner/live_availability.py`
- Modify: `D:/pythonProject/edge-scanner/tests/test_live_availability.py`

- [ ] Step 1: Write failing tests for JSON/Markdown output structure
- [ ] Step 2: Run `pytest tests/test_live_availability.py -q` and verify failure
- [ ] Step 3: Implement JSON/Markdown writers and timestamped file output helpers
- [ ] Step 4: Run `pytest tests/test_live_availability.py -q` and verify pass

### Task 4: Add Manual CLI Entry

**Files:**
- Modify: `D:/pythonProject/edge-scanner/live_availability.py`

- [ ] Step 1: Add argparse entry for manual report generation
- [ ] Step 2: Run manual command once and verify files are created in `data/`

### Task 5: Add Automatic Live-Scan Hook

**Files:**
- Modify: `D:/pythonProject/edge-scanner/app.py`
- Modify: `D:/pythonProject/edge-scanner/live_availability.py`

- [ ] Step 1: Write failing test or targeted regression for automatic live report call if practical
- [ ] Step 2: Add best-effort automatic report generation after live scans finish
- [ ] Step 3: Run targeted tests plus scanner regressions

### Task 6: Verify End-To-End

**Files:**
- Test: `D:/pythonProject/edge-scanner/tests/test_live_availability.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_scanner_regressions.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_provider_arb_pipeline.py`

- [ ] Step 1: Run `pytest tests/test_live_availability.py tests/test_provider_arb_pipeline.py tests/test_scanner_regressions.py -q`
- [ ] Step 2: Run manual live availability command and inspect output
