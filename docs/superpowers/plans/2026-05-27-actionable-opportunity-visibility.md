# Actionable Opportunity Visibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scan results explain why no real executable opportunity exists, mark +EV rows with DEX execution quality, and show paper-trade legs in plain actionable terms.

**Architecture:** Keep provider fetching unchanged. Add report-layer opportunity funnel metadata in `tools/hunt_dex_opportunities.py`, add +EV execution metadata in `scanner.py`, and render those fields in the existing Flask template/CSS.

**Tech Stack:** Python, Flask/Jinja template JavaScript, Node built-in test runner, pytest.

---

### Task 1: DEX Report Funnel

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [ ] **Step 1: Write the failing test**

Add a test that calls a new `_opportunity_funnel_summary` helper with scan diagnostics, model-only rows, and ticket counts. Assert it reports scan count, source-event coverage, cross-provider-match coverage, blockers, and the `no_execution_ready_opportunity` conclusion.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hunt_dex_opportunities.py::test_opportunity_funnel_summary_explains_no_execution_ready_opportunity -q`
Expected: FAIL because `_opportunity_funnel_summary` does not exist.

- [ ] **Step 3: Write minimal implementation**

Implement `_opportunity_funnel_summary` and include its result under `payload["summary"]["opportunity_funnel"]`. Add a small markdown section using the same data.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hunt_dex_opportunities.py::test_opportunity_funnel_summary_explains_no_execution_ready_opportunity -q`
Expected: PASS.

### Task 2: DEX +EV Quality Metadata

**Files:**
- Modify: `scanner.py`
- Test: `tests/test_scanner_regressions.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds a Polymarket + SX Bet +EV candidate and asserts the returned opportunity includes `execution_quality`, `bet.max_stake`, `bet.quote_source`, and execution identifiers.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner_regressions.py::ScannerRegressionTests::test_collect_plus_ev_marks_dex_execution_quality -q`
Expected: FAIL because +EV rows do not expose execution quality.

- [ ] **Step 3: Write minimal implementation**

Carry provider metadata through `_two_way_outcomes`, compute a compact quality object in `_collect_plus_ev_opportunities`, and attach it to each opportunity.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner_regressions.py::ScannerRegressionTests::test_collect_plus_ev_marks_dex_execution_quality -q`
Expected: PASS.

### Task 3: Frontend Paper Leg Clarity

**Files:**
- Modify: `templates/index.html`
- Modify: `static/style.css`
- Test: `tests/frontend_paper_trades.test.js`

- [ ] **Step 1: Write the failing test**

Add a Node test asserting `renderPaperTradeListHtml` shows outcome, market/line, stake, max stake, fee, quote time, and token/market identifiers.

- [ ] **Step 2: Run test to verify it fails**

Run: `node --test tests/frontend_paper_trades.test.js`
Expected: FAIL because the rendered leg lacks those labels/identifiers.

- [ ] **Step 3: Write minimal implementation**

Expand `renderPaperTradeLeg` and add compact styles so each leg reads like an execution draft instead of a vague row.

- [ ] **Step 4: Run test to verify it passes**

Run: `node --test tests/frontend_paper_trades.test.js`
Expected: PASS.

### Task 4: Final Verification

**Files:**
- Verify all modified behavior.

- [ ] Run: `pytest tests/test_hunt_dex_opportunities.py tests/test_scanner_regressions.py tests/test_paper_trading.py -q`
- [ ] Run: `node --test tests/frontend_paper_trades.test.js`
- [ ] Run a bounded DEX scan command or inspect the latest generated report to confirm the new funnel fields render in JSON/markdown.
