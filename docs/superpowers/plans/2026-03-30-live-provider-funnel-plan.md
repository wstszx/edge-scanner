# Live Provider Funnel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose a live custom-provider funnel that shows raw fetched events, live-filter drops, and post-filter merge inputs so weak live coverage can be attributed to source, filtering, or overlap.

**Architecture:** Extend the existing provider sport summaries produced in `scanner._scan_single_sport()` so live scans record provider-local raw counts and live filter stats before merge. Reuse those provider sport summaries in verification/reporting output instead of adding a second diagnostics system, then validate with one fresh real live scan that the new counters explain the failure stage.

**Tech Stack:** Python, existing scanner live filter helpers, existing provider verification/reporting pipeline, pytest

---

### Task 1: Lock The Live Funnel Contract In Scanner Tests

**Files:**
- Modify: `D:/pythonProject/edge-scanner/tests/test_scanner_regressions.py`
- Modify: `D:/pythonProject/edge-scanner/scanner.py`

- [ ] **Step 1: Write a failing regression test for live provider sport summaries**

Add a test that builds or exercises a live-scan provider update path and asserts the sport row includes:

```python
self.assertEqual(sport_row["events_fetched_raw"], 3)
self.assertEqual(sport_row["events_after_live_filter"], 1)
self.assertEqual(
    sport_row["live_filter_stats"],
    {
        "dropped_not_live_state": 1,
        "dropped_terminal_state": 0,
        "dropped_past": 1,
        "dropped_future": 0,
        "dropped_missing_time": 0,
        "suspicious_explicit_live_future": 0,
    },
)
```

- [ ] **Step 2: Run the new test to verify RED**

Run: `pytest tests/test_scanner_regressions.py -k "live_filter_stats or events_fetched_raw" -q`  
Expected: FAIL because the live provider sport rows do not expose the funnel fields yet.

- [ ] **Step 3: Add a prematch safety test if none exists nearby**

Write a second assertion or test that prematch rows still behave normally and are not required to carry live-only counts with misleading values.

- [ ] **Step 4: Run the focused scanner regression slice again**

Run: `pytest tests/test_scanner_regressions.py -k "live_filter_stats or prematch" -q`  
Expected: the new live test fails, and the prematch test either passes or fails only for the intended missing field behavior.

- [ ] **Step 5: Commit the test-only RED state**

```bash
git add D:/pythonProject/edge-scanner/tests/test_scanner_regressions.py
git commit -m "test: cover live provider funnel fields"
```

### Task 2: Implement Live Provider Funnel Recording In Scanner

**Files:**
- Modify: `D:/pythonProject/edge-scanner/scanner.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_scanner_regressions.py`

- [ ] **Step 1: Record raw and filtered provider event counts in the live provider branch**

In `scanner._scan_single_sport()`, keep the fetched provider event list intact long enough to measure:

```python
raw_provider_events = list(provider_events)
filtered_provider_events, live_filter_stats = _filter_live_events_for_scan(raw_provider_events)
events_fetched_raw = len(raw_provider_events)
events_after_live_filter = len(filtered_provider_events)
```

Only use the filtered provider list for merge during live scans. Prematch scans should keep using the current provider event flow.

- [ ] **Step 2: Attach the live funnel fields to the provider sport summary row**

Populate the existing provider sport summary dict with:

```python
"events_fetched_raw": events_fetched_raw,
"events_after_live_filter": events_after_live_filter,
"live_filter_stats": {
    "dropped_not_live_state": int(live_filter_stats.get("dropped_not_live_state", 0) or 0),
    "dropped_terminal_state": int(live_filter_stats.get("dropped_terminal_state", 0) or 0),
    "dropped_past": int(live_filter_stats.get("dropped_past", 0) or 0),
    "dropped_future": int(live_filter_stats.get("dropped_future", 0) or 0),
    "dropped_missing_time": int(live_filter_stats.get("dropped_missing_time", 0) or 0),
    "suspicious_explicit_live_future": int(live_filter_stats.get("suspicious_explicit_live_future", 0) or 0),
},
```

- [ ] **Step 3: Preserve the existing merge stats and scan diagnostics behavior**

Do not redesign `scan_diagnostics` in this step. The implementation goal is to make the provider sport rows explainable while leaving existing reason-code logic intact.

- [ ] **Step 4: Run the focused scanner tests to verify GREEN**

Run: `pytest tests/test_scanner_regressions.py -k "live_filter_stats or events_fetched_raw or prematch" -q`  
Expected: PASS.

- [ ] **Step 5: Commit the scanner implementation**

```bash
git add D:/pythonProject/edge-scanner/scanner.py D:/pythonProject/edge-scanner/tests/test_scanner_regressions.py
git commit -m "feat: record live provider funnel stats"
```

### Task 3: Surface Live Funnel Evidence In Verification Output

**Files:**
- Modify: `D:/pythonProject/edge-scanner/provider_verification.py`
- Modify: `D:/pythonProject/edge-scanner/tests/test_provider_verification.py`

- [ ] **Step 1: Write a failing verification-output test**

Add a test with a live provider sport row that includes:

```python
"events_fetched_raw": 3,
"events_after_live_filter": 1,
"live_filter_stats": {
    "dropped_not_live_state": 1,
    "dropped_past": 1,
}
```

Assert the verification summary or markdown output includes compact funnel evidence such as:

```python
self.assertIn("raw=3", summary)
self.assertIn("after_filter=1", summary)
self.assertIn("dropped_not_live_state=1", summary)
self.assertIn("dropped_past=1", summary)
```

- [ ] **Step 2: Run the provider verification test to verify RED**

Run: `pytest tests/test_provider_verification.py -k "live funnel" -q`  
Expected: FAIL because the current output does not expose the new funnel counters.

- [ ] **Step 3: Implement a compact live funnel renderer in provider verification**

Reuse the existing provider sport summary structure. Do not add a second data source. Keep the rendering compact, for example as a note/summary line per provider sport:

```python
f"live_funnel raw={raw_count} after_filter={filtered_count} "
f"dropped_not_live_state={...} dropped_past={...}"
```

- [ ] **Step 4: Run the focused provider verification tests to verify GREEN**

Run: `pytest tests/test_provider_verification.py -k "live funnel or report_to_markdown or build_console_summary" -q`  
Expected: PASS.

- [ ] **Step 5: Commit the verification output update**

```bash
git add D:/pythonProject/edge-scanner/provider_verification.py D:/pythonProject/edge-scanner/tests/test_provider_verification.py
git commit -m "feat: show live provider funnel in verification output"
```

### Task 4: Run Broader Regression Coverage

**Files:**
- Test: `D:/pythonProject/edge-scanner/tests/test_scanner_regressions.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_provider_verification.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_verification_matrix.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_history.py`
- Test: `D:/pythonProject/edge-scanner/tests/test_provider_arb_pipeline.py`

- [ ] **Step 1: Run scanner regressions**

Run: `pytest tests/test_scanner_regressions.py -q`  
Expected: PASS.

- [ ] **Step 2: Run verification and history/reporting tests**

Run: `pytest tests/test_provider_verification.py tests/test_verification_matrix.py tests/test_history.py -q`  
Expected: PASS.

- [ ] **Step 3: Run provider pipeline safety coverage**

Run: `pytest tests/test_provider_arb_pipeline.py -q`  
Expected: PASS.

- [ ] **Step 4: Commit the green verification checkpoint**

```bash
git add D:/pythonProject/edge-scanner/scanner.py D:/pythonProject/edge-scanner/provider_verification.py D:/pythonProject/edge-scanner/tests/test_scanner_regressions.py D:/pythonProject/edge-scanner/tests/test_provider_verification.py
git commit -m "test: verify live funnel diagnostics"
```

### Task 5: Validate With A Fresh Real Live Scan

**Files:**
- Modify: none
- Verify: `D:/pythonProject/edge-scanner/data/provider_verification/provider_verification_latest.json`
- Verify: `D:/pythonProject/edge-scanner/data/provider_snapshots/cross_provider_match_report.json`

- [ ] **Step 1: Run a fresh live provider verification command on a weak live pair**

Run:

```bash
python provider_verification.py --sport icehockey_nhl --providers artline bookmaker_xyz --skip-tests --summary-only
```

Expected: The summary explicitly reports whether the pair had:
- zero raw fetched live events
- non-zero raw events but filter drops
- post-filter events that still failed to overlap

- [ ] **Step 2: Inspect the latest JSON artifact**

Confirm `provider_verification_latest.json` contains provider sport rows with the new funnel fields and that the values explain the final `reason_code`.

- [ ] **Step 3: If the real scan shows filter-loss rather than source-loss, capture the exact bottleneck**

Document the highest drop counter from the fresh real scan in the task notes before attempting any follow-up optimization.

- [ ] **Step 4: Commit or document the real verification checkpoint**

```bash
git status --short
```

Expected: only the intended implementation files remain modified; include the real command output in the task handoff notes.

### Task 6: Handoff Summary

**Files:**
- Reference: `D:/pythonProject/edge-scanner/docs/superpowers/specs/2026-03-30-live-provider-funnel-design.md`
- Reference: `D:/pythonProject/edge-scanner/docs/superpowers/plans/2026-03-30-live-provider-funnel-plan.md`

- [ ] **Step 1: Summarize what stage the live bottleneck is in after the fresh scan**

Use one sentence each for source, filter, and merge:

```text
source: ...
filter: ...
merge: ...
```

- [ ] **Step 2: Decide the next optimization target based on the fresh evidence**

If raw live events are zero, next work should move to source/provider coverage.  
If raw events are present but dropped, next work should target live state/time mapping.  
If filtered events survive but fail to overlap, next work should target matching.

- [ ] **Step 3: Stop after evidence-based recommendation**

Do not bundle a second optimization into this plan. The purpose of this plan is to make the next live improvement choice evidence-based.
