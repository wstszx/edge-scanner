# DEX Latest Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the normal DEX report with quote-skew gates and write a concise Markdown summary next to the JSON report.

**Architecture:** Keep the JSON payload as source of truth. Add a small Markdown renderer in `tools/hunt_dex_opportunities.py` and write a summary path derived from `--out`, defaulting to `latest_actionable_summary.md`.

**Tech Stack:** Python, pytest, existing DEX runner.

---

### Task 1: Markdown Summary Renderer

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [x] **Step 1: Add renderer tests**

Assert a payload with actionable, execution-risky, and model-only rows renders the key counts and top risk reasons.

- [x] **Step 2: Implement `_render_markdown_summary()`**

Render summary counts, risk counts, and top rows without exposing secrets.

### Task 2: Write Summary During Runner Execution

**Files:**
- Modify: `tools/hunt_dex_opportunities.py`
- Test: `tests/test_hunt_dex_opportunities.py`

- [x] **Step 1: Add CLI summary test**

Assert `main()` writes both JSON and Markdown when `--summary-out` is provided.

- [x] **Step 2: Implement `--summary-out`**

Default the summary path to `data/provider_verification/latest_actionable_summary.md` and print both output paths.

- [x] **Step 3: Verify with real normal-mode scan**

Run normal DEX mode and inspect JSON plus Markdown for quote-skew demotion.
