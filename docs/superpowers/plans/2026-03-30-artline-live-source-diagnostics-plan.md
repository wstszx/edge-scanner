# Artline Live Source Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When Artline returns zero live events for a requested sport, record provider-native evidence about whether the broader live feed has other sports available so source loss can be distinguished from mapping/filter failures.

**Architecture:** Keep the enhancement inside `providers/artline.py` by adding a minimal live-all probe only when a live sport-specific request returns zero games. Reuse existing stats and provider verification note rendering so the evidence flows through the current reporting pipeline without adding a second diagnostics source.

**Tech Stack:** Python, existing Artline provider fetch flow, pytest

---

### Task 1: Add failing provider diagnostics tests

**Files:**
- Modify: `D:/pythonProject/edge-scanner/.worktrees/codex-live-provider-funnel/tests/test_provider_artline.py`
- Modify: `D:/pythonProject/edge-scanner/.worktrees/codex-live-provider-funnel/providers/artline.py`

- [ ] **Step 1: Write a failing Artline provider test for live sport-empty fallback evidence**
- [ ] **Step 2: Run the focused Artline test to verify RED**
- [ ] **Step 3: Write a failing provider verification note/output test for the new evidence**
- [ ] **Step 4: Run the focused verification test to verify RED**

### Task 2: Implement provider-native live source diagnostics

**Files:**
- Modify: `D:/pythonProject/edge-scanner/.worktrees/codex-live-provider-funnel/providers/artline.py`
- Modify: `D:/pythonProject/edge-scanner/.worktrees/codex-live-provider-funnel/provider_verification.py`
- Test: `D:/pythonProject/edge-scanner/.worktrees/codex-live-provider-funnel/tests/test_provider_artline.py`
- Test: `D:/pythonProject/edge-scanner/.worktrees/codex-live-provider-funnel/tests/test_provider_verification.py`

- [ ] **Step 1: When a live sport-specific Artline request returns zero games, probe live-all once and capture available sports/counts in stats**
- [ ] **Step 2: Surface that evidence through provider verification notes or compact output without creating a second diagnostics source**
- [ ] **Step 3: Run focused tests to verify GREEN**

### Task 3: Verify with a fresh real Artline live scan

**Files:**
- Verify: `D:/pythonProject/edge-scanner/.worktrees/codex-live-provider-funnel/data/provider_verification/provider_verification_latest.json`

- [ ] **Step 1: Run a fresh Artline-only live provider verification command**
- [ ] **Step 2: Confirm the artifact/report now explains whether live hockey is source-empty while other live sports exist**
