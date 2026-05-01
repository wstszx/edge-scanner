# Provider Capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize custom-provider sport and market capability metadata, verify provider live-state output stays compatible with scanner live filtering, and generate the provider support matrix from code instead of maintaining it manually.

**Architecture:** Introduce one normalized capability contract for all registered custom providers, then make the registry, verification flow, and documentation consume that contract. Add a provider/scanner live-state compatibility test suite so normalized provider output cannot silently drift away from the scanner's live filtering rules. Generate the support matrix from code so docs reflect the actual registry rather than manual summaries.

**Tech Stack:** Python 3.12, pytest, existing provider modules in `providers/`, scanner orchestration in `scanner.py`, verification/reporting in `provider_verification.py`, Markdown docs in `docs/`

---

## File Structure

**Create:**
- `D:\pythonProject\edge-scanner\providers\capabilities.py`
  Purpose: shared capability schema/helpers for provider declarations and registry export.
- `D:\pythonProject\edge-scanner\tests\test_provider_capabilities_registry.py`
  Purpose: regression coverage for centralized provider capability metadata.
- `D:\pythonProject\edge-scanner\tests\test_provider_live_state_contract.py`
  Purpose: regression coverage for provider normalized live-state output versus scanner live filtering.
- `D:\pythonProject\edge-scanner\scripts\generate_provider_support_matrix.py`
  Purpose: generate `docs/provider_support_matrix.md` from code-declared capability metadata.
- `D:\pythonProject\edge-scanner\tests\test_provider_support_matrix_generator.py`
  Purpose: regression coverage for generated support-matrix content.

**Modify:**
- `D:\pythonProject\edge-scanner\providers\__init__.py`
  Purpose: expose centralized provider capability registry alongside fetchers/titles.
- `D:\pythonProject\edge-scanner\providers\artline.py`
  Purpose: declare provider capability metadata from existing sport/market support.
- `D:\pythonProject\edge-scanner\providers\betdex.py`
  Purpose: declare provider capability metadata from existing sport/market support.
- `D:\pythonProject\edge-scanner\providers\bookmaker_xyz.py`
  Purpose: declare provider capability metadata from existing sport/market support.
- `D:\pythonProject\edge-scanner\providers\polymarket.py`
  Purpose: declare provider capability metadata from existing sport/market support.
- `D:\pythonProject\edge-scanner\providers\sx_bet.py`
  Purpose: declare provider capability metadata and keep live-state contract explicit.
- `D:\pythonProject\edge-scanner\provider_verification.py`
  Purpose: read provider list/title/docs/capabilities from the centralized registry instead of hard-coded duplicates where reasonable.
- `D:\pythonProject\edge-scanner\tests\test_provider_sport_coverage.py`
  Purpose: pivot existing sport coverage checks to the centralized capability contract.
- `D:\pythonProject\edge-scanner\tests\test_provider_market_segmentation.py`
  Purpose: keep provider-specific market tests but assert overlap with declared capabilities where useful.
- `D:\pythonProject\edge-scanner\docs\provider_support_matrix.md`
  Purpose: generated output from the support-matrix script, no longer hand-maintained.
- `D:\pythonProject\edge-scanner\README.md`
  Purpose: link to the generated support matrix and clarify it is code-declared support, not real-time availability.

**Reference:**
- `D:\pythonProject\edge-scanner\scanner.py`
  Purpose: scanner live filtering and provider registry consumers that must stay compatible.
- `D:\pythonProject\edge-scanner\docs/superpowers/plans/2026-03-30-live-provider-funnel-plan.md`
  Purpose: prior live-provider planning context for consistency.

---

### Task 1: Add A Central Provider Capability Contract

**Files:**
- Create: `D:\pythonProject\edge-scanner\providers\capabilities.py`
- Modify: `D:\pythonProject\edge-scanner\providers\__init__.py`
- Modify: `D:\pythonProject\edge-scanner\providers\artline.py`
- Modify: `D:\pythonProject\edge-scanner\providers\betdex.py`
- Modify: `D:\pythonProject\edge-scanner\providers\bookmaker_xyz.py`
- Modify: `D:\pythonProject\edge-scanner\providers\polymarket.py`
- Modify: `D:\pythonProject\edge-scanner\providers\sx_bet.py`
- Test: `D:\pythonProject\edge-scanner\tests\test_provider_capabilities_registry.py`

- [ ] **Step 1: Write the failing registry test**

```python
from providers import PROVIDER_CAPABILITIES


def test_registered_providers_expose_capability_metadata():
    assert set(PROVIDER_CAPABILITIES) == {"artline", "betdex", "bookmaker_xyz", "sx_bet", "polymarket"}
    assert PROVIDER_CAPABILITIES["sx_bet"].live_mode_supported is True
    assert "basketball_nba" in PROVIDER_CAPABILITIES["artline"].supported_sport_keys
    assert "both_teams_to_score" in PROVIDER_CAPABILITIES["polymarket"].supported_markets
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `pytest tests/test_provider_capabilities_registry.py -v`
Expected: FAIL because `PROVIDER_CAPABILITIES` and the shared schema do not exist yet.

- [ ] **Step 3: Add the shared capability schema**

Implement a small contract in `providers/capabilities.py`, for example:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    key: str
    title: str
    supported_sport_keys: tuple[str, ...]
    supported_markets: tuple[str, ...]
    live_mode_supported: bool
```

Include small helpers such as `normalize_supported_markets(...)` and `sorted_unique_keys(...)` so provider modules can build the contract from their existing constants.

- [ ] **Step 4: Declare capabilities in each provider module**

Use existing mappings rather than inventing support:

- `artline.py`: derive sport keys from `ARTLINE_SPORT_FILTERS`; markets from existing normalization branches (`h2h`, `h2h_3_way`, `spreads`, `totals`, `team_totals`).
- `betdex.py`: derive sport keys from `SPORT_SUBCATEGORY_DEFAULTS`/`SPORT_LEAGUE_HINTS`; markets from alias handling (`h2h`, `h2h_3_way`, `spreads`, `totals`, `both_teams_to_score`, `h2h_h1`, `spreads_h1`, `totals_h1`).
- `bookmaker_xyz.py`: derive sport keys from existing sport filter/hint maps; markets from normalized condition handling (`h2h`, `spreads`, `totals`, `both_teams_to_score`).
- `polymarket.py`: derive sport keys from `SPORT_ALIASES` and result hints; markets from `_requested_market_keys(...)`.
- `sx_bet.py`: derive sport keys from `SX_SPORT_ID_MAP` and `SPORT_LEAGUE_HINTS`; markets from `_market_type_aliases(...)` and supported normalized market families.

- [ ] **Step 5: Export the registry from `providers/__init__.py`**

Add `PROVIDER_CAPABILITIES` keyed by provider key, parallel to `PROVIDER_FETCHERS` and `PROVIDER_TITLES`.

- [ ] **Step 6: Run registry tests to verify they pass**

Run: `pytest tests/test_provider_capabilities_registry.py tests/test_provider_sport_coverage.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add providers/capabilities.py providers/__init__.py providers/artline.py providers/betdex.py providers/bookmaker_xyz.py providers/polymarket.py providers/sx_bet.py tests/test_provider_capabilities_registry.py tests/test_provider_sport_coverage.py
git commit -m "refactor: centralize provider capability metadata"
```

---

### Task 2: Add Provider/Scanner Live-State Contract Tests

**Files:**
- Create: `D:\pythonProject\edge-scanner\tests\test_provider_live_state_contract.py`
- Modify: `D:\pythonProject\edge-scanner\providers\sx_bet.py`
- Modify: `D:\pythonProject\edge-scanner\tests\test_provider_market_segmentation.py`
- Reference: `D:\pythonProject\edge-scanner\scanner.py`

- [ ] **Step 1: Write the failing contract test**

```python
import scanner


def test_sx_bet_scheduled_live_enabled_state_is_not_live_for_scanner():
    event = {
        "live_state": {
            "is_live": False,
            "status": "scheduled",
            "provider_status": "active",
            "market_status": "active",
            "live_enabled": True,
        },
        "bookmakers": [],
    }

    assert scanner._event_is_explicitly_live(event) is False
```

Also add at least one positive case:

```python
def test_explicit_live_state_survives_provider_to_scanner_contract():
    event = {
        "live_state": {"is_live": True, "status": "live"},
        "bookmakers": [],
    }

    assert scanner._event_is_explicitly_live(event) is True
```

- [ ] **Step 2: Run the contract test to verify it fails or exposes missing structure**

Run: `pytest tests/test_provider_live_state_contract.py -v`
Expected: FAIL until the test harness is wired to real provider-normalized fixtures or helper constructors.

- [ ] **Step 3: Build reusable normalized-event fixtures**

Add tiny helper builders in `tests/test_provider_live_state_contract.py` that model normalized provider events the scanner actually receives. Focus on contract behavior, not network mocks.

- [ ] **Step 4: Add provider-specific regression cases**

Cover at least:
- `sx_bet`: explicit scheduled/non-live state must not be treated as live even when the market is active.
- one positive provider case (`betdex` or `polymarket`): explicit live state survives scanner filtering.
- one fallback/non-explicit case: future event without explicit live state is still governed by scanner time-window behavior, not provider optimism.

- [ ] **Step 5: Align any provider-specific helper expectations**

If existing tests in `tests/test_provider_market_segmentation.py` assume looser provider live semantics, update those tests so provider behavior and scanner behavior match the intended contract.

- [ ] **Step 6: Run the focused test suite**

Run: `pytest tests/test_provider_live_state_contract.py tests/test_provider_market_segmentation.py tests/test_scanner_regressions.py -k "live or contract or segmentation" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_provider_live_state_contract.py tests/test_provider_market_segmentation.py providers/sx_bet.py
git commit -m "test: lock provider live-state contract to scanner rules"
```

---

### Task 3: Generate The Provider Support Matrix From Code

**Files:**
- Create: `D:\pythonProject\edge-scanner\scripts\generate_provider_support_matrix.py`
- Create: `D:\pythonProject\edge-scanner\tests\test_provider_support_matrix_generator.py`
- Modify: `D:\pythonProject\edge-scanner\provider_verification.py`
- Modify: `D:\pythonProject\edge-scanner\docs\provider_support_matrix.md`
- Modify: `D:\pythonProject\edge-scanner\README.md`

- [ ] **Step 1: Write the failing generator test**

```python
from scripts.generate_provider_support_matrix import build_matrix_markdown


def test_matrix_markdown_mentions_registered_providers_and_key_sports():
    markdown = build_matrix_markdown()
    assert "Custom Provider Support Matrix" in markdown
    assert "`bookmaker_xyz`" in markdown
    assert "`basketball_nba`" in markdown
```

- [ ] **Step 2: Run the generator test to verify it fails**

Run: `pytest tests/test_provider_support_matrix_generator.py -v`
Expected: FAIL because the generator module does not exist yet.

- [ ] **Step 3: Implement the generator**

Create `scripts/generate_provider_support_matrix.py` with:
- a pure `build_matrix_markdown()` function that reads `providers.PROVIDER_CAPABILITIES`
- a small CLI entry point that writes `docs/provider_support_matrix.md`
- stable ordering for providers and sport keys

- [ ] **Step 4: Replace manual matrix maintenance**

Regenerate `docs/provider_support_matrix.md` from the script and keep the same cautionary notes:
- code-declared support, not guaranteed real-time availability
- market support is family-level support, not every sport/provider combination

- [ ] **Step 5: Optionally reduce duplication in `provider_verification.py`**

Move any provider key/title/doc defaults that duplicate the registry into one place when it is low-risk to do so. Keep the change small; do not bundle unrelated refactors.

- [ ] **Step 6: Update README**

Add a short link to the generated support matrix and explain that the document describes declared support, not current live coverage.

- [ ] **Step 7: Run verification commands**

Run:

```bash
pytest tests/test_provider_support_matrix_generator.py tests/test_provider_sport_coverage.py tests/test_provider_market_segmentation.py -v
python scripts/generate_provider_support_matrix.py
git diff --check -- docs/provider_support_matrix.md
```

Expected:
- tests PASS
- support matrix regenerates without errors
- `git diff --check` is clean

- [ ] **Step 8: Commit**

```bash
git add scripts/generate_provider_support_matrix.py tests/test_provider_support_matrix_generator.py provider_verification.py docs/provider_support_matrix.md README.md
git commit -m "docs: generate provider support matrix from code"
```

---

## Final Verification

- [ ] Run the full focused suite:

```bash
pytest tests/test_provider_capabilities_registry.py tests/test_provider_live_state_contract.py tests/test_provider_support_matrix_generator.py tests/test_provider_sport_coverage.py tests/test_provider_market_segmentation.py tests/test_provider_verification.py tests/test_scanner_regressions.py -v
```

Expected: PASS

- [ ] Regenerate docs:

```bash
python scripts/generate_provider_support_matrix.py
git diff --check
```

Expected: clean output, no whitespace errors

- [ ] Smoke-check provider verification:

```bash
python provider_verification.py --sport basketball_nba --skip-tests --summary-only
```

Expected: command succeeds and still emits a coherent provider summary.

---

## Notes

- Keep the capability contract descriptive, not aspirational. Only declare support already evidenced by code.
- Do not try to normalize all providers to identical sport or market coverage in this work. This plan is about expressing and testing support correctly, not expanding provider functionality.
- Prefer small, reviewable commits in the order above.
- Delegated plan review was not dispatched here because this session was not explicitly authorized for subagent delegation.
