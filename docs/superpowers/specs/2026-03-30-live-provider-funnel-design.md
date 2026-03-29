# Live Provider Funnel Design

**Date:** 2026-03-30

**Goal**

Make live custom-provider scans explainable by exposing a per-provider, per-sport funnel that shows whether coverage is being lost at source fetch, live-state filtering, time filtering, or merge/overlap.

## Problem

Current live verification already shows that practical custom-provider coverage is much weaker than prematch coverage, but the system still collapses multiple failure modes into coarse outcomes such as `no_source_events` or `no_cross_provider_overlap`.

That is not enough to decide the next optimization step:

- If providers are returning zero raw live events, the bottleneck is source coverage.
- If providers are returning live events but they are removed by live-state or time filtering, the bottleneck is filtering/state mapping.
- If events survive filtering but still do not overlap, the bottleneck is merge/matching.

Without that funnel, live optimization risks targeting the wrong layer.

## Current Evidence

- Recent live provider-only scans frequently end with `no_source_events` or very low overlap.
- Real examples have already shown cases where providers report zero merged live events, but the product surface does not clearly separate "provider returned nothing" from "provider returned events that were filtered away."
- Prematch is already materially stronger than live, so the highest-value next step is narrowing the live bottleneck, not broad refactoring.

## Scope

This sub-project only adds observability for the live provider funnel and threads that evidence into existing diagnostics/reporting surfaces.

### In Scope

- Record raw provider event counts before live filtering.
- Record filtered counts after live-state/time filtering.
- Preserve live filter counters per `provider x sport`.
- Surface those counters through existing `custom_providers` sport summaries.
- Make the same counters visible through verification/reporting outputs so real scans can explain where live coverage is lost.

### Out of Scope

- Broad provider API rewrites.
- Match tolerance changes.
- Team alias normalization changes.
- UI redesign.
- Guarantees that live opportunity counts will increase in this iteration.

## Design

### 1. Per-provider live funnel counters

The live funnel will be recorded inside the existing provider sport summary entries that are built during `scanner._scan_single_sport()`.

For live scans, each provider sport entry should expose:

- `events_fetched_raw`
- `events_after_live_filter`
- `live_filter_stats`

`live_filter_stats` should carry the existing live filter counters already produced by `_filter_live_events_for_scan()` when available:

- `dropped_not_live_state`
- `dropped_terminal_state`
- `dropped_past`
- `dropped_future`
- `dropped_missing_time`
- `suspicious_explicit_live_future`

These values should be attached without changing prematch behavior.

### 2. Data flow

The intended data flow for live scans becomes:

1. Provider fetch returns `provider_events_raw`
2. Live filtering runs on the provider-local event list
3. Filter stats are stored on that provider sport row
4. Filtered events are merged into the shared event set
5. Existing merge stats and cross-provider diagnostics continue unchanged

This preserves the current scanner architecture while making the lost-event stage explicit.

### 3. Where the counters live

The counters should extend the existing `provider_updates[provider_key]["sports"]` entries rather than introducing a new diagnostics structure. That keeps the data reusable by:

- scan results
- provider verification
- verification matrix
- future UI/report surfaces

### 4. Reporting surfaces

This iteration should expose the new funnel evidence at least through verification/reporting paths that are already used to inspect live provider-only scans.

Recommended minimum:

- `provider_verification` output
- any diagnostics payloads already emitted from `scan_diagnostics` / provider summaries

The product UI can remain mostly unchanged as long as the raw diagnostic payload now contains the funnel counts needed for future presentation.

## Implementation Notes

### Scanner

Modify the live provider branch in `scanner._scan_single_sport()` so each provider's event list is measured twice:

- before provider-local live filtering
- after provider-local live filtering

The provider-local filtered list is what should be passed into `_merge_events_with_stats()` for live scans.

Prematch scans should keep using the existing flow.

### Verification Output

`provider_verification` should expose enough of the new provider sport counters to make the funnel visible during real live verification runs. The exact rendering can stay compact, but the output must distinguish:

- raw live events fetched
- live-filter drops
- post-filter events entering merge

## Testing Strategy

### Regression Tests

Add scanner regressions that verify:

- live provider sport summaries include `events_fetched_raw`
- live provider sport summaries include `events_after_live_filter`
- live provider sport summaries include `live_filter_stats`
- prematch summaries are not broken by the new live-only fields

Add verification/report tests that verify:

- the new live funnel counters are preserved when scan/provider summaries are surfaced

### Real Validation

Run at least one fresh live custom-provider scan after implementation and confirm the resulting evidence can answer which stage failed:

- no raw provider events
- raw events filtered away
- filtered events survive but fail to overlap

## Success Criteria

This work is successful when a live provider-only scan that currently ends in `no_source_events` or weak overlap can be explained with concrete counts instead of guesswork.

Specifically, after the change we should be able to answer, from real scan output alone:

- Did the provider return raw live events?
- Were those events removed by live-state/time filtering?
- Did filtered events reach merge but fail to overlap?

## Risks

- Provider-local filtering for live scans must not accidentally double-filter or diverge from the shared event filtering rules.
- Some providers may already pre-filter their feeds, so zero raw events can still be a true upstream limit.
- Output growth should stay modest; this is diagnostic metadata, not a full event dump.
