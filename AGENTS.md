## Edge Scanner Agent Guide

### Project Snapshot

- Python application centered on `app.py`
- Flask web UI for sportsbook scanning, arbitrage, middles, and +EV workflows
- Provider integrations live under `providers/`

### Common Commands

- Install dependencies: `pip install -r requirements.txt`
- Run locally: `python app.py`
- Run tests: `pytest`
- Run provider verification: `python provider_verification.py --sport basketball_nba`

### Superpowers

Superpowers is installed for this machine through Codex native skill discovery:

Installation paths can vary by machine. Do not assume a fixed clone path in this repository.

This repository does not vendor Superpowers as a Python dependency. It is integrated
through Codex native skill discovery, and the `superpowers` skills should be used when
they are available in the current user's local Codex environment.

When working in this repository, prefer the matching Superpowers workflow for non-trivial tasks:

- `brainstorming` before large feature work or design changes
- `writing-plans` before multi-step implementation
- `systematic-debugging` for bug investigation
- `verification-before-completion` before declaring a fix finished

### Working Notes

- Keep provider-specific logic isolated to the relevant module in `providers/`
- Prefer minimal, targeted changes over broad refactors
- Preserve existing deployment and scheduling behavior unless the task explicitly requests changes there
