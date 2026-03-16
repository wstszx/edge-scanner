# Provider API Rate Limits (Custom Providers)

Last updated: 2026-03-16 (UTC)

This document tracks public/official request-rate information for custom providers used by this project:
- `betdex`
- `bookmaker_xyz`
- `sx_bet`
- `polymarket`

## Summary

| Provider | Endpoint(s) used by this project | Official/Public rate limit | Practical interval |
|---|---|---|---|
| `polymarket` | Gamma API `GET /events` (see `providers/polymarket.py`) | `500 requests / 10s` for Gamma `/events` | ~`20 ms` per request (50 req/s) |
| `sx_bet` | `GET /summary/upcoming/...`, `GET /orders/odds/best` (see `providers/sx_bet.py`) | Public docs mention a baseline rate limiter, but no numeric limit published | No official numeric interval available; use conservative polling + backoff on 429 |
| `betdex` | Monaco API endpoints `/sessions`, `/events`, `/markets`, `/market-prices` (see `providers/betdex.py`) | Monaco public OpenAPI does not publish explicit numeric rate limits. The legacy website session endpoint `/api/session` may be front-end bot-protected. | No official numeric interval available; use conservative polling + backoff on 429 |
| `bookmaker_xyz` | Azuro public `market-manager` + official dictionaries package (see `providers/bookmaker_xyz.py`) | No public numeric rate-limit doc found | No official numeric interval available; use conservative polling + backoff on 429 |

## Suggested scanner policy (safe default)

If you want one global interval that stays compatible with the strictest known provider route:
- Start at `6-10 seconds` per full provider scan cycle.
- Keep exponential backoff enabled for retriable responses (`429`, `500`, `502`, `503`, `504`).

## Sources

- BetDEX docs entrypoint: https://docs.betdex.com/
- Polymarket rate limits: https://docs.polymarket.com/quickstart/introduction/rate-limits
- SX Bet API docs: https://api.docs.sx.bet/
- Monaco API OpenAPI (BetDEX backend): https://production.api.monacoprotocol.xyz/v3/api-docs
- bookmaker.xyz: https://bookmaker.xyz/
- Azuro public gateway docs: https://api.onchainfeed.org/api/v1/public/gateway/docs
