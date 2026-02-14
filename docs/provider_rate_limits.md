# Provider API Rate Limits (Custom Providers)

Last updated: 2026-02-14 (UTC)

This document tracks public/official request-rate information for custom providers used by this project:
- `betdex`
- `bookmaker_xyz`
- `sx_bet`
- `overtimemarkets_xyz`
- `polymarket`

## Summary

| Provider | Endpoint(s) used by this project | Official/Public rate limit | Practical interval |
|---|---|---|---|
| `polymarket` | Gamma API `GET /events` (see `providers/polymarket.py`) | `500 requests / 10s` for Gamma `/events` | ~`20 ms` per request (50 req/s) |
| `overtimemarkets_xyz` | `GET /networks/{id}/markets`, optional `/live-markets` (see `providers/overtimemarkets_xyz.py`) | Protected routes docs: `/markets` and `/live-markets` are `10 requests/minute`; `/sports`, `/users`, `/quote` are `300 requests/minute` | `/markets` and `/live-markets`: ~`6 s` per request; 300/min routes: ~`200 ms` |
| `sx_bet` | `GET /summary/upcoming/...`, `GET /orders/odds/best` (see `providers/sx_bet.py`) | Public docs mention a baseline rate limiter, but no numeric limit published | No official numeric interval available; use conservative polling + backoff on 429 |
| `betdex` | Monaco API endpoints `/events`, `/markets`, `/market-prices` (see `providers/betdex.py`) | Monaco public OpenAPI does not publish explicit numeric rate limits | No official numeric interval available; use conservative polling + backoff on 429 |
| `bookmaker_xyz` | GraphQL via Azuro subgraph + site dictionary bootstrap (see `providers/bookmaker_xyz.py`) | No public numeric rate-limit doc found | No official numeric interval available; use conservative polling + backoff on 429 |

## Suggested scanner policy (safe default)

If you want one global interval that stays compatible with the strictest known provider route:
- Start at `6-10 seconds` per full provider scan cycle.
- Keep exponential backoff enabled for retriable responses (`429`, `500`, `502`, `503`, `504`).

## Sources

- Polymarket rate limits: https://docs.polymarket.com/quickstart/introduction/rate-limits
- Overtime docs (protected routes): https://docs.overtime.io/overtime-v2-integration/overtime-v2-api/protected-routes
- SX Bet API docs: https://api.docs.sx.bet/
- Monaco API OpenAPI (BetDEX backend): https://production.api.monacoprotocol.xyz/v3/api-docs
- bookmaker.xyz: https://bookmaker.xyz/
