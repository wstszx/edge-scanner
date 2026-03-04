# Custom Provider Data API cURL Checklist

This document lists all data-fetch interfaces used by custom providers in `providers/`.
It is intended for endpoint health checks and manual API validation.

Registered custom providers:
- `purebet`
- `betdex`
- `bookmaker_xyz`
- `sx_bet`
- `polymarket`
- `dexsport_io` (proxy over `bookmaker_xyz`)
- `sportbet_one` (proxy over `bookmaker_xyz`)

## 0) Common Variables

```bash
# Purebet
PUREBET_API_BASE="https://v3api.purebet.io"
PUREBET_ORIGIN="https://purebet.io"
PUREBET_REFERER="https://purebet.io/"
PUREBET_UA="Mozilla/5.0"
PUREBET_EVENT_ID="REPLACE_EVENT_ID"

# BetDEX
BETDEX_SESSION_URL="https://www.betdex.com/api/session"
BETDEX_API_BASE="https://production.api.monacoprotocol.xyz"
BETDEX_UA="Mozilla/5.0"
BETDEX_SUBCATEGORY="FOOTBALL"
BETDEX_EVENT_ID_1="REPLACE_EVENT_ID_1"
BETDEX_EVENT_ID_2="REPLACE_EVENT_ID_2"
BETDEX_MARKET_ID_1="REPLACE_MARKET_ID_1"
BETDEX_MARKET_ID_2="REPLACE_MARKET_ID_2"

# bookmaker.xyz
BOOKMAKER_HOME="https://bookmaker.xyz/"
BOOKMAKER_PUBLIC_BASE="https://bookmaker.xyz"
BOOKMAKER_GRAPH_BASE="https://thegraph-1.onchainfeed.org/subgraphs/name/azuro-protocol"
BOOKMAKER_UA="Mozilla/5.0"
CHAIN_SLUG="polygon"  # polygon | gnosis | base
MIN_STARTS_AT="REPLACE_UNIX_TIMESTAMP"

# SX Bet
SX_BASE="https://api.sx.bet"
SX_UA="Mozilla/5.0"
SX_BEARER=""          # optional
SX_API_KEY=""         # optional
SX_COOKIE=""          # optional
SX_BASE_TOKEN="0x6629Ce1Cf35Cc1329ebB4F63202F3f197b3F050B"
SX_SPORT_ID="1"
SX_MARKET_HASHES="0xabc...,0xdef..."

# Polymarket
POLY_API_BASE="https://gamma-api.polymarket.com"
POLY_CLOB_BASE="https://clob.polymarket.com"
POLY_UA="Mozilla/5.0"
POLY_BEARER=""        # optional
POLY_API_KEY=""       # optional
POLY_COOKIE=""        # optional
POLY_TAG_ID="100639"
POLY_END_DATE_MIN="2026-01-01T00:00:00Z"
POLY_TOKEN_ID="REPLACE_CLOB_TOKEN_ID"
```

## 1) Purebet (`providers/purebet.py`)

### 1.1 Event list `/events`

```bash
curl -sS -G "$PUREBET_API_BASE/events" \
  -H "Origin: $PUREBET_ORIGIN" \
  -H "Referer: $PUREBET_REFERER" \
  -H "User-Agent: $PUREBET_UA" \
  --data-urlencode "live=false"
```

### 1.2 Active leagues `/activeLeagues`

```bash
curl -sS -G "$PUREBET_API_BASE/activeLeagues" \
  -H "Origin: $PUREBET_ORIGIN" \
  -H "Referer: $PUREBET_REFERER" \
  -H "User-Agent: $PUREBET_UA"
```

### 1.3 Event markets `/markets?event=<event_id>`

```bash
curl -sS -G "$PUREBET_API_BASE/markets" \
  -H "Origin: $PUREBET_ORIGIN" \
  -H "Referer: $PUREBET_REFERER" \
  -H "User-Agent: $PUREBET_UA" \
  --data-urlencode "event=$PUREBET_EVENT_ID"
```

## 2) BetDEX (`providers/betdex.py`)

### 2.1 Session token `/api/session`

```bash
curl -sS "$BETDEX_SESSION_URL" \
  -H "Accept: application/json" \
  -H "User-Agent: $BETDEX_UA"
```

Extract bearer token (requires `jq`):

```bash
BETDEX_TOKEN="$(curl -sS "$BETDEX_SESSION_URL" \
  -H "Accept: application/json" \
  -H "User-Agent: $BETDEX_UA" | jq -r '.sessions[0].accessToken')"
echo "$BETDEX_TOKEN"
```

### 2.2 Events `/events`

```bash
curl -sS -G "$BETDEX_API_BASE/events" \
  -H "Accept: application/json" \
  -H "User-Agent: $BETDEX_UA" \
  -H "Authorization: Bearer $BETDEX_TOKEN" \
  --data-urlencode "active=true" \
  --data-urlencode "page=0" \
  --data-urlencode "size=250" \
  --data-urlencode "subcategoryIds=$BETDEX_SUBCATEGORY"
```

### 2.3 Markets `/markets` (repeated `eventIds` and `statuses`)

```bash
curl -sS -G "$BETDEX_API_BASE/markets" \
  -H "Accept: application/json" \
  -H "User-Agent: $BETDEX_UA" \
  -H "Authorization: Bearer $BETDEX_TOKEN" \
  --data-urlencode "eventIds=$BETDEX_EVENT_ID_1" \
  --data-urlencode "eventIds=$BETDEX_EVENT_ID_2" \
  --data-urlencode "published=true" \
  --data-urlencode "page=0" \
  --data-urlencode "size=500" \
  --data-urlencode "statuses=Open"
```

### 2.4 Market prices `/market-prices` (repeated `marketIds`)

```bash
curl -sS -G "$BETDEX_API_BASE/market-prices" \
  -H "Accept: application/json" \
  -H "User-Agent: $BETDEX_UA" \
  -H "Authorization: Bearer $BETDEX_TOKEN" \
  --data-urlencode "marketIds=$BETDEX_MARKET_ID_1" \
  --data-urlencode "marketIds=$BETDEX_MARKET_ID_2"
```

## 3) bookmaker.xyz (`providers/bookmaker_xyz.py`)

### 3.1 Home page (used to discover const asset path)

```bash
curl -sS "$BOOKMAKER_HOME" \
  -H "User-Agent: $BOOKMAKER_UA"
```

### 3.2 Const asset (`/assets/const-*.js`)

```bash
CONST_ASSET_PATH="/assets/const-REPLACE_HASH.js"
curl -sS "${BOOKMAKER_PUBLIC_BASE}${CONST_ASSET_PATH}" \
  -H "User-Agent: $BOOKMAKER_UA"
```

### 3.3 GraphQL conditions endpoint

Endpoint format:
- `$BOOKMAKER_GRAPH_BASE/azuro-data-feed-polygon`
- `$BOOKMAKER_GRAPH_BASE/azuro-data-feed-gnosis`
- `$BOOKMAKER_GRAPH_BASE/azuro-data-feed-base`

```bash
curl -sS "${BOOKMAKER_GRAPH_BASE}/azuro-data-feed-${CHAIN_SLUG}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "User-Agent: $BOOKMAKER_UA" \
  --data-raw @- <<'JSON'
{
  "query": "query { conditions(first: 400, skip: 0, where: {state: \"Active\", game_: {state_in: [\"Prematch\", \"Live\"], startsAt_gt: \"REPLACE_MIN_STARTS_AT\"}}, orderBy: conditionId, orderDirection: desc, subgraphError: allow) { id conditionId state outcomes { outcomeId title currentOdds sortOrder } game { id gameId slug title startsAt state sport { sportId slug name } league { slug name } country { slug name } participants { name } } } }"
}
JSON
```

If you want to inject a runtime timestamp, replace `REPLACE_MIN_STARTS_AT` first.

## 4) SX Bet (`providers/sx_bet.py`)

Optional auth headers:
- `Authorization: Bearer $SX_BEARER`
- `X-API-Key: $SX_API_KEY`
- `Cookie: $SX_COOKIE`

### 4.1 Upcoming fixtures `/summary/upcoming/{baseToken}/{sportId}`

```bash
curl -sS "$SX_BASE/summary/upcoming/$SX_BASE_TOKEN/$SX_SPORT_ID" \
  -H "User-Agent: $SX_UA"
```

### 4.2 Best odds `/orders/odds/best`

```bash
curl -sS -G "$SX_BASE/orders/odds/best" \
  -H "User-Agent: $SX_UA" \
  -H "Authorization: Bearer $SX_BEARER" \
  -H "X-API-Key: $SX_API_KEY" \
  -H "Cookie: $SX_COOKIE" \
  --data-urlencode "marketHashes=$SX_MARKET_HASHES" \
  --data-urlencode "baseToken=$SX_BASE_TOKEN"
```

### 4.3 Orders `/orders`

```bash
curl -sS -G "$SX_BASE/orders" \
  -H "User-Agent: $SX_UA" \
  -H "Authorization: Bearer $SX_BEARER" \
  -H "X-API-Key: $SX_API_KEY" \
  -H "Cookie: $SX_COOKIE" \
  --data-urlencode "marketHashes=$SX_MARKET_HASHES" \
  --data-urlencode "baseToken=$SX_BASE_TOKEN"
```

## 5) Polymarket (`providers/polymarket.py`)

Optional auth headers:
- `Authorization: Bearer $POLY_BEARER`
- `X-API-Key: $POLY_API_KEY`
- `Cookie: $POLY_COOKIE`

### 5.1 Sports tags `/sports`

```bash
curl -sS "$POLY_API_BASE/sports" \
  -H "User-Agent: $POLY_UA" \
  -H "Authorization: Bearer $POLY_BEARER" \
  -H "X-API-Key: $POLY_API_KEY" \
  -H "Cookie: $POLY_COOKIE"
```

### 5.2 Active events `/events`

```bash
curl -sS -G "$POLY_API_BASE/events" \
  -H "User-Agent: $POLY_UA" \
  -H "Authorization: Bearer $POLY_BEARER" \
  -H "X-API-Key: $POLY_API_KEY" \
  -H "Cookie: $POLY_COOKIE" \
  --data-urlencode "tag_id=$POLY_TAG_ID" \
  --data-urlencode "active=true" \
  --data-urlencode "closed=false" \
  --data-urlencode "archived=false" \
  --data-urlencode "end_date_min=$POLY_END_DATE_MIN" \
  --data-urlencode "order=id" \
  --data-urlencode "ascending=false" \
  --data-urlencode "limit=200" \
  --data-urlencode "offset=0"
```

### 5.3 CLOB order book `/book?token_id=<token_id>`

```bash
curl -sS -G "$POLY_CLOB_BASE/book" \
  -H "User-Agent: $POLY_UA" \
  -H "Authorization: Bearer $POLY_BEARER" \
  -H "X-API-Key: $POLY_API_KEY" \
  -H "Cookie: $POLY_COOKIE" \
  --data-urlencode "token_id=$POLY_TOKEN_ID"
```

## 6) Dexsport and Sportbet Proxy Notes

### 6.1 `dexsport_io` (`providers/dexsport_io.py`)
- Default `DEXSPORT_SOURCE=bookmaker_xyz`.
- No independent HTTP data endpoint.
- Reuses `bookmaker_xyz` fetch path and only re-tags output key/title.

### 6.2 `sportbet_one` (`providers/sportbet_one.py`)
- Default `SPORTBET_ONE_SOURCE=bookmaker_xyz`.
- No independent HTTP data endpoint.
- Reuses `bookmaker_xyz` fetch path and only re-tags output key/title.

## 7) Quick Check Tips

Add `-i` to inspect HTTP status and headers:

```bash
curl -i -sS -G "$PUREBET_API_BASE/events" --data-urlencode "live=false" | head
```
