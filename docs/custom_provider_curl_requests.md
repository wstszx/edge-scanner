# Custom Provider Data API cURL Checklist

This document lists all data-fetch interfaces used by custom providers in `providers/`.
It is intended for endpoint health checks and manual API validation.

Registered custom providers:
- `betdex`
- `bookmaker_xyz`
- `sx_bet`
- `polymarket`

## 0) Common Variables

```bash
# BetDEX
BETDEX_API_BASE="https://production.api.monacoprotocol.xyz"
BETDEX_APP_ID="REPLACE_APP_ID"
BETDEX_API_KEY="REPLACE_API_KEY"
BETDEX_SESSION_URL="https://www.betdex.com/api/session"  # legacy website fallback, may be bot-protected
BETDEX_UA="Mozilla/5.0"
BETDEX_SUBCATEGORY="FOOTBALL"
BETDEX_EVENT_ID_1="REPLACE_EVENT_ID_1"
BETDEX_EVENT_ID_2="REPLACE_EVENT_ID_2"
BETDEX_MARKET_ID_1="REPLACE_MARKET_ID_1"
BETDEX_MARKET_ID_2="REPLACE_MARKET_ID_2"

# bookmaker.xyz
BOOKMAKER_HOME="https://bookmaker.xyz/"
BOOKMAKER_PUBLIC_BASE="https://bookmaker.xyz"
BOOKMAKER_MARKET_MANAGER_BASE="https://api.onchainfeed.org/api/v1/public/market-manager"
BOOKMAKER_UA="Mozilla/5.0"
BOOKMAKER_ENVIRONMENT="PolygonUSDT"  # PolygonUSDT | GnosisXDAI | BaseWETH
BOOKMAKER_SPORT_SLUG="basketball"
BOOKMAKER_LEAGUE_SLUG="nba"
BOOKMAKER_GAME_ID="REPLACE_GAME_ID"

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

## 1) BetDEX (`providers/betdex.py`)

### 1.1 Official session token `POST /sessions` (recommended)

```bash
curl -sS "$BETDEX_API_BASE/sessions" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -H "User-Agent: $BETDEX_UA" \
  --data "{\"appId\":\"$BETDEX_APP_ID\",\"apiKey\":\"$BETDEX_API_KEY\"}"
```

Extract bearer token (requires `jq`):

```bash
BETDEX_TOKEN="$(curl -sS "$BETDEX_API_BASE/sessions" \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  -H "User-Agent: $BETDEX_UA" \
  --data "{\"appId\":\"$BETDEX_APP_ID\",\"apiKey\":\"$BETDEX_API_KEY\"}" \
  | jq -r '.sessions[0].accessToken')"
echo "$BETDEX_TOKEN"
```

### 1.2 Legacy public website session `GET /api/session` (best-effort)

```bash
curl -i -sS "$BETDEX_SESSION_URL" \
  -H "Accept: application/json" \
  -H "User-Agent: $BETDEX_UA"
```

As of `2026-03-16`, this endpoint may return `429 Too Many Requests` with `x-vercel-mitigated: challenge`, which indicates front-end bot protection rather than a documented Monaco API limit.

### 1.3 Events `/events`

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

### 1.4 Markets `/markets` (repeated `eventIds` and `statuses`)

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

### 1.5 Market prices `/market-prices` (repeated `marketIds`)

```bash
curl -sS -G "$BETDEX_API_BASE/market-prices" \
  -H "Accept: application/json" \
  -H "User-Agent: $BETDEX_UA" \
  -H "Authorization: Bearer $BETDEX_TOKEN" \
  --data-urlencode "marketIds=$BETDEX_MARKET_ID_1" \
  --data-urlencode "marketIds=$BETDEX_MARKET_ID_2"
```

## 2) bookmaker.xyz (`providers/bookmaker_xyz.py`)

### 2.1 Home page (used to discover const asset path)

```bash
curl -sS "$BOOKMAKER_HOME" \
  -H "User-Agent: $BOOKMAKER_UA"
```

### 2.2 Const asset (`/assets/const-*.js`)

```bash
CONST_ASSET_PATH="/assets/const-REPLACE_HASH.js"
curl -sS "${BOOKMAKER_PUBLIC_BASE}${CONST_ASSET_PATH}" \
  -H "User-Agent: $BOOKMAKER_UA"
```

### 2.3 Official market-manager `/games-by-filters`

```bash
curl -sS -G "$BOOKMAKER_MARKET_MANAGER_BASE/games-by-filters" \
  -H "Accept: application/json" \
  -H "User-Agent: $BOOKMAKER_UA" \
  --data-urlencode "environment=$BOOKMAKER_ENVIRONMENT" \
  --data-urlencode "gameState=Prematch" \
  --data-urlencode "sportSlug=$BOOKMAKER_SPORT_SLUG" \
  --data-urlencode "leagueSlug=$BOOKMAKER_LEAGUE_SLUG" \
  --data-urlencode "conditionState=Active" \
  --data-urlencode "orderBy=startsAt" \
  --data-urlencode "orderDirection=asc" \
  --data-urlencode "page=1" \
  --data-urlencode "perPage=100"
```

### 2.4 Official market-manager `/conditions-by-game-ids`

```bash
curl -sS "$BOOKMAKER_MARKET_MANAGER_BASE/conditions-by-game-ids" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "User-Agent: $BOOKMAKER_UA" \
  --data-raw @- <<JSON
{
  "environment": "$BOOKMAKER_ENVIRONMENT",
  "gameIds": ["$BOOKMAKER_GAME_ID"]
}
JSON
```

## 3) SX Bet (`providers/sx_bet.py`)

Optional auth headers:
- `Authorization: Bearer $SX_BEARER`
- `X-API-Key: $SX_API_KEY`
- `Cookie: $SX_COOKIE`

### 3.1 Upcoming fixtures `/summary/upcoming/{baseToken}/{sportId}`

```bash
curl -sS "$SX_BASE/summary/upcoming/$SX_BASE_TOKEN/$SX_SPORT_ID" \
  -H "User-Agent: $SX_UA"
```

### 3.2 Best odds `/orders/odds/best`

```bash
curl -sS -G "$SX_BASE/orders/odds/best" \
  -H "User-Agent: $SX_UA" \
  -H "Authorization: Bearer $SX_BEARER" \
  -H "X-API-Key: $SX_API_KEY" \
  -H "Cookie: $SX_COOKIE" \
  --data-urlencode "marketHashes=$SX_MARKET_HASHES" \
  --data-urlencode "baseToken=$SX_BASE_TOKEN"
```

### 3.3 Orders `/orders`

```bash
curl -sS -G "$SX_BASE/orders" \
  -H "User-Agent: $SX_UA" \
  -H "Authorization: Bearer $SX_BEARER" \
  -H "X-API-Key: $SX_API_KEY" \
  -H "Cookie: $SX_COOKIE" \
  --data-urlencode "marketHashes=$SX_MARKET_HASHES" \
  --data-urlencode "baseToken=$SX_BASE_TOKEN"
```

## 4) Polymarket (`providers/polymarket.py`)

Optional auth headers:
- `Authorization: Bearer $POLY_BEARER`
- `X-API-Key: $POLY_API_KEY`
- `Cookie: $POLY_COOKIE`

### 4.1 Sports tags `/sports`

```bash
curl -sS "$POLY_API_BASE/sports" \
  -H "User-Agent: $POLY_UA" \
  -H "Authorization: Bearer $POLY_BEARER" \
  -H "X-API-Key: $POLY_API_KEY" \
  -H "Cookie: $POLY_COOKIE"
```

### 4.2 Active events `/events`

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

### 4.3 CLOB order book `/book?token_id=<token_id>`

```bash
curl -sS -G "$POLY_CLOB_BASE/book" \
  -H "User-Agent: $POLY_UA" \
  -H "Authorization: Bearer $POLY_BEARER" \
  -H "X-API-Key: $POLY_API_KEY" \
  -H "Cookie: $POLY_COOKIE" \
  --data-urlencode "token_id=$POLY_TOKEN_ID"
```

## 5) Quick Check Tips

Add `-i` to inspect HTTP status and headers:

```bash
curl -i -sS -G "$POLY_API_BASE/events" \
  --data-urlencode "tag_id=$POLY_TAG_ID" \
  --data-urlencode "active=true" \
  --data-urlencode "closed=false" | head
```
