# Custom Provider Support Matrix

## Scope

This document summarizes the custom-provider support that is declared in code.

It is generated from `providers.PROVIDER_CAPABILITIES`, which is the source of truth for declared provider support.

Use this as a code-level capability reference, not as a live availability report:

- Declared support means the provider has code-declared sport or market mappings.
- Declared support does not guarantee that the upstream provider currently has data.
- Declared support does not guarantee that every market is available for every supported sport.
- Live availability can still be empty even when a sport is supported in code.

## Registered Custom Providers

| Provider Key | Provider Name | Live Mode |
| --- | --- | --- |
| `artline` | Artline | Yes |
| `betdex` | BetDEX | Yes |
| `bookmaker_xyz` | bookmaker.xyz | Yes |
| `polymarket` | Polymarket | Yes |
| `sx_bet` | SX Bet | Yes |

## Market Support Matrix

| Provider | Declared Market Support |
| --- | --- |
| `artline` | `h2h`, `h2h_3_way`, `spreads`, `team_totals`, `totals` |
| `betdex` | `both_teams_to_score`, `h2h`, `h2h_3_way`, `h2h_h1`, `spreads`, `spreads_h1`, `totals`, `totals_h1` |
| `bookmaker_xyz` | `both_teams_to_score`, `h2h`, `spreads`, `totals` |
| `polymarket` | `both_teams_to_score`, `h2h`, `h2h_3_way`, `spreads`, `totals` |
| `sx_bet` | `both_teams_to_score`, `h2h`, `h2h_3_way`, `h2h_h1`, `spreads`, `spreads_h1`, `totals`, `totals_h1` |

## Sport Coverage Summary

Declared sport-key counts by provider:

| Provider | Declared Sport Keys |
| --- | ---: |
| `bookmaker_xyz` | 38 |
| `betdex` | 24 |
| `artline` | 20 |
| `polymarket` | 12 |
| `sx_bet` | 12 |

The union across all custom providers currently covers 38 distinct `sport_key` values.

## Sport Support Matrix

`Y` means the provider has code-declared support for the sport key.

| sport_key | artline | betdex | bookmaker_xyz | polymarket | sx_bet |
| --- | --- | --- | --- | --- | --- |
| `americanfootball_ncaaf` | - | Y | Y | Y | Y |
| `americanfootball_nfl` | - | Y | Y | Y | Y |
| `baseball_mlb` | Y | Y | Y | Y | Y |
| `baseball_mlb_spring_training` | - | - | Y | - | - |
| `basketball_euroleague` | Y | Y | Y | - | - |
| `basketball_france_pro_a` | Y | - | Y | - | - |
| `basketball_germany_bbl` | Y | Y | Y | - | - |
| `basketball_italy_serie_a` | - | - | Y | - | - |
| `basketball_nba` | Y | Y | Y | Y | Y |
| `basketball_ncaab` | - | Y | Y | Y | Y |
| `basketball_spain_liga_acb` | - | - | Y | - | - |
| `boxing_professional` | - | - | Y | - | - |
| `icehockey_ahl` | Y | - | Y | - | - |
| `icehockey_khl` | - | - | Y | - | - |
| `icehockey_nhl` | Y | Y | Y | Y | Y |
| `mma_ufc` | - | Y | Y | - | - |
| `rugby_league_nrl` | - | - | Y | - | - |
| `rugby_union` | - | Y | Y | - | - |
| `rugby_union_six_nations` | - | - | Y | - | - |
| `soccer_argentina_liga_profesional` | Y | Y | Y | - | - |
| `soccer_brazil_serie_a` | - | Y | Y | - | - |
| `soccer_england_championship` | Y | Y | Y | - | - |
| `soccer_england_league_one` | Y | Y | Y | - | - |
| `soccer_england_league_two` | Y | Y | Y | - | - |
| `soccer_epl` | Y | Y | Y | Y | Y |
| `soccer_france_ligue_one` | Y | Y | Y | Y | Y |
| `soccer_germany_bundesliga` | Y | Y | Y | Y | Y |
| `soccer_italy_serie_a` | Y | Y | Y | Y | Y |
| `soccer_mexico_liga_mx` | Y | - | Y | - | - |
| `soccer_netherlands_eredivisie` | Y | Y | Y | - | - |
| `soccer_portugal_primeira_liga` | Y | - | Y | - | - |
| `soccer_spain_la_liga` | Y | Y | Y | Y | Y |
| `soccer_turkey_super_lig` | - | - | Y | - | - |
| `soccer_usa_mls` | Y | Y | Y | Y | Y |
| `tennis_atp` | - | Y | Y | - | - |
| `tennis_atp_indian_wells` | - | - | Y | - | - |
| `tennis_wta` | - | Y | Y | - | - |
| `tennis_wta_indian_wells` | - | - | Y | - | - |

## Practical Notes

- This matrix is generated from code declarations, so regenerate it when provider capabilities change.
- Use live scans and provider verification reports to judge real-time availability or feed health.
