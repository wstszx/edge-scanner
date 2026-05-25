from tools import hunt_real_arbitrage


def test_default_sports_include_supported_dex_provider_long_tail() -> None:
    default_sports = hunt_real_arbitrage.DEFAULT_SPORTS

    assert default_sports[:3] == ["basketball_nba", "basketball_ncaab", "baseball_mlb"]
    assert len(default_sports) == len(set(default_sports))
    for sport_key in [
        "basketball_euroleague",
        "mma_ufc",
        "rugby_union",
        "soccer_england_championship",
        "tennis_atp",
        "tennis_wta",
    ]:
        assert sport_key in default_sports
