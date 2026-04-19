from __future__ import annotations

import unittest

from providers import PROVIDER_CAPABILITIES, PROVIDER_FETCHERS, PROVIDER_TITLES


class ProviderCapabilitiesRegistryTests(unittest.TestCase):
    def test_registry_matches_registered_custom_providers(self) -> None:
        self.assertEqual(set(PROVIDER_FETCHERS), set(PROVIDER_CAPABILITIES))
        self.assertEqual(set(PROVIDER_TITLES), set(PROVIDER_CAPABILITIES))

    def test_registry_entries_are_normalized_and_match_registry_keys_and_titles(self) -> None:
        for provider_key, capability in PROVIDER_CAPABILITIES.items():
            self.assertEqual(provider_key, capability.key)
            self.assertEqual(PROVIDER_TITLES[provider_key], capability.title)
            self.assertEqual(
                tuple(sorted(set(capability.supported_sport_keys))),
                capability.supported_sport_keys,
            )
            self.assertEqual(
                tuple(sorted(set(capability.supported_markets))),
                capability.supported_markets,
            )
            self.assertTrue(capability.supported_sport_keys)
            self.assertTrue(capability.supported_markets)
            self.assertIsInstance(capability.live_mode_supported, bool)

    def test_registry_preserves_shared_and_distinct_market_capabilities(self) -> None:
        for capability in PROVIDER_CAPABILITIES.values():
            self.assertIn("h2h", capability.supported_markets)
            self.assertIn("spreads", capability.supported_markets)
            self.assertIn("totals", capability.supported_markets)

        first_half_support = {
            provider_key
            for provider_key, capability in PROVIDER_CAPABILITIES.items()
            if {"h2h_h1", "spreads_h1", "totals_h1"}.issubset(capability.supported_markets)
        }
        self.assertEqual({"betdex", "sx_bet"}, first_half_support)

        both_teams_to_score_support = {
            provider_key
            for provider_key, capability in PROVIDER_CAPABILITIES.items()
            if "both_teams_to_score" in capability.supported_markets
        }
        self.assertEqual(
            {"betdex", "bookmaker_xyz", "polymarket", "sx_bet"},
            both_teams_to_score_support,
        )


if __name__ == "__main__":
    unittest.main()
