from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from providers import PROVIDER_CAPABILITIES
from scripts.generate_provider_support_matrix import (
    build_matrix_markdown,
    main,
)


class ProviderSupportMatrixGeneratorTests(unittest.TestCase):
    def test_build_matrix_markdown_uses_registry_with_stable_ordering(self) -> None:
        markdown = build_matrix_markdown()

        self.assertIn("# Custom Provider Support Matrix", markdown)
        self.assertIn(
            "This document summarizes the custom-provider support that is declared in code.",
            markdown,
        )
        self.assertIn(
            "Declared support does not guarantee that the upstream provider currently has data.",
            markdown,
        )

        expected_provider_order = sorted(PROVIDER_CAPABILITIES)
        provider_header = (
            "| sport_key | " + " | ".join(expected_provider_order) + " |"
        )
        self.assertIn(provider_header, markdown)

        sport_keys = sorted(
            {
                sport_key
                for capability in PROVIDER_CAPABILITIES.values()
                for sport_key in capability.supported_sport_keys
            }
        )
        basketball_nba_index = markdown.index("| `basketball_nba` |")
        basketball_ncaab_index = markdown.index("| `basketball_ncaab` |")
        self.assertLess(basketball_nba_index, basketball_ncaab_index)
        self.assertIn(f"The union across all custom providers currently covers {len(sport_keys)} distinct `sport_key` values.", markdown)

        bookmaker_xyz_line = next(
            line
            for line in markdown.splitlines()
            if line.startswith("| `bookmaker_xyz` |") and "`both_teams_to_score`" in line
        )
        self.assertIn("`both_teams_to_score`", bookmaker_xyz_line)
        self.assertIn("`h2h`", bookmaker_xyz_line)
        self.assertIn("`spreads`", bookmaker_xyz_line)
        self.assertIn("`totals`", bookmaker_xyz_line)

    def test_main_writes_generated_markdown_to_requested_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "provider_support_matrix.md"

            exit_code = main(["--output", str(output_path)])

            self.assertEqual(0, exit_code)
            self.assertTrue(output_path.exists())
            written = output_path.read_text(encoding="utf-8")
            self.assertEqual(build_matrix_markdown(), written)


if __name__ == "__main__":
    unittest.main()
