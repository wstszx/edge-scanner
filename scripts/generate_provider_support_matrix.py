from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from providers import PROVIDER_CAPABILITIES


DEFAULT_OUTPUT_PATH = Path("docs") / "provider_support_matrix.md"


def _sorted_provider_keys() -> list[str]:
    return sorted(PROVIDER_CAPABILITIES)


def _all_supported_sport_keys(provider_keys: Iterable[str]) -> list[str]:
    return sorted(
        {
            sport_key
            for provider_key in provider_keys
            for sport_key in PROVIDER_CAPABILITIES[provider_key].supported_sport_keys
        }
    )


def build_matrix_markdown() -> str:
    provider_keys = _sorted_provider_keys()
    sport_keys = _all_supported_sport_keys(provider_keys)

    lines = [
        "# Custom Provider Support Matrix",
        "",
        "## Scope",
        "",
        "This document summarizes the custom-provider support that is declared in code.",
        "",
        "It is generated from `providers.PROVIDER_CAPABILITIES`, which is the source of truth for declared provider support.",
        "",
        "Use this as a code-level capability reference, not as a live availability report:",
        "",
        "- Declared support means the provider has code-declared sport or market mappings.",
        "- Declared support does not guarantee that the upstream provider currently has data.",
        "- Declared support does not guarantee that every market is available for every supported sport.",
        "- Live availability can still be empty even when a sport is supported in code.",
        "",
        "## Registered Custom Providers",
        "",
        "| Provider Key | Provider Name | Live Mode |",
        "| --- | --- | --- |",
    ]

    for provider_key in provider_keys:
        capability = PROVIDER_CAPABILITIES[provider_key]
        live_mode = "Yes" if capability.live_mode_supported else "No"
        lines.append(f"| `{provider_key}` | {capability.title} | {live_mode} |")

    lines.extend(
        [
            "",
            "## Market Support Matrix",
            "",
            "| Provider | Declared Market Support |",
            "| --- | --- |",
        ]
    )

    for provider_key in provider_keys:
        capability = PROVIDER_CAPABILITIES[provider_key]
        markets = ", ".join(f"`{market}`" for market in capability.supported_markets)
        lines.append(f"| `{provider_key}` | {markets} |")

    lines.extend(
        [
            "",
            "## Sport Coverage Summary",
            "",
            "Declared sport-key counts by provider:",
            "",
            "| Provider | Declared Sport Keys |",
            "| --- | ---: |",
        ]
    )

    providers_by_coverage = sorted(
        provider_keys,
        key=lambda provider_key: (
            -len(PROVIDER_CAPABILITIES[provider_key].supported_sport_keys),
            provider_key,
        ),
    )
    for provider_key in providers_by_coverage:
        sport_count = len(PROVIDER_CAPABILITIES[provider_key].supported_sport_keys)
        lines.append(f"| `{provider_key}` | {sport_count} |")

    lines.extend(
        [
            "",
            f"The union across all custom providers currently covers {len(sport_keys)} distinct `sport_key` values.",
            "",
            "## Sport Support Matrix",
            "",
            "`Y` means the provider has code-declared support for the sport key.",
            "",
            "| sport_key | " + " | ".join(provider_keys) + " |",
            "| --- | " + " | ".join("---" for _ in provider_keys) + " |",
        ]
    )

    for sport_key in sport_keys:
        support_cells = []
        for provider_key in provider_keys:
            supported = sport_key in PROVIDER_CAPABILITIES[provider_key].supported_sport_keys
            support_cells.append("Y" if supported else "-")
        lines.append(f"| `{sport_key}` | " + " | ".join(support_cells) + " |")

    lines.extend(
        [
            "",
            "## Practical Notes",
            "",
            "- This matrix is generated from code declarations, so regenerate it when provider capabilities change.",
            "- Use live scans and provider verification reports to judge real-time availability or feed health.",
        ]
    )

    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the provider support matrix from the centralized provider capability registry.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Markdown file to write. Defaults to docs/provider_support_matrix.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_matrix_markdown(), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
