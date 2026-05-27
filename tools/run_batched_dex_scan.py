from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import hunt_dex_opportunities


DEFAULT_SPORTS = [
    "basketball_nba",
    "baseball_mlb",
    "icehockey_nhl",
    "soccer_epl",
    "tennis_atp",
    "tennis_wta",
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    "basketball_ncaab",
]


def build_hunt_args(args: argparse.Namespace, *, stamp: str) -> list[str]:
    output_dir = Path(args.out_dir)
    prefix = str(args.prefix or "dex_real_opportunity").strip() or "dex_real_opportunity"
    report_stem = f"{prefix}_{stamp}"
    out_path = output_dir / f"{report_stem}.json"
    summary_path = output_dir / f"{report_stem}.md"
    batch_dir = output_dir / report_stem
    hunt_args = [
        "--sports",
        *[str(item) for item in args.sports],
        "--provider-sets",
        str(args.provider_sets),
        "--stake",
        str(args.stake),
        "--min-roi",
        str(args.min_roi),
        "--allow-quality",
        *[str(item) for item in args.allow_quality],
        "--max-quote-skew-seconds",
        str(args.max_quote_skew_seconds),
        "--min-executable-stake",
        str(args.min_executable_stake),
        "--per-scan-timeout-seconds",
        str(args.per_scan_timeout_seconds),
        "--batch-out-dir",
        str(batch_dir),
        "--out",
        str(out_path),
        "--summary-out",
        str(summary_path),
    ]
    if args.include_all_markets:
        hunt_args.append("--include-all-markets")
    if args.all_markets_sports:
        hunt_args.extend(["--all-markets-sports", *[str(item) for item in args.all_markets_sports]])
    if args.require_explicit_liquidity:
        hunt_args.append("--require-explicit-liquidity")
    return hunt_args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard batched DEX opportunity scan.")
    parser.add_argument("--sports", nargs="*", default=DEFAULT_SPORTS)
    parser.add_argument("--provider-sets", default="all,pairs")
    parser.add_argument("--include-all-markets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--all-markets-sports", nargs="*", default=[])
    parser.add_argument("--require-explicit-liquidity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--min-roi", type=float, default=0.01)
    parser.add_argument("--allow-quality", nargs="*", default=["high"])
    parser.add_argument("--max-quote-skew-seconds", type=int, default=120)
    parser.add_argument("--min-executable-stake", type=float, default=25.0)
    parser.add_argument("--per-scan-timeout-seconds", type=float, default=35.0)
    parser.add_argument("--out-dir", default=str(Path("data") / "provider_verification"))
    parser.add_argument("--prefix", default="dex_real_opportunity")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hunt_args = build_hunt_args(args, stamp=stamp)
    return hunt_dex_opportunities.main(hunt_args)


if __name__ == "__main__":
    raise SystemExit(main())
