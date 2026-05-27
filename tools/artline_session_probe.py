from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from providers import artline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely inspect whether this machine has an Artline web session and "
            "optionally run a max-bet preflight. Cookie values are never printed."
        )
    )
    parser.add_argument("--sport", default="", help="Artline sport slug, for example tennis or hockey.")
    parser.add_argument("--game-id", default="", help="Artline game id from provider/book event.")
    parser.add_argument("--selection-id", default="", help="Artline betslip selection/event id.")
    parser.add_argument("--stake", type=float, default=5.0, help="Stake to validate against max-bet.")
    parser.add_argument("--event-url", default="", help="Optional Artline match URL used as referer.")
    parser.add_argument("--live", action="store_true", help="Use Artline live bet payload flag.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    probe = artline.resolve_artline_browser_cookie_header()
    payload: dict = {
        "cookie_source": probe.get("source"),
        "cookie_names": probe.get("cookie_names") or [],
        "errors": probe.get("errors") or [],
        "has_cookie_header": bool(probe.get("cookie_header")),
    }

    if args.sport and args.game_id and args.selection_id:
        os.environ["ARTLINE_AUTO_BROWSER_COOKIES"] = "1"
        preflight = artline.preflight_web_max_bet(
            sport=args.sport,
            game_id=args.game_id,
            selection_id=args.selection_id,
            is_live=args.live,
            stake=args.stake,
            event_url=args.event_url,
        )
        payload["preflight"] = preflight

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
