from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

SMALL_ROI_REPAIR_MAX_PERCENT = 1.0


def _compute_roi_percent_from_books(books: list[dict[str, Any]]) -> float | None:
    prices: list[float] = []
    for book in books:
        try:
            price = float(book.get("price") or 0)
        except (TypeError, ValueError):
            return None
        if price <= 1:
            return None
        prices.append(price)
    if len(prices) < 2:
        return None
    inverse_sum = sum(1.0 / price for price in prices)
    if inverse_sum <= 0:
        return None
    return round(((1.0 / inverse_sum) - 1.0) * 100.0, 2)


def _should_repair_roi(current_numeric: float | None, repaired_roi: float) -> bool:
    if current_numeric is None:
        return True
    if current_numeric == repaired_roi:
        return False
    return min(abs(current_numeric), abs(repaired_roi)) <= SMALL_ROI_REPAIR_MAX_PERCENT


def repair_history_roi_file(history_path: str | Path, *, backup: bool = True) -> dict[str, Any]:
    path = Path(history_path)
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    repaired_lines: list[str] = []
    scanned = 0
    updated = 0
    skipped = 0

    for raw_line in lines:
        if not raw_line.strip():
            continue
        scanned += 1
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            repaired_lines.append(raw_line)
            skipped += 1
            continue

        books = record.get("books")
        if not isinstance(books, list):
            repaired_lines.append(json.dumps(record, ensure_ascii=False))
            skipped += 1
            continue

        repaired_roi = _compute_roi_percent_from_books(books)
        if repaired_roi is None:
            repaired_lines.append(json.dumps(record, ensure_ascii=False))
            skipped += 1
            continue

        current_roi = record.get("roi_percent")
        try:
            current_numeric = round(float(current_roi), 2)
        except (TypeError, ValueError):
            current_numeric = None

        if _should_repair_roi(current_numeric, repaired_roi):
            record["roi_percent"] = repaired_roi
            updated += 1
        repaired_lines.append(json.dumps(record, ensure_ascii=False))

    if backup:
        backup_path = path.with_name(path.name + ".bak")
        shutil.copyfile(path, backup_path)

    path.write_text("\n".join(repaired_lines) + "\n", encoding="utf-8", newline="\n")
    return {
        "path": str(path),
        "scanned": scanned,
        "updated": updated,
        "skipped": skipped,
        "backup_created": backup,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair stored arbitrage history ROI values from book prices.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path("data") / "history" / "arbitrage_history.jsonl"),
        help="Path to arbitrage_history.jsonl",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak backup file before modifying the history file.",
    )
    args = parser.parse_args()

    result = repair_history_roi_file(args.path, backup=not args.no_backup)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
