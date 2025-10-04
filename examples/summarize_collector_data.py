#!/usr/bin/env python3
"""Summarise raw collector output files.

The collector stores one gzip-compressed file per symbol and day. Each line
contains the nanosecond receive timestamp followed by the JSON payload emitted
by the exchange websocket. This script walks one or more directories (or a list
of files), parses the messages, and prints high-level statistics that help you
verify what was captured.

Example
-------

.. code-block:: bash

    python examples/summarize_collector_data.py data/binance_cm

The script understands files produced by any connector that follows the same
"timestamp json" line format (Binance spot, Binance futures CM/UM, Hyperliquid,
etc.).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise gzip-compressed collector output files",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Directories or files produced by the collector",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after reading N messages (useful for quick previews)",
    )
    return parser.parse_args()


def iter_data_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield all gzip files found under the provided paths."""

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            for gz_path in sorted(path.rglob("*.gz")):
                if gz_path.is_file():
                    yield gz_path
        else:
            if path.suffix != ".gz":
                raise ValueError(f"Expected a .gz file, got {path}")
            yield path


def parse_line(line: str) -> tuple[int, Dict[str, object]]:
    """Split a collector line into timestamp and JSON payload."""

    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    try:
        ts_str, json_part = line.split(" ", 1)
    except ValueError as exc:  # not enough values to unpack
        raise ValueError(f"Line missing separator: {line!r}") from exc

    try:
        timestamp_ns = int(ts_str)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp {ts_str!r}") from exc

    try:
        payload = json.loads(json_part)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")

    return timestamp_ns, payload


def classify_message(payload: Dict[str, object]) -> tuple[Optional[str], str, Dict[str, object]]:
    """Extract symbol, message kind, and the nested data object."""

    stream = payload.get("stream")
    data = payload.get("data")

    if not isinstance(data, dict):
        data = {}

    symbol: Optional[str] = None
    message_type = "unknown"

    if isinstance(stream, str):
        if "@" in stream:
            symbol, message_type = stream.split("@", 1)
        else:
            symbol = stream
    
    if symbol is None:
        raw_symbol = data.get("s")
        if isinstance(raw_symbol, str):
            symbol = raw_symbol.lower()

    if symbol is None:
        raw_symbol = payload.get("symbol")
        if isinstance(raw_symbol, str):
            symbol = raw_symbol.lower()

    event_type = data.get("e")
    if isinstance(event_type, str):
        message_type = event_type

    if message_type == "unknown" and "lastUpdateId" in payload:
        message_type = "snapshot"

    if symbol is None:
        if "result" in payload or payload.get("id") is not None:
            symbol = "__meta__"
            message_type = "control"
        else:
            symbol = "__unknown__"

    return symbol, message_type, data


def to_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass
class SymbolStats:
    first_ns: Optional[int] = None
    last_ns: Optional[int] = None
    type_counts: Counter[str] = field(default_factory=Counter)
    trade_count: int = 0
    trade_qty: float = 0.0
    trade_quote: float = 0.0

    def observe(self, timestamp_ns: int, message_type: str, data: Dict[str, object]) -> None:
        self.type_counts[message_type] += 1
        if self.first_ns is None or timestamp_ns < self.first_ns:
            self.first_ns = timestamp_ns
        if self.last_ns is None or timestamp_ns > self.last_ns:
            self.last_ns = timestamp_ns

        if message_type in {"trade", "aggTrade"} or data.get("e") == "trade":
            qty = to_float(data.get("q") or data.get("quantity"))
            price = to_float(data.get("p") or data.get("price"))
            if qty is not None and price is not None:
                self.trade_count += 1
                self.trade_qty += qty
                self.trade_quote += qty * price


def format_timestamp(ts_ns: Optional[int]) -> str:
    if ts_ns is None:
        return "-"
    dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)
    return dt.isoformat()


def main() -> None:
    args = parse_args()
    stats: Dict[str, SymbolStats] = defaultdict(SymbolStats)

    total_messages = 0
    for path in iter_data_files(args.paths):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    try:
                        timestamp_ns, payload = parse_line(line)
                        symbol, message_type, data = classify_message(payload)
                    except ValueError as exc:
                        print(f"{path}:{line_no}: {exc}", file=sys.stderr)
                        continue

                    if symbol is None:
                        print(
                            f"{path}:{line_no}: unable to determine symbol, skipping",
                            file=sys.stderr,
                        )
                        continue

                    stats[symbol].observe(timestamp_ns, message_type, data)
                    total_messages += 1

                    if args.limit is not None and total_messages >= args.limit:
                        break

            if args.limit is not None and total_messages >= args.limit:
                break
        except OSError as exc:
            print(f"Failed to read {path}: {exc}", file=sys.stderr)

    if not stats:
        print("No messages processed. Check the provided paths.")
        return

    print("Summary by symbol")
    print("=" * 80)

    overall_trades = 0
    overall_qty = 0.0
    overall_quote = 0.0

    for symbol in sorted(stats):
        sym_stats = stats[symbol]
        total_per_symbol = sum(sym_stats.type_counts.values())
        type_summary = ", ".join(
            f"{kind}={count}" for kind, count in sym_stats.type_counts.most_common()
        )

        print(f"Symbol: {symbol}")
        print(f"  Messages    : {total_per_symbol}")
        print(f"  First (UTC) : {format_timestamp(sym_stats.first_ns)}")
        print(f"  Last  (UTC) : {format_timestamp(sym_stats.last_ns)}")
        print(f"  Trade count : {sym_stats.trade_count}")
        print(f"  Trade qty   : {sym_stats.trade_qty:.8f}")
        print(f"  Quote volume: {sym_stats.trade_quote:.2f}")
        if type_summary:
            print(f"  By type     : {type_summary}")
        print()

        overall_trades += sym_stats.trade_count
        overall_qty += sym_stats.trade_qty
        overall_quote += sym_stats.trade_quote

    print("Overall")
    print("-" * 80)
    print(f"Messages processed : {total_messages}")
    print(f"Trade messages     : {overall_trades}")
    print(f"Total trade qty    : {overall_qty:.8f}")
    print(f"Total quote volume : {overall_quote:.2f}")


if __name__ == "__main__":
    main()

