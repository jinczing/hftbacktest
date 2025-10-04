"""Visualize Binance coin-margined futures depth data as a price-level order book.

This example demonstrates how to load raw data collected with the
``collector`` tool (saved as ``.gz`` files under ``./data/binance_cm``) and
use the :mod:`hftbacktest` Python package to convert the stream into events,
reconstruct the limit order book, and plot the top price levels.

Usage
-----
The script defaults to the ``./data/binance_cm`` location.  To run the
example with the sample files under ``examples/cm`` you can execute::

    python examples/binance_cm_orderbook.py --glob "examples/cm/btcusd_perp_20240808.gz"

By default, the script converts the data, rebuilds the order book until the
1000th depth event, prints the top ten price levels, and shows a horizontal
bar plot of bid and ask liquidity.  You can change the number of processed
events, pick a specific exchange timestamp, or save the resulting plot to a
file via command-line options.
"""

from __future__ import annotations

import argparse
import glob
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the local hftbacktest package importable without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
PY_PKG = REPO_ROOT / "py-hftbacktest"
if PY_PKG.exists() and str(PY_PKG) not in sys.path:
    sys.path.insert(0, str(PY_PKG))


def _ensure_module(name: str, path: Path | None = None, package_paths: list[str] | None = None) -> types.ModuleType:
    """Load a module without executing package ``__init__`` files."""

    if name in sys.modules:
        return sys.modules[name]

    module = types.ModuleType(name)
    if package_paths is not None:
        module.__path__ = package_paths
    sys.modules[name] = module

    if path is not None:
        spec = spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {name} from {path}")
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


if PY_PKG.exists():
    _ensure_module("hftbacktest", package_paths=[str(PY_PKG / "hftbacktest")])
    _ensure_module("hftbacktest.types", PY_PKG / "hftbacktest" / "types.py")
    _ensure_module("hftbacktest.data", package_paths=[str(PY_PKG / "hftbacktest" / "data")])
    _ensure_module("hftbacktest.data.validation", PY_PKG / "hftbacktest" / "data" / "validation.py")
    _ensure_module("hftbacktest.data.utils", package_paths=[str(PY_PKG / "hftbacktest" / "data" / "utils")])
    _ensure_module(
        "hftbacktest.data.utils.binancefutures",
        PY_PKG / "hftbacktest" / "data" / "utils" / "binancefutures.py",
    )

from hftbacktest.data.utils.binancefutures import convert
from hftbacktest.types import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    SELL_EVENT,
)

# Only the lowest eight bits are used for the base event identifiers.
EVENT_KIND_MASK = 0xFF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        dest="patterns",
        nargs="*",
        default=["./data/binance_cm/*.gz"],
        help="One or more glob patterns pointing to raw Binance CM gzip files.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="Number of price levels per side to display.",
    )
    parser.add_argument(
        "--event-index",
        type=int,
        default=1000,
        help=(
            "Depth event index (0-based) to stop the reconstruction at.  "
            "Ignored when --timestamp is provided."
        ),
    )
    parser.add_argument(
        "--timestamp",
        type=int,
        help=(
            "Exchange timestamp (nanoseconds) at which to snapshot the book. "
            "The book is reconstructed using all events with timestamps up to "
            "and including this value."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated plot instead of opening a window.",
    )
    return parser.parse_args()


def resolve_files(patterns: Iterable[str]) -> List[Path]:
    """Expand glob patterns into an ordered list of files."""

    files: List[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if glob.has_magic(pattern) else [Path(pattern)]
        for match in matches:
            if match.is_file():
                files.append(match)
    return files


def load_events(files: Iterable[Path]) -> np.ndarray:
    """Convert raw Binance Futures stream dumps into the backtest event format."""

    arrays: List[np.ndarray] = []
    for filename in files:
        print(f"Converting {filename}")
        arrays.append(convert(str(filename)))
    if not arrays:
        raise FileNotFoundError("No input files were found. Check the --glob argument.")
    return np.concatenate(arrays)


def infer_tick_size(events: np.ndarray) -> float:
    """Infer the minimum price increment from the dataset."""

    kind = events["ev"] & EVENT_KIND_MASK
    mask = np.isin(kind, (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT))
    prices = np.unique(events["px"][mask])
    if prices.size <= 1:
        return 0.0
    diffs = np.diff(np.sort(prices))
    positive_diffs = diffs[diffs > 0]
    return float(positive_diffs.min()) if positive_diffs.size else 0.0


def snapshot_timestamp(events: np.ndarray, depth_index: int) -> int:
    """Return the exchange timestamp at the requested depth event index."""

    kind = events["ev"] & EVENT_KIND_MASK
    depth_positions = np.flatnonzero(np.isin(kind, (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT, DEPTH_CLEAR_EVENT)))
    if depth_positions.size == 0:
        raise ValueError("No depth-related events were found in the converted data.")
    idx = min(depth_index, depth_positions.size - 1)
    return int(events["exch_ts"][depth_positions[idx]])


def build_order_book(events: np.ndarray, upto_timestamp: int) -> Tuple[Dict[float, float], Dict[float, float]]:
    """Reconstruct the bid and ask books up to ``upto_timestamp`` (inclusive)."""

    bids: Dict[float, float] = {}
    asks: Dict[float, float] = {}
    for ev, exch_ts, price, qty in zip(events["ev"], events["exch_ts"], events["px"], events["qty"]):
        if exch_ts > upto_timestamp:
            break
        side_flag = ev & (BUY_EVENT | SELL_EVENT)
        book = bids if side_flag & BUY_EVENT else asks if side_flag & SELL_EVENT else None
        if book is None:
            continue

        event_kind = ev & EVENT_KIND_MASK
        if event_kind == DEPTH_CLEAR_EVENT:
            book.clear()
        elif event_kind in (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT):
            if qty <= 0:
                book.pop(price, None)
            else:
                book[price] = qty

    return bids, asks


def book_to_dataframe(
    bids: Dict[float, float], asks: Dict[float, float], levels: int
) -> pd.DataFrame:
    """Convert the order-book dictionaries into a price-aligned dataframe."""

    bid_levels = sorted(bids.items(), key=lambda item: item[0], reverse=True)[:levels]
    ask_levels = sorted(asks.items(), key=lambda item: item[0])[:levels]

    bid_df = pd.DataFrame(bid_levels, columns=["price", "bid_qty"]).set_index("price")
    ask_df = pd.DataFrame(ask_levels, columns=["price", "ask_qty"]).set_index("price")
    combined = bid_df.join(ask_df, how="outer").fillna(0.0)
    return combined.sort_index()


def plot_order_book(df: pd.DataFrame, upto_timestamp: int, tick_size: float, output_path: Path | None) -> None:
    """Render a horizontal bar chart with bid and ask depth."""

    if df.empty:
        print("No depth levels found at the requested snapshot.")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))

    prices = df.index.to_numpy()
    bid_qty = -df["bid_qty"].to_numpy()
    ask_qty = df["ask_qty"].to_numpy()

    ax.barh(prices, bid_qty, color="#2a9d8f", label="Bids")
    ax.barh(prices, ask_qty, color="#e76f51", label="Asks")
    ax.axvline(0.0, color="black", linewidth=0.8)

    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    ax.set_title(
        "Binance CM order book\n"
        f"Snapshot at {upto_timestamp} ns (tick size ≈ {tick_size or 'unknown'})"
    )
    ax.legend()
    ax.invert_yaxis()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    files = resolve_files(args.patterns)
    if not files:
        raise FileNotFoundError("No files matched the provided patterns.")

    events = load_events(files)
    tick_size = infer_tick_size(events)

    upto_timestamp = args.timestamp if args.timestamp is not None else snapshot_timestamp(events, args.event_index)
    bids, asks = build_order_book(events, upto_timestamp)

    df = book_to_dataframe(bids, asks, args.levels)
    if df.empty:
        print("The reconstructed book is empty at the requested snapshot.")
        return

    print(f"Snapshot exchange timestamp: {upto_timestamp}")
    print(df.sort_index(ascending=False))

    plot_order_book(df, upto_timestamp, tick_size, args.output)


if __name__ == "__main__":
    main()
