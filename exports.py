"""
exports.py — File export functions for trade instructions and aggregated trades.

Public API:
    export_trades_csv(trades, file_path, *, include_metadata=True) -> Path
    export_trades_json(trades, file_path, *, label=None, pretty=True) -> Path
    export_aggregated_csv(result, file_path) -> Path
    export_aggregated_json(result, file_path, *, pretty=True) -> Path
    export_full_package(trades, output_dir, *, label=None) -> dict[str, Path]

File inventory:
    ┌──────────────────────────────────┬────────────────────────────────────────┐
    │ Function                         │ Output                                 │
    ├──────────────────────────────────┼────────────────────────────────────────┤
    │ export_trades_csv                │ HIN-level trades, one row per trade    │
    │ export_trades_json               │ Same data as structured JSON           │
    │ export_aggregated_csv            │ Per-ticker bulk orders, one row/ticker │
    │ export_aggregated_json           │ Full AggregationResult as JSON         │
    │ export_full_package              │ All four files in one call             │
    └──────────────────────────────────┴────────────────────────────────────────┘

Column conventions (HIN-level CSV):
    account_id        — HIN or account identifier
    ticker            — ASX ticker code (uppercase)
    action            — BUY or SELL
    quantity          — Units to trade (positive, direction from action)
    estimated_value   — Indicative AUD value (2dp)
    model_id          — Model used for rebalance (if available)
    model_version     — Model version (if available)
    export_timestamp  — ISO 8601 UTC timestamp of file creation

Column conventions (aggregated CSV):
    ticker            — ASX ticker code
    net_action        — BUY / SELL / FLAT
    total_buy_qty     — Total buy units across all accounts
    total_sell_qty    — Total sell units across all accounts
    net_qty           — total_buy - total_sell (signed)
    total_buy_value   — Total AUD value of all buys (2dp)
    total_sell_value  — Total AUD value of all sells (2dp)
    net_value         — total_buy_value - total_sell_value (signed, 2dp)
    account_count     — Distinct accounts involved
    accounts          — Pipe-delimited HIN list for lineage

Downstream compatibility:
    - CSV files use UTF-8 encoding with BOM (utf-8-sig) for Excel compatibility
    - Dates are ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)
    - No trailing whitespace; all strings unquoted unless they contain commas
    - Numeric precision: quantities to 4dp, values to 2dp
    - JSON uses 2-space indentation and sorted keys for diffable audit files
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from models import TradeInstruction
from aggregation import AggregationResult, aggregate_trades

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions — frozen to prevent accidental reordering
# ---------------------------------------------------------------------------

#: HIN-level trade CSV columns in export order.
HIN_CSV_COLUMNS: tuple[str, ...] = (
    "account_id",
    "ticker",
    "action",
    "quantity",
    "estimated_value",
    "model_id",
    "model_version",
    "export_timestamp",
)

#: Aggregated bulk order CSV columns in export order.
AGG_CSV_COLUMNS: tuple[str, ...] = (
    "ticker",
    "net_action",
    "total_buy_qty",
    "total_sell_qty",
    "net_qty",
    "total_buy_value",
    "total_sell_value",
    "net_value",
    "account_count",
    "accounts",
)


# ---------------------------------------------------------------------------
# HIN-level trade exports
# ---------------------------------------------------------------------------

def export_trades_csv(
    trades: list[TradeInstruction],
    file_path: Union[str, Path],
    *,
    include_metadata: bool = True,
) -> Path:
    """
    Export a list of TradeInstruction objects to a CSV file.

    One row per trade. Sorted: SELLs first within each account, then BUYs,
    then by account_id — the standard ordering for DTR submission.

    Args:
        trades:           List of TradeInstruction objects to export.
        file_path:        Destination path. Parent directories are created
                          if they do not exist.
        include_metadata: If True, writes a two-line comment header with
                          export metadata before the column header row.
                          Set False for platforms that reject comment rows.

    Returns:
        Resolved absolute Path to the written file.

    Raises:
        ValueError:  If trades is not a list.
        OSError:     If the file cannot be written (permissions, disk full).
    """
    file_path = _resolve_path(file_path)
    timestamp = _utc_now()

    sorted_trades = _sort_hin_trades(trades)

    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        if include_metadata:
            writer.writerow([f"# Rebalancing Engine — HIN Trade Instructions"])
            writer.writerow([f"# Generated: {timestamp} | Records: {len(trades)}"])

        writer.writerow(HIN_CSV_COLUMNS)

        for t in sorted_trades:
            writer.writerow([
                t.account_id,
                t.ticker,
                t.action,
                _fmt_qty(t.quantity),
                _fmt_value(t.estimated_value),
                t.model_id or "",
                t.model_version or "",
                timestamp,
            ])

    logger.info("Exported %d HIN trades → %s", len(trades), file_path)
    return file_path


def export_trades_json(
    trades: list[TradeInstruction],
    file_path: Union[str, Path],
    *,
    label: Optional[str] = None,
    pretty: bool = True,
) -> Path:
    """
    Export a list of TradeInstruction objects to a JSON file.

    Produces a structured envelope with metadata, a summary block, and
    the full trade list. Designed for API ingestion and audit archiving.

    Args:
        trades:     List of TradeInstruction objects.
        file_path:  Destination path.
        label:      Optional descriptive label for the export batch.
        pretty:     If True, write indented JSON (2 spaces). Set False
                    for compact single-line output (e.g. log streaming).

    Returns:
        Resolved absolute Path to the written file.
    """
    file_path = _resolve_path(file_path)
    timestamp = _utc_now()

    accounts = sorted({t.account_id for t in trades})
    tickers = sorted({t.ticker for t in trades})
    buy_total = sum(t.estimated_value for t in trades if t.action == "BUY")
    sell_total = sum(t.estimated_value for t in trades if t.action == "SELL")

    payload = {
        "metadata": {
            "label": label,
            "export_timestamp": timestamp,
            "record_count": len(trades),
            "account_count": len(accounts),
            "ticker_count": len(tickers),
        },
        "summary": {
            "accounts": accounts,
            "tickers": tickers,
            "total_buy_value": round(buy_total, 2),
            "total_sell_value": round(sell_total, 2),
            "net_cash_flow": round(sell_total - buy_total, 2),
            "buy_count": sum(1 for t in trades if t.action == "BUY"),
            "sell_count": sum(1 for t in trades if t.action == "SELL"),
        },
        "trades": [
            {
                "account_id": t.account_id,
                "ticker": t.ticker,
                "action": t.action,
                "quantity": round(t.quantity, 4),
                "estimated_value": round(t.estimated_value, 2),
                "model_id": t.model_id,
                "model_version": t.model_version,
            }
            for t in _sort_hin_trades(trades)
        ],
    }

    _write_json(payload, file_path, pretty=pretty)
    logger.info("Exported %d HIN trades (JSON) → %s", len(trades), file_path)
    return file_path


# ---------------------------------------------------------------------------
# Aggregated trade exports
# ---------------------------------------------------------------------------

def export_aggregated_csv(
    result: AggregationResult,
    file_path: Union[str, Path],
    *,
    include_metadata: bool = True,
) -> Path:
    """
    Export an AggregationResult to a CSV file for bulk order submission.

    One row per ticker. Sorted by abs(net_value) descending so the execution
    desk sees the largest orders first.

    Column names use short forms (total_buy_qty, net_qty) for compatibility
    with platforms that enforce column width limits on bulk upload templates.

    Args:
        result:           AggregationResult from aggregation.aggregate_trades().
        file_path:        Destination path.
        include_metadata: If True, writes a metadata comment header.

    Returns:
        Resolved absolute Path to the written file.
    """
    file_path = _resolve_path(file_path)
    timestamp = _utc_now()

    df = result.to_dataframe()

    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

        if include_metadata:
            label_str = f" | Label: {result.label}" if result.label else ""
            writer.writerow([f"# Rebalancing Engine — Aggregated Bulk Orders{label_str}"])
            writer.writerow([
                f"# Generated: {timestamp} | Tickers: {result.ticker_count} "
                f"| Accounts: {result.total_accounts} "
                f"| Net cash flow: {_fmt_value(result.net_cash_flow)}"
            ])

        writer.writerow(AGG_CSV_COLUMNS)

        for _, row in df.iterrows():
            writer.writerow([
                row["ticker"],
                row["net_action"],
                _fmt_qty(row["total_buy_quantity"]),
                _fmt_qty(row["total_sell_quantity"]),
                _fmt_qty(row["net_quantity"]),
                _fmt_value(row["total_buy_value"]),
                _fmt_value(row["total_sell_value"]),
                _fmt_value(row["net_value"]),
                int(row["account_count"]),
                row["accounts"],
            ])

    logger.info(
        "Exported %d aggregated tickers → %s", result.ticker_count, file_path
    )
    return file_path


def export_aggregated_json(
    result: AggregationResult,
    file_path: Union[str, Path],
    *,
    pretty: bool = True,
) -> Path:
    """
    Export a full AggregationResult to JSON, including per-ticker detail
    and the portfolio-level summary block.

    Args:
        result:     AggregationResult to serialise.
        file_path:  Destination path.
        pretty:     Indented (True) or compact (False) JSON.

    Returns:
        Resolved absolute Path to the written file.
    """
    file_path = _resolve_path(file_path)
    timestamp = _utc_now()

    payload = result.to_json()
    payload["metadata"] = {
        "export_timestamp": timestamp,
        "generator": "rebalance_engine.exports",
    }

    # Move metadata to top of dict for readability (Python 3.7+ preserves order)
    ordered = {"metadata": payload.pop("metadata")}
    ordered.update(payload)

    _write_json(ordered, file_path, pretty=pretty)
    logger.info(
        "Exported aggregated result (JSON) → %s", file_path
    )
    return file_path


# ---------------------------------------------------------------------------
# Convenience: export all four files in one call
# ---------------------------------------------------------------------------

def export_full_package(
    trades: list[TradeInstruction],
    output_dir: Union[str, Path],
    *,
    label: Optional[str] = None,
    file_stem: str = "rebalance",
) -> dict[str, Path]:
    """
    Export all four output files (HIN CSV, HIN JSON, aggregated CSV,
    aggregated JSON) to a directory in a single call.

    File names:
        {file_stem}_hin_trades.csv
        {file_stem}_hin_trades.json
        {file_stem}_bulk_orders.csv
        {file_stem}_bulk_orders.json

    Args:
        trades:      Complete list of TradeInstruction objects (all accounts).
        output_dir:  Directory to write files into. Created if absent.
        label:       Optional label stamped into JSON metadata.
        file_stem:   Prefix for all output filenames. Default: "rebalance".

    Returns:
        Dict mapping file type key to resolved Path:
            {
                "hin_csv":  Path(..._hin_trades.csv),
                "hin_json": Path(..._hin_trades.json),
                "agg_csv":  Path(..._bulk_orders.csv),
                "agg_json": Path(..._bulk_orders.json),
            }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    agg_result = aggregate_trades(trades, label=label)

    paths = {
        "hin_csv": export_trades_csv(
            trades,
            output_dir / f"{file_stem}_hin_trades.csv",
        ),
        "hin_json": export_trades_json(
            trades,
            output_dir / f"{file_stem}_hin_trades.json",
            label=label,
        ),
        "agg_csv": export_aggregated_csv(
            agg_result,
            output_dir / f"{file_stem}_bulk_orders.csv",
        ),
        "agg_json": export_aggregated_json(
            agg_result,
            output_dir / f"{file_stem}_bulk_orders.json",
        ),
    }

    logger.info(
        "Export package complete (%d trades, %d tickers) → %s",
        len(trades),
        agg_result.ticker_count,
        output_dir,
    )
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_path(file_path: Union[str, Path]) -> Path:
    """Resolve to absolute Path, creating parent directories if needed."""
    p = Path(file_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _utc_now() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_qty(qty: float) -> str:
    """
    Format a quantity for CSV output.
    Whole numbers export without decimals (100, not 100.0000).
    Fractional quantities export to 4dp (229.9610).
    """
    if qty == int(qty):
        return str(int(qty))
    return f"{qty:.4f}"


def _fmt_value(value: float) -> str:
    """Format an AUD value to 2dp for CSV output."""
    return f"{value:.2f}"


def _sort_hin_trades(trades: list[TradeInstruction]) -> list[TradeInstruction]:
    """
    Sort trades for export:
      1. By account_id alphabetically
      2. Within account: SELLs before BUYs (execution order)
      3. Within action: by ticker alphabetically
    """
    action_order = {"SELL": 0, "BUY": 1}
    return sorted(
        trades,
        key=lambda t: (t.account_id, action_order.get(t.action, 9), t.ticker),
    )


def _write_json(payload: dict, file_path: Path, *, pretty: bool) -> None:
    """Write a JSON-serialisable dict to file."""
    indent = 2 if pretty else None
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
        if pretty:
            f.write("\n")  # POSIX-standard trailing newline
