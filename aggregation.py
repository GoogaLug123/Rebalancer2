"""
aggregation.py — Multi-account trade aggregation for bulk execution.

Public API:
    aggregate_trades(
        trades  : list[TradeInstruction],
        label   : str | None = None,
    ) -> AggregationResult

This module is the final stage in the rebalancing pipeline before trade
output leaves the system. It consumes TradeInstruction objects from one
or many accounts and produces two outputs:

  1. A Pandas DataFrame — one row per ticker, ready for CSV export or
     direct upload to a bulk trading platform (HUB24, BT Panorama, etc.)

  2. A JSON-serialisable dict — for the API layer, Streamlit dashboard,
     or any downstream consumer that doesn't want a DataFrame dependency.

Aggregation logic:
    For each ticker across all accounts:

        total_buy_quantity   = sum of quantity  where action == "BUY"
        total_sell_quantity  = sum of quantity  where action == "SELL"
        net_quantity         = total_buy_quantity - total_sell_quantity
        total_buy_value      = sum of estimated_value where action == "BUY"
        total_sell_value     = sum of estimated_value where action == "SELL"
        net_value            = total_buy_value - total_sell_value
                               Positive → net buyer of this security
                               Negative → net seller
        account_count        = number of distinct accounts trading this ticker
        net_action           = "BUY"  if net_quantity > 0
                               "SELL" if net_quantity < 0
                               "FLAT" if net_quantity == 0 (buys and sells offset)

    Portfolio-level summary:
        total_buy_value_all   = sum of all buy estimated_values
        total_sell_value_all  = sum of all sell estimated_values
        net_cash_flow         = total_sell_value_all - total_buy_value_all
                                Positive → net cash inflow across all accounts
                                Negative → net cash outflow (more buying than selling)

Netting semantics:
    net_quantity and net_value represent the *minimum* market activity needed
    to achieve all rebalances. If HIN001 needs to BUY 100 CBA and HIN002 needs
    to SELL 100 CBA, the net_quantity = 0 and net_action = "FLAT" — in theory
    these could be crossed internally. In practice, Australian retail wrap
    platforms do not support internal crossing; the gross quantities
    (total_buy_quantity, total_sell_quantity) are what get routed to market.
    Both are included in the output so the execution desk can choose.

DataFrame column order is fixed and documented below so downstream consumers
(Excel uploads, platform APIs) can rely on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from models import TradeInstruction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

# Fixed column order for the aggregated DataFrame and all exports.
# Do not reorder — platform bulk-upload templates depend on this sequence.
DATAFRAME_COLUMNS: list[str] = [
    "ticker",
    "net_action",
    "total_buy_quantity",
    "total_sell_quantity",
    "net_quantity",
    "total_buy_value",
    "total_sell_value",
    "net_value",
    "account_count",
    "accounts",
]


@dataclass
class TickerAggregate:
    """
    Aggregated trade data for a single security across all accounts.

    Attributes:
        ticker:               ASX ticker code.
        total_buy_quantity:   Total units to BUY across all accounts.
        total_sell_quantity:  Total units to SELL across all accounts.
        net_quantity:         total_buy - total_sell. Positive = net buyer.
        total_buy_value:      Total estimated AUD value of all BUY trades.
        total_sell_value:     Total estimated AUD value of all SELL trades.
        net_value:            total_buy_value - total_sell_value.
        account_count:        Number of distinct accounts trading this ticker.
        accounts:             Sorted list of account IDs involved in trades
                              for this ticker. Provides full lineage back to
                              individual HIN-level instructions.
    """
    ticker: str
    total_buy_quantity: float
    total_sell_quantity: float
    net_quantity: float
    total_buy_value: float
    total_sell_value: float
    net_value: float
    account_count: int
    accounts: list[str]

    @property
    def net_action(self) -> str:
        """
        Net direction of market activity for this ticker.
        "BUY"  → more buying than selling across all accounts.
        "SELL" → more selling than buying.
        "FLAT" → buys and sells offset exactly (potential internal cross).
        """
        if self.net_quantity > 0:
            return "BUY"
        elif self.net_quantity < 0:
            return "SELL"
        return "FLAT"

    @property
    def is_flat(self) -> bool:
        """True when BUY and SELL quantities exactly offset."""
        return self.net_quantity == 0.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "net_action": self.net_action,
            "total_buy_quantity": self.total_buy_quantity,
            "total_sell_quantity": self.total_sell_quantity,
            "net_quantity": self.net_quantity,
            "total_buy_value": round(self.total_buy_value, 2),
            "total_sell_value": round(self.total_sell_value, 2),
            "net_value": round(self.net_value, 2),
            "account_count": self.account_count,
            "accounts": self.accounts,
        }

    def __repr__(self) -> str:
        return (
            f"TickerAggregate({self.ticker}: {self.net_action} "
            f"net_qty={self.net_quantity:+,.0f}, "
            f"net_value=${self.net_value:+,.2f}, "
            f"accounts={self.account_count})"
        )


@dataclass
class AggregationResult:
    """
    Complete output of aggregate_trades() — ticker-level aggregates plus
    a portfolio-wide summary.

    Attributes:
        label:                Optional descriptive label (e.g. "GROWTH_70 2024-Q2
                              rebalance" or a batch run timestamp). Included in
                              JSON output for traceability.
        tickers:              Per-ticker TickerAggregate objects, sorted by
                              abs(net_value) descending (largest trades first).
        total_accounts:       Number of distinct accounts in the input.
        total_instructions:   Number of raw TradeInstruction objects processed.
        total_buy_value:      Sum of all buy estimated_values across all tickers
                              and accounts.
        total_sell_value:     Sum of all sell estimated_values.
        net_cash_flow:        total_sell_value - total_buy_value.
                              Positive → portfolio is a net seller overall (cash inflow).
                              Negative → portfolio is a net buyer (cash outflow).
        flat_tickers:         Tickers where buys and sells exactly offset. These
                              are flagged for the execution desk — they could be
                              left untouched if the platform supports netting.
    """
    label: Optional[str]
    tickers: list[TickerAggregate]
    total_accounts: int
    total_instructions: int
    total_buy_value: float
    total_sell_value: float

    @property
    def net_cash_flow(self) -> float:
        """
        Net cash flow across all accounts and tickers.
        Positive = net cash inflow (more selling than buying).
        Negative = net cash outflow (more buying than selling).
        """
        return self.total_sell_value - self.total_buy_value

    @property
    def flat_tickers(self) -> list[str]:
        """Tickers where gross buys and sells exactly offset (net_quantity = 0)."""
        return [t.ticker for t in self.tickers if t.is_flat]

    @property
    def ticker_count(self) -> int:
        """Total number of distinct tickers in the aggregated output."""
        return len(self.tickers)

    @property
    def buy_ticker_count(self) -> int:
        return sum(1 for t in self.tickers if t.net_action == "BUY")

    @property
    def sell_ticker_count(self) -> int:
        return sum(1 for t in self.tickers if t.net_action == "SELL")

    # ------------------------------------------------------------------
    # Primary outputs: DataFrame and JSON
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a Pandas DataFrame with one row per ticker, columns in the
        fixed order defined by DATAFRAME_COLUMNS.

        Numeric columns are float64. The 'accounts' column contains
        pipe-delimited account IDs (e.g. "HIN001|HIN002") for CSV export
        compatibility — list representation would break single-column CSVs.

        Returns:
            pd.DataFrame with shape (n_tickers, len(DATAFRAME_COLUMNS)).
            Empty DataFrame (0 rows, same columns) if no trades were input.
        """
        if not self.tickers:
            return pd.DataFrame(columns=DATAFRAME_COLUMNS)

        rows = []
        for ta in self.tickers:
            rows.append({
                "ticker": ta.ticker,
                "net_action": ta.net_action,
                "total_buy_quantity": ta.total_buy_quantity,
                "total_sell_quantity": ta.total_sell_quantity,
                "net_quantity": ta.net_quantity,
                "total_buy_value": round(ta.total_buy_value, 2),
                "total_sell_value": round(ta.total_sell_value, 2),
                "net_value": round(ta.net_value, 2),
                "account_count": ta.account_count,
                "accounts": "|".join(ta.accounts),  # pipe-delimited for CSV safety
            })

        df = pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)

        # Enforce dtypes explicitly — avoids silent object-column issues
        float_cols = [
            "total_buy_quantity", "total_sell_quantity", "net_quantity",
            "total_buy_value", "total_sell_value", "net_value",
        ]
        for col in float_cols:
            df[col] = df[col].astype("float64")
        df["account_count"] = df["account_count"].astype("int64")

        return df

    def to_json(self) -> dict:
        """
        Return a JSON-serialisable dict representation of the full result.

        Structure:
            {
                "label": str | None,
                "summary": {
                    "total_accounts": int,
                    "total_instructions": int,
                    "ticker_count": int,
                    "buy_ticker_count": int,
                    "sell_ticker_count": int,
                    "flat_ticker_count": int,
                    "total_buy_value": float,
                    "total_sell_value": float,
                    "net_cash_flow": float,
                    "flat_tickers": [str, ...]
                },
                "tickers": [
                    {
                        "ticker": str,
                        "net_action": str,
                        "total_buy_quantity": float,
                        "total_sell_quantity": float,
                        "net_quantity": float,
                        "total_buy_value": float,
                        "total_sell_value": float,
                        "net_value": float,
                        "account_count": int,
                        "accounts": [str, ...]
                    },
                    ...
                ]
            }
        """
        return {
            "label": self.label,
            "summary": {
                "total_accounts": self.total_accounts,
                "total_instructions": self.total_instructions,
                "ticker_count": self.ticker_count,
                "buy_ticker_count": self.buy_ticker_count,
                "sell_ticker_count": self.sell_ticker_count,
                "flat_ticker_count": len(self.flat_tickers),
                "total_buy_value": round(self.total_buy_value, 2),
                "total_sell_value": round(self.total_sell_value, 2),
                "net_cash_flow": round(self.net_cash_flow, 2),
                "flat_tickers": self.flat_tickers,
            },
            "tickers": [t.to_dict() for t in self.tickers],
        }

    def __repr__(self) -> str:
        label_str = f" [{self.label!r}]" if self.label else ""
        return (
            f"AggregationResult{label_str}("
            f"accounts={self.total_accounts}, "
            f"instructions={self.total_instructions}, "
            f"tickers={self.ticker_count} "
            f"[{self.buy_ticker_count}B/{self.sell_ticker_count}S/{len(self.flat_tickers)}F], "
            f"net_cash=${self.net_cash_flow:+,.2f})"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def aggregate_trades(
    trades: list[TradeInstruction],
    label: Optional[str] = None,
) -> AggregationResult:
    """
    Aggregate a list of TradeInstruction objects across all accounts into
    per-ticker totals suitable for bulk execution.

    Args:
        trades: TradeInstruction objects from one or many accounts.
                May be empty — returns an empty AggregationResult.
        label:  Optional descriptive label for this aggregation run.
                Included in JSON output for audit trail purposes.
                Example: "GROWTH_70 rebalance 2024-Q2"

    Returns:
        AggregationResult containing:
          - Per-ticker TickerAggregate objects (sorted by abs(net_value) desc)
          - Portfolio-wide summary statistics
          - .to_dataframe() → pd.DataFrame
          - .to_json()      → dict

    Raises:
        TypeError: If any element of trades is not a TradeInstruction.
    """
    if not trades:
        logger.info("aggregate_trades called with empty trade list.")
        return AggregationResult(
            label=label,
            tickers=[],
            total_accounts=0,
            total_instructions=0,
            total_buy_value=0.0,
            total_sell_value=0.0,
        )

    _validate_inputs(trades)

    # -----------------------------------------------------------------------
    # Step 1: Convert to DataFrame for vectorised aggregation
    # -----------------------------------------------------------------------
    raw_df = _instructions_to_dataframe(trades)

    # -----------------------------------------------------------------------
    # Step 2: Pivot into per-ticker BUY / SELL columns
    # -----------------------------------------------------------------------
    ticker_aggs = _aggregate_by_ticker(raw_df)

    # -----------------------------------------------------------------------
    # Step 3: Build summary statistics
    # -----------------------------------------------------------------------
    total_accounts = raw_df["account_id"].nunique()
    total_buy_value = raw_df.loc[raw_df["action"] == "BUY", "estimated_value"].sum()
    total_sell_value = raw_df.loc[raw_df["action"] == "SELL", "estimated_value"].sum()

    result = AggregationResult(
        label=label,
        tickers=ticker_aggs,
        total_accounts=total_accounts,
        total_instructions=len(trades),
        total_buy_value=float(total_buy_value),
        total_sell_value=float(total_sell_value),
    )

    logger.info(
        "Aggregated %d instructions across %d accounts → %d tickers "
        "[%dB/%dS/%dF], net_cash=$%+,.2f%s",
        len(trades),
        total_accounts,
        result.ticker_count,
        result.buy_ticker_count,
        result.sell_ticker_count,
        len(result.flat_tickers),
        result.net_cash_flow,
        f" label={label!r}" if label else "",
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(trades: list) -> None:
    for i, t in enumerate(trades):
        if not isinstance(t, TradeInstruction):
            raise TypeError(
                f"trades[{i}] is {type(t).__name__!r}, expected TradeInstruction."
            )


def _instructions_to_dataframe(trades: list[TradeInstruction]) -> pd.DataFrame:
    """
    Convert the list of TradeInstruction objects into a flat DataFrame
    for vectorised aggregation. Signed columns are not used here —
    aggregation uses separate BUY/SELL pivoting to keep values unambiguous.
    """
    return pd.DataFrame([
        {
            "account_id": t.account_id,
            "ticker": t.ticker,
            "action": t.action,
            "quantity": t.quantity,
            "estimated_value": t.estimated_value,
        }
        for t in trades
    ])


def _aggregate_by_ticker(df: pd.DataFrame) -> list[TickerAggregate]:
    """
    Group trades by ticker and compute per-direction totals using
    a pivot approach: split BUY and SELL into separate sub-DataFrames,
    aggregate each, then merge — avoiding any ambiguity from signed arithmetic.

    Returns:
        List of TickerAggregate sorted by abs(net_value) descending.
    """
    # Separate BUY and SELL sub-frames
    buys = (
        df[df["action"] == "BUY"]
        .groupby("ticker", sort=False)
        .agg(
            total_buy_quantity=("quantity", "sum"),
            total_buy_value=("estimated_value", "sum"),
        )
    )

    sells = (
        df[df["action"] == "SELL"]
        .groupby("ticker", sort=False)
        .agg(
            total_sell_quantity=("quantity", "sum"),
            total_sell_value=("estimated_value", "sum"),
        )
    )

    # Account lists per ticker (for lineage)
    accounts_per_ticker = (
        df.groupby("ticker")["account_id"]
        .apply(lambda s: sorted(s.unique().tolist()))
    )

    # Merge: outer join so tickers that only buy OR only sell are retained
    merged = (
        buys
        .join(sells, how="outer")
        .join(accounts_per_ticker.rename("accounts"), how="left")
        .fillna(0.0)
        .reset_index()
    )

    # Compute net columns
    merged["net_quantity"] = merged["total_buy_quantity"] - merged["total_sell_quantity"]
    merged["net_value"] = merged["total_buy_value"] - merged["total_sell_value"]
    merged["account_count"] = merged["accounts"].apply(len)

    # Sort by abs(net_value) descending — largest trades first
    merged["abs_net_value"] = merged["net_value"].abs()
    merged = merged.sort_values("abs_net_value", ascending=False).drop(columns="abs_net_value")

    # Convert to TickerAggregate objects
    result: list[TickerAggregate] = []
    for _, row in merged.iterrows():
        result.append(TickerAggregate(
            ticker=str(row["ticker"]),
            total_buy_quantity=float(row["total_buy_quantity"]),
            total_sell_quantity=float(row["total_sell_quantity"]),
            net_quantity=float(row["net_quantity"]),
            total_buy_value=float(row["total_buy_value"]),
            total_sell_value=float(row["total_sell_value"]),
            net_value=float(row["net_value"]),
            account_count=int(row["account_count"]),
            accounts=list(row["accounts"]),
        ))

    return result
