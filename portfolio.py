"""
portfolio.py — Portfolio ingestion, validation, and construction.

Public API:
    load_portfolios(file_path) -> list[Portfolio]

Ingestion pipeline (in order):
    1. Read CSV into a DataFrame
    2. Validate required columns are present
    3. Validate no missing account_id values
    4. Validate numeric fields are parseable and within range
    5. Resolve cash_balance per account (first non-null value wins)
    6. Build SecurityHolding objects row-by-row
    7. Group holdings into Portfolio objects by account_id
    8. Return sorted list of Portfolio objects

All validation errors are collected before raising — the caller receives
a complete picture of what is wrong in the file, not just the first error.
This is important for adviser-facing tooling where a batch upload may have
multiple issues that all need fixing before re-submission.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd

from models import Portfolio, SecurityHolding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"account_id", "ticker", "quantity", "price", "cash_balance"}
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_portfolios(file_path: Union[str, Path]) -> list[Portfolio]:
    """
    Load and validate a holdings CSV file into a list of Portfolio objects.

    Each row in the CSV represents one security holding. Multiple rows with
    the same account_id are grouped into a single Portfolio. Cash balance
    is taken from the first row encountered for each account — it must be
    consistent across all rows for the same account (a warning is emitted
    if it is not).

    Args:
        file_path: Absolute or relative path to the holdings CSV file.

    Returns:
        List of Portfolio objects, one per unique account_id, sorted
        alphabetically by account_id.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If required columns are missing, or any row fails
                    validation. All validation errors are collected and
                    raised together in a single exception message.

    CSV format (header row required):
        account_id  : str   — HIN or account identifier (required, non-empty)
        ticker      : str   — ASX ticker code (required, non-empty)
        quantity    : float — Units held (required, >= 0)
        price       : float — Last price in AUD (required, >= 0)
        cash_balance: float — Uninvested cash in AUD (required, >= 0)
                              Must be the same on every row for the same account.
    """
    file_path = Path(file_path)
    _assert_file_exists(file_path)

    df = _read_csv(file_path)
    _validate_columns(df)
    df = _normalise_types(df)
    errors = _collect_validation_errors(df)

    if errors:
        error_block = "\n  ".join(errors)
        raise ValueError(
            f"Holdings file '{file_path.name}' failed validation "
            f"with {len(errors)} error(s):\n  {error_block}"
        )

    portfolios = _build_portfolios(df)
    logger.info(
        "Loaded %d portfolio(s) from '%s' (%d holdings total)",
        len(portfolios),
        file_path.name,
        sum(len(p.holdings) for p in portfolios),
    )
    return portfolios


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _assert_file_exists(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Holdings file not found: '{file_path}'")
    if not file_path.is_file():
        raise FileNotFoundError(f"Path exists but is not a file: '{file_path}'")


def _read_csv(file_path: Path) -> pd.DataFrame:
    """Read CSV; strip leading/trailing whitespace from all string columns."""
    df = pd.read_csv(file_path, dtype=str)  # read everything as str first
    df.columns = [c.strip().lower() for c in df.columns]
    # Strip whitespace from all cells before type conversion
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df


# ---------------------------------------------------------------------------
# Column validation
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise immediately if required columns are absent — nothing else is safe."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Holdings CSV is missing required column(s): "
            f"{sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Type normalisation
# ---------------------------------------------------------------------------


def _normalise_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to their target dtypes in-place.
    Coerce numeric columns — invalid values become NaN so that the
    validation step can report them with row numbers rather than
    crashing the conversion itself.
    """
    df = df.copy()
    df["account_id"] = df["account_id"].astype(str)
    df["ticker"] = df["ticker"].str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["cash_balance"] = pd.to_numeric(df["cash_balance"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Row-level validation (collect-all strategy)
# ---------------------------------------------------------------------------


def _collect_validation_errors(df: pd.DataFrame) -> list[str]:
    """
    Walk every row and collect all validation failures.

    Using a collect-all approach (rather than fail-fast) so that an adviser
    uploading a file with multiple issues receives a complete error report
    in a single attempt.

    Returns:
        List of human-readable error strings, one per failure.
        Empty list means the DataFrame is valid.
    """
    errors: list[str] = []

    for idx, row in df.iterrows():
        row_num = idx + 2  # +1 for 0-index, +1 for header row → 1-based CSV line

        # --- account_id ---
        account_id = str(row["account_id"]).strip()
        if not account_id or account_id.lower() in ("nan", "none", ""):
            errors.append(f"Row {row_num}: account_id is missing or empty.")
            # Can't usefully validate anything else without an account_id
            continue

        # --- ticker ---
        ticker = str(row["ticker"]).strip()
        if not ticker or ticker.lower() in ("nan", "none", ""):
            errors.append(
                f"Row {row_num} [{account_id}]: ticker is missing or empty."
            )

        # --- quantity ---
        if pd.isna(row["quantity"]):
            errors.append(
                f"Row {row_num} [{account_id}|{ticker}]: "
                f"quantity is not a valid number."
            )
        elif row["quantity"] < 0:
            errors.append(
                f"Row {row_num} [{account_id}|{ticker}]: "
                f"quantity must be >= 0, got {row['quantity']}."
            )

        # --- price ---
        if pd.isna(row["price"]):
            errors.append(
                f"Row {row_num} [{account_id}|{ticker}]: "
                f"price is not a valid number."
            )
        elif row["price"] < 0:
            errors.append(
                f"Row {row_num} [{account_id}|{ticker}]: "
                f"price must be >= 0, got {row['price']}."
            )

        # --- cash_balance ---
        if pd.isna(row["cash_balance"]):
            errors.append(
                f"Row {row_num} [{account_id}]: "
                f"cash_balance is not a valid number."
            )
        elif row["cash_balance"] < 0:
            errors.append(
                f"Row {row_num} [{account_id}]: "
                f"cash_balance must be >= 0, got {row['cash_balance']}."
            )

    # --- Cross-row: cash_balance must be consistent per account ---
    errors.extend(_validate_cash_balance_consistency(df))

    # --- Cross-row: duplicate ticker within the same account ---
    errors.extend(_validate_no_duplicate_tickers(df))

    return errors


def _validate_cash_balance_consistency(df: pd.DataFrame) -> list[str]:
    """
    Warn (as errors) if the same account_id has different cash_balance values
    across its rows. This catches data entry mistakes where a holdings export
    has been partially edited.
    """
    errors: list[str] = []
    for account_id, group in df.groupby("account_id"):
        unique_balances = group["cash_balance"].dropna().unique()
        if len(unique_balances) > 1:
            errors.append(
                f"Account '{account_id}': inconsistent cash_balance values "
                f"across rows: {sorted(unique_balances)}. "
                f"All rows for the same account must carry the same cash_balance."
            )
    return errors


def _validate_no_duplicate_tickers(df: pd.DataFrame) -> list[str]:
    """
    Flag duplicate (account_id, ticker) pairs. A holding file should have
    exactly one row per security per account — duplicates usually indicate
    a double-export or merge error.
    """
    errors: list[str] = []
    dupes = df[df.duplicated(subset=["account_id", "ticker"], keep=False)]
    if not dupes.empty:
        for (account_id, ticker), _ in dupes.groupby(["account_id", "ticker"]):
            errors.append(
                f"Account '{account_id}': ticker '{ticker}' appears more than once. "
                f"Each security must have exactly one row per account."
            )
    return errors


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------


def _build_portfolios(df: pd.DataFrame) -> list[Portfolio]:
    """
    Convert a validated DataFrame into Portfolio objects.

    Cash balance strategy: take the first non-null value for each account.
    After _validate_cash_balance_consistency passes, all rows for an account
    carry the same value, so first-wins is safe and deterministic.

    Returns:
        Portfolios sorted by account_id.
    """
    holdings_by_account: dict[str, list[SecurityHolding]] = defaultdict(list)
    cash_by_account: dict[str, float] = {}

    for _, row in df.iterrows():
        account_id: str = str(row["account_id"]).strip()
        ticker: str = str(row["ticker"]).strip().upper()

        holding = SecurityHolding(
            account_id=account_id,
            ticker=ticker,
            quantity=float(row["quantity"]),
            price=float(row["price"]),
        )
        holdings_by_account[account_id].append(holding)

        # Capture cash balance once per account (first row wins)
        if account_id not in cash_by_account:
            cash_by_account[account_id] = float(row["cash_balance"])

    portfolios = [
        Portfolio(
            account_id=account_id,
            holdings=holdings,
            cash_balance=cash_by_account.get(account_id, 0.0),
        )
        for account_id, holdings in holdings_by_account.items()
    ]

    return sorted(portfolios, key=lambda p: p.account_id)
