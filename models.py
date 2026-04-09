"""
models.py — Canonical data models for the portfolio rebalancing engine.

All models use Python dataclasses for lightweight, typed, introspectable
data structures with no external dependencies. No business logic lives here
beyond simple value calculations that are intrinsic to the model itself.

Data flow:
    SecurityHolding + price data  →  Portfolio
    ModelHolding                  →  ModelPortfolio
    Drift calculation             →  TradeInstruction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Security-level holding (one line in a client's portfolio)
# ---------------------------------------------------------------------------

@dataclass
class SecurityHolding:
    """
    Represents a single security position held within a client account.

    Attributes:
        account_id:  The HIN or account identifier this holding belongs to.
        ticker:      ASX ticker code (e.g. "CBA", "BHP"). Always stored
                     uppercased — normalised at ingestion time by portfolio.py.
        quantity:    Number of units held. Float to support managed funds /
                     fractional instruments; standard ASX equities will carry
                     whole numbers in practice.
        price:       Last known price per unit in AUD. Used for market value
                     calculations. Set by portfolio.py from a price feed or
                     the ingested holdings file; never modified by trade logic.
    """

    account_id: str
    ticker: str
    quantity: float
    price: float

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()

        if self.quantity < 0:
            raise ValueError(
                f"[{self.ticker}] quantity must be non-negative, got {self.quantity}"
            )
        if self.price < 0:
            raise ValueError(
                f"[{self.ticker}] price must be non-negative, got {self.price}"
            )

    @property
    def market_value(self) -> float:
        """Current market value of this holding: quantity × price (AUD)."""
        return self.quantity * self.price

    def __repr__(self) -> str:
        return (
            f"SecurityHolding(account={self.account_id!r}, ticker={self.ticker!r}, "
            f"qty={self.quantity:,.4f}, price={self.price:,.4f}, "
            f"value={self.market_value:,.2f})"
        )


# ---------------------------------------------------------------------------
# Client portfolio (all holdings for one account)
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """
    A client's complete investment portfolio, identified by account / HIN.

    Holds an ordered list of SecurityHolding objects plus an uninvested
    cash balance. Provides convenience methods for valuation queries used
    throughout the drift and trade generation pipeline.

    Attributes:
        account_id:    The HIN or account identifier.
        holdings:      All security positions for this account.
        cash_balance:  Uninvested cash held in the account (AUD). Defaults
                       to 0.0. Included in total_value() so the rebalance
                       engine can allocate cash into new positions.
    """

    account_id: str
    holdings: list[SecurityHolding] = field(default_factory=list)
    cash_balance: float = 0.0

    def __post_init__(self) -> None:
        if self.cash_balance < 0:
            raise ValueError(
                f"[{self.account_id}] cash_balance must be non-negative, "
                f"got {self.cash_balance}"
            )

    # ------------------------------------------------------------------
    # Valuation helpers
    # ------------------------------------------------------------------

    def total_value(self) -> float:
        """
        Total portfolio value in AUD: sum of all holding market values
        plus uninvested cash.

        Returns:
            float: Total portfolio value. Returns 0.0 for an empty portfolio
                   with no cash (no ZeroDivisionError risk downstream when
                   callers guard on this value).
        """
        securities_value = sum(h.market_value for h in self.holdings)
        return securities_value + self.cash_balance

    def holding_value(self, ticker: str) -> float:
        """
        Market value of a specific security within this portfolio.

        Performs a case-insensitive ticker lookup. Returns 0.0 if the
        security is not held — this is intentional: the drift engine
        needs a numeric zero for securities in the model but not held.

        Args:
            ticker: ASX ticker code to look up.

        Returns:
            float: Market value of the holding, or 0.0 if not held.
        """
        ticker = ticker.upper().strip()
        for h in self.holdings:
            if h.ticker == ticker:
                return h.market_value
        return 0.0

    def holding_quantity(self, ticker: str) -> float:
        """
        Number of units held for a specific security.

        Args:
            ticker: ASX ticker code to look up.

        Returns:
            float: Quantity held, or 0.0 if not held.
        """
        ticker = ticker.upper().strip()
        for h in self.holdings:
            if h.ticker == ticker:
                return h.quantity
        return 0.0

    def holding_price(self, ticker: str) -> Optional[float]:
        """
        Last known price for a specific security.

        Args:
            ticker: ASX ticker code to look up.

        Returns:
            float: Last price if held, None if not held (distinct from
                   price = 0.0, which would indicate a data quality issue).
        """
        ticker = ticker.upper().strip()
        for h in self.holdings:
            if h.ticker == ticker:
                return h.price
        return None

    @property
    def tickers(self) -> list[str]:
        """Sorted list of all tickers currently held."""
        return sorted(h.ticker for h in self.holdings)

    @property
    def securities_value(self) -> float:
        """Total value of invested securities, excluding cash."""
        return sum(h.market_value for h in self.holdings)

    def __repr__(self) -> str:
        return (
            f"Portfolio(account={self.account_id!r}, "
            f"holdings={len(self.holdings)}, "
            f"securities={self.securities_value:,.2f}, "
            f"cash={self.cash_balance:,.2f}, "
            f"total={self.total_value():,.2f})"
        )


# ---------------------------------------------------------------------------
# Model portfolio target weight (one line in a model)
# ---------------------------------------------------------------------------

@dataclass
class ModelHolding:
    """
    A single security's target weight within a model portfolio.

    Attributes:
        ticker:        ASX ticker code. Uppercased on init.
        target_weight: Desired portfolio weight expressed as a decimal
                       fraction (e.g. 0.05 = 5%). Validation that weights
                       sum to 1.0 across a ModelPortfolio is enforced in
                       ModelPortfolio.__post_init__, not here — a single
                       ModelHolding has no knowledge of its siblings.
    """

    ticker: str
    target_weight: float

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()

        if not (0.0 < self.target_weight <= 1.0):
            raise ValueError(
                f"[{self.ticker}] target_weight must be in (0, 1], "
                f"got {self.target_weight}"
            )

    def __repr__(self) -> str:
        return (
            f"ModelHolding(ticker={self.ticker!r}, "
            f"target_weight={self.target_weight:.4%})"
        )


# ---------------------------------------------------------------------------
# Model portfolio (the target construct used to rebalance against)
# ---------------------------------------------------------------------------

@dataclass
class ModelPortfolio:
    """
    A named, versioned set of target weights defining a model portfolio.

    Used by advisers to express how a client's portfolio *should* be
    allocated. The rebalancing engine computes drift against this and
    generates trades to move toward it.

    Attributes:
        model_id:  Unique identifier for this model (e.g. "GROWTH_70",
                   "CONSERVATIVE_30"). Arbitrary string — defined by the
                   adviser platform.
        version:   Version string for the model (e.g. "2024-Q1", "v3.2").
                   Allows audit trail of which model version was in effect
                   at rebalance time. Stored on the TradeInstruction output.
        holdings:  List of ModelHolding objects. Weights must sum to
                   within WEIGHT_SUM_TOLERANCE of 1.0 (see __post_init__).
    """

    # Tolerance for floating-point weight sum validation (±0.0001 = ±0.01%)
    _WEIGHT_SUM_TOLERANCE: float = field(default=0.0001, init=False, repr=False)

    model_id: str
    version: str
    holdings: list[ModelHolding] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.holdings:
            self._validate_weights()

    def _validate_weights(self) -> None:
        """
        Assert that model weights sum to 1.0 within floating-point tolerance.
        Called on init and after any mutation via add_holding().
        """
        total = sum(h.target_weight for h in self.holdings)
        if abs(total - 1.0) > self._WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"ModelPortfolio '{self.model_id}' weights sum to {total:.6f}, "
                f"expected 1.0 (±{self._WEIGHT_SUM_TOLERANCE}). "
                f"Difference: {total - 1.0:+.6f}"
            )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def target_weight(self, ticker: str) -> float:
        """
        Target weight for a specific ticker.

        Args:
            ticker: ASX ticker code.

        Returns:
            float: Target weight as a decimal, or 0.0 if the security
                   is not in this model (i.e. it should be fully sold).
        """
        ticker = ticker.upper().strip()
        for h in self.holdings:
            if h.ticker == ticker:
                return h.target_weight
        return 0.0

    @property
    def tickers(self) -> list[str]:
        """Sorted list of all tickers included in this model."""
        return sorted(h.ticker for h in self.holdings)

    @property
    def weight_sum(self) -> float:
        """Sum of all target weights. Should equal 1.0 after validation."""
        return sum(h.target_weight for h in self.holdings)

    def __repr__(self) -> str:
        return (
            f"ModelPortfolio(id={self.model_id!r}, version={self.version!r}, "
            f"holdings={len(self.holdings)}, weight_sum={self.weight_sum:.6f})"
        )


# ---------------------------------------------------------------------------
# Trade instruction (output of the trade generation engine)
# ---------------------------------------------------------------------------

@dataclass
class TradeInstruction:
    """
    A single trade instruction generated by the rebalancing engine.

    Represents one BUY or SELL order for a specific security within a
    specific account. Produced by trades.py and consumed by aggregation.py
    (for bulk order assembly) and the API / UI layers for display.

    Attributes:
        account_id:       The HIN or account this trade is for.
        ticker:           ASX ticker to trade.
        action:           "BUY" or "SELL". Validated on init.
        quantity:         Number of units to trade. Always a positive number —
                          direction is expressed by `action`, not sign.
                          ASX equities will be whole numbers after rounding
                          in trades.py; stored as float for managed fund support.
        estimated_value:  Estimated AUD value of the trade (qty × price at
                          calculation time). Indicative only — actual fill
                          price will differ. Used for aggregation reporting
                          and minimum trade size filtering.
        model_id:         Optional: the model portfolio this trade rebalances
                          toward. Provides audit lineage on the output file.
        model_version:    Optional: the model version in effect at rebalance
                          time. Stored for regulatory audit trail.
    """

    VALID_ACTIONS: frozenset[str] = field(
        default=frozenset({"BUY", "SELL"}), init=False, repr=False
    )

    account_id: str
    ticker: str
    action: str
    quantity: float
    estimated_value: float
    model_id: Optional[str] = None
    model_version: Optional[str] = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper().strip()
        self.action = self.action.upper().strip()

        if self.action not in self.VALID_ACTIONS:
            raise ValueError(
                f"[{self.ticker}] action must be 'BUY' or 'SELL', got {self.action!r}"
            )
        if self.quantity <= 0:
            raise ValueError(
                f"[{self.ticker}] quantity must be positive, got {self.quantity}. "
                f"Use action='BUY'/'SELL' to express direction."
            )
        if self.estimated_value < 0:
            raise ValueError(
                f"[{self.ticker}] estimated_value must be non-negative, "
                f"got {self.estimated_value}"
            )

    @property
    def signed_quantity(self) -> float:
        """
        Signed quantity: positive for BUY, negative for SELL.
        Convenience property for aggregation netting calculations.
        """
        return self.quantity if self.action == "BUY" else -self.quantity

    @property
    def signed_value(self) -> float:
        """
        Signed estimated value: positive for BUY (cash outflow),
        negative for SELL (cash inflow).
        """
        return self.estimated_value if self.action == "BUY" else -self.estimated_value

    def __repr__(self) -> str:
        model_str = f", model={self.model_id!r}" if self.model_id else ""
        return (
            f"TradeInstruction(account={self.account_id!r}, "
            f"ticker={self.ticker!r}, action={self.action!r}, "
            f"qty={self.quantity:,.4f}, "
            f"est_value={self.estimated_value:,.2f}{model_str})"
        )
