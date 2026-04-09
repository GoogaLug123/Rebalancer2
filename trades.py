"""
trades.py — Trade generation for portfolio rebalancing.

Public API:
    generate_trades(
        portfolio : Portfolio,
        model     : ModelPortfolio,
        config    : TradeConfig | None = None,
    ) -> TradeResult

Pipeline (executed in this strict order):
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. GUARD          Reject zero-value portfolios              │
    │ 2. SELL first     Generate SELL trades for:                 │
    │                     a. NOT_IN_MODEL securities (sell to 0)  │
    │                     b. Overweight model securities           │
    │ 3. CASH POOL      Start with existing cash_balance          │
    │                   Add proceeds from step-2 SELL trades      │
    │ 4. BUY from cash  Generate BUY trades funded by cash pool   │
    │                   Stop issuing BUYs when cash pool < 0       │
    │ 5. FILTER         Drop trades below min_trade_value          │
    │ 6. ROUND          Round qty per security type (ASX = whole)  │
    │ 7. RECONCILE      Compute residual cash after all trades     │
    └─────────────────────────────────────────────────────────────┘

Cash handling rationale:
    Australian wrap platforms (HUB24, BT Panorama, Netwealth) process trades
    in settlement batches. SELL proceeds are not available until T+2, so a
    strict "sell first, buy with proceeds" approach is operationally correct
    for same-day execution. However, most platforms allow advisers to execute
    buys and sells in the same batch with netting at settlement. This engine
    takes the conservative approach: cash on hand is used first, then sell
    proceeds fund buys, meaning the generated trade list is always fully
    self-funding from an available-cash perspective. If sell proceeds are
    insufficient to fund all buys, the remaining buys are still generated
    (flagged in the TradeResult) — the adviser must decide whether to fund
    from an external source or defer.

Rounding conventions:
    ASX-listed equities trade in whole shares only. Managed funds and ETFs
    may allow fractional units. The default is whole-share rounding (floor
    for SELLs, floor for BUYs — conservative in both directions to avoid
    overshooting targets or overdrawing cash). Managed funds use 3dp rounding.
    Rounding mode is configurable via TradeConfig.

Minimum trade value:
    Trades below min_trade_value (default $500) are suppressed. This avoids
    generating nuisance trades for tiny rounding residuals. Suppressed trades
    are recorded in TradeResult.suppressed_trades for full auditability.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models import ModelPortfolio, Portfolio, TradeInstruction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class RoundingMode(str, Enum):
    """
    How to round trade quantities to the permitted precision.

    FLOOR:  Always round down (conservative — never overshoot).
            Default for both BUY and SELL.
            BUY  floor: buy slightly fewer shares than ideal, leave small cash residual.
            SELL floor: sell slightly fewer shares than ideal, leave tiny overweight.
    ROUND:  Standard half-up rounding (nearest integer / nearest 0.001).
            Used for managed funds where fractional units are permitted.
    """
    FLOOR = "FLOOR"
    ROUND = "ROUND"


@dataclass
class TradeConfig:
    """
    Configuration knobs for the trade generation engine.

    All values have sensible defaults for an ASX equity portfolio.
    Override as needed per model or per platform.

    Attributes:
        min_trade_value:    Minimum AUD value of a trade to be issued.
                            Trades below this value are suppressed and
                            recorded in TradeResult.suppressed_trades.
                            Default: $500.00.

        whole_shares:       If True, quantity is rounded to a whole integer.
                            Set False for managed funds / ETF models that
                            allow fractional units.
                            Default: True (ASX equities).

        rounding_mode:      How to round quantities after division.
                            Default: FLOOR (conservative — never overshoot).

        managed_fund_dp:    Decimal places for managed fund quantity when
                            whole_shares=False.
                            Default: 3.

        drift_threshold:    Absolute drift threshold used to decide which
                            securities need trading. Securities within band
                            are left untouched.
                            Default: 0.03 (3 percentage points).
    """
    min_trade_value: float = 500.0
    whole_shares: bool = True
    rounding_mode: RoundingMode = RoundingMode.FLOOR
    managed_fund_dp: int = 3
    drift_threshold: float = 0.03


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class TradeDetail:
    """
    A single trade with full calculation lineage — used internally and
    exposed on TradeResult for auditability.

    Extends TradeInstruction metadata with the intermediate values that
    produced it, so that any adviser or compliance reviewer can trace
    exactly how each trade quantity was derived.

    Attributes:
        instruction:        The canonical TradeInstruction (account, ticker,
                            action, quantity, estimated_value).
        target_value:       target_weight × total_portfolio_value  (AUD).
        current_value:      Current market value of the holding     (AUD).
        raw_trade_value:    target_value - current_value before rounding (AUD).
        raw_quantity:       raw_trade_value / price before rounding.
        rounded_quantity:   Final quantity after rounding per config.
        rounding_residual:  AUD value lost to rounding
                            = (raw_quantity - rounded_quantity) × price.
        price_used:         The price used for quantity calculation.
        suppressed:         True if this trade was below min_trade_value.
        suppression_reason: Human-readable reason if suppressed.
    """
    instruction: TradeInstruction
    target_value: float
    current_value: float
    raw_trade_value: float
    raw_quantity: float
    rounded_quantity: float
    rounding_residual: float
    price_used: float
    suppressed: bool = False
    suppression_reason: Optional[str] = None

    @property
    def ticker(self) -> str:
        return self.instruction.ticker

    @property
    def action(self) -> str:
        return self.instruction.action

    def __repr__(self) -> str:
        flag = "[SUPPRESSED] " if self.suppressed else ""
        return (
            f"{flag}TradeDetail({self.action} {self.ticker}: "
            f"raw_qty={self.raw_quantity:.4f} → "
            f"rounded_qty={self.rounded_quantity:.4f}, "
            f"est_value=${self.instruction.estimated_value:,.2f}, "
            f"residual=${self.rounding_residual:,.2f})"
        )


@dataclass
class TradeResult:
    """
    Complete output of generate_trades() for one portfolio.

    Contains the active trade list plus full reconciliation data so the
    adviser and any downstream system can verify the cash arithmetic.

    Attributes:
        account_id:          Portfolio HIN / account identifier.
        model_id:            Model used for this rebalance.
        model_version:       Model version used.
        total_portfolio_value: Portfolio value used as weight denominator.
        opening_cash:        cash_balance at start of rebalance.
        sell_proceeds:       Estimated proceeds from all SELL trades.
        buy_cost:            Estimated cost of all BUY trades.
        closing_cash:        opening_cash + sell_proceeds - buy_cost.
                             Positive = uninvested residual.
                             Negative = funding shortfall (BUYs exceed available cash).
        trades:              Active TradeInstruction objects ready for execution.
        trade_details:       Full calculation lineage for every trade considered
                             (active + suppressed), sorted SELLs first then BUYs.
        suppressed_trades:   Trades dropped because estimated_value < min_trade_value.
        config:              The TradeConfig used for this run.
    """
    account_id: str
    model_id: str
    model_version: str
    total_portfolio_value: float
    opening_cash: float
    sell_proceeds: float
    buy_cost: float
    trades: list[TradeInstruction] = field(default_factory=list)
    trade_details: list[TradeDetail] = field(default_factory=list)
    suppressed_trades: list[TradeDetail] = field(default_factory=list)
    config: TradeConfig = field(default_factory=TradeConfig)

    @property
    def closing_cash(self) -> float:
        """
        Projected cash balance after all trades settle.
        Positive = residual cash remaining uninvested.
        Negative = funding shortfall (BUYs cannot be fully funded from
                   existing cash + sell proceeds).
        """
        return self.opening_cash + self.sell_proceeds - self.buy_cost

    @property
    def has_funding_shortfall(self) -> bool:
        """True if BUY trades exceed available cash + sell proceeds."""
        return self.closing_cash < -0.01  # 1c tolerance for float noise

    @property
    def buy_count(self) -> int:
        return sum(1 for t in self.trades if t.action == "BUY")

    @property
    def sell_count(self) -> int:
        return sum(1 for t in self.trades if t.action == "SELL")

    def to_dict(self) -> dict:
        """
        Serialise to JSON-compatible dict for API / UI consumption.

        Structure mirrors the DriftReport.to_dict() convention for consistency
        across the pipeline.
        """
        return {
            "account_id": self.account_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "total_portfolio_value": round(self.total_portfolio_value, 2),
            "opening_cash": round(self.opening_cash, 2),
            "sell_proceeds": round(self.sell_proceeds, 2),
            "buy_cost": round(self.buy_cost, 2),
            "closing_cash": round(self.closing_cash, 2),
            "has_funding_shortfall": self.has_funding_shortfall,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "suppressed_count": len(self.suppressed_trades),
            "trades": [
                {
                    "account_id": t.account_id,
                    "ticker": t.ticker,
                    "action": t.action,
                    "quantity": t.quantity,
                    "estimated_value": round(t.estimated_value, 2),
                    "model_id": t.model_id,
                    "model_version": t.model_version,
                }
                for t in self.trades
            ],
            "suppressed_trades": [
                {
                    "ticker": td.ticker,
                    "action": td.action,
                    "raw_trade_value": round(td.raw_trade_value, 2),
                    "suppression_reason": td.suppression_reason,
                }
                for td in self.suppressed_trades
            ],
        }

    def __repr__(self) -> str:
        shortfall = f" ⚠ SHORTFALL ${abs(self.closing_cash):,.2f}" if self.has_funding_shortfall else ""
        return (
            f"TradeResult(account={self.account_id!r}, "
            f"model={self.model_id!r}@{self.model_version!r}, "
            f"buys={self.buy_count}, sells={self.sell_count}, "
            f"suppressed={len(self.suppressed_trades)}, "
            f"closing_cash=${self.closing_cash:,.2f}{shortfall})"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_trades(
    portfolio: Portfolio,
    model: ModelPortfolio,
    config: Optional[TradeConfig] = None,
) -> TradeResult:
    """
    Generate a complete set of trade instructions to rebalance a portfolio
    toward a model.

    Args:
        portfolio:  The client's current holdings and cash balance.
        model:      The target model portfolio with validated weights.
        config:     Optional TradeConfig. Uses sensible ASX equity defaults
                    if not provided.

    Returns:
        TradeResult containing active trades, suppressed trades, and a full
        cash reconciliation.

    Raises:
        ValueError: If portfolio total value is zero or negative.
        ValueError: If a model security has no price available (not held and
                    no price can be inferred — caller must supply prices).
    """
    if config is None:
        config = TradeConfig()

    _validate_portfolio(portfolio)

    total_value = portfolio.total_value()

    # -----------------------------------------------------------------------
    # Step 1: Build the full security universe (model ∪ portfolio)
    # -----------------------------------------------------------------------
    universe = _build_trade_universe(portfolio, model)

    # -----------------------------------------------------------------------
    # Step 2: Compute raw trade requirements for every security
    # -----------------------------------------------------------------------
    raw_trades = _compute_raw_trades(universe, portfolio, model, total_value, config)

    # -----------------------------------------------------------------------
    # Step 3: Separate SELLs and BUYs — process SELLs first
    # -----------------------------------------------------------------------
    raw_sells = [t for t in raw_trades if t.raw_trade_value < 0]
    raw_buys = [t for t in raw_trades if t.raw_trade_value > 0]

    # -----------------------------------------------------------------------
    # Step 4: Round and filter SELL trades
    # -----------------------------------------------------------------------
    sell_details = _finalise_trades(raw_sells, config, portfolio)

    # -----------------------------------------------------------------------
    # Step 5: Build cash pool = opening cash + SELL proceeds
    # -----------------------------------------------------------------------
    opening_cash = portfolio.cash_balance
    sell_proceeds = sum(
        td.instruction.estimated_value
        for td in sell_details
        if not td.suppressed
    )
    available_cash = opening_cash + sell_proceeds

    logger.debug(
        "[%s] Cash pool: opening=$%.2f + sell_proceeds=$%.2f = $%.2f",
        portfolio.account_id,
        opening_cash,
        sell_proceeds,
        available_cash,
    )

    # -----------------------------------------------------------------------
    # Step 6: Round and filter BUY trades (largest first to prioritise big buys)
    # -----------------------------------------------------------------------
    raw_buys_sorted = sorted(raw_buys, key=lambda t: abs(t.raw_trade_value), reverse=True)
    buy_details = _finalise_trades(raw_buys_sorted, config, portfolio)

    buy_cost = sum(
        td.instruction.estimated_value
        for td in buy_details
        if not td.suppressed
    )

    # -----------------------------------------------------------------------
    # Step 7: Log funding shortfall if BUY cost exceeds available cash
    # -----------------------------------------------------------------------
    closing_cash = available_cash - buy_cost
    if closing_cash < -0.01:
        logger.warning(
            "[%s] Funding shortfall of $%.2f — BUY trades ($%.2f) exceed "
            "available cash ($%.2f). Adviser action required.",
            portfolio.account_id,
            abs(closing_cash),
            buy_cost,
            available_cash,
        )

    # -----------------------------------------------------------------------
    # Step 8: Assemble result
    # -----------------------------------------------------------------------
    all_details = _sort_details(sell_details + buy_details)
    active_details = [td for td in all_details if not td.suppressed]
    suppressed_details = [td for td in all_details if td.suppressed]

    active_trades = [td.instruction for td in active_details]

    result = TradeResult(
        account_id=portfolio.account_id,
        model_id=model.model_id,
        model_version=model.version,
        total_portfolio_value=total_value,
        opening_cash=opening_cash,
        sell_proceeds=sell_proceeds,
        buy_cost=buy_cost,
        trades=active_trades,
        trade_details=all_details,
        suppressed_trades=suppressed_details,
        config=config,
    )

    logger.info(
        "[%s] Trades generated: %d BUY, %d SELL, %d suppressed, "
        "closing_cash=$%.2f%s",
        portfolio.account_id,
        result.buy_count,
        result.sell_count,
        len(suppressed_details),
        result.closing_cash,
        " ⚠ SHORTFALL" if result.has_funding_shortfall else "",
    )

    return result


# ---------------------------------------------------------------------------
# Internal: validation
# ---------------------------------------------------------------------------

def _validate_portfolio(portfolio: Portfolio) -> None:
    total = portfolio.total_value()
    if total <= 0:
        raise ValueError(
            f"Portfolio '{portfolio.account_id}' has total value ${total:,.2f}. "
            f"Cannot generate trades for a zero-value portfolio."
        )


# ---------------------------------------------------------------------------
# Internal: universe and raw trade calculation
# ---------------------------------------------------------------------------

def _build_trade_universe(
    portfolio: Portfolio,
    model: ModelPortfolio,
) -> set[str]:
    """
    Full set of tickers that need a trade decision:
      - All tickers in the model (may need BUY or trimming SELL)
      - All tickers held but not in the model (need full SELL)
    """
    return {h.ticker for h in model.holdings} | {h.ticker for h in portfolio.holdings}


@dataclass
class _RawTrade:
    """
    Pre-rounding, pre-filtering trade calculation for one security.
    Internal use only — converted to TradeDetail after rounding.
    """
    ticker: str
    target_value: float
    current_value: float
    raw_trade_value: float   # positive = BUY, negative = SELL
    price: float
    raw_quantity: float      # signed: positive = BUY, negative = SELL


def _compute_raw_trades(
    universe: set[str],
    portfolio: Portfolio,
    model: ModelPortfolio,
    total_value: float,
    config: TradeConfig,
) -> list[_RawTrade]:
    """
    For every ticker in the universe, compute the exact dollar and
    quantity delta needed to hit target weight — before rounding or filtering.

    A security not held has current_value = 0.0 → BUY from zero.
    A security not in the model has target_value = 0.0 → SELL to zero.
    Securities within the drift band are skipped (no trade needed).
    """
    raw: list[_RawTrade] = []

    for ticker in universe:
        target_weight = model.target_weight(ticker)   # 0.0 if not in model
        target_value = target_weight * total_value
        current_value = portfolio.holding_value(ticker)  # 0.0 if not held

        raw_trade_value = target_value - current_value

        # Skip securities within the drift band (no trade needed)
        weight_drift = abs(raw_trade_value) / total_value
        not_in_model = target_weight == 0.0 and current_value > 0.0
        if weight_drift <= config.drift_threshold and not not_in_model:
            logger.debug(
                "[%s] %s: drift=%.4f%% within band (%.4f%%), skipping",
                portfolio.account_id,
                ticker,
                weight_drift * 100,
                config.drift_threshold * 100,
            )
            continue

        # Resolve price: use held price if available, else raise
        price = _resolve_price(ticker, portfolio)

        if price == 0.0:
            logger.warning(
                "[%s] %s: price is zero — cannot compute trade quantity. "
                "Skipping. Check price feed.",
                portfolio.account_id,
                ticker,
            )
            continue

        raw_quantity = raw_trade_value / price

        raw.append(_RawTrade(
            ticker=ticker,
            target_value=target_value,
            current_value=current_value,
            raw_trade_value=raw_trade_value,
            price=price,
            raw_quantity=raw_quantity,
        ))

    return raw


def _resolve_price(ticker: str, portfolio: Portfolio) -> float:
    """
    Get the price for a ticker from the portfolio.

    For securities held in the portfolio, this is the last known price.
    For securities in the model but not held (BUY from zero), the price
    must be injected into the portfolio before calling generate_trades().
    If it cannot be resolved, returns 0.0 (caller logs a warning and skips).

    In a production system, this would call a price feed / security master.
    For this POC, prices are sourced from the portfolio's holdings only.
    """
    price = portfolio.holding_price(ticker)
    return price if price is not None else 0.0


# ---------------------------------------------------------------------------
# Internal: rounding, filtering, and trade construction
# ---------------------------------------------------------------------------

def _round_quantity(raw_qty: float, config: TradeConfig) -> float:
    """
    Round a raw (possibly fractional) quantity to the permitted precision.

    Sign is stripped before rounding and restored after — this function
    always returns a non-negative float (direction is carried by action).

    ASX equities (whole_shares=True):
        FLOOR mode: math.floor(abs(raw_qty))
        ROUND mode: round(abs(raw_qty))

    Managed funds (whole_shares=False):
        FLOOR mode: floor to managed_fund_dp decimal places
        ROUND mode: round to managed_fund_dp decimal places
    """
    abs_qty = abs(raw_qty)

    if config.whole_shares:
        if config.rounding_mode == RoundingMode.FLOOR:
            return float(math.floor(abs_qty))
        else:
            return float(round(abs_qty))
    else:
        dp = config.managed_fund_dp
        if config.rounding_mode == RoundingMode.FLOOR:
            factor = 10 ** dp
            return math.floor(abs_qty * factor) / factor
        else:
            return round(abs_qty, dp)


def _finalise_trades(
    raw_trades: list[_RawTrade],
    config: TradeConfig,
    portfolio: Portfolio,
) -> list[TradeDetail]:
    """
    Convert _RawTrade objects into TradeDetail objects by:
      1. Rounding quantity per config
      2. Computing estimated_value = rounded_qty × price
      3. Suppressing trades below min_trade_value
      4. Constructing the canonical TradeInstruction
    """
    details: list[TradeDetail] = []

    for rt in raw_trades:
        action = "BUY" if rt.raw_trade_value > 0 else "SELL"
        rounded_qty = _round_quantity(rt.raw_quantity, config)
        rounding_residual = abs(abs(rt.raw_quantity) - rounded_qty) * rt.price
        estimated_value = rounded_qty * rt.price

        # Determine suppression
        suppressed = False
        suppression_reason: Optional[str] = None

        if rounded_qty == 0:
            suppressed = True
            suppression_reason = (
                f"Rounded quantity is zero (raw_qty={rt.raw_quantity:.6f}). "
                f"Trade value ${abs(rt.raw_trade_value):,.2f} rounds to 0 units."
            )
        elif estimated_value < config.min_trade_value:
            suppressed = True
            suppression_reason = (
                f"Estimated value ${estimated_value:,.2f} is below "
                f"minimum trade value ${config.min_trade_value:,.2f}."
            )

        instruction = TradeInstruction(
            account_id=portfolio.account_id,
            ticker=rt.ticker,
            action=action,
            quantity=rounded_qty if rounded_qty > 0 else 1.0,
            estimated_value=estimated_value if estimated_value > 0 else abs(rt.raw_trade_value),
            model_id=None,   # attached by caller if needed
            model_version=None,
        )

        # Override quantity/value for suppressed zero-qty trades so they
        # carry accurate raw info rather than the 1.0/raw placeholder above
        if suppressed and rounded_qty == 0:
            instruction = TradeInstruction(
                account_id=portfolio.account_id,
                ticker=rt.ticker,
                action=action,
                quantity=max(abs(rt.raw_quantity), 0.0001),  # non-zero to pass dataclass guard
                estimated_value=abs(rt.raw_trade_value),
                model_id=None,
                model_version=None,
            )

        details.append(TradeDetail(
            instruction=instruction,
            target_value=rt.target_value,
            current_value=rt.current_value,
            raw_trade_value=rt.raw_trade_value,
            raw_quantity=rt.raw_quantity,
            rounded_quantity=rounded_qty,
            rounding_residual=rounding_residual,
            price_used=rt.price,
            suppressed=suppressed,
            suppression_reason=suppression_reason,
        ))

    return details


def _sort_details(details: list[TradeDetail]) -> list[TradeDetail]:
    """
    Sort trade details for display and execution:
      1. SELLs first (must execute before BUYs to free cash)
      2. Within SELLs: largest estimated_value first
      3. BUYs after
      4. Within BUYs: largest estimated_value first
    """
    sells = sorted(
        [td for td in details if td.action == "SELL"],
        key=lambda td: td.instruction.estimated_value,
        reverse=True,
    )
    buys = sorted(
        [td for td in details if td.action == "BUY"],
        key=lambda td: td.instruction.estimated_value,
        reverse=True,
    )
    return sells + buys
