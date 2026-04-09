"""
drift.py — Drift calculation between a client portfolio and a model portfolio.

Public API:
    calculate_drift(
        portfolio:  Portfolio,
        model:      ModelPortfolio,
        threshold:  float = DEFAULT_DRIFT_THRESHOLD,
    ) -> DriftReport

Calculation pipeline:
    1.  Guard: reject zero-value portfolios (nothing to rebalance)
    2.  Compute actual weight per holding  = market_value / total_portfolio_value
    3.  Build a unified security universe: union of portfolio tickers and model tickers
    4.  For every ticker in the universe, compute:
            current_weight  — actual weight (0.0 if not held)
            target_weight   — model weight  (0.0 if not in model → must be sold)
            drift           — current_weight - target_weight
            abs_drift       — abs(drift)
            exceeds_threshold — abs_drift > threshold
    5.  Classify each position:
            OVERWEIGHT      drift >  +threshold
            UNDERWEIGHT     drift <  -threshold
            IN_BAND         abs_drift <= threshold
            NOT_IN_MODEL    not in model  (target = 0, must be sold; always flags)
    6.  Attach summary statistics to the report

No trade logic lives here. This module only reads weights and returns
observations — what to *do* about drift is trade.py's responsibility.

Threshold semantics:
    threshold is an absolute weight difference expressed as a decimal.
    threshold=0.03 means "flag if actual weight deviates by more than 3 percentage
    points from target". This is the most common convention in Australian wrap
    platforms (e.g. HUB24, BT Panorama, Netwealth).

    Relative drift (drift / target_weight) is also computed and included in each
    HoldingDrift record for informational purposes, but threshold breaches are
    evaluated on absolute drift only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from models import ModelPortfolio, Portfolio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default absolute drift threshold (3 percentage points).
#: Override per-call via the `threshold` argument of calculate_drift().
DEFAULT_DRIFT_THRESHOLD: float = 0.03


# ---------------------------------------------------------------------------
# Enums and output types
# ---------------------------------------------------------------------------


class DriftStatus(str, Enum):
    """
    Classification of a single security's drift relative to its target weight.

    String enum so that DriftStatus values serialise cleanly to JSON / CSV
    without extra conversion.
    """

    OVERWEIGHT = "OVERWEIGHT"
    """Current weight exceeds target by more than the threshold."""

    UNDERWEIGHT = "UNDERWEIGHT"
    """Current weight is below target by more than the threshold."""

    IN_BAND = "IN_BAND"
    """Drift is within the acceptable threshold — no action required."""

    NOT_IN_MODEL = "NOT_IN_MODEL"
    """
    Security is held in the portfolio but has no target weight in the model.
    Always flagged regardless of threshold: it must be sold to zero.
    """


@dataclass
class HoldingDrift:
    """
    Drift observation for a single security within a portfolio.

    All weight fields are expressed as decimal fractions (0.0 – 1.0).
    Percentage representations are available via convenience properties.

    Attributes:
        ticker:              ASX ticker code.
        current_weight:      Actual portfolio weight at time of calculation.
        target_weight:       Model target weight (0.0 if not in model).
        drift:               current_weight - target_weight.
                             Positive → overweight; negative → underweight.
        abs_drift:           abs(drift). Compared against threshold.
        relative_drift:      drift / target_weight when target > 0, else None.
                             Informational only; threshold is applied to abs_drift.
        status:              DriftStatus classification.
        exceeds_threshold:   True when abs_drift > threshold OR status is NOT_IN_MODEL.
        market_value:        Current AUD market value of the holding (0.0 if not held).
        threshold_used:      The threshold value used to classify this holding.
                             Stored for auditability.
    """

    ticker: str
    current_weight: float
    target_weight: float
    drift: float
    abs_drift: float
    relative_drift: Optional[float]
    status: DriftStatus
    exceeds_threshold: bool
    market_value: float
    threshold_used: float

    # ------------------------------------------------------------------
    # Convenience percentage properties (for display / reporting)
    # ------------------------------------------------------------------

    @property
    def current_weight_pct(self) -> float:
        """current_weight expressed as a percentage (e.g. 0.12 → 12.0)."""
        return self.current_weight * 100

    @property
    def target_weight_pct(self) -> float:
        """target_weight expressed as a percentage."""
        return self.target_weight * 100

    @property
    def drift_pct(self) -> float:
        """drift expressed as a percentage (e.g. 0.04 → 4.0)."""
        return self.drift * 100

    @property
    def abs_drift_pct(self) -> float:
        """abs_drift expressed as a percentage."""
        return self.abs_drift * 100

    def __repr__(self) -> str:
        flag = "⚑ " if self.exceeds_threshold else "  "
        return (
            f"{flag}HoldingDrift({self.ticker}: "
            f"current={self.current_weight_pct:.2f}%, "
            f"target={self.target_weight_pct:.2f}%, "
            f"drift={self.drift_pct:+.2f}%, "
            f"status={self.status.value})"
        )


@dataclass
class DriftReport:
    """
    Complete drift analysis for one client portfolio against one model.

    Attributes:
        account_id:           The portfolio's HIN / account identifier.
        model_id:             The model portfolio's identifier.
        model_version:        The model portfolio's version string.
        total_portfolio_value: Total AUD value used as the denominator for weights.
        threshold:            The absolute drift threshold used for this report.
        holdings:             One HoldingDrift record per security in the universe.
                              Sorted: breaching holdings first, then by ticker.
        flagged_count:        Number of holdings exceeding the threshold.
        requires_rebalance:   True if any holding exceeds the threshold.
    """

    account_id: str
    model_id: str
    model_version: str
    total_portfolio_value: float
    threshold: float
    holdings: list[HoldingDrift] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived summary properties
    # ------------------------------------------------------------------

    @property
    def flagged_count(self) -> int:
        """Number of holdings that exceed the drift threshold."""
        return sum(1 for h in self.holdings if h.exceeds_threshold)

    @property
    def requires_rebalance(self) -> bool:
        """True if at least one holding exceeds the drift threshold."""
        return self.flagged_count > 0

    @property
    def flagged_holdings(self) -> list[HoldingDrift]:
        """Subset of holdings that exceed the threshold, sorted by abs_drift desc."""
        return sorted(
            [h for h in self.holdings if h.exceeds_threshold],
            key=lambda h: h.abs_drift,
            reverse=True,
        )

    @property
    def in_band_holdings(self) -> list[HoldingDrift]:
        """Subset of holdings that are within the acceptable band."""
        return [h for h in self.holdings if not h.exceeds_threshold]

    def to_dict(self) -> dict:
        """
        Serialise the report to the canonical dict format consumed by the
        API layer, Streamlit UI, and CSV exporters.

        Structure:
            {
                "account_id": str,
                "model_id": str,
                "model_version": str,
                "total_portfolio_value": float,
                "threshold": float,
                "requires_rebalance": bool,
                "flagged_count": int,
                "holdings": [
                    {
                        "ticker": str,
                        "current_weight": float,
                        "target_weight": float,
                        "drift": float,
                        "abs_drift": float,
                        "relative_drift": float | None,
                        "status": str,
                        "exceeds_threshold": bool,
                        "market_value": float,
                    },
                    ...
                ]
            }
        """
        return {
            "account_id": self.account_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "total_portfolio_value": round(self.total_portfolio_value, 2),
            "threshold": self.threshold,
            "requires_rebalance": self.requires_rebalance,
            "flagged_count": self.flagged_count,
            "holdings": [
                {
                    "ticker": h.ticker,
                    "current_weight": round(h.current_weight, 6),
                    "target_weight": round(h.target_weight, 6),
                    "drift": round(h.drift, 6),
                    "abs_drift": round(h.abs_drift, 6),
                    "relative_drift": (
                        round(h.relative_drift, 6)
                        if h.relative_drift is not None
                        else None
                    ),
                    "status": h.status.value,
                    "exceeds_threshold": h.exceeds_threshold,
                    "market_value": round(h.market_value, 2),
                }
                for h in self.holdings
            ],
        }

    def __repr__(self) -> str:
        return (
            f"DriftReport(account={self.account_id!r}, "
            f"model={self.model_id!r}@{self.model_version!r}, "
            f"total=${self.total_portfolio_value:,.2f}, "
            f"threshold={self.threshold:.1%}, "
            f"flagged={self.flagged_count}/{len(self.holdings)}, "
            f"requires_rebalance={self.requires_rebalance})"
        )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def calculate_drift(
    portfolio: Portfolio,
    model: ModelPortfolio,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> DriftReport:
    """
    Calculate the drift between a client portfolio and a model portfolio.

    Args:
        portfolio:  The client's current holdings. Must have total_value() > 0.
        model:      The target model portfolio with validated weights summing to 1.0.
        threshold:  Absolute drift threshold as a decimal fraction.
                    Default is 0.03 (3 percentage points).
                    Holdings with abs(drift) > threshold are flagged.

    Returns:
        DriftReport containing per-holding drift observations and summary stats.

    Raises:
        ValueError: If portfolio has zero or negative total value (unresolvable
                    — weights cannot be computed against a zero denominator).
        ValueError: If threshold is not in the range (0, 1).
    """
    _validate_inputs(portfolio, model, threshold)

    total_value = portfolio.total_value()

    # Step 1: Build actual weight map  { ticker → current_weight }
    actual_weights = _compute_actual_weights(portfolio, total_value)

    # Step 2: Build unified security universe (portfolio ∪ model tickers)
    universe = _build_universe(portfolio, model)

    # Step 3: Compute drift for every ticker in the universe
    holding_drifts = []
    for ticker in sorted(universe):
        hd = _compute_holding_drift(
            ticker=ticker,
            actual_weights=actual_weights,
            model=model,
            portfolio=portfolio,
            threshold=threshold,
        )
        holding_drifts.append(hd)

    # Step 4: Sort — flagged first (by abs_drift desc), then in-band by ticker
    holding_drifts = _sort_holdings(holding_drifts)

    report = DriftReport(
        account_id=portfolio.account_id,
        model_id=model.model_id,
        model_version=model.version,
        total_portfolio_value=total_value,
        threshold=threshold,
        holdings=holding_drifts,
    )

    logger.info(
        "Drift calculated — account=%s model=%s@%s total=$%.2f "
        "securities=%d flagged=%d threshold=%.1f%%",
        portfolio.account_id,
        model.model_id,
        model.version,
        total_value,
        len(holding_drifts),
        report.flagged_count,
        threshold * 100,
    )

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    portfolio: Portfolio,
    model: ModelPortfolio,
    threshold: float,
) -> None:
    total_value = portfolio.total_value()
    if total_value <= 0:
        raise ValueError(
            f"Portfolio '{portfolio.account_id}' has total value "
            f"${total_value:,.2f} — cannot compute weights against a "
            f"zero or negative denominator."
        )
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"threshold must be in (0, 1), got {threshold}. "
            f"Example: 0.03 means flag drift greater than 3 percentage points."
        )


def _compute_actual_weights(
    portfolio: Portfolio,
    total_value: float,
) -> dict[str, float]:
    """
    Compute the actual portfolio weight for each holding.

    Returns:
        Dict mapping ticker → weight (decimal fraction).
        Cash is excluded — weights represent invested securities only
        from the perspective of security-level drift.
        The cash weight is implicitly (1 - sum(security_weights)).
    """
    return {
        h.ticker: h.market_value / total_value
        for h in portfolio.holdings
    }


def _build_universe(
    portfolio: Portfolio,
    model: ModelPortfolio,
) -> set[str]:
    """
    Build the full security universe as the union of:
      - tickers currently held in the portfolio
      - tickers targeted by the model

    Securities only in the portfolio (not in model) will have target_weight=0.0
    and will be classified as NOT_IN_MODEL.

    Securities only in the model (not in portfolio) will have current_weight=0.0
    and will be classified as UNDERWEIGHT.
    """
    portfolio_tickers = {h.ticker for h in portfolio.holdings}
    model_tickers = {h.ticker for h in model.holdings}
    return portfolio_tickers | model_tickers


def _compute_holding_drift(
    ticker: str,
    actual_weights: dict[str, float],
    model: ModelPortfolio,
    portfolio: Portfolio,
    threshold: float,
) -> HoldingDrift:
    """
    Compute the complete drift observation for a single ticker.

    The `model.target_weight(ticker)` method returns 0.0 for tickers not in
    the model — this is intentional and drives the NOT_IN_MODEL classification.
    """
    current_weight: float = actual_weights.get(ticker, 0.0)
    target_weight: float = model.target_weight(ticker)

    drift: float = current_weight - target_weight
    abs_drift: float = abs(drift)

    # Relative drift: only meaningful when there is a non-zero target
    relative_drift: Optional[float] = (
        drift / target_weight if target_weight > 0.0 else None
    )

    # Classification
    in_model = target_weight > 0.0

    if not in_model:
        # Held but not in the model — always flag for full divestment
        status = DriftStatus.NOT_IN_MODEL
        exceeds_threshold = True
    elif abs_drift > threshold:
        status = DriftStatus.OVERWEIGHT if drift > 0 else DriftStatus.UNDERWEIGHT
        exceeds_threshold = True
    else:
        status = DriftStatus.IN_BAND
        exceeds_threshold = False

    market_value = portfolio.holding_value(ticker)

    return HoldingDrift(
        ticker=ticker,
        current_weight=current_weight,
        target_weight=target_weight,
        drift=drift,
        abs_drift=abs_drift,
        relative_drift=relative_drift,
        status=status,
        exceeds_threshold=exceeds_threshold,
        market_value=market_value,
        threshold_used=threshold,
    )


def _sort_holdings(holdings: list[HoldingDrift]) -> list[HoldingDrift]:
    """
    Sort holdings for display:
      1. Flagged holdings first, sorted by abs_drift descending (worst first)
      2. In-band holdings after, sorted alphabetically by ticker

    This ordering means the most urgent rebalancing actions appear at the top
    of any report or UI table.
    """
    flagged = sorted(
        [h for h in holdings if h.exceeds_threshold],
        key=lambda h: h.abs_drift,
        reverse=True,
    )
    in_band = sorted(
        [h for h in holdings if not h.exceeds_threshold],
        key=lambda h: h.ticker,
    )
    return flagged + in_band
