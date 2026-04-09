"""
app.py — Streamlit UI for the portfolio rebalancing engine.

Run:
    streamlit run app.py

Features:
    1. Upload portfolio CSV (or load sample data)
    2. Input model weights via editable table
    3. Display per-account drift results with status badges
    4. Generate trade instructions
    5. Download trades as CSV
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drift import DEFAULT_DRIFT_THRESHOLD, calculate_drift
from exports import export_trades_csv
from models import ModelHolding, ModelPortfolio, Portfolio, SecurityHolding
from portfolio import load_portfolios
from trades import TradeConfig, generate_trades

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rebalancing Engine",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Minimal custom CSS — clean monochrome, no gradients
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Tighten sidebar */
    section[data-testid="stSidebar"] { min-width: 300px; max-width: 340px; }

    /* Status badge helpers used inline via st.markdown */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-over  { background: #fff3cd; color: #7d5a00; }
    .badge-under { background: #cfe2ff; color: #084298; }
    .badge-model { background: #f8d7da; color: #842029; }
    .badge-band  { background: #d1e7dd; color: #0a3622; }

    /* Reduce top padding on metric labels */
    div[data-testid="metric-container"] { padding-top: 0.4rem; }

    /* Slightly soften table header */
    thead tr th { background-color: #f8f9fa !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "portfolios": [],           # list[Portfolio]
        "model": None,              # ModelPortfolio | None
        "drift_reports": [],        # list[DriftReport]
        "trade_results": [],        # list[TradeResult]
        "active_account": None,     # str | None  (which account tab is active)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
account_id,ticker,quantity,price,cash_balance
HIN001,CBA,420.0000,121.5000,5000.0000
HIN001,BHP,180.0000,46.2000,5000.0000
HIN001,CSL,55.2500,289.0000,5000.0000
HIN001,WES,90.0000,68.5000,5000.0000
HIN001,ANZ,200.0000,29.4000,5000.0000
HIN002,CBA,600.0000,121.5000,8500.0000
HIN002,BHP,300.0000,46.2000,8500.0000
HIN002,CSL,90.7500,289.0000,8500.0000
HIN002,WES,220.0000,68.5000,8500.0000
HIN002,ANZ,410.0000,29.4000,8500.0000
HIN002,WBC,350.0000,29.1000,8500.0000
HIN003,CBA,200.0000,121.5000,2000.0000
HIN003,BHP,150.3300,46.2000,2000.0000
HIN003,CSL,0.0000,289.0000,2000.0000
HIN003,WES,0.0000,68.5000,2000.0000
HIN003,ANZ,0.0000,29.4000,2000.0000
"""

SAMPLE_WEIGHTS = [
    ("CBA", 25.0),
    ("BHP", 20.0),
    ("CSL", 15.0),
    ("WES", 15.0),
    ("ANZ", 12.0),
    ("MQG", 8.0),
    ("FMG", 5.0),
]

# ---------------------------------------------------------------------------
# Sidebar — Step 1: Upload portfolio
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚖ Rebalancing Engine")
    st.caption("Model portfolio drift & trade generation")
    st.divider()

    # ── Step 1 ──────────────────────────────────────────────────────────
    st.subheader("1 · Portfolio")

    use_sample = st.checkbox("Load sample data", value=False)

    if use_sample:
        if st.button("Load sample portfolio", use_container_width=True):
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as f:
                f.write(SAMPLE_CSV)
                tmp_path = f.name
            try:
                portfolios = load_portfolios(tmp_path)
                st.session_state.portfolios = portfolios
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Loaded {len(portfolios)} accounts.")
            except Exception as exc:
                st.error(str(exc))
            finally:
                os.unlink(tmp_path)
    else:
        uploaded = st.file_uploader(
            "Upload holdings CSV",
            type=["csv"],
            help="Required columns: account_id, ticker, quantity, price, cash_balance",
        )
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".csv", delete=False
            ) as f:
                f.write(uploaded.read())
                tmp_path = f.name
            try:
                portfolios = load_portfolios(tmp_path)
                st.session_state.portfolios = portfolios
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Loaded {len(portfolios)} account(s).")
            except ValueError as exc:
                st.error(f"Validation error:\n\n{exc}")
            finally:
                os.unlink(tmp_path)

    # CSV format reference
    with st.expander("CSV format"):
        st.code(
            "account_id,ticker,quantity,price,cash_balance\n"
            "HIN001,CBA,150.0000,121.5000,5000.0000\n"
            "HIN001,BHP,200.2500,46.2000,5000.0000",
            language="text",
        )

    st.divider()

    # ── Step 2 — Model weights ───────────────────────────────────────────
    st.subheader("2 · Model portfolio")

    model_id = st.text_input("Model ID", value="GROWTH_70")
    model_version = st.text_input("Version", value="2024-Q2")

    st.caption("Enter target weights (must sum to 100%)")

    # Editable weight table
    default_df = pd.DataFrame(SAMPLE_WEIGHTS, columns=["Ticker", "Weight"])
    weight_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Weight": st.column_config.NumberColumn(
                "Weight (%)",
                min_value=0.01,
                max_value=100.0,
                step=0.01,
                format="%.4f",
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="weight_editor",
    )

    # Weight sum indicator
    weight_sum = weight_df["Weight"].sum() if not weight_df.empty else 0.0
    delta = weight_sum - 100.0
    if abs(delta) < 0.05:
        st.success(f"Weights sum: {weight_sum:.2f}% ✓")
    else:
        st.warning(f"Weights sum: {weight_sum:.2f}%  (need 100%,  {delta:+.2f}%)")

    if st.button("Save model", use_container_width=True, type="primary"):
        try:
            holdings = [
                ModelHolding(
                    ticker=str(row["Ticker"]).strip().upper(),
                    target_weight=float(row["Weight"]) / 100.0,
                )
                for _, row in weight_df.iterrows()
                if str(row["Ticker"]).strip()
            ]
            model = ModelPortfolio(
                model_id=model_id.strip(),
                version=model_version.strip(),
                holdings=holdings,
            )
            st.session_state.model = model
            st.session_state.drift_reports = []
            st.session_state.trade_results = []
            st.success(f"Model '{model_id}' saved.")
        except ValueError as exc:
            st.error(str(exc))

    st.divider()

    # ── Step 3 — Run settings ────────────────────────────────────────────
    st.subheader("3 · Run settings")

    drift_threshold = st.slider(
        "Drift threshold (%)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Flag holdings deviating by more than this many percentage points",
    ) / 100.0

    min_trade_value = st.number_input(
        "Min trade value (AUD)",
        min_value=0.0,
        value=500.0,
        step=100.0,
        help="Suppress trades smaller than this amount",
    )

    fractional_shares = st.checkbox(
        "Allow fractional shares",
        value=True,
        help="Enable fractional unit trading (e.g. managed funds, ETFs)",
    )

    if fractional_shares:
        decimal_places = st.number_input(
            "Decimal places for quantities",
            min_value=1,
            max_value=6,
            value=4,
            step=1,
            help="Number of decimal places for fractional quantities",
        )
    else:
        decimal_places = 0

    ready = bool(st.session_state.portfolios and st.session_state.model)
    run_btn = st.button(
        "▶  Run rebalance",
        use_container_width=True,
        type="primary",
        disabled=not ready,
    )
    if not ready:
        if not st.session_state.portfolios:
            st.caption("⚠ Upload a portfolio first.")
        if not st.session_state.model:
            st.caption("⚠ Save a model first.")

# ---------------------------------------------------------------------------
# Run pipeline when button clicked
# ---------------------------------------------------------------------------

if run_btn and ready:
    model = st.session_state.model
    config = TradeConfig(
        drift_threshold=drift_threshold,
        min_trade_value=min_trade_value,
        whole_shares=not fractional_shares,
        managed_fund_dp=int(decimal_places) if fractional_shares else 0,
    )
    drift_reports = []
    trade_results = []

    for portfolio in st.session_state.portfolios:
        try:
            dr = calculate_drift(portfolio, model, threshold=drift_threshold)
            drift_reports.append(dr)
        except ValueError as exc:
            st.error(f"{portfolio.account_id}: drift error — {exc}")
            continue

        try:
            tr = generate_trades(portfolio, model, config)
            trade_results.append(tr)
        except ValueError as exc:
            st.error(f"{portfolio.account_id}: trade error — {exc}")

    st.session_state.drift_reports = drift_reports
    st.session_state.trade_results = trade_results
    if drift_reports:
        st.session_state.active_account = drift_reports[0].account_id

# ---------------------------------------------------------------------------
# Main panel — only shown once pipeline has run
# ---------------------------------------------------------------------------

if not st.session_state.drift_reports:
    # Welcome / instructions
    st.markdown("## Portfolio Rebalancing Engine")
    st.markdown(
        "Upload a holdings CSV, configure target weights, then click "
        "**▶ Run rebalance** to see drift and trades."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nUpload a portfolio CSV or load sample data from the sidebar.")
    with col2:
        st.info("**Step 2**\nEnter target weights in the model table. Weights must sum to 100%.")
    with col3:
        st.info("**Step 3**\nSet your drift threshold and minimum trade value, then run.")

    if not st.session_state.portfolios:
        st.markdown("---")
        st.caption("Sample CSV format:")
        st.code(SAMPLE_CSV, language="text")
    st.stop()


# ---------------------------------------------------------------------------
# Summary bar
# ---------------------------------------------------------------------------

drift_reports = st.session_state.drift_reports
trade_results  = st.session_state.trade_results
model          = st.session_state.model

accounts_requiring = sum(1 for dr in drift_reports if dr.requires_rebalance)
total_flagged      = sum(dr.flagged_count for dr in drift_reports)
total_trades       = sum(len(tr.trades) for tr in trade_results)
all_trades         = [t for tr in trade_results for t in tr.trades]
gross_buy          = sum(t.estimated_value for t in all_trades if t.action == "BUY")
gross_sell         = sum(t.estimated_value for t in all_trades if t.action == "SELL")

st.markdown(f"### {model.model_id}  ·  {model.version}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accounts", len(drift_reports))
c2.metric("Need rebalance", accounts_requiring)
c3.metric("Flagged holdings", total_flagged)
c4.metric("Total trades", total_trades)
c5.metric("Net cash flow", f"${gross_sell - gross_buy:+,.4f}")

st.divider()

# ---------------------------------------------------------------------------
# Account tabs
# ---------------------------------------------------------------------------

account_ids = [dr.account_id for dr in drift_reports]
tabs = st.tabs([f"  {aid}  " for aid in account_ids])

for tab, drift_report in zip(tabs, drift_reports):
    with tab:
        trade_result = next(
            (tr for tr in trade_results if tr.account_id == drift_report.account_id),
            None,
        )

        # ── Account summary row ──────────────────────────────────────────
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Portfolio value", f"${drift_report.total_portfolio_value:,.4f}")
        col_b.metric("Holdings", len(drift_report.holdings))
        col_c.metric("Flagged", drift_report.flagged_count)
        if trade_result:
            shortfall_label = "⚠ Shortfall" if trade_result.has_funding_shortfall else "Closing cash"
            col_d.metric(shortfall_label, f"${trade_result.closing_cash:,.4f}")

        st.divider()

        # ── Drift table ──────────────────────────────────────────────────
        st.markdown("#### Drift analysis")

        STATUS_BADGE = {
            "OVERWEIGHT":   '<span class="badge badge-over">OVER</span>',
            "UNDERWEIGHT":  '<span class="badge badge-under">UNDER</span>',
            "NOT_IN_MODEL": '<span class="badge badge-model">DIVEST</span>',
            "IN_BAND":      '<span class="badge badge-band">OK</span>',
        }

        drift_rows = []
        for h in drift_report.holdings:
            drift_rows.append({
                "Ticker":         h.ticker,
                "Status":         h.status.value,
                "Current (%)":    round(h.current_weight * 100, 4),
                "Target (%)":     round(h.target_weight * 100, 4),
                "Drift (pp)":     round(h.drift * 100, 4),
                "Market Value":   round(h.market_value, 4),
                "Flag":           h.exceeds_threshold,
            })

        drift_df = pd.DataFrame(drift_rows)

        # Colour-code the drift column
        def _colour_drift(val: float) -> str:
            if val > 0:
                return "background-color: #fff8e1; color: #5d4200"
            elif val < 0:
                return "background-color: #e8f0fe; color: #1a237e"
            return ""

        styled = (
            drift_df
            .style
            .map(_colour_drift, subset=["Drift (pp)"])
            .format({
                "Current (%)":  "{:.4f}%",
                "Target (%)":   "{:.4f}%",
                "Drift (pp)":   "{:+.4f}pp",
                "Market Value": "${:,.4f}",
            })
            .hide(axis="index")
        )

        st.dataframe(styled, use_container_width=True, height=280)

        if drift_report.requires_rebalance:
            st.caption(
                f"⚑  {drift_report.flagged_count} holding(s) exceed the "
                f"{drift_threshold*100:.1f}pp threshold."
            )
        else:
            st.caption("✓  All holdings are within the drift band. No rebalance required.")

        st.divider()

        # ── Trade table ──────────────────────────────────────────────────
        if trade_result is None:
            st.info("No trade result available for this account.")
        elif not trade_result.trades and not trade_result.suppressed_trades:
            st.success("No trades required — portfolio is within tolerance.")
        else:
            st.markdown("#### Trade instructions")

            # Cash flow summary
            cf1, cf2, cf3, cf4 = st.columns(4)
            cf1.metric("Opening cash",  f"${trade_result.opening_cash:,.4f}")
            cf2.metric("Sell proceeds", f"${trade_result.sell_proceeds:,.4f}")
            cf3.metric("Buy cost",      f"${trade_result.buy_cost:,.4f}")
            cf4.metric(
                "Closing cash" if not trade_result.has_funding_shortfall else "⚠ Shortfall",
                f"${trade_result.closing_cash:,.4f}",
            )

            if trade_result.has_funding_shortfall:
                st.warning(
                    f"Funding shortfall of ${abs(trade_result.closing_cash):,.4f}. "
                    "Buy cost exceeds available cash. Adviser review required."
                )

            if trade_result.trades:
                trade_rows = [
                    {
                        "Action":     t.action,
                        "Ticker":     t.ticker,
                        "Quantity":   round(t.quantity, 4),
                        "Est. Value": round(t.estimated_value, 4),
                    }
                    for t in trade_result.trades
                ]
                trade_df = pd.DataFrame(trade_rows)

                def _colour_action(val: str) -> str:
                    if val == "BUY":
                        return "color: #155724; font-weight: 600"
                    if val == "SELL":
                        return "color: #721c24; font-weight: 600"
                    return ""

                styled_trades = (
                    trade_df
                    .style
                    .map(_colour_action, subset=["Action"])
                    .format({
                        "Quantity":   "{:.4f}",
                        "Est. Value": "${:,.4f}",
                    })
                    .hide(axis="index")
                )
                st.dataframe(styled_trades, use_container_width=True, height=220)

            if trade_result.suppressed_trades:
                with st.expander(
                    f"Suppressed trades ({len(trade_result.suppressed_trades)})"
                ):
                    sup_rows = [
                        {
                            "Action":    td.action,
                            "Ticker":    td.ticker,
                            "Raw value": round(td.raw_trade_value, 4),
                            "Reason":    td.suppression_reason or "",
                        }
                        for td in trade_result.suppressed_trades
                    ]
                    st.dataframe(
                        pd.DataFrame(sup_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

# ---------------------------------------------------------------------------
# Download section — all trades as CSV
# ---------------------------------------------------------------------------

st.divider()
st.markdown("#### Download")

if all_trades:
    col_dl1, col_dl2 = st.columns([2, 5])

    with col_dl1:
        # Write to in-memory bytes via a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
       
