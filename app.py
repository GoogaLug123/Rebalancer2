"""
app.py — Portfolio Rebalancing Engine
Focused rebalancing tool with optional client profiling.

Tab structure:
    1. Rebalance      — Core workflow: model → portfolio → drift → trades → download
    2. Models         — Create and manage saved model portfolios
    3. Client Profile — Optional risk questionnaire
    4. Settings       — Admin: weighting config, change log, lock
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drift import DEFAULT_DRIFT_THRESHOLD, calculate_drift
from exports import export_trades_csv
from models import ModelHolding, ModelPortfolio, Portfolio, SecurityHolding
from portfolio import load_portfolios
from trades import TradeConfig, generate_trades

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rebalancing Engine",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Typography ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 2px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-over  { background: #fff3cd; color: #7d5a00; border: 1px solid #ffc107; }
.badge-under { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
.badge-divest{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.badge-ok    { background: #dcfce7; color: #166534; border: 1px solid #86efac; }

/* ── Status bar ── */
.status-bar {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 1rem;
    display: flex;
    gap: 2rem;
    align-items: center;
    font-size: 0.82rem;
}
.status-item { display: flex; align-items: center; gap: 0.4rem; }
.status-done { color: #166534; font-weight: 600; }
.status-pending { color: #6b7280; }
.status-warn { color: #92400e; font-weight: 600; }

/* ── Step headers ── */
.step-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}
.step-num {
    background: #1e293b;
    color: white;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
}
.step-num-done {
    background: #166534;
    color: white;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-title { font-size: 1rem; font-weight: 600; color: #1e293b; }
.step-subtitle { font-size: 0.8rem; color: #6b7280; margin-top: 0; }

/* ── Conflict box ── */
.conflict-box {
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-left: 4px solid #f59e0b;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.875rem;
}

/* ── Risk scale ── */
.risk-scale {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.risk-seg-1 { background: #22c55e; flex: 1; }
.risk-seg-2 { background: #84cc16; flex: 1; }
.risk-seg-3 { background: #eab308; flex: 1; }
.risk-seg-4 { background: #f97316; flex: 1; }
.risk-seg-5 { background: #ef4444; flex: 1; }
.risk-marker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #6b7280;
    margin-top: 0.2rem;
}

/* ── Metric tweaks ── */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.75rem 1rem !important;
}
div[data-testid="metric-container"] label {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Table tweaks ── */
thead tr th {
    background-color: #f1f5f9 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #475569 !important;
}
tbody tr td { font-size: 0.875rem !important; }

/* ── Primary button ── */
div[data-testid="stButton"] button[kind="primary"] {
    background: #1e293b;
    border: none;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #334155;
}

/* ── Download button ── */
div[data-testid="stDownloadButton"] button {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
}

/* ── Sidebar gone ── */
section[data-testid="stSidebar"] { display: none; }

/* ── Tab style ── */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTIONNAIRE_VERSION = "1.0"

QUESTIONS = [
    {
        "id": "Q1", "section": "A", "label": "Time Horizon",
        "text": "When do you expect to start drawing on this investment?",
        "options": [
            ("Less than 2 years", 1),
            ("2–5 years", 2),
            ("5–10 years", 3),
            ("10–15 years", 4),
            ("More than 15 years", 5),
        ],
    },
    {
        "id": "Q2", "section": "B", "label": "Risk Tolerance",
        "text": "If your portfolio lost 20% of its value in a market downturn, what would you most likely do?",
        "options": [
            ("Sell everything immediately", 1),
            ("Sell some to reduce exposure", 2),
            ("Do nothing and wait", 3),
            ("Buy a little more while prices are low", 4),
            ("Significantly increase my investment", 5),
        ],
    },
    {
        "id": "Q3", "section": "B", "label": "Risk Tolerance",
        "text": "Which statement best describes your attitude toward investment risk?",
        "options": [
            ("I cannot accept any loss of capital", 1),
            ("I can accept small short-term losses for modest returns", 2),
            ("I can accept moderate losses for reasonable returns", 3),
            ("I can accept significant short-term losses for higher long-term returns", 4),
            ("I am comfortable with high volatility in pursuit of maximum returns", 5),
        ],
    },
    {
        "id": "Q4", "section": "C", "label": "Risk Capacity",
        "text": "If you lost this entire investment, how would it affect your lifestyle?",
        "options": [
            ("Severely — I depend on it", 1),
            ("Significantly — it would cause real hardship", 2),
            ("Moderately — I would need to adjust my plans", 3),
            ("Mildly — I have other assets to fall back on", 4),
            ("Minimally — this is discretionary money", 5),
        ],
    },
    {
        "id": "Q5", "section": "D", "label": "Income Stability",
        "text": "How would you describe your current and expected future income?",
        "options": [
            ("Unstable or uncertain", 1),
            ("Variable with some uncertainty", 2),
            ("Stable but could change", 3),
            ("Stable and likely to continue", 4),
            ("Very stable or I have multiple income sources", 5),
        ],
    },
    {
        "id": "Q6", "section": "E", "label": "Investment Experience",
        "text": "Which best describes your investment experience?",
        "options": [
            ("None — I have never invested before", 1),
            ("Limited — I have a savings account or term deposit", 2),
            ("Moderate — I have invested in managed funds or super", 3),
            ("Experienced — I have invested in shares or ETFs directly", 4),
            ("Advanced — I actively manage a diversified investment portfolio", 5),
        ],
    },
    {
        "id": "Q7", "section": "F", "label": "Goals",
        "text": "What is the primary purpose of this investment?",
        "options": [
            ("Preserve my capital above all else", 1),
            ("Generate regular income", 2),
            ("Balanced mix of income and growth", 3),
            ("Grow my wealth over the long term", 4),
            ("Maximise long-term growth, I do not need income", 5),
        ],
        "flags": {1: "income"},
    },
    {
        "id": "Q8", "section": "F", "label": "ESG Preferences",
        "text": "Do you have any ethical or ESG investment preferences?",
        "options": [
            ("No preference", 0),
            ("Somewhat important", 0),
            ("Very important — I want to exclude certain industries", 0),
        ],
        "flags": {1: "esg", 2: "esg"},
        "no_score": True,
    },
]

SCORE_MAP = [
    (6,  10, "Conservative"),
    (11, 16, "Moderate / Balanced"),
    (17, 22, "Growth"),
    (23, 28, "High Growth"),
    (29, 30, "Aggressive"),
]

RISK_LEVEL = {
    "Conservative":       1,
    "Moderate / Balanced": 2,
    "Growth":             3,
    "High Growth":        4,
    "Aggressive":         5,
}

CONFLICT_RULES = [
    ("Q2", [0, 1], "Q7", [3, 4],
     "Low risk tolerance (Q2) conflicts with high growth goals (Q7)."),
    ("Q1", [0], "Q7", [3, 4],
     "Short time horizon (Q1) conflicts with growth-oriented goals (Q7)."),
    ("Q4", [0], None, None,
     "Client is financially dependent on this investment (Q4). Consider capping at Conservative."),
]

DEFAULT_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0}
WEIGHT_MIN = 0.5
WEIGHT_MAX = 2.0

SECTION_LABELS = {
    "A": "Time Horizon",
    "B": "Risk Tolerance",
    "C": "Risk Capacity",
    "D": "Income Stability",
    "E": "Investment Experience",
    "F": "Goals",
}

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

SAMPLE_MODEL_WEIGHTS = [
    ("CBA", 25.0), ("BHP", 20.0), ("CSL", 15.0),
    ("WES", 15.0), ("ANZ", 12.0), ("MQG", 8.0), ("FMG", 5.0),
]

# ---------------------------------------------------------------------------
# Predefined model portfolios
# All weights sum to 100%. ETF-based where appropriate for wrap platform use.
# ---------------------------------------------------------------------------

PREDEFINED_MODELS: list = [
    {
        "model_id": "CONSERVATIVE",
        "version":  "2024-Q2",
        "holdings": [
            ("AGG",  40.0),
            ("VAF",  20.0),
            ("VAS",  20.0),
            ("VGS",  10.0),
            ("VHY",  10.0),
        ],
    },
    {
        "model_id": "MODERATE",
        "version":  "2024-Q2",
        "holdings": [
            ("VAS",  25.0),
            ("VGS",  25.0),
            ("VAF",  25.0),
            ("AGG",  15.0),
            ("VHY",  10.0),
        ],
    },
    {
        "model_id": "GROWTH",
        "version":  "2024-Q2",
        "holdings": [
            ("VAS",  30.0),
            ("VGS",  25.0),
            ("VGE",  15.0),
            ("VAF",  20.0),
            ("VHY",  10.0),
        ],
    },
    {
        "model_id": "HIGH_GROWTH",
        "version":  "2024-Q2",
        "holdings": [
            ("VAS",  30.0),
            ("VGS",  30.0),
            ("VGE",  15.0),
            ("NDQ",  15.0),
            ("VHY",  10.0),
        ],
    },
    {
        "model_id": "AGGRESSIVE",
        "version":  "2024-Q2",
        "holdings": [
            ("VAS",  30.0),
            ("VGS",  30.0),
            ("NDQ",  20.0),
            ("VGE",  20.0),
        ],
    },
    {
        "model_id": "INCOME",
        "version":  "2024-Q2",
        "holdings": [
            ("VHY",  30.0),
            ("VAF",  25.0),
            ("AFI",  15.0),
            ("ARG",  15.0),
            ("VAP",  15.0),
        ],
    },
    {
        "model_id": "ESG",
        "version":  "2024-Q2",
        "holdings": [
            ("ETHI", 35.0),
            ("FAIR", 30.0),
            ("VESG", 20.0),
            ("VBND", 15.0),
        ],
    },
    {
        "model_id": "AUSTRALIAN_EQ",
        "version":  "2024-Q2",
        "holdings": [
            ("CBA",  25.0),
            ("BHP",  20.0),
            ("CSL",  15.0),
            ("WES",  15.0),
            ("ANZ",  12.0),
            ("MQG",   8.0),
            ("FMG",   5.0),
        ],
    },
    {
        "model_id": "INDEX_PASSIVE",
        "version":  "2024-Q2",
        "holdings": [
            ("VAS",  40.0),
            ("VGS",  40.0),
            ("VAF",  20.0),
        ],
    },
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "portfolios":      [],
        "model":           None,
        "saved_models":    {},
        "drift_reports":   [],
        "trade_results":   [],
        "q_answers":       {},
        "q_result":        None,
        "q_client_name":   "",
        "q_step":          0,
        "weights":         dict(DEFAULT_WEIGHTS),
        "weight_log":      [],
        "weights_locked":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

if not st.session_state.saved_models:
    for _pm in PREDEFINED_MODELS:
        try:
            _h = [ModelHolding(ticker=t, target_weight=w / 100.0)
                  for t, w in _pm["holdings"]]
            _m = ModelPortfolio(
                model_id=_pm["model_id"],
                version=_pm["version"],
                holdings=_h,
            )
            _label = f"{_pm['model_id']} ({_pm['version']})"
            st.session_state.saved_models[_label] = _m
        except Exception:
            pass
    # Set GROWTH as the default active model
    _default_key = "GROWTH (2024-Q2)"
    if _default_key in st.session_state.saved_models:
        st.session_state.model = st.session_state.saved_models[_default_key]
    elif st.session_state.saved_models:
        st.session_state.model = list(st.session_state.saved_models.values())[0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _step_header(num: str, title: str, subtitle: str = "", done: bool = False) -> None:
    icon = f'<span class="step-num-done">✓</span>' if done else f'<span class="step-num">{num}</span>'
    st.markdown(
        f'<div class="step-header">'
        f'{icon}'
        f'<div>'
        f'<div class="step-title">{title}</div>'
        f'{"<div class=step-subtitle>" + subtitle + "</div>" if subtitle else ""}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _status_bar() -> None:
    model = st.session_state.model
    portfolios = st.session_state.portfolios
    drift_reports = st.session_state.drift_reports
    trade_results = st.session_state.trade_results

    m_str = (f'<span class="status-done">✓ {model.model_id} v{model.version}</span>'
             if model else '<span class="status-pending">No model selected</span>')
    p_str = (f'<span class="status-done">✓ {len(portfolios)} account(s)</span>'
             if portfolios else '<span class="status-pending">No portfolio loaded</span>')

    if drift_reports:
        needs = sum(1 for dr in drift_reports if dr.requires_rebalance)
        d_str = (f'<span class="status-warn">⚑ {needs} account(s) need rebalancing</span>'
                 if needs else '<span class="status-done">✓ All accounts in band</span>')
    else:
        d_str = '<span class="status-pending">Not run</span>'

    all_trades = [t for tr in trade_results for t in tr.trades]
    t_str = (f'<span class="status-done">✓ {len(all_trades)} trade(s) generated</span>'
             if all_trades else '<span class="status-pending">No trades generated</span>')

    st.markdown(
        f'<div class="status-bar">'
        f'<div class="status-item"><b>Model:</b>&nbsp;{m_str}</div>'
        f'<div class="status-item"><b>Portfolio:</b>&nbsp;{p_str}</div>'
        f'<div class="status-item"><b>Drift:</b>&nbsp;{d_str}</div>'
        f'<div class="status-item"><b>Trades:</b>&nbsp;{t_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _compute_score(answers: dict, weights: dict) -> tuple:
    section_raw: dict = {}
    flags: set = set()
    for q in QUESTIONS:
        qid = q["id"]
        if qid not in answers:
            continue
        idx = answers[qid]
        pts = q["options"][idx][1]
        if not q.get("no_score", False):
            section_raw.setdefault(q["section"], []).append(pts)
        if "flags" in q and idx in q["flags"]:
            flags.add(q["flags"][idx])

    total_w = weighted_sum = 0.0
    for sec, pts_list in section_raw.items():
        avg = sum(pts_list) / len(pts_list)
        w = weights.get(sec, 1.0)
        weighted_sum += avg * w
        total_w += w

    score = round((weighted_sum / total_w) * 6, 2) if total_w else 0.0

    conflicts = []
    for q1_id, q1_idxs, q2_id, q2_idxs, msg in CONFLICT_RULES:
        q1_ans = answers.get(q1_id)
        if q1_ans is None or q1_ans not in q1_idxs:
            continue
        if q2_id is None:
            conflicts.append(msg)
            continue
        q2_ans = answers.get(q2_id)
        if q2_ans is not None and q2_ans in q2_idxs:
            conflicts.append(msg)

    return score, flags, conflicts


def _score_to_model(score: float, answers: dict) -> str:
    base = "Conservative"
    for lo, hi, name in SCORE_MAP:
        if lo <= score <= hi:
            base = name
            break
    order = ["Conservative", "Moderate / Balanced", "Growth", "High Growth", "Aggressive"]
    q1 = answers.get("Q1", 4)
    if q1 == 0:
        base = "Conservative"
    elif q1 == 1:
        base = order[min(order.index(base), 1)]
    if answers.get("Q4", 4) == 0:
        base = order[max(order.index(base) - 1, 0)]
    return base


def _risk_scale_html(level: int) -> str:
    labels = ["Conservative", "Moderate", "Growth", "High Growth", "Aggressive"]
    colours = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
    segs = ""
    for i in range(1, 6):
        border = "3px solid #1e293b" if i == level else "none"
        segs += (
            f'<div style="flex:1;background:{colours[i-1]};'
            f'outline:{border};outline-offset:-2px;"></div>'
        )
    label = labels[level - 1] if 1 <= level <= 5 else ""
    return (
        f'<div style="display:flex;height:10px;border-radius:4px;overflow:hidden;'
        f'margin:0.4rem 0;">{segs}</div>'
        f'<div style="font-size:0.72rem;color:#6b7280;font-family:IBM Plex Mono,monospace;">'
        f'Risk level: <strong>{label}</strong></div>'
    )


def _colour_drift(val: float) -> str:
    if val > 0:
        return "background-color: #fffbeb; color: #92400e"
    if val < 0:
        return "background-color: #eff6ff; color: #1e40af"
    return ""


def _colour_action(val: str) -> str:
    if val == "BUY":
        return "color: #166534; font-weight: 600"
    if val == "SELL":
        return "color: #991b1b; font-weight: 600"
    return ""


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

col_logo, col_spacer = st.columns([3, 7])
with col_logo:
    st.markdown(
        '<h2 style="font-family:IBM Plex Mono,monospace;font-weight:600;'
        'color:#1e293b;margin:0;letter-spacing:-0.02em;">⚖ Rebalancing Engine</h2>',
        unsafe_allow_html=True,
    )

_status_bar()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_rebalance, tab_models, tab_profile, tab_settings = st.tabs([
    "  Rebalance  ",
    "  Models  ",
    "  Client Profile  ",
    "  Settings  ",
])

# ===========================================================================
# TAB 1 — REBALANCE  (core workflow)
# ===========================================================================

with tab_rebalance:

    # ── Step 1: Select model ─────────────────────────────────────────────
    has_model = bool(st.session_state.model)
    _step_header("1", "Select Model", "Choose the target model portfolio for this rebalance.", done=has_model)

    saved_keys = list(st.session_state.saved_models.keys())

    if not saved_keys:
        st.warning("No models saved yet. Go to the **Models** tab to create one.")
    else:
        current_label = ""
        if st.session_state.model:
            current_label = (
                f"{st.session_state.model.model_id} "
                f"({st.session_state.model.version})"
            )
        default_idx = saved_keys.index(current_label) if current_label in saved_keys else 0

        sel_col1, sel_col2 = st.columns([3, 2])
        with sel_col1:
            selected_key = st.selectbox(
                "Model portfolio",
                options=saved_keys,
                index=default_idx,
                label_visibility="collapsed",
            )
        with sel_col2:
            if st.button("Use this model", type="primary", use_container_width=True):
                st.session_state.model = st.session_state.saved_models[selected_key]
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.rerun()

        if st.session_state.model:
            m = st.session_state.model
            with st.expander(
                f"Model: {m.model_id} v{m.version}  —  {len(m.holdings)} holdings"
            ):
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Ticker": h.ticker,
                            "Target (%)": f"{h.target_weight*100:.4f}%",
                        }
                        for h in m.holdings
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

    st.divider()

    # ── Step 2: Load portfolio ───────────────────────────────────────────
    has_portfolio = bool(st.session_state.portfolios)
    _step_header("2", "Load Portfolio", "Upload a holdings CSV or use the built-in sample data.", done=has_portfolio)

    up_col1, up_col2 = st.columns([3, 2])
    with up_col1:
        use_sample = st.checkbox("Use sample data")
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
                label_visibility="collapsed",
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
                    st.error(f"Validation error: {exc}")
                finally:
                    os.unlink(tmp_path)

    with up_col2:
        with st.expander("CSV format"):
            st.code(
                "account_id,ticker,quantity,price,cash_balance\n"
                "HIN001,CBA,150.0000,121.5000,5000.0000\n"
                "HIN001,BHP,200.2500,46.2000,5000.0000",
                language="text",
            )

    if has_portfolio:
        with st.expander(
            f"{len(st.session_state.portfolios)} account(s) loaded — click to preview"
        ):
            for p in st.session_state.portfolios:
                st.caption(
                    f"**{p.account_id}**  |  "
                    f"{len(p.holdings)} holdings  |  "
                    f"Total value: ${p.total_value():,.4f}  |  "
                    f"Cash: ${p.cash_balance:,.4f}"
                )

    st.divider()

    # ── Step 3: Run settings + rebalance ─────────────────────────────────
    has_results = bool(st.session_state.drift_reports)
    _step_header("3", "Run Rebalance", "Configure parameters and run.", done=has_results)

    rs1, rs2, rs3, rs4 = st.columns(4)
    with rs1:
        drift_threshold = st.slider(
            "Drift threshold (%)",
            0.5, 10.0, 3.0, 0.5,
            help="Flag holdings deviating more than this many percentage points from target",
        ) / 100.0
    with rs2:
        min_trade_value = st.number_input(
            "Min trade value (AUD)",
            min_value=0.0, value=500.0, step=100.0,
        )
    with rs3:
        fractional_shares = st.checkbox("Fractional shares", value=True)
    with rs4:
        decimal_places = int(st.number_input(
            "Qty decimal places", min_value=1, max_value=6, value=4, step=1,
        )) if fractional_shares else 0

    ready = has_model and has_portfolio

    with st.spinner("Calculating drift and generating trades..."):
        run_clicked = st.button(
            "Run Rebalance",
            type="primary",
            disabled=not ready,
            use_container_width=False,
        )

    if run_clicked and ready:
        config = TradeConfig(
            drift_threshold=drift_threshold,
            min_trade_value=min_trade_value,
            whole_shares=not fractional_shares,
            managed_fund_dp=decimal_places,
        )
        d_reports, t_results = [], []
        for portfolio in st.session_state.portfolios:
            try:
                dr = calculate_drift(
                    portfolio, st.session_state.model, threshold=drift_threshold
                )
                d_reports.append(dr)
            except ValueError as exc:
                st.error(f"{portfolio.account_id}: {exc}")
                continue
            try:
                tr = generate_trades(portfolio, st.session_state.model, config)
                t_results.append(tr)
            except ValueError as exc:
                st.error(f"{portfolio.account_id}: {exc}")
        st.session_state.drift_reports = d_reports
        st.session_state.trade_results = t_results
        st.rerun()

    if not ready:
        missing = []
        if not has_model:
            missing.append("select a model")
        if not has_portfolio:
            missing.append("load a portfolio")
        st.caption(f"To run: {' and '.join(missing)} above.")

    # ── Step 4: Results ──────────────────────────────────────────────────
    drift_reports = st.session_state.drift_reports
    trade_results = st.session_state.trade_results
    all_trades = [t for tr in trade_results for t in tr.trades]

    if drift_reports:
        st.divider()
        _step_header("4", "Results", "Review drift and trade instructions for each account.", done=bool(all_trades))

        model = st.session_state.model
        gross_buy  = sum(t.estimated_value for t in all_trades if t.action == "BUY")
        gross_sell = sum(t.estimated_value for t in all_trades if t.action == "SELL")

        # Summary metrics — two rows
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Accounts processed", len(drift_reports))
        sm2.metric(
            "Need rebalancing",
            sum(1 for dr in drift_reports if dr.requires_rebalance),
        )
        sm3.metric(
            "Total flagged holdings",
            sum(dr.flagged_count for dr in drift_reports),
        )

        sm4, sm5, sm6 = st.columns(3)
        sm4.metric("Total trades", len(all_trades))
        sm5.metric("Gross buy value", f"${gross_buy:,.2f}")
        sm6.metric("Net cash flow", f"${gross_sell - gross_buy:+,.2f}")

        st.write("")

        # Per-account tabs
        acct_tabs = st.tabs([f"  {dr.account_id}  " for dr in drift_reports])

        for tab_a, dr in zip(acct_tabs, drift_reports):
            with tab_a:
                tr = next(
                    (x for x in trade_results if x.account_id == dr.account_id),
                    None,
                )

                # Account header metrics
                ca, cb, cc, cd = st.columns(4)
                ca.metric("Portfolio value", f"${dr.total_portfolio_value:,.2f}")
                cb.metric("Holdings", len(dr.holdings))
                cc.metric(
                    "Flagged",
                    dr.flagged_count,
                )
                if tr:
                    shortfall = tr.has_funding_shortfall
                    cd.metric(
                        "⚠ Shortfall" if shortfall else "Closing cash",
                        f"${tr.closing_cash:,.2f}",
                    )
                    if shortfall:
                        st.warning(
                            f"Funding shortfall of ${abs(tr.closing_cash):,.2f}. "
                            "Buy cost exceeds available cash. Adviser review required."
                        )

                st.divider()

                # Drift table
                st.markdown("**Drift Analysis**")

                drift_rows = [{
                    "Ticker":       h.ticker,
                    "Status":       h.status.value,
                    "Current (%)":  round(h.current_weight * 100, 4),
                    "Target (%)":   round(h.target_weight * 100, 4),
                    "Drift (pp)":   round(h.drift * 100, 4),
                    "Market Value": round(h.market_value, 4),
                    "Flag":         "Yes" if h.exceeds_threshold else "No",
                } for h in dr.holdings]

                st.dataframe(
                    pd.DataFrame(drift_rows)
                    .style
                    .map(_colour_drift, subset=["Drift (pp)"])
                    .format({
                        "Current (%)":  "{:.4f}%",
                        "Target (%)":   "{:.4f}%",
                        "Drift (pp)":   "{:+.4f}pp",
                        "Market Value": "${:,.4f}",
                    })
                    .hide(axis="index"),
                    use_container_width=True,
                    height=min(60 + len(drift_rows) * 35, 400),
                )

                if dr.requires_rebalance:
                    st.caption(
                        f"⚑  {dr.flagged_count} holding(s) exceed "
                        f"the {drift_threshold*100:.1f}pp threshold."
                    )
                else:
                    st.caption("All holdings are within the drift band. No rebalance required.")

                # Trades
                if tr and tr.trades:
                    st.write("")
                    st.markdown("**Trade Instructions**")

                    tf1, tf2, tf3, tf4 = st.columns(4)
                    tf1.metric("Opening cash",  f"${tr.opening_cash:,.2f}")
                    tf2.metric("Sell proceeds", f"${tr.sell_proceeds:,.2f}")
                    tf3.metric("Buy cost",      f"${tr.buy_cost:,.2f}")
                    tf4.metric(
                        "⚠ Shortfall" if tr.has_funding_shortfall else "Closing cash",
                        f"${tr.closing_cash:,.2f}",
                    )

                    trows = [{
                        "Action":     t.action,
                        "Ticker":     t.ticker,
                        "Quantity":   round(t.quantity, 4),
                        "Est. Value": round(t.estimated_value, 4),
                    } for t in tr.trades]

                    st.dataframe(
                        pd.DataFrame(trows)
                        .style
                        .map(_colour_action, subset=["Action"])
                        .format({
                            "Quantity":   "{:.4f}",
                            "Est. Value": "${:,.4f}",
                        })
                        .hide(axis="index"),
                        use_container_width=True,
                        height=min(60 + len(trows) * 35, 320),
                    )

                if tr and tr.suppressed_trades:
                    with st.expander(
                        f"Suppressed trades ({len(tr.suppressed_trades)})"
                    ):
                        st.dataframe(
                            pd.DataFrame([{
                                "Action":    td.action,
                                "Ticker":    td.ticker,
                                "Raw value": round(td.raw_trade_value, 4),
                                "Reason":    td.suppression_reason or "",
                            } for td in tr.suppressed_trades]),
                            use_container_width=True,
                            hide_index=True,
                        )

        # ── Step 5: Download ─────────────────────────────────────────────
        st.divider()
        _step_header("5", "Download", "Export trade instructions for execution.", done=False)

        if all_trades:
            dl1, dl2 = st.columns([2, 5])
            with dl1:
                tmp_dl = tempfile.mktemp(suffix=".csv")
                export_trades_csv(all_trades, tmp_dl, include_metadata=True)
                with open(tmp_dl, "rb") as f:
                    csv_bytes = f.read()
                os.unlink(tmp_dl)
                st.download_button(
                    label="Download all trades (CSV)",
                    data=csv_bytes,
                    file_name=(
                        f"trades_{st.session_state.model.model_id}_"
                        f"{st.session_state.model.version}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                    type="primary",
                )
            with dl2:
                st.caption(
                    f"{len(all_trades)} trade(s) across "
                    f"{len(set(t.account_id for t in all_trades))} account(s)  |  "
                    f"Gross buy: ${gross_buy:,.2f}  |  "
                    f"Gross sell: ${gross_sell:,.2f}  |  "
                    f"Net: ${gross_sell - gross_buy:+,.2f}"
                )
        else:
            st.caption("No active trades were generated.")


# ===========================================================================
# TAB 2 — MODELS
# ===========================================================================

with tab_models:
    st.markdown("### Model Portfolios")
    st.caption("Create and manage model portfolios. The active model is used in the Rebalance tab.")

    saved_models = st.session_state.saved_models
    saved_keys = list(saved_models.keys())

    if saved_keys:
        st.markdown("#### Saved models")

        current_label = ""
        if st.session_state.model:
            current_label = (
                f"{st.session_state.model.model_id} "
                f"({st.session_state.model.version})"
            )
        default_idx = saved_keys.index(current_label) if current_label in saved_keys else 0

        sel_model_key = st.selectbox(
            "Select model to view",
            options=saved_keys,
            index=default_idx,
            key="models_tab_select",
        )
        sel_model = saved_models[sel_model_key]

        st.dataframe(
            pd.DataFrame([
                {
                    "Ticker": h.ticker,
                    "Target Weight (%)": round(h.target_weight * 100, 4),
                }
                for h in sel_model.holdings
            ]),
            use_container_width=True,
            hide_index=True,
        )

        mc1, mc2 = st.columns([3, 1])
        with mc1:
            if st.button("Set as active model", type="primary", use_container_width=True):
                st.session_state.model = sel_model
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Active model set to: {sel_model_key}")
        with mc2:
            if st.button("Delete", use_container_width=True):
                if len(saved_keys) == 1:
                    st.error("Cannot delete the only saved model.")
                else:
                    confirm_key = f"confirm_delete_{sel_model_key}"
                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = True
                        st.warning(
                            f"Click Delete again to confirm deletion of '{sel_model_key}'."
                        )
                    else:
                        del st.session_state.saved_models[sel_model_key]
                        del st.session_state[confirm_key]
                        if st.session_state.model:
                            lbl = (
                                f"{st.session_state.model.model_id} "
                                f"({st.session_state.model.version})"
                            )
                            if lbl == sel_model_key:
                                st.session_state.model = None
                                st.session_state.drift_reports = []
                                st.session_state.trade_results = []
                        st.rerun()

        if st.session_state.model:
            st.caption(
                f"Active: **{st.session_state.model.model_id}** "
                f"v{st.session_state.model.version}  |  "
                f"{len(st.session_state.model.holdings)} holdings"
            )
    else:
        st.info("No saved models. Create one below.")

    st.divider()
    st.markdown("#### Create new model")

    mc_a, mc_b = st.columns(2)
    with mc_a:
        new_id = st.text_input("Model ID", placeholder="e.g. CONSERVATIVE_30")
    with mc_b:
        new_ver = st.text_input("Version", placeholder="e.g. 2024-Q3")

    st.caption("Enter target weights — must sum to exactly 100%")

    m_weight_df = st.data_editor(
        pd.DataFrame(SAMPLE_MODEL_WEIGHTS, columns=["Ticker", "Weight"]),
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Weight": st.column_config.NumberColumn(
                "Weight (%)", min_value=0.0001, max_value=100.0,
                step=0.01, format="%.4f",
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="weight_editor_models",
    )

    wsum = m_weight_df["Weight"].sum() if not m_weight_df.empty else 0.0
    wdelta = wsum - 100.0
    if abs(wdelta) < 0.05:
        st.success(f"Weights sum: {wsum:.4f}%  ✓")
    else:
        st.warning(f"Weights sum: {wsum:.4f}%  (need 100%,  {wdelta:+.4f}%)")

    if st.button("Save model", type="primary"):
        if not new_id.strip():
            st.error("Model ID is required.")
        elif not new_ver.strip():
            st.error("Version is required.")
        else:
            try:
                holdings = [
                    ModelHolding(
                        ticker=str(r["Ticker"]).strip().upper(),
                        target_weight=float(r["Weight"]) / 100.0,
                    )
                    for _, r in m_weight_df.iterrows()
                    if str(r["Ticker"]).strip()
                    and str(r["Ticker"]).strip().lower() != "nan"
                ]
                new_model = ModelPortfolio(
                    model_id=new_id.strip(),
                    version=new_ver.strip(),
                    holdings=holdings,
                )
                label = f"{new_model.model_id} ({new_model.version})"
                st.session_state.saved_models[label] = new_model
                st.session_state.model = new_model
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Saved and activated: {label}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


# ===========================================================================
# TAB 3 — CLIENT PROFILE  (optional)
# ===========================================================================

with tab_profile:
    st.markdown("### Client Risk Profile")
    st.caption(
        "Optional — use this questionnaire to determine an appropriate model "
        "portfolio for a client. Results can be exported for the client file."
    )

    scored_qs = [q for q in QUESTIONS if not q.get("no_score", False)]
    total_qs = len(QUESTIONS)

    # One-question-at-a-time flow
    step = st.session_state.q_step
    answers = dict(st.session_state.q_answers)

    col_cn, col_dt = st.columns([3, 2])
    with col_cn:
        client_name = st.text_input(
            "Client name / reference",
            value=st.session_state.q_client_name,
            placeholder="e.g. John Smith or HIN001",
        )
        st.session_state.q_client_name = client_name
    with col_dt:
        st.metric("Date", _utc_now_str().split(" ")[0])

    st.divider()

    # Progress bar
    answered = sum(1 for q in QUESTIONS if q["id"] in answers)
    st.progress(answered / total_qs, text=f"Question {min(step+1, total_qs)} of {total_qs}")

    # Current question
    if step < total_qs:
        q = QUESTIONS[step]
        st.markdown(
            f"**Section {q['section']} — {q['label']}**",
        )
        st.markdown(f"### {q['text']}")

        options = [opt[0] for opt in q["options"]]
        current_idx = answers.get(q["id"], None)

        selected = st.radio(
            "Select an answer",
            options=options,
            index=current_idx,
            key=f"q_radio_{step}",
            label_visibility="collapsed",
        )

        st.write("")
        nav1, nav2, nav3 = st.columns([1, 1, 5])

        with nav1:
            if step > 0:
                if st.button("Back", use_container_width=True):
                    st.session_state.q_step = step - 1
                    st.rerun()

        with nav2:
            label = "Next" if step < total_qs - 1 else "See results"
            if st.button(label, type="primary", use_container_width=True,
                         disabled=selected is None):
                if selected is not None:
                    answers[q["id"]] = options.index(selected)
                    st.session_state.q_answers = answers
                st.session_state.q_step = step + 1
                st.rerun()

    else:
        # Results page
        all_scored_answered = all(q["id"] in answers for q in scored_qs)

        if not all_scored_answered:
            st.warning("Some scored questions were not answered. Please go back and complete them.")
            if st.button("Back to questions"):
                st.session_state.q_step = 0
                st.rerun()
        else:
            score, flags, conflicts = _compute_score(answers, st.session_state.weights)
            base_model = _score_to_model(score, answers)

            display_model = base_model
            if "income" in flags:
                display_model += " — Income / Yield"
            if "esg" in flags:
                display_model += " (ESG)"

            st.markdown("#### Assessment complete")

            level = RISK_LEVEL.get(base_model, 3)
            r1, r2 = st.columns([2, 3])
            with r1:
                st.metric("Recommended model", base_model)
                st.markdown(_risk_scale_html(level), unsafe_allow_html=True)
                if flags:
                    st.caption(f"Flags: {', '.join(f.upper() for f in flags)}")
            with r2:
                with st.expander("Score breakdown"):
                    rows = []
                    for sec, label in SECTION_LABELS.items():
                        sec_qs = [q for q in QUESTIONS
                                  if q["section"] == sec and not q.get("no_score", False)]
                        pts = [q["options"][answers[q["id"]]][1]
                               for q in sec_qs if q["id"] in answers]
                        if pts:
                            avg = sum(pts) / len(pts)
                            w = st.session_state.weights.get(sec, 1.0)
                            rows.append({
                                "Section": f"{sec} — {label}",
                                "Score (1–5)": round(avg, 2),
                                "Weight": w,
                            })
                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width=True,
                            hide_index=True,
                        )

            if conflicts:
                st.markdown(
                    '<div class="conflict-box">'
                    '<strong>Conflicts detected — adviser review required</strong><br>'
                    + "<br>".join(f"• {c}" for c in conflicts)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("#### Adviser override")
            st.caption(
                "Override the recommended model if required. "
                "A reason must be provided and will be recorded."
            )

            override_options = [
                "— Use recommended model —",
                "Conservative", "Moderate / Balanced", "Growth",
                "High Growth", "Aggressive", "Income / Yield",
                "ESG", "Australian Equities", "Index / Passive",
            ]
            override_sel = st.selectbox(
                "Override selection", options=override_options, index=0,
            )
            override_reason = ""
            if override_sel != "— Use recommended model —":
                override_reason = st.text_area(
                    "Reason for override (required)",
                    placeholder="Document why the recommended model is not appropriate...",
                )
                if not override_reason.strip():
                    st.warning("A reason must be provided before saving.")

            final_model = (
                override_sel if override_sel != "— Use recommended model —"
                else display_model
            )
            is_overridden = override_sel != "— Use recommended model —"

            st.info(
                f"**Final model: {final_model}**"
                + ("  *(adviser override)*" if is_overridden else "  *(system recommended)*")
            )

            can_save = not is_overridden or bool(override_reason.strip())

            save_col, export_col, restart_col = st.columns(3)

            with save_col:
                if st.button(
                    "Save profile",
                    type="primary",
                    use_container_width=True,
                    disabled=not can_save,
                ):
                    st.session_state.q_result = {
                        "client_name":     client_name,
                        "timestamp":       _utc_now_str(),
                        "version":         QUESTIONNAIRE_VERSION,
                        "answers":         dict(answers),
                        "score":           score,
                        "flags":           list(flags),
                        "conflicts":       conflicts,
                        "base_model":      base_model,
                        "display_model":   display_model,
                        "final_model":     final_model,
                        "overridden":      is_overridden,
                        "override_reason": override_reason.strip(),
                        "weights_used":    dict(st.session_state.weights),
                    }
                    st.success("Profile saved.")

            with export_col:
                if st.session_state.q_result:
                    res = st.session_state.q_result
                    lines = [
                        ["Risk Profile Assessment"],
                        ["Client", res["client_name"]],
                        ["Date", res["timestamp"]],
                        ["Questionnaire version", res["version"]],
                        [],
                        ["Question", "Answer", "Points"],
                    ]
                    for q in QUESTIONS:
                        qid = q["id"]
                        if qid in res["answers"]:
                            idx = res["answers"][qid]
                            opt_text, pts = q["options"][idx]
                            lines.append([
                                q["text"], opt_text,
                                "" if q.get("no_score") else pts,
                            ])
                    lines += [
                        [],
                        ["Weighted score", res["score"]],
                        ["Recommended model", res["display_model"]],
                        ["Final model", res["final_model"]],
                        ["Override", "Yes" if res["overridden"] else "No"],
                    ]
                    if res["overridden"]:
                        lines.append(["Override reason", res["override_reason"]])
                    if res["conflicts"]:
                        lines.append(["Conflicts", "; ".join(res["conflicts"])])
                    buf = io.StringIO()
                    csv.writer(buf).writerows(lines)
                    fname = (
                        f"risk_profile_"
                        f"{client_name.replace(' ', '_') or 'client'}_"
                        f"{res['timestamp'][:10]}.csv"
                    )
                    st.download_button(
                        label="Download summary (CSV)",
                        data=buf.getvalue().encode("utf-8-sig"),
                        file_name=fname,
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "Download summary (CSV)",
                        disabled=True,
                        use_container_width=True,
                        help="Save profile first",
                    )

            with restart_col:
                if st.button("Start new assessment", use_container_width=True):
                    st.session_state.q_answers = {}
                    st.session_state.q_step = 0
                    st.session_state.q_result = None
                    st.session_state.q_client_name = ""
                    st.rerun()

            if st.session_state.q_result:
                st.divider()
                st.caption(
                    f"To apply this profile, go to **Models** and activate a model "
                    f"matching **{final_model}**, then run the rebalance."
                )


# ===========================================================================
# TAB 4 — SETTINGS
# ===========================================================================

with tab_settings:
    st.markdown("### Settings")
    st.caption(
        "Configure questionnaire section weightings. "
        "All changes are timestamped and logged for compliance purposes."
    )

    locked = st.session_state.weights_locked

    if locked:
        st.error("Weightings are locked by the practice principal. Unlock below to make changes.")

    st.divider()
    st.markdown("#### Questionnaire section weights")
    st.caption(
        f"Adjust how much each section contributes to the final risk score.  "
        f"Range: {WEIGHT_MIN}x – {WEIGHT_MAX}x.  Default: 1.0x (equal weight)."
    )

    new_weights = {}
    wc1, wc2 = st.columns(2)
    for i, (sec, label) in enumerate(SECTION_LABELS.items()):
        col = wc1 if i % 2 == 0 else wc2
        with col:
            new_weights[sec] = st.slider(
                f"{sec}  —  {label}",
                min_value=WEIGHT_MIN,
                max_value=WEIGHT_MAX,
                value=float(st.session_state.weights.get(sec, 1.0)),
                step=0.1,
                disabled=locked,
                key=f"weight_slider_{sec}",
            )

    total_w = sum(new_weights.values())
    st.markdown("**Effective contribution to final score**")
    st.dataframe(
        pd.DataFrame([{
            "Section": f"{s}  —  {SECTION_LABELS[s]}",
            "Weight": new_weights[s],
            "Contribution": f"{new_weights[s]/total_w*100:.1f}%",
        } for s in SECTION_LABELS]),
        use_container_width=False,
        hide_index=True,
    )

    st.write("")
    note_col, btn_col = st.columns([3, 2])
    with note_col:
        change_note = st.text_input(
            "Reason for change (required)",
            placeholder="e.g. Practice policy update — increase time horizon weighting",
            disabled=locked,
        )
    with btn_col:
        st.write("")
        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button(
                "Save",
                type="primary",
                disabled=locked or not change_note.strip(),
                use_container_width=True,
            ):
                st.session_state.weight_log.append({
                    "timestamp": _utc_now_str(),
                    "weights":   dict(new_weights),
                    "note":      change_note.strip(),
                })
                st.session_state.weights = dict(new_weights)
                st.success("Weightings saved and logged.")
        with b2:
            if st.button("Reset", disabled=locked, use_container_width=True):
                st.session_state.weights = dict(DEFAULT_WEIGHTS)
                st.session_state.weight_log.append({
                    "timestamp": _utc_now_str(),
                    "weights":   dict(DEFAULT_WEIGHTS),
                    "note":      "Restored to default equal weighting",
                })
                st.success("Reset to defaults.")
                st.rerun()

    st.divider()
    st.markdown("#### Principal controls")
    st.caption(
        "Lock weightings to prevent individual advisers from making changes. "
        "Only the practice principal should use this control."
    )

    if locked:
        if st.button("Unlock weightings"):
            st.session_state.weights_locked = False
            st.rerun()
    else:
        if st.button("Lock weightings"):
            st.session_state.weights_locked = True
            st.rerun()

    st.divider()
    st.markdown("#### Change log")
    if st.session_state.weight_log:
        log_rows = []
        for entry in reversed(st.session_state.weight_log):
            row = {"Timestamp": entry["timestamp"], "Note": entry["note"]}
            row.update({s: entry["weights"].get(s, 1.0) for s in SECTION_LABELS})
            log_rows.append(row)
        st.dataframe(
            pd.DataFrame(log_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No changes recorded yet.")
