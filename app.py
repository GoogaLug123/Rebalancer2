"""
app.py — Portfolio Rebalancing Engine
Adviser-grade rebalancing tool with optional client profiling.

Navigation: persistent left sidebar
Layout:     card-based sections
Colours:    deep navy (#0f1f3d) + blue accent (#2754F5)
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drift import DEFAULT_DRIFT_THRESHOLD, calculate_drift
from exports import export_trades_csv
from models import ModelHolding, ModelPortfolio, Portfolio, SecurityHolding
from portfolio import load_portfolios
from trades import TradeConfig, generate_trades

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rebalancing Engine",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
NAVY   = "#0f1f3d"
BLUE   = "#2754F5"
BLUE_L = "#e8effe"
OFF_W  = "#f7f8fc"
WHITE  = "#ffffff"
BORDER = "#e4e7ef"
MUTED  = "#6b7280"
TEXT   = "#1a2233"
LIGHT  = "#f0f3ff"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    color: {TEXT};
}}
.stApp {{ background: {OFF_W}; }}
.block-container {{
    padding: 2rem 2.5rem 6rem 2.5rem !important;
    max-width: 1200px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {WHITE};
    border-right: 1px solid {BORDER};
    min-width: 200px !important;
    max-width: 200px !important;
}}
section[data-testid="stSidebar"] > div {{ padding: 0 !important; }}

.sidebar-logo {{
    padding: 1.75rem 1.25rem 1.25rem 1.25rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 0.5rem;
}}
.sidebar-logo h2 {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: {NAVY};
    margin: 0;
    letter-spacing: 0.01em;
}}
.sidebar-logo p {{
    font-size: 0.67rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin: 0.2rem 0 0 0;
}}

/* ── Sidebar nav links ── */
.nav-link {{
    display: block;
    padding: 0.5rem 1.25rem;
    font-size: 0.82rem;
    font-weight: 500;
    color: {MUTED};
    text-decoration: none;
    cursor: pointer;
    border-left: 2px solid transparent;
    transition: color 0.12s, border-color 0.12s;
    line-height: 1.5;
    margin: 0.1rem 0;
}}
.nav-link:hover {{ color: {NAVY}; }}
.nav-link.active {{
    color: {BLUE};
    border-left-color: {BLUE};
    font-weight: 600;
    background: {LIGHT};
}}

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] div[data-testid="stButton"] {{
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
    all: unset !important;
    display: block !important;
    width: 100% !important;
    box-sizing: border-box !important;
    padding: 0.55rem 1.25rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {MUTED} !important;
    text-align: left !important;
    cursor: pointer !important;
    border-left: 2px solid transparent !important;
    background: transparent !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
    background: {OFF_W} !important;
    color: {NAVY} !important;
    border-left-color: {BORDER} !important;
}}
section[data-testid="stSidebar"] div[data-testid="stButton"] button p {{
    margin: 0 !important;
    text-align: left !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: inherit !important;
}}

/* ── Sidebar status ── */
.sidebar-status {{
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: {WHITE};
    padding: 0.85rem 1.25rem;
    border-top: 1px solid {BORDER};
}}
.sidebar-status-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
    font-size: 0.68rem;
    font-family: 'IBM Plex Mono', monospace;
}}
.sidebar-status-row:last-child {{ margin-bottom: 0; }}
.sidebar-status-label {{ color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}
.s-done   {{ color: {BLUE}; font-weight: 600; }}
.s-warn   {{ color: #dc2626; font-weight: 600; }}
.s-idle   {{ color: #c8cdd8; }}

/* ── Page heading ── */
.page-heading {{
    margin-bottom: 2rem;
}}
.page-heading h1 {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 2rem;
    font-weight: 600;
    color: {NAVY};
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.page-heading .sub {{
    font-size: 0.82rem;
    color: {MUTED};
    margin: 0;
}}

/* ── Cards ── */
.card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    animation: fadeUp 0.2s ease;
}}
.card-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid {BORDER};
}}
.card-step {{
    width: 24px; height: 24px;
    border-radius: 50%;
    border: 1.5px solid {BLUE};
    color: {BLUE};
    background: {WHITE};
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.68rem;
    font-weight: 700;
    flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
}}
.card-step-done {{
    width: 24px; height: 24px;
    border-radius: 50%;
    background: {BLUE};
    color: {WHITE};
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.68rem;
    font-weight: 700;
    flex-shrink: 0;
}}
.card-title   {{ font-size: 0.9rem; font-weight: 600; color: {NAVY}; }}
.card-subtitle {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.1rem; }}

/* ── Metric cards ── */
div[data-testid="metric-container"] {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 1rem 1.25rem !important;
}}
div[data-testid="metric-container"] label {{
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {MUTED} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.35rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: {NAVY} !important;
    font-weight: 600 !important;
}}

/* ── Status pills ── */
.pill {{
    display: inline-flex;
    align-items: center;
    padding: 1px 8px;
    border-radius: 20px;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
}}
.pill-over   {{ background: #fef3c7; color: #92400e; }}
.pill-under  {{ background: {BLUE_L}; color: {BLUE}; }}
.pill-divest {{ background: #fee2e2; color: #991b1b; }}
.pill-ok     {{ background: #f0fdf4; color: #166534; }}

/* ── Tables ── */
thead tr th {{
    background: {LIGHT} !important;
    color: {NAVY} !important;
    font-size: 0.67rem !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-family: 'IBM Plex Mono', monospace !important;
    border-bottom: 1px solid {BORDER} !important;
    padding: 0.55rem 0.75rem !important;
    font-weight: 600 !important;
}}
tbody tr td {{
    font-size: 0.82rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-bottom: 1px solid #f5f6fa !important;
    padding: 0.5rem 0.75rem !important;
    color: {TEXT} !important;
}}
tbody tr:hover td {{ background: #fafbff !important; }}

/* ── Buttons — general reset ── */
div[data-testid="stButton"] button {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    border-radius: 5px !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background: {NAVY} !important;
    color: {WHITE} !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{
    background: #1a3260 !important;
}}
div[data-testid="stButton"] button[kind="primary"]:disabled {{
    background: #c8cdd8 !important;
    color: {WHITE} !important;
    cursor: not-allowed !important;
}}
div[data-testid="stButton"] button[kind="secondary"] {{
    background: {WHITE} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    padding: 0.45rem 1.1rem !important;
}}
div[data-testid="stButton"] button[kind="secondary"]:hover {{
    border-color: {BLUE} !important;
    color: {BLUE} !important;
}}

/* ── Run button ── */
.run-btn-wrap div[data-testid="stButton"] button {{
    width: 100% !important;
    height: 46px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    background: {BLUE} !important;
    border: none !important;
    letter-spacing: 0.02em !important;
}}
.run-btn-wrap div[data-testid="stButton"] button:hover {{
    background: #1f44d6 !important;
}}
.run-btn-wrap div[data-testid="stButton"] button:disabled {{
    background: #c8cdd8 !important;
}}

/* ── Download button ── */
div[data-testid="stDownloadButton"] button {{
    background: {WHITE} !important;
    color: {BLUE} !important;
    border: 1.5px solid {BLUE} !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 5px !important;
    padding: 0.45rem 1.1rem !important;
}}
div[data-testid="stDownloadButton"] button:hover {{
    background: {BLUE_L} !important;
}}

/* ── Sticky download bar ── */
.sticky-download {{
    position: fixed;
    bottom: 0; left: 200px; right: 0;
    background: {WHITE};
    border-top: 1px solid {BORDER};
    padding: 0.75rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 999;
    animation: slideUp 0.25s ease;
}}
.dl-summary {{
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    color: {MUTED};
}}
.dl-summary strong {{ color: {NAVY}; }}

/* ── Model description ── */
.model-desc {{
    background: {LIGHT};
    border: 1px solid {BORDER};
    border-left: 2px solid {BLUE};
    border-radius: 4px;
    padding: 0.55rem 0.85rem;
    font-size: 0.78rem;
    color: {MUTED};
    margin-top: 0.5rem;
    animation: fadeUp 0.15s ease;
}}
.model-desc strong {{ color: {TEXT}; font-weight: 600; }}

/* ── Empty states ── */
.empty-state {{
    text-align: center;
    padding: 2rem 1rem;
    border: 1.5px dashed {BORDER};
    border-radius: 8px;
    background: {WHITE};
    margin: 0.5rem 0;
}}
.empty-state .icon {{ font-size: 1.75rem; margin-bottom: 0.4rem; }}
.empty-state .msg  {{ font-size: 0.88rem; color: {TEXT}; font-weight: 500; }}
.empty-state .hint {{ font-size: 0.75rem; color: {MUTED}; margin-top: 0.2rem; }}

/* ── Conflict box ── */
.conflict-box {{
    background: #fffbeb;
    border: 1px solid #f59e0b;
    border-left: 3px solid #d97706;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.8rem;
    color: #451a03;
}}

/* ── Risk bar ── */
.risk-marker {{
    font-size: 0.7rem;
    color: {MUTED};
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}}

/* ── Tabs ── */
button[data-baseweb="tab"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {MUTED} !important;
    padding: 0.5rem 0.85rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {NAVY} !important;
    border-bottom-color: {BLUE} !important;
    font-weight: 600 !important;
}}

/* ── Inputs ── */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    border-radius: 5px !important;
    border-color: {BORDER} !important;
}}
div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    font-size: 0.82rem !important;
    border-radius: 5px !important;
}}

/* ── Captions ── */
div[data-testid="stCaptionContainer"] {{
    font-size: 0.75rem !important;
    color: {MUTED} !important;
}}

/* ── Dividers ── */
hr {{ border-color: {BORDER} !important; margin: 1.25rem 0 !important; }}

/* ── Expanders ── */
details summary {{
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: {TEXT} !important;
}}

/* ── Accessibility ── */
button:focus-visible {{
    outline: 2px solid {BLUE} !important;
    outline-offset: 2px !important;
}}

/* ── Animations ── */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes slideUp {{
    from {{ transform: translateY(100%); }}
    to   {{ transform: translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTIONNAIRE_VERSION = "1.0"

MODEL_DESCRIPTIONS = {
    "CONSERVATIVE":   ("30% growth / 70% defensive", "Capital preservation. Bonds and cash-like ETFs dominate.", 1),
    "MODERATE":       ("50% growth / 50% defensive", "Balanced income and growth across Australian and global assets.", 2),
    "GROWTH":         ("70% growth / 30% defensive", "Long-term wealth accumulation with meaningful defensive buffer.", 3),
    "HIGH_GROWTH":    ("90% growth / 10% defensive", "Maximum equity exposure. Long time horizon required.", 4),
    "AGGRESSIVE":     ("100% growth",                "Pure equity. No defensive allocation. High volatility expected.", 5),
    "INCOME":         ("Income focused",             "High-dividend stocks, LICs, and fixed interest for regular income.", 2),
    "ESG":            ("Ethical / screened growth",  "Excludes fossil fuels, weapons, tobacco. ESG-screened ETFs only.", 3),
    "AUSTRALIAN_EQ":  ("100% Australian equities",  "Direct ASX blue-chip exposure. No international or fixed income.", 4),
    "INDEX_PASSIVE":  ("Low-cost index",             "Three-fund portfolio. Broad market exposure at minimum cost.", 3),
}

QUESTIONS = [
    {
        "id": "Q1", "section": "A", "label": "Time Horizon",
        "text": "When do you expect to start drawing on this investment?",
        "options": [
            ("Less than 2 years", 1), ("2–5 years", 2), ("5–10 years", 3),
            ("10–15 years", 4), ("More than 15 years", 5),
        ],
    },
    {
        "id": "Q2", "section": "B", "label": "Risk Tolerance",
        "text": "If your portfolio lost 20% of its value in a market downturn, what would you most likely do?",
        "options": [
            ("Sell everything immediately", 1), ("Sell some to reduce exposure", 2),
            ("Do nothing and wait", 3), ("Buy a little more while prices are low", 4),
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
            ("Severely — I depend on it", 1), ("Significantly — it would cause real hardship", 2),
            ("Moderately — I would need to adjust my plans", 3),
            ("Mildly — I have other assets to fall back on", 4),
            ("Minimally — this is discretionary money", 5),
        ],
    },
    {
        "id": "Q5", "section": "D", "label": "Income Stability",
        "text": "How would you describe your current and expected future income?",
        "options": [
            ("Unstable or uncertain", 1), ("Variable with some uncertainty", 2),
            ("Stable but could change", 3), ("Stable and likely to continue", 4),
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
            ("Preserve my capital above all else", 1), ("Generate regular income", 2),
            ("Balanced mix of income and growth", 3), ("Grow my wealth over the long term", 4),
            ("Maximise long-term growth, I do not need income", 5),
        ],
        "flags": {1: "income"},
    },
    {
        "id": "Q8", "section": "F", "label": "ESG Preferences",
        "text": "Do you have any ethical or ESG investment preferences?",
        "options": [
            ("No preference", 0), ("Somewhat important", 0),
            ("Very important — I want to exclude certain industries", 0),
        ],
        "flags": {1: "esg", 2: "esg"},
        "no_score": True,
    },
]

SCORE_MAP = [
    (6, 10, "Conservative"), (11, 16, "Moderate / Balanced"),
    (17, 22, "Growth"), (23, 28, "High Growth"), (29, 30, "Aggressive"),
]
RISK_LEVEL = {
    "Conservative": 1, "Moderate / Balanced": 2, "Growth": 3,
    "High Growth": 4, "Aggressive": 5,
}
CONFLICT_RULES = [
    ("Q2", [0,1], "Q7", [3,4], "Low risk tolerance (Q2) conflicts with high growth goals (Q7)."),
    ("Q1", [0], "Q7", [3,4], "Short time horizon (Q1) conflicts with growth-oriented goals (Q7)."),
    ("Q4", [0], None, None, "Client is financially dependent on this investment (Q4). Consider capping at Conservative."),
]
DEFAULT_WEIGHTS = {"A":1.0,"B":1.0,"C":1.0,"D":1.0,"E":1.0,"F":1.0}
WEIGHT_MIN = 0.5
WEIGHT_MAX = 2.0
SECTION_LABELS = {
    "A":"Time Horizon","B":"Risk Tolerance","C":"Risk Capacity",
    "D":"Income Stability","E":"Investment Experience","F":"Goals",
}

SAMPLE_CSV = """\
account_id,ticker,quantity,price,cash_balance
HIN001,VAS,480.0000,104.00,3500.0000
HIN001,VGS,265.0000,140.50,3500.0000
HIN001,VAF,280.0000,47.80,3500.0000
HIN001,AGG,100.0000,110.30,3500.0000
HIN001,VHY,0.0000,72.40,3500.0000
HIN002,VAS,480.0000,104.00,5000.0000
HIN002,VGS,280.0000,140.50,5000.0000
HIN002,VAF,420.0000,47.80,5000.0000
HIN002,AGG,320.0000,110.30,5000.0000
HIN002,VHY,120.0000,72.40,5000.0000
HIN002,NDQ,250.0000,48.60,5000.0000
HIN003,VAS,120.0000,104.00,25000.0000
HIN003,VGS,80.0000,140.50,25000.0000
HIN003,VAF,100.0000,47.80,25000.0000
HIN003,AGG,0.0000,110.30,25000.0000
HIN003,VHY,0.0000,72.40,25000.0000
"""

SAMPLE_MODEL_WEIGHTS = [
    ("CBA",25.0),("BHP",20.0),("CSL",15.0),
    ("WES",15.0),("ANZ",12.0),("MQG",8.0),("FMG",5.0),
]

PREDEFINED_MODELS: list = [
    {"model_id":"CONSERVATIVE","version":"2024-Q2","holdings":[("AGG",40.0),("VAF",20.0),("VAS",20.0),("VGS",10.0),("VHY",10.0)]},
    {"model_id":"MODERATE","version":"2024-Q2","holdings":[("VAS",25.0),("VGS",25.0),("VAF",25.0),("AGG",15.0),("VHY",10.0)]},
    {"model_id":"GROWTH","version":"2024-Q2","holdings":[("VAS",30.0),("VGS",25.0),("VGE",15.0),("VAF",20.0),("VHY",10.0)]},
    {"model_id":"HIGH_GROWTH","version":"2024-Q2","holdings":[("VAS",30.0),("VGS",30.0),("VGE",15.0),("NDQ",15.0),("VHY",10.0)]},
    {"model_id":"AGGRESSIVE","version":"2024-Q2","holdings":[("VAS",30.0),("VGS",30.0),("NDQ",20.0),("VGE",20.0)]},
    {"model_id":"INCOME","version":"2024-Q2","holdings":[("VHY",30.0),("VAF",25.0),("AFI",15.0),("ARG",15.0),("VAP",15.0)]},
    {"model_id":"ESG","version":"2024-Q2","holdings":[("ETHI",35.0),("FAIR",30.0),("VESG",20.0),("VBND",15.0)]},
    {"model_id":"AUSTRALIAN_EQ","version":"2024-Q2","holdings":[("CBA",25.0),("BHP",20.0),("CSL",15.0),("WES",15.0),("ANZ",12.0),("MQG",8.0),("FMG",5.0)]},
    {"model_id":"INDEX_PASSIVE","version":"2024-Q2","holdings":[("VAS",40.0),("VGS",40.0),("VAF",20.0)]},
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "page":           "rebalance",
        "portfolios":     [],
        "model":          None,
        "saved_models":   {},
        "drift_reports":  [],
        "trade_results":  [],
        "q_answers":      {},
        "q_result":       None,
        "q_client_name":  "",
        "q_step":         0,
        "weights":        dict(DEFAULT_WEIGHTS),
        "weight_log":     [],
        "weights_locked": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

if not st.session_state.saved_models:
    for _pm in PREDEFINED_MODELS:
        try:
            _h = [ModelHolding(ticker=t, target_weight=w/100.0) for t,w in _pm["holdings"]]
            _m = ModelPortfolio(model_id=_pm["model_id"], version=_pm["version"], holdings=_h)
            st.session_state.saved_models[f"{_pm['model_id']} ({_pm['version']})"] = _m
        except Exception:
            pass
    _dk = "GROWTH (2024-Q2)"
    st.session_state.model = st.session_state.saved_models.get(
        _dk, list(st.session_state.saved_models.values())[0] if st.session_state.saved_models else None
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _settle_date() -> str:
    d = datetime.now(timezone.utc)
    added = 0
    while added < 2:
        d += timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d.strftime("%d %b %Y")


def _card_header(step: str, title: str, subtitle: str = "", done: bool = False) -> None:
    icon = f'<span class="card-step-done">✓</span>' if done else f'<span class="card-step">{step}</span>'
    st.markdown(
        f'<div class="card-header">{icon}'
        f'<div><div class="card-title">{title}</div>'
        f'{"<div class=card-subtitle>" + subtitle + "</div>" if subtitle else ""}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _pill(label: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'


def _status_pill(status: str) -> str:
    mapping = {
        "OVERWEIGHT":   ("OVER",   "over"),
        "UNDERWEIGHT":  ("UNDER",  "under"),
        "NOT_IN_MODEL": ("DIVEST", "divest"),
        "IN_BAND":      ("OK",     "ok"),
    }
    label, kind = mapping.get(status, (status, "ok"))
    return _pill(label, kind)


def _risk_bar(level: int) -> str:
    colours = ["#22c55e","#84cc16","#eab308","#f97316","#ef4444"]
    labels  = ["Conservative","Moderate","Growth","High Growth","Aggressive"]
    segs = ""
    for i in range(1, 6):
        outline = f"outline: 2.5px solid {NAVY}; outline-offset: -2px;" if i == level else ""
        # Add pattern stripe for accessibility (every other segment)
        segs += f'<div style="flex:1;background:{colours[i-1]};{outline}"></div>'
    return (
        f'<div style="display:flex;height:10px;border-radius:4px;overflow:hidden;margin:0.4rem 0;">{segs}</div>'
        f'<div style="font-size:0.7rem;color:{MUTED};font-family:IBM Plex Mono,monospace;">'
        f'Risk profile: <strong style="color:{NAVY}">{labels[level-1] if 1<=level<=5 else ""}</strong></div>'
    )


def _colour_drift_row(row) -> list:
    s = row.get("Status","")
    if s == "NOT_IN_MODEL":
        return ["background:#fff1f2;border-left:4px solid #dc2626"] * len(row)
    if s == "OVERWEIGHT":
        return ["background:#fffbeb;border-left:4px solid #d97706"] * len(row)
    if s == "UNDERWEIGHT":
        return ["background:#eff6ff;border-left:4px solid #2563eb"] * len(row)
    return [""] * len(row)


def _colour_drift_val(val: float) -> str:
    if val > 0: return "font-weight:700;color:#92400e"
    if val < 0: return "font-weight:700;color:#1e3a8a"
    return ""


def _colour_trade_row(row) -> list:
    a = row.get("Action","")
    if a == "BUY":  return ["background:#f0fdf4;border-left:4px solid #16a34a"] * len(row)
    if a == "SELL": return ["background:#fff1f2;border-left:4px solid #dc2626"] * len(row)
    return [""] * len(row)


def _colour_action(val: str) -> str:
    if val == "BUY":  return "color:#065f46;font-weight:700"
    if val == "SELL": return "color:#7f1d1d;font-weight:700"
    return ""


def _model_desc_html(model_id: str) -> str:
    if model_id not in MODEL_DESCRIPTIONS:
        return ""
    alloc, desc, _ = MODEL_DESCRIPTIONS[model_id]
    return (
        f'<div class="model-desc">'
        f'<strong>{alloc}</strong> — {desc}'
        f'</div>'
    )


def _compute_score(answers: dict, weights: dict) -> tuple:
    section_raw: dict = {}
    flags: set = set()
    for q in QUESTIONS:
        qid = q["id"]
        if qid not in answers: continue
        idx = answers[qid]
        pts = q["options"][idx][1]
        if not q.get("no_score", False):
            section_raw.setdefault(q["section"], []).append(pts)
        if "flags" in q and idx in q["flags"]:
            flags.add(q["flags"][idx])
    total_w = weighted_sum = 0.0
    for sec, pts_list in section_raw.items():
        avg = sum(pts_list)/len(pts_list)
        w = weights.get(sec, 1.0)
        weighted_sum += avg * w
        total_w += w
    score = round((weighted_sum/total_w)*6, 2) if total_w else 0.0
    conflicts = []
    for q1_id, q1_idxs, q2_id, q2_idxs, msg in CONFLICT_RULES:
        q1_ans = answers.get(q1_id)
        if q1_ans is None or q1_ans not in q1_idxs: continue
        if q2_id is None: conflicts.append(msg); continue
        q2_ans = answers.get(q2_id)
        if q2_ans is not None and q2_ans in q2_idxs: conflicts.append(msg)
    return score, flags, conflicts


def _score_to_model(score: float, answers: dict) -> str:
    base = "Conservative"
    for lo, hi, name in SCORE_MAP:
        if lo <= score <= hi: base = name; break
    order = ["Conservative","Moderate / Balanced","Growth","High Growth","Aggressive"]
    q1 = answers.get("Q1", 4)
    if q1 == 0: base = "Conservative"
    elif q1 == 1: base = order[min(order.index(base),1)]
    if answers.get("Q4",4) == 0: base = order[max(order.index(base)-1,0)]
    return base


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

# Handle query param navigation
qp = st.query_params.get("page", None)
if qp and qp != st.session_state.page:
    st.session_state.page = qp

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        '<h2>Rebalancing Engine</h2>'
        '<p>Portfolio &amp; Trade Management</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    pg = st.session_state.page
    def _nav(pid, label):
        cls = "nav-active" if pg == pid else "nav-inactive"
        return (
            f'<a class="nav-link {cls}" ' 
            f'href="?page={pid}" target="_self">{label}</a>'
        )

    st.markdown(
        '<div class="nav-block">' +
        _nav("rebalance", "Rebalance") +
        _nav("models",    "Models") +
        _nav("profile",   "Client Profile") +
        _nav("settings",  "Settings") +
        '</div>',
        unsafe_allow_html=True,
    )

    # Status panel — live session summary
    model      = st.session_state.model
    portfolios = st.session_state.portfolios
    drift_reports = st.session_state.drift_reports
    all_trades_s  = [t for tr in st.session_state.trade_results for t in tr.trades]

    p_cls = "s-done" if portfolios else "s-idle"
    p_val = f"{len(portfolios)} loaded" if portfolios else "—"
    m_cls = "s-done" if model else "s-idle"
    m_val = model.model_id if model else "—"

    if drift_reports:
        needs = sum(1 for dr in drift_reports if dr.requires_rebalance)
        d_cls = "s-warn" if needs else "s-done"
        d_val = f"{needs} flagged" if needs else "All in band"
    else:
        d_cls, d_val = "s-idle", "—"

    t_cls = "s-done" if all_trades_s else "s-idle"
    t_val = f"{len(all_trades_s)} trades" if all_trades_s else "—"

    st.markdown(
        f'<div class="sidebar-status">'
        f'<div class="sidebar-status-row"><span class="sidebar-status-label">Portfolio</span><span class="{p_cls}">{p_val}</span></div>'
        f'<div class="sidebar-status-row"><span class="sidebar-status-label">Model</span><span class="{m_cls}">{m_val}</span></div>'
        f'<div class="sidebar-status-row"><span class="sidebar-status-label">Drift</span><span class="{d_cls}">{d_val}</span></div>'
        f'<div class="sidebar-status-row"><span class="sidebar-status-label">Trades</span><span class="{t_cls}">{t_val}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

page = st.session_state.page

# ===========================================================================
# PAGE: REBALANCE
# ===========================================================================

if page == "rebalance":

    st.markdown(
        '<div class="page-heading">'
        '<div><h1>Rebalance</h1>'
        '<div class="sub">Load a portfolio, select a model, generate trades</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Step 1: Portfolio ────────────────────────────────────────────────
    has_portfolio = bool(st.session_state.portfolios)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    _card_header("1", "Load Client Portfolio",
                 "Upload a holdings CSV or use the built-in sample data.", done=has_portfolio)

    up1, up2 = st.columns([3, 2])
    with up1:
        use_sample = st.checkbox("Use built-in sample data")
        if use_sample:
            if st.button("Load sample portfolio", type="primary"):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                                  delete=False, encoding="utf-8") as f:
                    f.write(SAMPLE_CSV); tmp = f.name
                try:
                    ps = load_portfolios(tmp)
                    st.session_state.portfolios = ps
                    st.session_state.drift_reports = []
                    st.session_state.trade_results = []
                    st.success(f"Loaded {len(ps)} account(s).")
                except Exception as exc:
                    st.error(str(exc))
                finally:
                    os.unlink(tmp)
        else:
            up = st.file_uploader("Upload holdings CSV", type=["csv"],
                                   label_visibility="collapsed",
                                   help="Required: account_id, ticker, quantity, price, cash_balance")
            if up:
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
                    f.write(up.read()); tmp = f.name
                try:
                    ps = load_portfolios(tmp)
                    st.session_state.portfolios = ps
                    st.session_state.drift_reports = []
                    st.session_state.trade_results = []
                    st.success(f"Loaded {len(ps)} account(s).")
                except ValueError as exc:
                    st.error(str(exc))
                finally:
                    os.unlink(tmp)

    with up2:
        with st.expander("CSV format reference"):
            st.code(
                "account_id,ticker,quantity,price,cash_balance\n"
                "HIN001,VAS,480.0000,104.0000,5000.0000\n"
                "HIN001,VGS,265.0000,140.5000,5000.0000",
                language="text",
            )

    if has_portfolio:
        with st.expander(f"{len(st.session_state.portfolios)} account(s) loaded"):
            for p in st.session_state.portfolios:
                st.caption(
                    f"**{p.account_id}**  ·  {len(p.holdings)} holdings  ·  "
                    f"Value: ${p.total_value():,.2f}  ·  Cash: ${p.cash_balance:,.2f}"
                )
    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">📂</div>'
            '<div class="msg">No portfolio loaded</div>'
            '<div class="hint">Upload a CSV file or use the sample data above</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 2: Model ────────────────────────────────────────────────────
    has_model = bool(st.session_state.model)
    saved_keys = list(st.session_state.saved_models.keys())

    st.markdown('<div class="card">', unsafe_allow_html=True)
    _card_header("2", "Select Model Portfolio",
                 "Choose the target model to rebalance toward.", done=has_model)

    if not saved_keys:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">📁</div>'
            '<div class="msg">No models saved</div>'
            '<div class="hint">Go to Models to create one</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        current_label = (
            f"{st.session_state.model.model_id} ({st.session_state.model.version})"
            if st.session_state.model else saved_keys[0]
        )
        default_idx = saved_keys.index(current_label) if current_label in saved_keys else 0

        mc1, mc2 = st.columns([3, 2])
        with mc1:
            sel_key = st.selectbox("Model", options=saved_keys, index=default_idx,
                                    label_visibility="collapsed")
            sel_model = st.session_state.saved_models[sel_key]
            mid = sel_model.model_id
            if mid in MODEL_DESCRIPTIONS:
                _, level = MODEL_DESCRIPTIONS[mid][1], MODEL_DESCRIPTIONS[mid][2]
                st.markdown(_model_desc_html(mid), unsafe_allow_html=True)
                st.markdown(_risk_bar(level), unsafe_allow_html=True)

        with mc2:
            st.write("")
            if st.button("Use this model", type="primary", use_container_width=True):
                st.session_state.model = sel_model
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.rerun()
            if st.session_state.model:
                m = st.session_state.model
                st.caption(f"Active: **{m.model_id}** v{m.version} · {len(m.holdings)} holdings")

        with st.expander("View holdings"):
            st.dataframe(
                pd.DataFrame([
                    {"Ticker": h.ticker, "Target (%)": f"{h.target_weight*100:.4f}%"}
                    for h in sel_model.holdings
                ]),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 3: Run ──────────────────────────────────────────────────────
    has_results = bool(st.session_state.drift_reports)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    _card_header("3", "Run Rebalance", "Configure parameters and execute.", done=has_results)

    rs1, rs2, rs3, rs4 = st.columns(4)
    with rs1:
        drift_threshold = st.slider("Drift threshold (%)", 0.5, 10.0, 3.0, 0.5,
                                     help="Holdings deviating more than this are flagged") / 100.0
    with rs2:
        min_trade_value = st.number_input("Min trade value (AUD)",
                                           min_value=0.0, value=500.0, step=100.0)
    with rs3:
        fractional = st.checkbox("Fractional shares", value=True)
    with rs4:
        dp = int(st.number_input("Qty decimal places", 1, 6, 4, 1)) if fractional else 0

    ready = has_model and has_portfolio
    st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)

    if st.button(
        "▶  Run Rebalance" if not has_results else "↺  Re-run Rebalance",
        type="primary",
        disabled=not ready,
        use_container_width=True,
    ):
        config = TradeConfig(
            drift_threshold=drift_threshold,
            min_trade_value=min_trade_value,
            whole_shares=not fractional,
            managed_fund_dp=dp,
        )
        with st.spinner("Calculating drift and generating trades..."):
            d_reports, t_results = [], []
            for portfolio in st.session_state.portfolios:
                try:
                    dr = calculate_drift(portfolio, st.session_state.model, threshold=drift_threshold)
                    d_reports.append(dr)
                except ValueError as exc:
                    st.error(f"{portfolio.account_id}: {exc}"); continue
                try:
                    tr = generate_trades(portfolio, st.session_state.model, config)
                    t_results.append(tr)
                except ValueError as exc:
                    st.error(f"{portfolio.account_id}: {exc}")
            st.session_state.drift_reports = d_reports
            st.session_state.trade_results = t_results
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if not ready:
        missing = []
        if not has_portfolio: missing.append("load a portfolio")
        if not has_model: missing.append("select a model")
        st.caption(f"Complete steps above to run: {' and '.join(missing)}.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 4: Results ──────────────────────────────────────────────────
    drift_reports = st.session_state.drift_reports
    trade_results = st.session_state.trade_results
    all_trades = [t for tr in trade_results for t in tr.trades]

    if not drift_reports and ready:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">📊</div>'
            '<div class="msg">No results yet</div>'
            '<div class="hint">Click Run Rebalance above to calculate drift and generate trades</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if drift_reports:
        gross_buy  = sum(t.estimated_value for t in all_trades if t.action == "BUY")
        gross_sell = sum(t.estimated_value for t in all_trades if t.action == "SELL")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        _card_header("4", "Results", "Review drift analysis and trade instructions.", done=bool(all_trades))

        # 3 clean summary metrics
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric(
            "Accounts needing rebalance",
            f"{sum(1 for dr in drift_reports if dr.requires_rebalance)} / {len(drift_reports)}",
        )
        sm2.metric("Total trades generated", len(all_trades))
        sm3.metric("Net cash flow", f"${gross_sell - gross_buy:+,.2f}")

        st.write("")

        # Collapsible account cards
        for dr in drift_reports:
            tr = next((x for x in trade_results if x.account_id == dr.account_id), None)
            needs = dr.requires_rebalance
            shortfall = tr.has_funding_shortfall if tr else False

            flags_html = ""
            if needs:
                flags_html += f'&nbsp;{_pill(f"{dr.flagged_count} flagged", "over")}'
            else:
                flags_html += f'&nbsp;{_pill("In band", "ok")}'
            if shortfall:
                flags_html += f'&nbsp;{_pill("Shortfall", "divest")}'

            header_cls = "flagged" if needs else "ok"

            with st.expander(
                f"**{dr.account_id}**  ·  ${dr.total_portfolio_value:,.2f}  ·  {len(dr.holdings)} holdings",
                expanded=needs,
            ):
                # Account metrics
                ca, cb, cc, cd = st.columns(4)
                ca.metric("Portfolio value", f"${dr.total_portfolio_value:,.2f}")
                cb.metric("Holdings", len(dr.holdings))
                cc.metric("Flagged", dr.flagged_count)
                if tr:
                    cd.metric(
                        "⚠ Shortfall" if shortfall else "Closing cash",
                        f"${tr.closing_cash:,.2f}",
                    )
                if shortfall:
                    st.warning(
                        f"Funding shortfall of ${abs(tr.closing_cash):,.2f}. "
                        "Buy cost exceeds available cash. Adviser review required."
                    )

                st.write("")
                st.markdown("**Drift Analysis**")

                # Build drift rows with Status as first data col after Ticker
                drift_rows = []
                for h in dr.holdings:
                    drift_rows.append({
                        "Ticker":       h.ticker,
                        "Status":       h.status.value,
                        "Current (%)":  round(h.current_weight * 100, 4),
                        "Target (%)":   round(h.target_weight * 100, 4),
                        "Drift (pp)":   round(h.drift * 100, 4),
                        "Mkt Value ($)": round(h.market_value, 2),
                    })

                drift_df = pd.DataFrame(drift_rows)
                st.dataframe(
                    drift_df.style
                    .apply(_colour_drift_row, axis=1)
                    .map(_colour_drift_val, subset=["Drift (pp)"])
                    .format({
                        "Current (%)":  "{:.4f}%",
                        "Target (%)":   "{:.4f}%",
                        "Drift (pp)":   "{:+.4f}pp",
                        "Mkt Value ($)": "${:,.2f}",
                    })
                    .hide(axis="index"),
                    use_container_width=True,
                    height=min(60 + len(drift_rows) * 38, 440),
                )

                if dr.requires_rebalance:
                    st.caption(
                        f"⚑  {dr.flagged_count} holding(s) exceed "
                        f"the {drift_threshold*100:.1f}pp threshold. "
                        f"Row colour: amber = overweight, blue = underweight, red = not in model."
                    )
                else:
                    st.caption("✓  All holdings within drift band. No rebalance required.")

                # Trades
                if tr and tr.trades:
                    st.write("")
                    st.markdown("**Trade Instructions**")

                    tf1, tf2, tf3, tf4 = st.columns(4)
                    tf1.metric("Opening cash",  f"${tr.opening_cash:,.2f}")
                    tf2.metric("Sell proceeds", f"${tr.sell_proceeds:,.2f}")
                    tf3.metric("Buy cost",      f"${tr.buy_cost:,.2f}")
                    tf4.metric(
                        "⚠ Shortfall" if shortfall else "Closing cash",
                        f"${tr.closing_cash:,.2f}",
                    )

                    settle = _settle_date()
                    trows = [{
                        "Action":      t.action,
                        "Ticker":      t.ticker,
                        "Quantity":    round(t.quantity, 4),
                        "Est. Value":  round(t.estimated_value, 2),
                        "Settlement":  settle,
                    } for t in tr.trades]

                    st.dataframe(
                        pd.DataFrame(trows)
                        .style
                        .apply(_colour_trade_row, axis=1)
                        .map(_colour_action, subset=["Action"])
                        .format({
                            "Quantity":   "{:.4f}",
                            "Est. Value": "${:,.2f}",
                        })
                        .hide(axis="index"),
                        use_container_width=True,
                        height=min(60 + len(trows) * 38, 340),
                    )

                if tr and tr.suppressed_trades:
                    with st.expander(f"Suppressed trades ({len(tr.suppressed_trades)})"):
                        st.dataframe(
                            pd.DataFrame([{
                                "Action":    td.action,
                                "Ticker":    td.ticker,
                                "Raw value": round(td.raw_trade_value, 2),
                                "Reason":    td.suppression_reason or "",
                            } for td in tr.suppressed_trades]),
                            use_container_width=True, hide_index=True,
                        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Sticky download bar ──────────────────────────────────────────────
    if all_trades and st.session_state.model:
        model = st.session_state.model
        tmp_dl = tempfile.mktemp(suffix=".csv")
        export_trades_csv(all_trades, tmp_dl, include_metadata=True)
        with open(tmp_dl, "rb") as f:
            csv_bytes = f.read()
        os.unlink(tmp_dl)

        gross_buy  = sum(t.estimated_value for t in all_trades if t.action == "BUY")
        gross_sell = sum(t.estimated_value for t in all_trades if t.action == "SELL")

        st.markdown('<div class="sticky-download">', unsafe_allow_html=True)
        dl1, dl2 = st.columns([2, 5])
        with dl1:
            st.download_button(
                label="⬇  Download Trade Instructions (CSV)",
                data=csv_bytes,
                file_name=f"trades_{model.model_id}_{model.version}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.markdown(
                f'<div class="dl-summary">'
                f'<strong>{len(all_trades)}</strong> trades  ·  '
                f'<strong>{len(set(t.account_id for t in all_trades))}</strong> accounts  ·  '
                f'Gross buy <strong>${gross_buy:,.2f}</strong>  ·  '
                f'Gross sell <strong>${gross_sell:,.2f}</strong>  ·  '
                f'Net <strong>${gross_sell - gross_buy:+,.2f}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: MODELS
# ===========================================================================

elif page == "models":

    st.markdown(
        '<div class="page-heading">'
        '<div><h1>Model Portfolios</h1>'
        '<div class="sub">Create and manage target model portfolios</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    saved_models = st.session_state.saved_models
    saved_keys = list(saved_models.keys())

    if saved_keys:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Saved Models")

        current_label = (
            f"{st.session_state.model.model_id} ({st.session_state.model.version})"
            if st.session_state.model else saved_keys[0]
        )
        default_idx = saved_keys.index(current_label) if current_label in saved_keys else 0

        sel_key = st.selectbox("Select model", options=saved_keys, index=default_idx)
        sel_model = saved_models[sel_key]
        mid = sel_model.model_id

        if mid in MODEL_DESCRIPTIONS:
            _, level = MODEL_DESCRIPTIONS[mid][1], MODEL_DESCRIPTIONS[mid][2]
            st.markdown(_model_desc_html(mid), unsafe_allow_html=True)
            st.markdown(_risk_bar(level), unsafe_allow_html=True)
            st.write("")

        st.dataframe(
            pd.DataFrame([
                {"Ticker": h.ticker, "Target Weight (%)": round(h.target_weight*100, 4)}
                for h in sel_model.holdings
            ]),
            use_container_width=True, hide_index=True,
        )

        mc1, mc2 = st.columns([3, 1])
        with mc1:
            if st.button("Set as active model", type="primary", use_container_width=True):
                st.session_state.model = sel_model
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Active model set to: {sel_key}")
        with mc2:
            if len(saved_keys) > 1:
                confirm_key = f"confirm_del_{sel_key}"
                if st.button("Delete", use_container_width=True):
                    if confirm_key not in st.session_state:
                        st.session_state[confirm_key] = True
                        st.warning("Click Delete again to confirm.")
                    else:
                        del st.session_state.saved_models[sel_key]
                        del st.session_state[confirm_key]
                        if st.session_state.model:
                            lbl = f"{st.session_state.model.model_id} ({st.session_state.model.version})"
                            if lbl == sel_key:
                                st.session_state.model = None
                                st.session_state.drift_reports = []
                                st.session_state.trade_results = []
                        st.rerun()

        if st.session_state.model:
            st.caption(
                f"Active: **{st.session_state.model.model_id}** "
                f"v{st.session_state.model.version}  ·  "
                f"{len(st.session_state.model.holdings)} holdings"
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Create New Model")

    nm1, nm2 = st.columns(2)
    with nm1:
        new_id = st.text_input("Model ID", placeholder="e.g. CONSERVATIVE_30")
    with nm2:
        new_ver = st.text_input("Version", placeholder="e.g. 2024-Q3")

    st.caption("Enter target weights — must sum to 100%")

    m_wdf = st.data_editor(
        pd.DataFrame(SAMPLE_MODEL_WEIGHTS, columns=["Ticker", "Weight"]),
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Weight": st.column_config.NumberColumn(
                "Weight (%)", min_value=0.0001, max_value=100.0, step=0.01, format="%.4f"),
        },
        hide_index=True, use_container_width=True, key="wdf_models",
    )

    wsum = m_wdf["Weight"].sum() if not m_wdf.empty else 0.0
    wdelta = wsum - 100.0
    if abs(wdelta) < 0.05:
        st.success(f"Weights sum: {wsum:.4f}% ✓")
    else:
        st.warning(f"Weights sum: {wsum:.4f}%  (need 100%,  {wdelta:+.4f}%)")

    if st.button("Save Model", type="primary"):
        if not new_id.strip():
            st.error("Model ID is required.")
        elif not new_ver.strip():
            st.error("Version is required.")
        else:
            try:
                holdings = [
                    ModelHolding(ticker=str(r["Ticker"]).strip().upper(), target_weight=float(r["Weight"])/100.0)
                    for _, r in m_wdf.iterrows()
                    if str(r["Ticker"]).strip() and str(r["Ticker"]).strip().lower() != "nan"
                ]
                new_model = ModelPortfolio(model_id=new_id.strip(), version=new_ver.strip(), holdings=holdings)
                label = f"{new_model.model_id} ({new_model.version})"
                st.session_state.saved_models[label] = new_model
                st.session_state.model = new_model
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Saved and activated: {label}")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: CLIENT PROFILE
# ===========================================================================

elif page == "profile":

    st.markdown(
        '<div class="page-heading">'
        '<div><h1>Client Profile</h1>'
        '<div class="sub">Optional — risk questionnaire to determine suitable model portfolio</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    scored_qs = [q for q in QUESTIONS if not q.get("no_score", False)]
    step = st.session_state.q_step
    answers = dict(st.session_state.q_answers)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col_cn, col_dt = st.columns([3, 2])
    with col_cn:
        client_name = st.text_input(
            "Client name / reference",
            value=st.session_state.q_client_name,
            placeholder="e.g. John Smith or HIN001",
        )
        st.session_state.q_client_name = client_name
    with col_dt:
        st.metric("Assessment date", _utc_now().split(" ")[0])

    st.progress(min(step, len(QUESTIONS)) / len(QUESTIONS),
                text=f"Question {min(step+1, len(QUESTIONS))} of {len(QUESTIONS)}")

    st.markdown('</div>', unsafe_allow_html=True)

    if step < len(QUESTIONS):
        q = QUESTIONS[step]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**Section {q['section']} — {q['label']}**")
        st.markdown(f"### {q['text']}")

        options = [opt[0] for opt in q["options"]]
        current_idx = answers.get(q["id"], None)
        selected = st.radio("Select an answer", options=options,
                             index=current_idx, key=f"qr_{step}",
                             label_visibility="collapsed")

        st.write("")
        n1, n2, n3 = st.columns([1, 1, 5])
        with n1:
            if step > 0 and st.button("← Back", use_container_width=True):
                st.session_state.q_step = step - 1
                st.rerun()
        with n2:
            label = "Next →" if step < len(QUESTIONS)-1 else "See Results →"
            if st.button(label, type="primary", use_container_width=True, disabled=selected is None):
                if selected is not None:
                    answers[q["id"]] = options.index(selected)
                    st.session_state.q_answers = answers
                st.session_state.q_step = step + 1
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        all_answered = all(q["id"] in answers for q in scored_qs)
        if not all_answered:
            st.warning("Some scored questions were not answered.")
            if st.button("← Back to questions"):
                st.session_state.q_step = 0; st.rerun()
        else:
            score, flags, conflicts = _compute_score(answers, st.session_state.weights)
            base_model = _score_to_model(score, answers)
            display_model = base_model
            if "income" in flags: display_model += " — Income / Yield"
            if "esg"    in flags: display_model += " (ESG)"

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Assessment Result")

            level = RISK_LEVEL.get(base_model, 3)
            r1, r2 = st.columns([2, 3])
            with r1:
                st.metric("Recommended model", base_model)
                st.markdown(_risk_bar(level), unsafe_allow_html=True)
                if flags:
                    st.caption(f"Flags: {', '.join(f.upper() for f in flags)}")
            with r2:
                with st.expander("Score breakdown"):
                    rows = []
                    for sec, lbl in SECTION_LABELS.items():
                        sec_qs = [q for q in QUESTIONS if q["section"]==sec and not q.get("no_score",False)]
                        pts = [q["options"][answers[q["id"]]][1] for q in sec_qs if q["id"] in answers]
                        if pts:
                            avg = sum(pts)/len(pts)
                            w = st.session_state.weights.get(sec,1.0)
                            rows.append({"Section": f"{sec} — {lbl}", "Score (1–5)": round(avg,2), "Weight": w})
                    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if conflicts:
                st.markdown(
                    '<div class="conflict-box"><strong>Conflicts detected — adviser review required</strong><br>'
                    + "<br>".join(f"• {c}" for c in conflicts)
                    + '</div>', unsafe_allow_html=True,
                )

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Adviser Override")
            st.caption("Override the recommended model if required. A reason must be documented.")

            ov_opts = [
                "— Use recommended model —","Conservative","Moderate / Balanced",
                "Growth","High Growth","Aggressive","Income / Yield",
                "ESG","Australian Equities","Index / Passive",
            ]
            ov_sel = st.selectbox("Override selection", options=ov_opts, index=0)
            ov_reason = ""
            if ov_sel != "— Use recommended model —":
                ov_reason = st.text_area("Reason for override (required)",
                                          placeholder="Document why the recommended model is not appropriate...")
                if not ov_reason.strip(): st.warning("A reason must be provided.")

            final_model = ov_sel if ov_sel != "— Use recommended model —" else display_model
            is_overridden = ov_sel != "— Use recommended model —"
            st.info(f"**Final model: {final_model}**"
                    + ("  *(adviser override)*" if is_overridden else "  *(system recommended)*"))

            can_save = not is_overridden or bool(ov_reason.strip())
            sc1, sc2, sc3 = st.columns(3)

            with sc1:
                if st.button("Save Profile", type="primary", use_container_width=True, disabled=not can_save):
                    st.session_state.q_result = {
                        "client_name": client_name, "timestamp": _utc_now(),
                        "version": QUESTIONNAIRE_VERSION, "answers": dict(answers),
                        "score": score, "flags": list(flags), "conflicts": conflicts,
                        "base_model": base_model, "display_model": display_model,
                        "final_model": final_model, "overridden": is_overridden,
                        "override_reason": ov_reason.strip(),
                        "weights_used": dict(st.session_state.weights),
                    }
                    st.success("Profile saved.")

            with sc2:
                if st.session_state.q_result:
                    res = st.session_state.q_result
                    lines = [["Risk Profile Assessment"],["Client",res["client_name"]],
                             ["Date",res["timestamp"]],["Version",res["version"]],[],
                             ["Question","Answer","Points"]]
                    for q in QUESTIONS:
                        if q["id"] in res["answers"]:
                            idx = res["answers"][q["id"]]
                            ot, pts = q["options"][idx]
                            lines.append([q["text"], ot, "" if q.get("no_score") else pts])
                    lines += [[],["Score",res["score"]],["Recommended",res["display_model"]],
                              ["Final",res["final_model"]],["Override","Yes" if res["overridden"] else "No"]]
                    if res["overridden"]: lines.append(["Override reason",res["override_reason"]])
                    if res["conflicts"]:  lines.append(["Conflicts","; ".join(res["conflicts"])])
                    buf = io.StringIO(); csv.writer(buf).writerows(lines)
                    fname = f"risk_profile_{client_name.replace(' ','_') or 'client'}_{res['timestamp'][:10]}.csv"
                    st.download_button("Download Summary (CSV)", data=buf.getvalue().encode("utf-8-sig"),
                                       file_name=fname, mime="text/csv", use_container_width=True)
                else:
                    st.button("Download Summary (CSV)", disabled=True, use_container_width=True)

            with sc3:
                if st.button("Start New Assessment", use_container_width=True):
                    st.session_state.q_answers = {}
                    st.session_state.q_step = 0
                    st.session_state.q_result = None
                    st.session_state.q_client_name = ""
                    st.rerun()

            if st.session_state.q_result:
                st.divider()
                st.caption(
                    f"Go to **Models** to activate a model matching **{final_model}**, "
                    f"then run the rebalance."
                )

            st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# PAGE: SETTINGS
# ===========================================================================

elif page == "settings":

    st.markdown(
        '<div class="page-heading">'
        '<div><h1>Settings</h1>'
        '<div class="sub">Questionnaire weighting and principal controls</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    locked = st.session_state.weights_locked
    if locked:
        st.error("Weightings are locked by the practice principal.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Section Weights")
    st.caption(
        f"Adjust how much each section contributes to the final risk score. "
        f"Range: {WEIGHT_MIN}x – {WEIGHT_MAX}x. Default: 1.0x (equal weight)."
    )

    new_weights = {}
    wc1, wc2 = st.columns(2)
    for i, (sec, label) in enumerate(SECTION_LABELS.items()):
        col = wc1 if i % 2 == 0 else wc2
        with col:
            new_weights[sec] = st.slider(
                f"{sec} — {label}", WEIGHT_MIN, WEIGHT_MAX,
                float(st.session_state.weights.get(sec, 1.0)), 0.1,
                disabled=locked, key=f"ws_{sec}",
            )

    total_w = sum(new_weights.values())
    st.markdown("**Effective contribution to final score**")
    st.dataframe(
        pd.DataFrame([{
            "Section": f"{s} — {SECTION_LABELS[s]}",
            "Weight": new_weights[s],
            "Contribution": f"{new_weights[s]/total_w*100:.1f}%",
        } for s in SECTION_LABELS]),
        use_container_width=False, hide_index=True,
    )

    note_col, btn_col = st.columns([3, 2])
    with note_col:
        change_note = st.text_input(
            "Reason for change (required)",
            placeholder="e.g. Practice policy update",
            disabled=locked,
        )
    with btn_col:
        st.write(""); st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Save", type="primary",
                         disabled=locked or not change_note.strip(),
                         use_container_width=True):
                st.session_state.weight_log.append({
                    "timestamp": _utc_now(), "weights": dict(new_weights), "note": change_note.strip()
                })
                st.session_state.weights = dict(new_weights)
                st.success("Saved.")
        with b2:
            if st.button("Reset", disabled=locked, use_container_width=True):
                st.session_state.weights = dict(DEFAULT_WEIGHTS)
                st.session_state.weight_log.append({
                    "timestamp": _utc_now(), "weights": dict(DEFAULT_WEIGHTS),
                    "note": "Restored to default equal weighting"
                })
                st.success("Reset."); st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Principal Controls")
    st.caption("Lock weightings to prevent advisers from making changes.")

    if locked:
        if st.button("🔓 Unlock Weightings"):
            st.session_state.weights_locked = False; st.rerun()
    else:
        if st.button("🔒 Lock Weightings"):
            st.session_state.weights_locked = True; st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Change Log")
    if st.session_state.weight_log:
        log_rows = []
        for entry in reversed(st.session_state.weight_log):
            row = {"Timestamp": entry["timestamp"], "Note": entry["note"]}
            row.update({s: entry["weights"].get(s, 1.0) for s in SECTION_LABELS})
            log_rows.append(row)
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No changes recorded yet.")
    st.markdown('</div>', unsafe_allow_html=True)
