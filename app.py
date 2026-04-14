"""
app.py — Portfolio Rebalancing Engine
Full-featured adviser tool.

Features:
  1.  Client records
  2.  Rebalance history + audit log
  3.  Trade status tracking
  4.  Pre/post comparison
  5.  Multi-account per client
  6.  Cash flow input
  7.  Drift alert dashboard
  8.  CGT flag on sells
  9.  Model change workflow
  10. Bulk rebalance
  11. Per-client ticker exclusions
  13. Rebalance scheduler
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

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
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
NAVY  = "#0f1f3d"
BLUE  = "#2754F5"
BLUE_L= "#e8effe"
OFF_W = "#f7f8fc"
WHITE = "#ffffff"
BORDER= "#e4e7ef"
MUTED = "#6b7280"
TEXT  = "#1a2233"
LIGHT = "#f0f3ff"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family:'IBM Plex Sans',sans-serif; font-size:14px; color:{TEXT}; }}
.stApp {{ background:{OFF_W}; }}
.block-container {{ padding:2rem 2.5rem 6rem 2.5rem !important; max-width:1280px; }}

/* Hide sidebar toggle */
section[data-testid="stSidebar"] {{ display:none !important; }}
button[data-testid="baseButton-headerNoPadding"] {{ display:none !important; }}
div[data-testid="collapsedControl"] {{ display:none !important; }}

/* Tabs */
button[data-baseweb="tab"] {{
    font-family:'IBM Plex Sans',sans-serif !important;
    font-size:0.82rem !important; font-weight:500 !important;
    color:{MUTED} !important; padding:0.6rem 1rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color:{NAVY} !important; font-weight:600 !important;
    border-bottom-color:{BLUE} !important;
}}

/* Page heading */
.page-heading {{ margin-bottom:1.5rem; }}
.page-heading h1 {{
    font-size:1.75rem; font-weight:600; color:{NAVY};
    margin:0 0 0.2rem 0; letter-spacing:-0.02em;
}}
.page-heading .sub {{ font-size:0.82rem; color:{MUTED}; }}

/* Section headers */
.section-title {{
    font-size:0.72rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; color:{MUTED}; margin:1.5rem 0 0.75rem 0;
    padding-bottom:0.4rem; border-bottom:1px solid {BORDER};
}}

/* Step headers */
.step-header {{
    display:flex; align-items:center; gap:0.65rem;
    margin:0 0 1rem 0; padding-bottom:0.75rem; border-bottom:1px solid {BORDER};
}}
.step-num {{
    width:22px; height:22px; border-radius:50%;
    border:1.5px solid {BLUE}; color:{BLUE}; background:{WHITE};
    display:inline-flex; align-items:center; justify-content:center;
    font-size:0.65rem; font-weight:700; flex-shrink:0;
    font-family:'IBM Plex Mono',monospace;
}}
.step-num-done {{
    width:22px; height:22px; border-radius:50%;
    background:{BLUE}; color:{WHITE};
    display:inline-flex; align-items:center; justify-content:center;
    font-size:0.65rem; font-weight:700; flex-shrink:0;
}}
.step-title {{ font-size:0.88rem; font-weight:600; color:{NAVY}; }}
.step-sub   {{ font-size:0.73rem; color:{MUTED}; margin-top:0.1rem; }}

/* Metrics */
div[data-testid="metric-container"] {{
    background:{WHITE}; border:1px solid {BORDER};
    border-top:2px solid {BLUE}; border-radius:6px;
    padding:0.85rem 1rem !important;
}}
div[data-testid="metric-container"] label {{
    font-size:0.65rem !important; text-transform:uppercase;
    letter-spacing:0.07em; color:{MUTED} !important;
    font-family:'IBM Plex Mono',monospace !important;
}}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size:1.25rem !important; font-family:'IBM Plex Mono',monospace !important;
    color:{NAVY} !important; font-weight:600 !important;
}}

/* Pills */
.pill {{
    display:inline-flex; align-items:center; padding:1px 8px;
    border-radius:20px; font-size:0.67rem; font-weight:700;
    letter-spacing:0.05em; text-transform:uppercase;
    font-family:'IBM Plex Mono',monospace; white-space:nowrap;
}}
.pill-over  {{ background:#fef3c7; color:#92400e; }}
.pill-under {{ background:{BLUE_L}; color:{BLUE}; }}
.pill-divest{{ background:#fee2e2; color:#991b1b; }}
.pill-ok    {{ background:#f0fdf4; color:#166534; }}
.pill-warn  {{ background:#fff3cd; color:#7d5a00; }}
.pill-info  {{ background:{BLUE_L}; color:{BLUE}; }}
.pill-pending {{ background:#f3f4f6; color:{MUTED}; }}
.pill-submitted {{ background:#fef3c7; color:#92400e; }}
.pill-executed  {{ background:#f0fdf4; color:#166534; }}

/* Drift status colours */
.drift-red    {{ color:#dc2626; font-weight:700; }}
.drift-amber  {{ color:#d97706; font-weight:700; }}
.drift-green  {{ color:#16a34a; font-weight:700; }}

/* Tables */
thead tr th {{
    background:{LIGHT} !important; color:{NAVY} !important;
    font-size:0.67rem !important; text-transform:uppercase;
    letter-spacing:0.07em; font-family:'IBM Plex Mono',monospace !important;
    border-bottom:1px solid {BORDER} !important; padding:0.5rem 0.75rem !important;
    font-weight:600 !important;
}}
tbody tr td {{
    font-size:0.81rem !important; font-family:'IBM Plex Mono',monospace !important;
    border-bottom:1px solid #f5f6fa !important; padding:0.45rem 0.75rem !important;
}}
tbody tr:hover td {{ background:#fafbff !important; }}

/* Buttons */
div[data-testid="stButton"] button {{
    font-family:'IBM Plex Sans',sans-serif !important;
    font-size:0.82rem !important; font-weight:500 !important;
    border-radius:5px !important; box-shadow:none !important;
}}
div[data-testid="stButton"] button[kind="primary"] {{
    background:{NAVY} !important; color:{WHITE} !important;
    border:none !important; font-weight:600 !important;
    padding:0.45rem 1.1rem !important;
}}
div[data-testid="stButton"] button[kind="primary"]:hover {{ background:#1a3260 !important; }}
div[data-testid="stButton"] button[kind="primary"]:disabled {{ background:#c8cdd8 !important; }}
div[data-testid="stButton"] button[kind="secondary"] {{
    background:{WHITE} !important; color:{TEXT} !important;
    border:1px solid {BORDER} !important; padding:0.45rem 1.1rem !important;
}}
div[data-testid="stButton"] button[kind="secondary"]:hover {{
    border-color:{BLUE} !important; color:{BLUE} !important;
}}

/* Run button */
.run-btn-wrap div[data-testid="stButton"] button {{
    width:100% !important; height:46px !important;
    font-size:0.9rem !important; font-weight:600 !important;
    background:{BLUE} !important; border:none !important;
}}
.run-btn-wrap div[data-testid="stButton"] button:hover {{ background:#1f44d6 !important; }}
.run-btn-wrap div[data-testid="stButton"] button:disabled {{ background:#c8cdd8 !important; }}

/* Download button */
div[data-testid="stDownloadButton"] button {{
    background:{WHITE} !important; color:{BLUE} !important;
    border:1.5px solid {BLUE} !important; font-weight:600 !important;
    border-radius:5px !important; padding:0.45rem 1.1rem !important;
}}
div[data-testid="stDownloadButton"] button:hover {{ background:{BLUE_L} !important; }}

/* Sticky download */
.sticky-download {{
    position:fixed; bottom:0; left:0; right:0;
    background:{WHITE}; border-top:1px solid {BORDER};
    padding:0.65rem 2.5rem; display:flex; align-items:center;
    justify-content:space-between; z-index:999;
}}
.dl-summary {{ font-size:0.75rem; font-family:'IBM Plex Mono',monospace; color:{MUTED}; }}
.dl-summary strong {{ color:{NAVY}; }}

/* Client card */
.client-card {{
    background:{WHITE}; border:1px solid {BORDER}; border-radius:8px;
    padding:1rem 1.25rem; margin-bottom:0.75rem;
    display:flex; align-items:center; justify-content:space-between;
    cursor:pointer; transition:border-color 0.15s;
}}
.client-card:hover {{ border-color:{BLUE}; }}
.client-card.selected {{ border-color:{BLUE}; background:{LIGHT}; }}
.client-name {{ font-weight:600; color:{NAVY}; font-size:0.9rem; }}
.client-meta {{ font-size:0.73rem; color:{MUTED}; margin-top:0.15rem; }}

/* Alert box */
.alert-box {{
    background:{LIGHT}; border:1px solid {BLUE};
    border-left:3px solid {BLUE}; border-radius:4px;
    padding:0.65rem 0.9rem; font-size:0.8rem; color:{NAVY};
    margin:0.5rem 0;
}}
.alert-warn {{
    background:#fffbeb; border:1px solid #f59e0b;
    border-left:3px solid #f59e0b; border-radius:4px;
    padding:0.65rem 0.9rem; font-size:0.8rem; color:#451a03;
    margin:0.5rem 0;
}}
.alert-ok {{
    background:#f0fdf4; border:1px solid #86efac;
    border-left:3px solid #16a34a; border-radius:4px;
    padding:0.65rem 0.9rem; font-size:0.8rem; color:#14532d;
    margin:0.5rem 0;
}}

/* Model desc */
.model-desc {{
    background:{LIGHT}; border:1px solid {BORDER};
    border-left:2px solid {BLUE}; border-radius:4px;
    padding:0.5rem 0.8rem; font-size:0.77rem; color:{MUTED};
    margin-top:0.4rem;
}}
.model-desc strong {{ color:{TEXT}; }}

/* Empty state */
.empty-state {{
    text-align:center; padding:2rem 1rem;
    border:1.5px dashed {BORDER}; border-radius:8px;
    background:{WHITE}; margin:0.5rem 0;
}}
.empty-state .icon {{ font-size:1.75rem; margin-bottom:0.4rem; }}
.empty-state .msg  {{ font-size:0.88rem; color:{TEXT}; font-weight:500; }}
.empty-state .hint {{ font-size:0.75rem; color:{MUTED}; margin-top:0.2rem; }}

/* Conflict box */
.conflict-box {{
    background:#fffbeb; border:1px solid #f59e0b;
    border-left:3px solid #d97706; border-radius:4px;
    padding:0.75rem 1rem; margin:0.75rem 0;
    font-size:0.8rem; color:#451a03;
}}

hr {{ border-color:{BORDER} !important; margin:1.25rem 0 !important; }}
div[data-testid="stCaptionContainer"] {{ font-size:0.73rem !important; color:{MUTED} !important; }}
details summary {{ font-size:0.82rem !important; font-weight:500 !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DESCRIPTIONS = {
    "CONSERVATIVE":  ("30% growth / 70% defensive", 1),
    "MODERATE":      ("50% growth / 50% defensive", 2),
    "GROWTH":        ("70% growth / 30% defensive", 3),
    "HIGH_GROWTH":   ("90% growth / 10% defensive", 4),
    "AGGRESSIVE":    ("100% growth",                 5),
    "INCOME":        ("Income / yield focused",       2),
    "ESG":           ("Ethical / ESG screened",       3),
    "AUSTRALIAN_EQ": ("100% Australian equities",     4),
    "INDEX_PASSIVE": ("Low-cost index portfolio",     3),
}

QUESTIONNAIRE_VERSION = "1.0"

QUESTIONS = [
    {"id":"Q1","section":"A","label":"Time Horizon","text":"When do you expect to start drawing on this investment?","options":[("Less than 2 years",1),("2–5 years",2),("5–10 years",3),("10–15 years",4),("More than 15 years",5)]},
    {"id":"Q2","section":"B","label":"Risk Tolerance","text":"If your portfolio lost 20% of its value in a market downturn, what would you most likely do?","options":[("Sell everything immediately",1),("Sell some to reduce exposure",2),("Do nothing and wait",3),("Buy a little more while prices are low",4),("Significantly increase my investment",5)]},
    {"id":"Q3","section":"B","label":"Risk Tolerance","text":"Which statement best describes your attitude toward investment risk?","options":[("I cannot accept any loss of capital",1),("I can accept small short-term losses for modest returns",2),("I can accept moderate losses for reasonable returns",3),("I can accept significant short-term losses for higher long-term returns",4),("I am comfortable with high volatility in pursuit of maximum returns",5)]},
    {"id":"Q4","section":"C","label":"Risk Capacity","text":"If you lost this entire investment, how would it affect your lifestyle?","options":[("Severely — I depend on it",1),("Significantly — it would cause real hardship",2),("Moderately — I would need to adjust my plans",3),("Mildly — I have other assets to fall back on",4),("Minimally — this is discretionary money",5)]},
    {"id":"Q5","section":"D","label":"Income Stability","text":"How would you describe your current and expected future income?","options":[("Unstable or uncertain",1),("Variable with some uncertainty",2),("Stable but could change",3),("Stable and likely to continue",4),("Very stable or I have multiple income sources",5)]},
    {"id":"Q6","section":"E","label":"Investment Experience","text":"Which best describes your investment experience?","options":[("None — I have never invested before",1),("Limited — I have a savings account or term deposit",2),("Moderate — I have invested in managed funds or super",3),("Experienced — I have invested in shares or ETFs directly",4),("Advanced — I actively manage a diversified investment portfolio",5)]},
    {"id":"Q7","section":"F","label":"Goals","text":"What is the primary purpose of this investment?","options":[("Preserve my capital above all else",1),("Generate regular income",2),("Balanced mix of income and growth",3),("Grow my wealth over the long term",4),("Maximise long-term growth, I do not need income",5)],"flags":{1:"income"}},
    {"id":"Q8","section":"F","label":"ESG Preferences","text":"Do you have any ethical or ESG investment preferences?","options":[("No preference",0),("Somewhat important",0),("Very important — I want to exclude certain industries",0)],"flags":{1:"esg",2:"esg"},"no_score":True},
]
SCORE_MAP = [(6,10,"Conservative"),(11,16,"Moderate / Balanced"),(17,22,"Growth"),(23,28,"High Growth"),(29,30,"Aggressive")]
RISK_LEVEL = {"Conservative":1,"Moderate / Balanced":2,"Growth":3,"High Growth":4,"Aggressive":5}
CONFLICT_RULES = [
    ("Q2",[0,1],"Q7",[3,4],"Low risk tolerance (Q2) conflicts with high growth goals (Q7)."),
    ("Q1",[0],"Q7",[3,4],"Short time horizon (Q1) conflicts with growth-oriented goals (Q7)."),
    ("Q4",[0],None,None,"Client is financially dependent on this investment (Q4). Consider capping at Conservative."),
]
DEFAULT_WEIGHTS = {"A":1.0,"B":1.0,"C":1.0,"D":1.0,"E":1.0,"F":1.0}
WEIGHT_MIN, WEIGHT_MAX = 0.5, 2.0
SECTION_LABELS = {"A":"Time Horizon","B":"Risk Tolerance","C":"Risk Capacity","D":"Income Stability","E":"Investment Experience","F":"Goals"}

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

SAMPLE_MODEL_WEIGHTS = [("CBA",25.0),("BHP",20.0),("CSL",15.0),("WES",15.0),("ANZ",12.0),("MQG",8.0),("FMG",5.0)]

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

REVIEW_FREQUENCIES = ["Monthly", "Quarterly", "Half-yearly", "Annually"]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "saved_models":    {},
        "model":           None,
        "portfolios":      [],
        "drift_reports":   [],
        "trade_results":   [],
        "clients":         {},       # {client_id: client_dict}
        "active_client":   None,     # client_id
        "rebalance_history": [],     # list of history_entry dicts
        "q_answers":       {},
        "q_result":        None,
        "q_client_name":   "",
        "q_step":          0,
        "weights":         dict(DEFAULT_WEIGHTS),
        "weight_log":      [],
        "weights_locked":  False,
        "scheduled_reviews": {},     # {client_id: {frequency, next_date}}
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

# Seed demo clients
if not st.session_state.clients:
    for _cid, _cname, _hin, _risk, _mid in [
        ("c1", "Sarah Mitchell",  "HIN001", "MODERATE",  "MODERATE (2024-Q2)"),
        ("c2", "David Chen",      "HIN002", "GROWTH",    "GROWTH (2024-Q2)"),
        ("c3", "Emma Thompson",   "HIN003", "MODERATE",  "MODERATE (2024-Q2)"),
    ]:
        st.session_state.clients[_cid] = {
            "id":            _cid,
            "name":          _cname,
            "accounts":      [{"hin": _hin, "model_id": _mid, "label": "Investment"}],
            "risk_profile":  _risk,
            "created":       "2024-01-15",
            "last_rebalance": "2024-09-30",
            "review_freq":   "Quarterly",
            "next_review":   "2025-03-31",
            "exclusions":    [],
        }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _settle_date() -> str:
    from datetime import datetime, timezone, timedelta
    d = datetime.now(timezone.utc)
    added = 0
    while added < 2:
        d += timedelta(days=1)
        if d.weekday() < 5: added += 1
    return d.strftime("%d %b %Y")

def _step_header(num: str, title: str, subtitle: str = "", done: bool = False) -> None:
    icon = f'<span class="step-num-done">✓</span>' if done else f'<span class="step-num">{num}</span>'
    st.markdown(
        f'<div class="step-header">{icon}'
        f'<div><div class="step-title">{title}</div>'
        f'{"<div class=step-sub>" + subtitle + "</div>" if subtitle else ""}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

def _pill(label: str, kind: str) -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'

def _model_desc(model_id: str) -> str:
    if model_id not in MODEL_DESCRIPTIONS: return ""
    desc, _ = MODEL_DESCRIPTIONS[model_id]
    return f'<div class="model-desc"><strong>{model_id}</strong> — {desc}</div>'

def _risk_bar(level: int) -> str:
    colours = ["#22c55e","#84cc16","#eab308","#f97316","#ef4444"]
    labels  = ["Conservative","Moderate","Growth","High Growth","Aggressive"]
    segs = "".join(
        f'<div style="flex:1;background:{colours[i]};{"outline:2px solid #0f1f3d;outline-offset:-2px;" if i+1==level else ""}"></div>'
        for i in range(5)
    )
    lbl = labels[level-1] if 1<=level<=5 else ""
    return (
        f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin:0.35rem 0;">{segs}</div>'
        f'<div style="font-size:0.7rem;color:{MUTED};font-family:IBM Plex Mono,monospace;">Profile: <strong style="color:{NAVY}">{lbl}</strong></div>'
    )

def _colour_drift_row(row) -> list:
    s = row.get("Status","")
    if s == "NOT_IN_MODEL": return [f"background:#fff1f2;border-left:3px solid #dc2626"] * len(row)
    if s == "OVERWEIGHT":   return [f"background:#fffbeb;border-left:3px solid #d97706"] * len(row)
    if s == "UNDERWEIGHT":  return [f"background:#eff6ff;border-left:3px solid #2563eb"] * len(row)
    return [""] * len(row)

def _colour_drift_val(val: float) -> str:
    if val > 0: return "font-weight:700;color:#92400e"
    if val < 0: return "font-weight:700;color:#1e3a8a"
    return ""

def _colour_trade_row(row) -> list:
    a = row.get("Action","")
    if a == "BUY":  return ["background:#f0fdf4;border-left:3px solid #16a34a"] * len(row)
    if a == "SELL": return ["background:#fff1f2;border-left:3px solid #dc2626"] * len(row)
    return [""] * len(row)

def _colour_action(val: str) -> str:
    if val == "BUY":  return "color:#065f46;font-weight:700"
    if val == "SELL": return "color:#7f1d1d;font-weight:700"
    return ""

def _drift_status(max_drift_pp: float, threshold: float = 0.03) -> tuple[str, str]:
    """Returns (status_label, pill_kind) based on max drift."""
    if max_drift_pp >= 10.0: return "Critical", "divest"
    if max_drift_pp >= threshold * 100: return "Action needed", "over"
    if max_drift_pp >= (threshold * 100) * 0.7: return "Watch", "warn"
    return "In band", "ok"

def _compute_score(answers: dict, weights: dict) -> tuple:
    section_raw: dict = {}; flags: set = set()
    for q in QUESTIONS:
        qid = q["id"]
        if qid not in answers: continue
        idx = answers[qid]; pts = q["options"][idx][1]
        if not q.get("no_score", False): section_raw.setdefault(q["section"], []).append(pts)
        if "flags" in q and idx in q["flags"]: flags.add(q["flags"][idx])
    total_w = weighted_sum = 0.0
    for sec, pts_list in section_raw.items():
        avg = sum(pts_list)/len(pts_list); w = weights.get(sec, 1.0)
        weighted_sum += avg * w; total_w += w
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

def _run_rebalance_for_portfolio(portfolio: Portfolio, model: ModelPortfolio,
                                  config: TradeConfig, cash_addition: float = 0.0,
                                  exclusions: list = None) -> tuple:
    """Run drift + trades for one portfolio, optionally with cash injection and exclusions."""
    if cash_addition > 0:
        portfolio = Portfolio(
            account_id=portfolio.account_id,
            holdings=portfolio.holdings,
            cash_balance=portfolio.cash_balance + cash_addition,
        )
    if exclusions:
        # Remove excluded tickers from model for this run (sell them, don't rebuy)
        filtered_holdings = [h for h in model.holdings if h.ticker not in exclusions]
        if filtered_holdings:
            total_w = sum(h.target_weight for h in filtered_holdings)
            renorm = [ModelHolding(ticker=h.ticker, target_weight=h.target_weight/total_w)
                      for h in filtered_holdings]
            try:
                model = ModelPortfolio(model_id=model.model_id, version=model.version, holdings=renorm)
            except Exception:
                pass
    dr = calculate_drift(portfolio, model, threshold=config.drift_threshold)
    tr = generate_trades(portfolio, model, config)
    return dr, tr

# ---------------------------------------------------------------------------
# App header + tabs
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="page-heading" style="margin-bottom:0.75rem;">'
    f'<h1 style="font-size:1.4rem;margin-bottom:0.1rem;">Rebalancing Engine</h1>'
    f'<div class="sub">Portfolio Drift &amp; Trade Generation</div>'
    f'</div>',
    unsafe_allow_html=True,
)

tab_dash, tab_clients, tab_rebalance, tab_models, tab_profile, tab_settings = st.tabs([
    "Dashboard", "Clients", "Rebalance", "Models", "Client Profile", "Settings"
])

# ===========================================================================
# TAB: DASHBOARD  (Feature 7 — drift alert dashboard)
# ===========================================================================

with tab_dash:
    st.markdown('<div class="page-heading"><h1>Dashboard</h1><div class="sub">Portfolio drift status across all clients</div></div>', unsafe_allow_html=True)

    clients = st.session_state.clients
    saved_models = st.session_state.saved_models

    if not clients:
        st.markdown('<div class="empty-state"><div class="icon">👥</div><div class="msg">No clients yet</div><div class="hint">Add clients in the Clients tab</div></div>', unsafe_allow_html=True)
    else:
        # Summary metrics
        today = datetime.now(timezone.utc).date()
        overdue = []
        for cid, c in clients.items():
            if c.get("next_review"):
                try:
                    nr = datetime.strptime(c["next_review"], "%Y-%m-%d").date()
                    if nr <= today: overdue.append(c["name"])
                except Exception: pass

        dm1, dm2, dm3, dm4 = st.columns(4)
        dm1.metric("Total clients", len(clients))
        dm2.metric("Reviews overdue", len(overdue))
        dm3.metric("Models in use", len({a["model_id"] for c in clients.values() for a in c.get("accounts",[])}))
        dm4.metric("Total accounts", sum(len(c.get("accounts",[])) for c in clients.values()))

        st.markdown('<div class="section-title">Client drift status</div>', unsafe_allow_html=True)
        st.caption("Load each client's portfolio in the Rebalance tab to calculate live drift. Showing last known status.")

        dash_rows = []
        for cid, c in clients.items():
            for acct in c.get("accounts", []):
                mid = acct.get("model_id","")
                m = saved_models.get(mid)
                last_rb = c.get("last_rebalance","—")
                next_rev = c.get("next_review","—")
                overdue_flag = ""
                if next_rev != "—":
                    try:
                        nr = datetime.strptime(next_rev, "%Y-%m-%d").date()
                        if nr <= today: overdue_flag = "⚠ Overdue"
                    except Exception: pass

                dash_rows.append({
                    "Client":        c["name"],
                    "HIN":           acct["hin"],
                    "Account":       acct.get("label","Investment"),
                    "Model":         mid.split(" (")[0] if mid else "—",
                    "Risk profile":  c.get("risk_profile","—"),
                    "Last rebalance":last_rb,
                    "Next review":   next_rev,
                    "Review status": overdue_flag if overdue_flag else "On track",
                })

        dash_df = pd.DataFrame(dash_rows)

        def _colour_review(val: str) -> str:
            if "Overdue" in val: return "color:#dc2626;font-weight:700"
            if val == "On track": return "color:#166534"
            return ""

        st.dataframe(
            dash_df.style.map(_colour_review, subset=["Review status"]).hide(axis="index"),
            use_container_width=True,
            height=min(60 + len(dash_rows)*38, 400),
        )

        if overdue:
            st.markdown(
                f'<div class="alert-warn">⚠ <strong>{len(overdue)} client(s)</strong> have overdue reviews: {", ".join(overdue)}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-title">Rebalance history</div>', unsafe_allow_html=True)
        history = st.session_state.rebalance_history
        if not history:
            st.caption("No rebalances recorded yet. Run a rebalance in the Rebalance tab to see history here.")
        else:
            hist_rows = []
            for h in reversed(history[-50:]):
                hist_rows.append({
                    "Timestamp":  h["timestamp"],
                    "Client":     h["client_name"],
                    "HIN":        h["account_id"],
                    "Model":      h["model_id"],
                    "Trades":     h["trade_count"],
                    "Buy value":  f"${h['buy_value']:,.2f}",
                    "Sell value": f"${h['sell_value']:,.2f}",
                    "Net":        f"${h['net']:+,.2f}",
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True,
                         height=min(60+len(hist_rows)*38, 350))

# ===========================================================================
# TAB: CLIENTS  (Features 1, 2, 3, 5, 9)
# ===========================================================================

with tab_clients:
    st.markdown('<div class="page-heading"><h1>Clients</h1><div class="sub">Manage client records, accounts, and rebalance history</div></div>', unsafe_allow_html=True)

    clients = st.session_state.clients
    saved_keys = list(st.session_state.saved_models.keys())

    cl1, cl2 = st.columns([2, 3])

    # ── Left: client list ────────────────────────────────────────────────
    with cl1:
        st.markdown('<div class="section-title">Client list</div>', unsafe_allow_html=True)

        if not clients:
            st.markdown('<div class="empty-state"><div class="icon">👥</div><div class="msg">No clients yet</div></div>', unsafe_allow_html=True)
        else:
            for cid, c in clients.items():
                selected = st.session_state.active_client == cid
                acct_count = len(c.get("accounts", []))
                next_rev = c.get("next_review","")
                overdue = ""
                if next_rev:
                    try:
                        nr = datetime.strptime(next_rev, "%Y-%m-%d").date()
                        if nr <= datetime.now(timezone.utc).date(): overdue = " · ⚠ Review overdue"
                    except Exception: pass

                if st.button(
                    f"**{c['name']}**\n{c.get('risk_profile','—')} · {acct_count} account(s){overdue}",
                    key=f"client_btn_{cid}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.active_client = cid
                    st.rerun()

        st.write("")
        with st.expander("+ Add new client"):
            new_name = st.text_input("Full name", key="new_client_name")
            new_hin  = st.text_input("Primary HIN", key="new_client_hin")
            new_risk = st.selectbox("Risk profile", ["Conservative","Moderate / Balanced","Growth","High Growth","Aggressive"], key="new_client_risk")
            new_model = st.selectbox("Default model", saved_keys, key="new_client_model") if saved_keys else None
            new_freq = st.selectbox("Review frequency", REVIEW_FREQUENCIES, key="new_client_freq")

            if st.button("Save client", type="primary", key="save_new_client"):
                if new_name.strip() and new_hin.strip():
                    cid = str(uuid.uuid4())[:8]
                    next_rev_date = (datetime.now(timezone.utc) + timedelta(days=90)).strftime("%Y-%m-%d")
                    st.session_state.clients[cid] = {
                        "id": cid, "name": new_name.strip(),
                        "accounts": [{"hin": new_hin.strip().upper(), "model_id": new_model or "", "label": "Investment"}],
                        "risk_profile": new_risk,
                        "created": _today(), "last_rebalance": "—",
                        "review_freq": new_freq, "next_review": next_rev_date,
                        "exclusions": [],
                    }
                    st.session_state.active_client = cid
                    st.success(f"Client '{new_name}' added.")
                    st.rerun()
                else:
                    st.error("Name and HIN are required.")

    # ── Right: client detail ─────────────────────────────────────────────
    with cl2:
        active_id = st.session_state.active_client
        if not active_id or active_id not in clients:
            st.markdown('<div class="empty-state"><div class="icon">👤</div><div class="msg">Select a client</div><div class="hint">Click a client on the left to view details</div></div>', unsafe_allow_html=True)
        else:
            c = clients[active_id]
            st.markdown(f'<div class="section-title">{c["name"]}</div>', unsafe_allow_html=True)

            # Client summary
            cd1, cd2, cd3 = st.columns(3)
            cd1.metric("Risk profile", c.get("risk_profile","—"))
            cd2.metric("Last rebalance", c.get("last_rebalance","—"))
            cd3.metric("Next review", c.get("next_review","—"))

            # ── Accounts (feature 5: multi-account) ─────────────────────
            st.markdown('<div class="section-title">Accounts</div>', unsafe_allow_html=True)
            for i, acct in enumerate(c.get("accounts",[])):
                with st.expander(f"{acct.get('label','Account')} — {acct['hin']}"):
                    a1, a2 = st.columns(2)
                    with a1:
                        new_label = st.text_input("Account label", value=acct.get("label","Investment"), key=f"acct_label_{active_id}_{i}")
                        new_hin   = st.text_input("HIN", value=acct["hin"], key=f"acct_hin_{active_id}_{i}")
                    with a2:
                        cur_mid = acct.get("model_id","")
                        cur_idx = saved_keys.index(cur_mid) if cur_mid in saved_keys else 0
                        new_model = st.selectbox("Model", saved_keys, index=cur_idx, key=f"acct_model_{active_id}_{i}") if saved_keys else ""
                        st.markdown(_model_desc(new_model.split(" (")[0]) if new_model else "", unsafe_allow_html=True)
                    if st.button("Update account", key=f"upd_acct_{active_id}_{i}"):
                        st.session_state.clients[active_id]["accounts"][i] = {
                            "hin": new_hin.strip().upper(),
                            "label": new_label.strip(),
                            "model_id": new_model,
                        }
                        st.success("Account updated.")

            if st.button("+ Add account", key=f"add_acct_{active_id}"):
                st.session_state.clients[active_id]["accounts"].append(
                    {"hin":"","model_id":saved_keys[0] if saved_keys else "","label":"New account"}
                )
                st.rerun()

            # ── Exclusions (feature 11) ──────────────────────────────────
            st.markdown('<div class="section-title">Ticker exclusions</div>', unsafe_allow_html=True)
            st.caption("Tickers listed here will not be bought when rebalancing this client.")
            excl = c.get("exclusions",[])
            excl_str = st.text_input("Excluded tickers (comma-separated)", value=", ".join(excl), key=f"excl_{active_id}")
            if st.button("Save exclusions", key=f"save_excl_{active_id}"):
                new_excl = [t.strip().upper() for t in excl_str.split(",") if t.strip()]
                st.session_state.clients[active_id]["exclusions"] = new_excl
                st.success("Exclusions saved.")

            # ── Model change workflow (feature 9) ────────────────────────
            st.markdown('<div class="section-title">Model change</div>', unsafe_allow_html=True)
            st.caption("Use this to plan a transition between models before executing.")
            mc1, mc2 = st.columns(2)
            with mc1:
                from_model_key = st.selectbox("From model", saved_keys, key=f"from_model_{active_id}") if saved_keys else None
            with mc2:
                to_model_key = st.selectbox("To model", saved_keys, key=f"to_model_{active_id}") if saved_keys else None

            if from_model_key and to_model_key and from_model_key != to_model_key:
                if st.button("Preview transition", key=f"preview_transition_{active_id}"):
                    fm = st.session_state.saved_models[from_model_key]
                    tm = st.session_state.saved_models[to_model_key]
                    from_w = {h.ticker: h.target_weight for h in fm.holdings}
                    to_w   = {h.ticker: h.target_weight for h in tm.holdings}
                    all_t  = set(from_w)|set(to_w)
                    rows = []
                    for t in sorted(all_t):
                        fw = from_w.get(t, 0.0)*100
                        tw = to_w.get(t, 0.0)*100
                        diff = tw - fw
                        action = "BUY" if diff > 0 else ("SELL" if diff < 0 else "—")
                        rows.append({"Ticker":t,"From (%)":round(fw,2),"To (%)":round(tw,2),"Change (pp)":round(diff,2),"Direction":action})
                    df_trans = pd.DataFrame(rows)
                    st.dataframe(
                        df_trans.style.map(_colour_action, subset=["Direction"])
                        .format({"From (%)":"{:.2f}%","To (%)":"{:.2f}%","Change (pp)":"{:+.2f}pp"})
                        .hide(axis="index"),
                        use_container_width=True, hide_index=True,
                    )
                    st.caption(f"Transitioning from {from_model_key.split(' (')[0]} to {to_model_key.split(' (')[0]}. Apply this change in the Rebalance tab.")

            # ── Review schedule (feature 13) ─────────────────────────────
            st.markdown('<div class="section-title">Review schedule</div>', unsafe_allow_html=True)
            rs1, rs2 = st.columns(2)
            with rs1:
                cur_freq = c.get("review_freq","Quarterly")
                freq_idx = REVIEW_FREQUENCIES.index(cur_freq) if cur_freq in REVIEW_FREQUENCIES else 1
                new_freq = st.selectbox("Review frequency", REVIEW_FREQUENCIES, index=freq_idx, key=f"freq_{active_id}")
            with rs2:
                next_rev = c.get("next_review","")
                new_next = st.text_input("Next review date (YYYY-MM-DD)", value=next_rev, key=f"next_rev_{active_id}")
            if st.button("Save schedule", key=f"save_sched_{active_id}"):
                st.session_state.clients[active_id]["review_freq"] = new_freq
                st.session_state.clients[active_id]["next_review"] = new_next
                st.success("Schedule saved.")

            # ── Rebalance history (feature 2) ─────────────────────────────
            st.markdown('<div class="section-title">Rebalance history</div>', unsafe_allow_html=True)
            client_history = [h for h in st.session_state.rebalance_history
                              if h.get("client_id") == active_id]
            if not client_history:
                st.caption("No rebalances recorded for this client yet.")
            else:
                for h in reversed(client_history[-10:]):
                    with st.expander(f"{h['timestamp'][:16]}  ·  {h['account_id']}  ·  {h['trade_count']} trades"):
                        hc1, hc2, hc3 = st.columns(3)
                        hc1.metric("Buy value",  f"${h['buy_value']:,.2f}")
                        hc2.metric("Sell value", f"${h['sell_value']:,.2f}")
                        hc3.metric("Net",        f"${h['net']:+,.2f}")
                        if h.get("trades"):
                            trade_rows = []
                            for t in h["trades"]:
                                trade_rows.append({
                                    "Action":   t["action"],
                                    "Ticker":   t["ticker"],
                                    "Quantity": t["quantity"],
                                    "Est. Value": t["estimated_value"],
                                    "Status":   t.get("status","Pending"),
                                })
                            tdf = pd.DataFrame(trade_rows)

                            def _status_colour(val):
                                if val == "Executed": return "color:#166534;font-weight:600"
                                if val == "Submitted": return "color:#92400e;font-weight:600"
                                return f"color:{MUTED}"

                            st.dataframe(
                                tdf.style
                                .map(_colour_action, subset=["Action"])
                                .map(_status_colour, subset=["Status"])
                                .format({"Quantity":"{:.4f}","Est. Value":"${:,.2f}"})
                                .hide(axis="index"),
                                use_container_width=True, hide_index=True,
                            )

                            # Feature 3: Update trade status
                            st.caption("Update trade status:")
                            ts1, ts2, ts3 = st.columns(3)
                            h_idx = st.session_state.rebalance_history.index(h)
                            with ts1:
                                if st.button("Mark all Submitted", key=f"sub_{h_idx}"):
                                    for t in st.session_state.rebalance_history[h_idx]["trades"]:
                                        t["status"] = "Submitted"
                                    st.rerun()
                            with ts2:
                                if st.button("Mark all Executed", key=f"exe_{h_idx}"):
                                    for t in st.session_state.rebalance_history[h_idx]["trades"]:
                                        t["status"] = "Executed"
                                    st.rerun()
                            with ts3:
                                if st.button("Reset to Pending", key=f"rst_{h_idx}"):
                                    for t in st.session_state.rebalance_history[h_idx]["trades"]:
                                        t["status"] = "Pending"
                                    st.rerun()


# ===========================================================================
# TAB: REBALANCE  (Features 4, 6, 8, 10, 11)
# ===========================================================================

with tab_rebalance:
    st.markdown('<div class="page-heading"><h1>Rebalance</h1><div class="sub">Load a portfolio, select a model, generate trades</div></div>', unsafe_allow_html=True)

    saved_keys = list(st.session_state.saved_models.keys())

    # ── Step 1: Portfolio ────────────────────────────────────────────────
    has_portfolio = bool(st.session_state.portfolios)
    _step_header("1", "Load Portfolio", "Upload a CSV or use the built-in sample data.", done=has_portfolio)

    up1, up2 = st.columns([3, 2])
    with up1:
        use_sample = st.checkbox("Use built-in sample data", value=True)
        if use_sample:
            if st.button("Load sample portfolio", type="primary"):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
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
            up = st.file_uploader("Upload holdings CSV", type=["csv"], label_visibility="collapsed",
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
        with st.expander("CSV format"):
            st.code("account_id,ticker,quantity,price,cash_balance\nHIN001,VAS,480.0000,104.0000,5000.0000", language="text")

    if has_portfolio:
        with st.expander(f"{len(st.session_state.portfolios)} account(s) loaded"):
            for p in st.session_state.portfolios:
                st.caption(f"**{p.account_id}**  ·  {len(p.holdings)} holdings  ·  ${p.total_value():,.2f}  ·  Cash: ${p.cash_balance:,.2f}")

    st.divider()

    # ── Step 2: Model ────────────────────────────────────────────────────
    has_model = bool(st.session_state.model)
    _step_header("2", "Select Model", "Choose the target model portfolio.", done=has_model)

    if not saved_keys:
        st.warning("No models saved. Go to the Models tab to create one.")
    else:
        current_label = f"{st.session_state.model.model_id} ({st.session_state.model.version})" if st.session_state.model else saved_keys[0]
        default_idx = saved_keys.index(current_label) if current_label in saved_keys else 0
        mc1, mc2 = st.columns([3, 2])
        with mc1:
            sel_key = st.selectbox("Model", options=saved_keys, index=default_idx, label_visibility="collapsed")
            sel_model_obj = st.session_state.saved_models[sel_key]
            mid = sel_model_obj.model_id
            if mid in MODEL_DESCRIPTIONS:
                _, level = MODEL_DESCRIPTIONS[mid]
                st.markdown(_model_desc(mid), unsafe_allow_html=True)
                st.markdown(_risk_bar(level), unsafe_allow_html=True)
        with mc2:
            st.write("")
            if st.button("Use this model", type="primary", use_container_width=True):
                st.session_state.model = sel_model_obj
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.rerun()
            if st.session_state.model:
                st.caption(f"Active: **{st.session_state.model.model_id}**")

        with st.expander("View model holdings"):
            st.dataframe(
                pd.DataFrame([{"Ticker":h.ticker,"Target (%)":f"{h.target_weight*100:.4f}%"} for h in sel_model_obj.holdings]),
                use_container_width=True, hide_index=True,
            )

    st.divider()

    # ── Step 3: Run settings ─────────────────────────────────────────────
    _step_header("3", "Run Settings", "Configure parameters before running.")

    rs1, rs2, rs3, rs4 = st.columns(4)
    with rs1:
        drift_threshold = st.slider("Drift threshold (%)", 0.5, 10.0, 3.0, 0.5) / 100.0
    with rs2:
        min_trade_value = st.number_input("Min trade value (AUD)", min_value=0.0, value=500.0, step=100.0)
    with rs3:
        fractional = st.checkbox("Fractional shares", value=True)
    with rs4:
        dp = int(st.number_input("Qty decimal places", 1, 6, 4, 1)) if fractional else 0

    # Feature 6: Cash flow input
    with st.expander("Cash flow adjustment (optional)"):
        st.caption("Add a cash contribution or withdrawal before rebalancing. Applied to all loaded accounts equally unless specified.")
        cf1, cf2 = st.columns(2)
        with cf1:
            cash_addition = st.number_input("Cash to add / withdraw (AUD)", value=0.0, step=1000.0,
                                             help="Positive = contribution, negative = withdrawal")
        with cf2:
            st.write("")
            if cash_addition != 0:
                direction = "contribution" if cash_addition > 0 else "withdrawal"
                st.markdown(f'<div class="alert-box">A <strong>${abs(cash_addition):,.2f}</strong> {direction} will be applied before generating trades.</div>', unsafe_allow_html=True)

    # Feature 11: Exclusions from active client
    active_client_excl = []
    active_id = st.session_state.active_client
    if active_id and active_id in st.session_state.clients:
        active_client_excl = st.session_state.clients[active_id].get("exclusions", [])
        if active_client_excl:
            st.markdown(f'<div class="alert-box">Client exclusions active: <strong>{", ".join(active_client_excl)}</strong> will not be purchased.</div>', unsafe_allow_html=True)

    ready = has_model and has_portfolio
    st.write("")
    st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)

    bulk_mode = st.checkbox("Bulk rebalance (feature 10) — run all loaded accounts at once", value=True)

    run_clicked = st.button(
        "▶  Run Rebalance" if not st.session_state.drift_reports else "↺  Re-run Rebalance",
        type="primary", disabled=not ready, use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if run_clicked and ready:
        config = TradeConfig(drift_threshold=drift_threshold, min_trade_value=min_trade_value,
                              whole_shares=not fractional, managed_fund_dp=dp)
        d_reports, t_results = [], []
        portfolios_to_run = st.session_state.portfolios if bulk_mode else st.session_state.portfolios[:1]

        with st.spinner("Calculating drift and generating trades..."):
            for portfolio in portfolios_to_run:
                try:
                    dr, tr = _run_rebalance_for_portfolio(
                        portfolio, st.session_state.model, config,
                        cash_addition=cash_addition,
                        exclusions=active_client_excl,
                    )
                    d_reports.append(dr)
                    t_results.append(tr)
                except ValueError as exc:
                    st.error(f"{portfolio.account_id}: {exc}")

        st.session_state.drift_reports = d_reports
        st.session_state.trade_results = t_results

        # Save to history
        for dr, tr in zip(d_reports, t_results):
            client_name = "—"
            client_id = None
            for cid, c in st.session_state.clients.items():
                if any(a["hin"] == dr.account_id for a in c.get("accounts",[])):
                    client_name = c["name"]; client_id = cid; break

            history_entry = {
                "timestamp":   _utc_now(),
                "client_id":   client_id,
                "client_name": client_name,
                "account_id":  dr.account_id,
                "model_id":    st.session_state.model.model_id,
                "trade_count": len(tr.trades),
                "buy_value":   sum(t.estimated_value for t in tr.trades if t.action=="BUY"),
                "sell_value":  sum(t.estimated_value for t in tr.trades if t.action=="SELL"),
                "net":         sum(t.estimated_value if t.action=="BUY" else -t.estimated_value for t in tr.trades),
                "trades":      [{"action":t.action,"ticker":t.ticker,"quantity":t.quantity,
                                  "estimated_value":t.estimated_value,"status":"Pending"} for t in tr.trades],
            }
            st.session_state.rebalance_history.append(history_entry)
            if client_id:
                st.session_state.clients[client_id]["last_rebalance"] = _utc_now()[:10]

        st.rerun()

    if not ready:
        missing = []
        if not has_portfolio: missing.append("load a portfolio")
        if not has_model: missing.append("select a model")
        st.caption(f"To run: {' and '.join(missing)} above.")

    # ── Step 4: Results ──────────────────────────────────────────────────
    drift_reports = st.session_state.drift_reports
    trade_results = st.session_state.trade_results
    all_trades = [t for tr in trade_results for t in tr.trades]

    if drift_reports:
        st.divider()
        _step_header("4", "Results", "Review drift and trade instructions.", done=bool(all_trades))

        gross_buy  = sum(t.estimated_value for t in all_trades if t.action=="BUY")
        gross_sell = sum(t.estimated_value for t in all_trades if t.action=="SELL")

        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Accounts needing rebalance",
                   f"{sum(1 for dr in drift_reports if dr.requires_rebalance)} / {len(drift_reports)}")
        sm2.metric("Total trades generated", len(all_trades))
        sm3.metric("Net cash flow", f"${gross_sell - gross_buy:+,.2f}")

        st.write("")

        for dr in drift_reports:
            tr = next((x for x in trade_results if x.account_id == dr.account_id), None)
            shortfall = tr.has_funding_shortfall if tr else False

            with st.expander(
                f"**{dr.account_id}**  ·  ${dr.total_portfolio_value:,.2f}  ·  "
                f"{dr.flagged_count} flagged  {'· ⚠ Shortfall' if shortfall else ''}",
                expanded=dr.requires_rebalance,
            ):
                ca, cb, cc, cd = st.columns(4)
                ca.metric("Portfolio value", f"${dr.total_portfolio_value:,.2f}")
                cb.metric("Holdings", len(dr.holdings))
                cc.metric("Flagged", dr.flagged_count)
                if tr: cd.metric("⚠ Shortfall" if shortfall else "Closing cash", f"${tr.closing_cash:,.2f}")

                if shortfall:
                    st.warning(f"Funding shortfall of ${abs(tr.closing_cash):,.2f}. Adviser review required.")

                st.write("")
                st.markdown("**Drift Analysis**")

                # Feature 4: Pre/post comparison
                settle = _settle_date()
                pre_post_rows = []
                model_weights = {h.ticker: h.target_weight for h in st.session_state.model.holdings}

                # Compute post-trade weights
                post_values = {h.ticker: h.market_value for h in dr.holdings if h.market_value > 0}
                if tr:
                    for t in tr.trades:
                        price = dr.total_portfolio_value  # approximate
                        for h in dr.holdings:
                            if h.ticker == t.ticker and h.market_value > 0:
                                price = h.market_value / max(h.market_value / dr.total_portfolio_value * dr.total_portfolio_value, 1)
                                break
                        # Simpler: adjust by estimated value
                        current_val = post_values.get(t.ticker, 0)
                        if t.action == "BUY":  post_values[t.ticker] = current_val + t.estimated_value
                        elif t.action == "SELL": post_values[t.ticker] = max(0, current_val - t.estimated_value)

                post_total = sum(post_values.values()) + (tr.closing_cash if tr else 0)

                drift_rows = []
                for h in dr.holdings:
                    post_val = post_values.get(h.ticker, 0)
                    post_w = (post_val / post_total * 100) if post_total > 0 else 0
                    drift_rows.append({
                        "Ticker":        h.ticker,
                        "Status":        h.status.value,
                        "Current (%)":   round(h.current_weight * 100, 2),
                        "Target (%)":    round(h.target_weight * 100, 2),
                        "Post-trade (%)":round(post_w, 2),
                        "Drift (pp)":    round(h.drift * 100, 2),
                        "Mkt Value":     round(h.market_value, 2),
                    })

                st.dataframe(
                    pd.DataFrame(drift_rows)
                    .style.apply(_colour_drift_row, axis=1)
                    .map(_colour_drift_val, subset=["Drift (pp)"])
                    .format({
                        "Current (%)":   "{:.2f}%",
                        "Target (%)":    "{:.2f}%",
                        "Post-trade (%)":"{:.2f}%",
                        "Drift (pp)":    "{:+.2f}pp",
                        "Mkt Value":     "${:,.2f}",
                    })
                    .hide(axis="index"),
                    use_container_width=True,
                    height=min(60 + len(drift_rows)*38, 420),
                )
                st.caption("Row colours: amber = overweight, blue = underweight, red = not in model. Post-trade (%) shows projected weight after executing all trades.")

                if tr and tr.trades:
                    st.write("")
                    st.markdown("**Trade Instructions**")

                    tf1, tf2, tf3, tf4 = st.columns(4)
                    tf1.metric("Opening cash",  f"${tr.opening_cash:,.2f}")
                    tf2.metric("Sell proceeds", f"${tr.sell_proceeds:,.2f}")
                    tf3.metric("Buy cost",      f"${tr.buy_cost:,.2f}")
                    tf4.metric("⚠ Shortfall" if shortfall else "Closing cash", f"${tr.closing_cash:,.2f}")

                    trows = []
                    for t in tr.trades:
                        # Feature 8: CGT flag — flag SELL trades
                        cgt_flag = ""
                        if t.action == "SELL":
                            # Simple heuristic: if estimated value > $10k flag as review
                            if t.estimated_value > 10000:
                                cgt_flag = "Review CGT"

                        trows.append({
                            "Action":     t.action,
                            "Ticker":     t.ticker,
                            "Quantity":   round(t.quantity, 4),
                            "Est. Value": round(t.estimated_value, 2),
                            "Settlement": settle,
                            "CGT":        cgt_flag,
                        })

                    def _cgt_colour(val: str) -> str:
                        if val == "Review CGT": return "color:#d97706;font-weight:600"
                        return ""

                    st.dataframe(
                        pd.DataFrame(trows)
                        .style.apply(_colour_trade_row, axis=1)
                        .map(_colour_action, subset=["Action"])
                        .map(_cgt_colour, subset=["CGT"])
                        .format({"Quantity":"{:.4f}","Est. Value":"${:,.2f}"})
                        .hide(axis="index"),
                        use_container_width=True,
                        height=min(60 + len(trows)*38, 340),
                    )
                    st.caption("CGT column flags SELL trades >$10k estimated value for adviser review. Confirm tax position before executing.")

                if tr and tr.suppressed_trades:
                    with st.expander(f"Suppressed trades ({len(tr.suppressed_trades)})"):
                        st.dataframe(pd.DataFrame([{
                            "Action": td.action, "Ticker": td.ticker,
                            "Raw value": round(td.raw_trade_value, 2), "Reason": td.suppression_reason or "",
                        } for td in tr.suppressed_trades]), use_container_width=True, hide_index=True)

        # Sticky download
        if all_trades and st.session_state.model:
            model = st.session_state.model
            tmp_dl = tempfile.mktemp(suffix=".csv")
            export_trades_csv(all_trades, tmp_dl, include_metadata=True)
            with open(tmp_dl, "rb") as f: csv_bytes = f.read()
            os.unlink(tmp_dl)

            st.markdown('<div class="sticky-download">', unsafe_allow_html=True)
            dl1, dl2 = st.columns([2, 5])
            with dl1:
                st.download_button(
                    "⬇  Download Trade Instructions (CSV)",
                    data=csv_bytes,
                    file_name=f"trades_{model.model_id}_{model.version}.csv",
                    mime="text/csv", use_container_width=True,
                )
            with dl2:
                st.markdown(
                    f'<div class="dl-summary"><strong>{len(all_trades)}</strong> trades  ·  '
                    f'Buy <strong>${gross_buy:,.2f}</strong>  ·  '
                    f'Sell <strong>${gross_sell:,.2f}</strong>  ·  '
                    f'Net <strong>${gross_sell-gross_buy:+,.2f}</strong></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB: MODELS
# ===========================================================================

with tab_models:
    st.markdown('<div class="page-heading"><h1>Models</h1><div class="sub">Create and manage model portfolios</div></div>', unsafe_allow_html=True)

    saved_models = st.session_state.saved_models
    saved_keys = list(saved_models.keys())

    if saved_keys:
        st.markdown('<div class="section-title">Saved models</div>', unsafe_allow_html=True)
        current_label = f"{st.session_state.model.model_id} ({st.session_state.model.version})" if st.session_state.model else saved_keys[0]
        default_idx = saved_keys.index(current_label) if current_label in saved_keys else 0
        sel_key = st.selectbox("Select model", options=saved_keys, index=default_idx)
        sel_model_v = saved_models[sel_key]
        mid = sel_model_v.model_id
        if mid in MODEL_DESCRIPTIONS:
            desc, level = MODEL_DESCRIPTIONS[mid]
            st.markdown(_model_desc(mid), unsafe_allow_html=True)
            st.markdown(_risk_bar(level), unsafe_allow_html=True)

        st.dataframe(
            pd.DataFrame([{"Ticker":h.ticker,"Target Weight (%)":round(h.target_weight*100,4)} for h in sel_model_v.holdings]),
            use_container_width=True, hide_index=True,
        )

        mc1, mc2 = st.columns([3, 1])
        with mc1:
            if st.button("Set as active model", type="primary", use_container_width=True):
                st.session_state.model = sel_model_v
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Active: {sel_key}")
        with mc2:
            if len(saved_keys) > 1 and st.button("Delete", use_container_width=True):
                ck = f"confirm_del_{sel_key}"
                if ck not in st.session_state:
                    st.session_state[ck] = True
                    st.warning("Click Delete again to confirm.")
                else:
                    del st.session_state.saved_models[sel_key]
                    del st.session_state[ck]
                    if st.session_state.model:
                        lbl = f"{st.session_state.model.model_id} ({st.session_state.model.version})"
                        if lbl == sel_key:
                            st.session_state.model = None
                    st.rerun()

        if st.session_state.model:
            st.caption(f"Active: **{st.session_state.model.model_id}** v{st.session_state.model.version}")

    st.markdown('<div class="section-title">Create new model</div>', unsafe_allow_html=True)
    nm1, nm2 = st.columns(2)
    with nm1: new_id  = st.text_input("Model ID", placeholder="e.g. CONSERVATIVE_30")
    with nm2: new_ver = st.text_input("Version",  placeholder="e.g. 2024-Q3")
    st.caption("Enter target weights — must sum to 100%")
    m_wdf = st.data_editor(
        pd.DataFrame(SAMPLE_MODEL_WEIGHTS, columns=["Ticker","Weight"]),
        num_rows="dynamic",
        column_config={"Ticker":st.column_config.TextColumn("Ticker",width="small"),
                       "Weight":st.column_config.NumberColumn("Weight (%)",min_value=0.0001,max_value=100.0,step=0.01,format="%.4f")},
        hide_index=True, use_container_width=True, key="wdf_models",
    )
    wsum = m_wdf["Weight"].sum() if not m_wdf.empty else 0.0
    if abs(wsum-100.0) < 0.05: st.success(f"Weights sum: {wsum:.4f}% ✓")
    else: st.warning(f"Weights sum: {wsum:.4f}%  (need 100%,  {wsum-100:+.4f}%)")

    if st.button("Save Model", type="primary"):
        if not new_id.strip(): st.error("Model ID required.")
        elif not new_ver.strip(): st.error("Version required.")
        else:
            try:
                holdings = [ModelHolding(ticker=str(r["Ticker"]).strip().upper(), target_weight=float(r["Weight"])/100.0)
                            for _,r in m_wdf.iterrows() if str(r["Ticker"]).strip() and str(r["Ticker"]).strip().lower()!="nan"]
                nm = ModelPortfolio(model_id=new_id.strip(), version=new_ver.strip(), holdings=holdings)
                label = f"{nm.model_id} ({nm.version})"
                st.session_state.saved_models[label] = nm
                st.session_state.model = nm
                st.success(f"Saved: {label}"); st.rerun()
            except ValueError as exc:
                st.error(str(exc))


# ===========================================================================
# TAB: CLIENT PROFILE
# ===========================================================================

with tab_profile:
    st.markdown('<div class="page-heading"><h1>Client Profile</h1><div class="sub">Optional risk questionnaire to determine suitable model</div></div>', unsafe_allow_html=True)

    scored_qs = [q for q in QUESTIONS if not q.get("no_score",False)]
    step = st.session_state.q_step
    answers = dict(st.session_state.q_answers)

    cp1, cp2 = st.columns([3,2])
    with cp1:
        client_name = st.text_input("Client name / reference", value=st.session_state.q_client_name, placeholder="e.g. John Smith")
        st.session_state.q_client_name = client_name
    with cp2:
        st.metric("Assessment date", _utc_now().split(" ")[0])

    st.progress(min(step, len(QUESTIONS)) / len(QUESTIONS), text=f"Question {min(step+1,len(QUESTIONS))} of {len(QUESTIONS)}")

    if step < len(QUESTIONS):
        q = QUESTIONS[step]
        st.markdown(f"**Section {q['section']} — {q['label']}**")
        st.markdown(f"### {q['text']}")
        options = [opt[0] for opt in q["options"]]
        selected = st.radio("Select answer", options=options, index=answers.get(q["id"],None), key=f"qr_{step}", label_visibility="collapsed")
        st.write("")
        n1, n2, _ = st.columns([1,1,5])
        with n1:
            if step > 0 and st.button("← Back", use_container_width=True):
                st.session_state.q_step = step-1; st.rerun()
        with n2:
            lbl = "Next →" if step < len(QUESTIONS)-1 else "See Results →"
            if st.button(lbl, type="primary", use_container_width=True, disabled=selected is None):
                if selected is not None:
                    answers[q["id"]] = options.index(selected)
                    st.session_state.q_answers = answers
                st.session_state.q_step = step+1; st.rerun()
    else:
        all_answered = all(q["id"] in answers for q in scored_qs)
        if not all_answered:
            st.warning("Some questions were not answered.")
            if st.button("← Back"): st.session_state.q_step = 0; st.rerun()
        else:
            score, flags, conflicts = _compute_score(answers, st.session_state.weights)
            base_model = _score_to_model(score, answers)
            display_model = base_model
            if "income" in flags: display_model += " — Income / Yield"
            if "esg"    in flags: display_model += " (ESG)"
            level = RISK_LEVEL.get(base_model, 3)

            r1, r2 = st.columns([2,3])
            with r1:
                st.metric("Recommended model", base_model)
                st.markdown(_risk_bar(level), unsafe_allow_html=True)
                if flags: st.caption(f"Flags: {', '.join(f.upper() for f in flags)}")
            with r2:
                with st.expander("Score breakdown"):
                    rows = []
                    for sec, lbl in SECTION_LABELS.items():
                        sec_qs = [q for q in QUESTIONS if q["section"]==sec and not q.get("no_score",False)]
                        pts = [q["options"][answers[q["id"]]][1] for q in sec_qs if q["id"] in answers]
                        if pts:
                            rows.append({"Section":f"{sec} — {lbl}","Score":round(sum(pts)/len(pts),2),"Weight":st.session_state.weights.get(sec,1.0)})
                    if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if conflicts:
                st.markdown('<div class="conflict-box"><strong>Conflicts detected</strong><br>' + "<br>".join(f"• {c}" for c in conflicts) + '</div>', unsafe_allow_html=True)

            ov_opts = ["— Use recommended —","Conservative","Moderate / Balanced","Growth","High Growth","Aggressive","Income / Yield","ESG","Australian Equities","Index / Passive"]
            ov_sel = st.selectbox("Override", options=ov_opts, index=0)
            ov_reason = ""
            if ov_sel != "— Use recommended —":
                ov_reason = st.text_area("Override reason (required)", placeholder="Document why...")
                if not ov_reason.strip(): st.warning("Reason required.")

            final_model = ov_sel if ov_sel != "— Use recommended —" else display_model
            is_overridden = ov_sel != "— Use recommended —"
            st.info(f"**Final model: {final_model}**" + ("  *(override)*" if is_overridden else "  *(recommended)*"))

            can_save = not is_overridden or bool(ov_reason.strip())
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                if st.button("Save Profile", type="primary", use_container_width=True, disabled=not can_save):
                    st.session_state.q_result = {
                        "client_name":client_name,"timestamp":_utc_now(),
                        "version":QUESTIONNAIRE_VERSION,"answers":dict(answers),
                        "score":score,"flags":list(flags),"conflicts":conflicts,
                        "base_model":base_model,"display_model":display_model,
                        "final_model":final_model,"overridden":is_overridden,
                        "override_reason":ov_reason,"weights_used":dict(st.session_state.weights),
                    }
                    st.success("Profile saved.")
            with sc2:
                if st.session_state.q_result:
                    res = st.session_state.q_result
                    lines = [["Risk Profile Assessment"],["Client",res["client_name"]],["Date",res["timestamp"]],[],["Question","Answer","Points"]]
                    for q in QUESTIONS:
                        if q["id"] in res["answers"]:
                            idx = res["answers"][q["id"]]; ot,pts = q["options"][idx]
                            lines.append([q["text"],ot,"" if q.get("no_score") else pts])
                    lines += [[],["Score",res["score"]],["Recommended",res["display_model"]],["Final",res["final_model"]],["Override","Yes" if res["overridden"] else "No"]]
                    if res["overridden"]: lines.append(["Override reason",res["override_reason"]])
                    buf = io.StringIO(); csv.writer(buf).writerows(lines)
                    st.download_button("Download CSV", data=buf.getvalue().encode("utf-8-sig"),
                                       file_name=f"profile_{client_name.replace(' ','_')}_{_today()}.csv",
                                       mime="text/csv", use_container_width=True)
                else:
                    st.button("Download CSV", disabled=True, use_container_width=True)
            with sc3:
                if st.button("New Assessment", use_container_width=True):
                    st.session_state.q_answers = {}; st.session_state.q_step = 0
                    st.session_state.q_result = None; st.session_state.q_client_name = ""
                    st.rerun()


# ===========================================================================
# TAB: SETTINGS
# ===========================================================================

with tab_settings:
    st.markdown('<div class="page-heading"><h1>Settings</h1><div class="sub">Questionnaire weighting and principal controls</div></div>', unsafe_allow_html=True)

    locked = st.session_state.weights_locked
    if locked: st.error("Weightings are locked.")

    st.markdown('<div class="section-title">Section weights</div>', unsafe_allow_html=True)
    st.caption(f"Adjust section importance. Range: {WEIGHT_MIN}x – {WEIGHT_MAX}x. Default: 1.0x.")

    new_weights = {}
    wc1, wc2 = st.columns(2)
    for i, (sec, label) in enumerate(SECTION_LABELS.items()):
        col = wc1 if i % 2 == 0 else wc2
        with col:
            new_weights[sec] = st.slider(f"{sec} — {label}", WEIGHT_MIN, WEIGHT_MAX,
                                          float(st.session_state.weights.get(sec,1.0)), 0.1,
                                          disabled=locked, key=f"ws_{sec}")

    total_w = sum(new_weights.values())
    st.dataframe(pd.DataFrame([{"Section":f"{s} — {SECTION_LABELS[s]}","Weight":new_weights[s],"Contribution":f"{new_weights[s]/total_w*100:.1f}%"} for s in SECTION_LABELS]),
                 use_container_width=False, hide_index=True)

    note_col, btn_col = st.columns([3,2])
    with note_col:
        change_note = st.text_input("Reason for change", placeholder="e.g. Practice policy update", disabled=locked)
    with btn_col:
        st.write(""); st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Save", type="primary", disabled=locked or not change_note.strip(), use_container_width=True):
                st.session_state.weight_log.append({"timestamp":_utc_now(),"weights":dict(new_weights),"note":change_note.strip()})
                st.session_state.weights = dict(new_weights); st.success("Saved.")
        with b2:
            if st.button("Reset", disabled=locked, use_container_width=True):
                st.session_state.weights = dict(DEFAULT_WEIGHTS)
                st.session_state.weight_log.append({"timestamp":_utc_now(),"weights":dict(DEFAULT_WEIGHTS),"note":"Restored defaults"})
                st.success("Reset."); st.rerun()

    st.markdown('<div class="section-title">Principal controls</div>', unsafe_allow_html=True)
    if locked:
        if st.button("Unlock Weightings"): st.session_state.weights_locked = False; st.rerun()
    else:
        if st.button("Lock Weightings"): st.session_state.weights_locked = True; st.rerun()

    st.markdown('<div class="section-title">Change log</div>', unsafe_allow_html=True)
    if st.session_state.weight_log:
        log_rows = [{"Timestamp":e["timestamp"],"Note":e["note"],**{s:e["weights"].get(s,1.0) for s in SECTION_LABELS}} for e in reversed(st.session_state.weight_log)]
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No changes recorded yet.")
