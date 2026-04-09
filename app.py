"""
app.py — Streamlit UI for the portfolio rebalancing engine.

Tabs:
    1. Client Profiling   — Risk questionnaire, model recommendation, override, export
    2. Model Portfolios   — Create, select, delete model portfolios
    3. Portfolio & Drift  — Upload CSV, run drift, generate trades, download
    4. Settings           — Weighting configuration, change log, principal controls

Run:
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from datetime import datetime, timezone
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

st.markdown("""
<style>
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 3px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    }
    .badge-over  { background: #fff3cd; color: #7d5a00; }
    .badge-under { background: #cfe2ff; color: #084298; }
    .badge-model { background: #f8d7da; color: #842029; }
    .badge-band  { background: #d1e7dd; color: #0a3622; }
    .conflict-box {
        background: #fff3cd; border: 1px solid #ffc107;
        border-radius: 6px; padding: 0.75rem 1rem; margin: 0.5rem 0;
    }
    div[data-testid="metric-container"] { padding-top: 0.4rem; }
    thead tr th { background-color: #f8f9fa !important; }
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
            ("2-5 years", 2),
            ("5-10 years", 3),
            ("10-15 years", 4),
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
            ("Severely - I depend on it", 1),
            ("Significantly - it would cause real hardship", 2),
            ("Moderately - I would need to adjust my plans", 3),
            ("Mildly - I have other assets to fall back on", 4),
            ("Minimally - this is discretionary money", 5),
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
            ("None - I have never invested before", 1),
            ("Limited - I have a savings account or term deposit", 2),
            ("Moderate - I have invested in managed funds or super", 3),
            ("Experienced - I have invested in shares or ETFs directly", 4),
            ("Advanced - I actively manage a diversified investment portfolio", 5),
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
            ("Maximise long-term growth, I don't need income", 5),
        ],
        "flags": {1: "income"},
    },
    {
        "id": "Q8", "section": "F", "label": "ESG Preferences",
        "text": "Do you have any ethical or ESG investment preferences?",
        "options": [
            ("No preference", 0),
            ("Somewhat important", 0),
            ("Very important - I want to exclude certain industries", 0),
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

CONFLICT_RULES = [
    ("Q2", [0, 1], "Q7", [3, 4],
     "Low risk tolerance (Q2) conflicts with high growth goals (Q7)."),
    ("Q1", [0], "Q7", [3, 4],
     "Short time horizon (Q1) conflicts with growth-oriented goals (Q7)."),
    ("Q4", [0], None, None,
     "Client is financially dependent on this investment (Q4). Consider capping at Conservative."),
]

DEFAULT_WEIGHTS = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "D": 1.0,
    "E": 1.0,
    "F": 1.0,
}

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

SAMPLE_WEIGHTS_MODEL = [
    ("CBA", 25.0), ("BHP", 20.0), ("CSL", 15.0),
    ("WES", 15.0), ("ANZ", 12.0), ("MQG", 8.0), ("FMG", 5.0),
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        "portfolios":        [],
        "model":             None,
        "saved_models":      {},
        "drift_reports":     [],
        "trade_results":     [],
        "q_answers":         {},
        "q_result":          None,
        "q_client_name":     "",
        "weights":           dict(DEFAULT_WEIGHTS),
        "weight_log":        [],
        "weights_locked":    False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

if not st.session_state.saved_models:
    try:
        _h = [ModelHolding(ticker=t, target_weight=w / 100.0)
              for t, w in SAMPLE_WEIGHTS_MODEL]
        _m = ModelPortfolio(model_id="GROWTH_70", version="2024-Q2", holdings=_h)
        st.session_state.saved_models["GROWTH_70 (2024-Q2)"] = _m
        st.session_state.model = _m
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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
            sec = q["section"]
            section_raw.setdefault(sec, [])
            section_raw[sec].append(pts)

        if "flags" in q and idx in q["flags"]:
            flags.add(q["flags"][idx])

    total_weight = 0.0
    weighted_sum = 0.0
    for sec, pts_list in section_raw.items():
        avg = sum(pts_list) / len(pts_list)
        w = weights.get(sec, 1.0)
        weighted_sum += avg * w
        total_weight += w

    if total_weight == 0:
        return 0.0, flags, []

    normalised = (weighted_sum / total_weight) * 6
    score = round(normalised, 2)

    conflicts = []
    for rule in CONFLICT_RULES:
        q1_id, q1_idxs, q2_id, q2_idxs, msg = rule
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
        idx = order.index(base)
        base = order[min(idx, 1)]

    q4 = answers.get("Q4", 4)
    if q4 == 0:
        idx = order.index(base)
        base = order[max(idx - 1, 0)]

    return base


# ---------------------------------------------------------------------------
# App header
# ---------------------------------------------------------------------------

st.title("Rebalancing Engine")

tab_profile, tab_models, tab_drift, tab_settings = st.tabs([
    "  Client Profiling  ",
    "  Model Portfolios  ",
    "  Portfolio & Drift  ",
    "  Settings  ",
])

# ===========================================================================
# TAB 1 — CLIENT PROFILING
# ===========================================================================

with tab_profile:
    st.markdown("### Risk Profile Questionnaire")
    st.caption(
        f"Version {QUESTIONNAIRE_VERSION}  |  "
        "Complete all questions then review the recommended model. "
        "All responses are recorded for compliance purposes."
    )

    col_name, col_date = st.columns([3, 2])
    with col_name:
        client_name = st.text_input(
            "Client name / reference",
            value=st.session_state.q_client_name,
            placeholder="e.g. John Smith or HIN001",
        )
        st.session_state.q_client_name = client_name
    with col_date:
        st.metric("Assessment date", _utc_now_str().split(" ")[0])

    st.divider()

    answers = dict(st.session_state.q_answers)
    current_section = None

    for q in QUESTIONS:
        if q["section"] != current_section:
            current_section = q["section"]
            st.markdown(
                f"**Section {current_section} — {SECTION_LABELS[current_section]}**"
            )

        options = [opt[0] for opt in q["options"]]
        current_idx = answers.get(q["id"], None)

        selected = st.radio(
            q["text"],
            options=options,
            index=current_idx,
            key=f"radio_{q['id']}",
        )

        if selected is not None:
            answers[q["id"]] = options.index(selected)

        st.write("")

    st.session_state.q_answers = answers
    answered_count = sum(1 for q in QUESTIONS if q["id"] in answers)
    st.progress(answered_count / len(QUESTIONS))
    st.caption(f"{answered_count} of {len(QUESTIONS)} questions answered")

    st.divider()

    scored_qs = [q for q in QUESTIONS if not q.get("no_score", False)]
    all_scored_answered = all(q["id"] in answers for q in scored_qs)

    if not all_scored_answered:
        st.info("Answer all questions to see the recommended risk profile.")
    else:
        score, flags, conflicts = _compute_score(answers, st.session_state.weights)
        base_model = _score_to_model(score, answers)

        display_model = base_model
        if "income" in flags:
            display_model += " - Income / Yield"
        if "esg" in flags:
            display_model += " (ESG)"

        st.markdown("#### Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Weighted score", f"{score:.1f} / 30")
        r2.metric("Recommended model", base_model)
        r3.metric("Flags", ", ".join(flags).upper() if flags else "None")

        if conflicts:
            st.markdown('<div class="conflict-box">', unsafe_allow_html=True)
            st.markdown("**Conflicts detected - adviser review required**")
            for c in conflicts:
                st.markdown(f"- {c}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()

        with st.expander("Score breakdown by section"):
            breakdown_rows = []
            for sec, label in SECTION_LABELS.items():
                sec_qs = [q for q in QUESTIONS
                          if q["section"] == sec and not q.get("no_score", False)]
                if not sec_qs:
                    continue
                pts = [q["options"][answers[q["id"]]][1]
                       for q in sec_qs if q["id"] in answers]
                if pts:
                    avg = sum(pts) / len(pts)
                    w = st.session_state.weights.get(sec, 1.0)
                    breakdown_rows.append({
                        "Section": f"{sec} - {label}",
                        "Raw avg (1-5)": round(avg, 2),
                        "Weight": w,
                        "Weighted avg": round(avg * w, 2),
                    })
            if breakdown_rows:
                st.dataframe(
                    pd.DataFrame(breakdown_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()
        st.markdown("#### Adviser override")
        st.caption(
            "If the recommended model is not appropriate, select an alternative "
            "and provide a mandatory reason. This will be recorded."
        )

        override_options = [
            "- Use recommended model -",
            "Conservative",
            "Moderate / Balanced",
            "Growth",
            "High Growth",
            "Aggressive",
            "Income / Yield",
            "ESG",
            "Australian Equities",
            "Index / Passive",
        ]

        override_sel = st.selectbox(
            "Override model selection",
            options=override_options,
            index=0,
        )

        override_reason = ""
        if override_sel != "- Use recommended model -":
            override_reason = st.text_area(
                "Reason for override (required)",
                placeholder="Document why the recommended model is not appropriate...",
            )
            if not override_reason.strip():
                st.warning("A reason must be provided before saving.")

        final_model = (
            override_sel if override_sel != "- Use recommended model -"
            else display_model
        )
        is_overridden = override_sel != "- Use recommended model -"

        st.info(
            f"**Final model: {final_model}**"
            + ("  *(adviser override)*" if is_overridden else "  *(system recommended)*")
        )

        col_save, col_export = st.columns(2)

        with col_save:
            can_save = not is_overridden or bool(override_reason.strip())
            if st.button(
                "Save profile result",
                use_container_width=True,
                type="primary",
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

        with col_export:
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
                    label="Download profile summary (CSV)",
                    data=buf.getvalue().encode("utf-8-sig"),
                    file_name=fname,
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button(
                    "Download profile summary (CSV)",
                    disabled=True,
                    use_container_width=True,
                    help="Save the profile first",
                )

        if st.session_state.q_result:
            st.divider()
            st.caption(
                f"To apply this profile, go to Model Portfolios and select "
                f"a model matching **{final_model}**, then run drift analysis "
                f"in Portfolio & Drift."
            )


# ===========================================================================
# TAB 2 — MODEL PORTFOLIOS
# ===========================================================================

with tab_models:
    st.markdown("### Model Portfolios")

    saved_models: dict = st.session_state.saved_models
    saved_keys = list(saved_models.keys())

    if saved_keys:
        st.markdown("#### Saved models")

        current_key = saved_keys[0]
        if st.session_state.model:
            lbl = f"{st.session_state.model.model_id} ({st.session_state.model.version})"
            if lbl in saved_keys:
                current_key = lbl

        selected_key = st.selectbox(
            "Select model",
            options=saved_keys,
            index=saved_keys.index(current_key),
        )

        selected_model = saved_models[selected_key]
        holdings_df = pd.DataFrame([
            {"Ticker": h.ticker, "Target Weight (%)": round(h.target_weight * 100, 4)}
            for h in selected_model.holdings
        ])
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)

        col_use, col_del = st.columns([3, 1])
        with col_use:
            if st.button("Activate this model", use_container_width=True, type="primary"):
                st.session_state.model = selected_model
                st.session_state.drift_reports = []
                st.session_state.trade_results = []
                st.success(f"Active model: {selected_key}")
        with col_del:
            if st.button("Delete", use_container_width=True):
                del st.session_state.saved_models[selected_key]
                if st.session_state.model:
                    lbl = f"{st.session_state.model.model_id} ({st.session_state.model.version})"
                    if lbl == selected_key:
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

    mc1, mc2 = st.columns(2)
    with mc1:
        new_id = st.text_input("Model ID", placeholder="e.g. CONSERVATIVE_30")
    with mc2:
        new_ver = st.text_input("Version", placeholder="e.g. 2024-Q3")

    st.caption("Enter target weights - must sum to 100%")

    m_weight_df = st.data_editor(
        pd.DataFrame(SAMPLE_WEIGHTS_MODEL, columns=["Ticker", "Weight"]),
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
        st.success(f"Weights sum: {wsum:.4f}% ✓")
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
# TAB 3 — PORTFOLIO & DRIFT
# ===========================================================================

with tab_drift:
    st.markdown("### Portfolio & Drift Analysis")

    col_up1, col_up2 = st.columns([3, 2])

    with col_up1:
        st.markdown("#### 1 - Load portfolio")
        use_sample = st.checkbox("Use built-in sample data")

        if use_sample:
            if st.button("Load sample portfolio"):
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

        with st.expander("CSV format"):
            st.code(
                "account_id,ticker,quantity,price,cash_balance\n"
                "HIN001,CBA,150.0000,121.5000,5000.0000\n"
                "HIN001,BHP,200.2500,46.2000,5000.0000",
                language="text",
            )

    with col_up2:
        st.markdown("#### 2 - Active model")
        if st.session_state.model:
            m = st.session_state.model
            st.success(f"**{m.model_id}** v{m.version}  |  {len(m.holdings)} holdings")
            st.dataframe(
                pd.DataFrame([
                    {"Ticker": h.ticker, "Target (%)": f"{h.target_weight*100:.4f}%"}
                    for h in m.holdings
                ]),
                use_container_width=True,
                hide_index=True,
                height=220,
            )
        else:
            st.warning("No model selected. Go to Model Portfolios to activate one.")

    st.divider()
    st.markdown("#### 3 - Run settings")

    rs1, rs2, rs3 = st.columns(3)
    with rs1:
        drift_threshold = st.slider(
            "Drift threshold (%)", 0.5, 10.0, 3.0, 0.5,
        ) / 100.0
    with rs2:
        min_trade_value = st.number_input(
            "Min trade value (AUD)", min_value=0.0, value=500.0, step=100.0,
        )
    with rs3:
        fractional_shares = st.checkbox("Allow fractional shares", value=True)
        decimal_places = int(st.number_input(
            "Quantity decimal places", min_value=1, max_value=6, value=4, step=1,
        )) if fractional_shares else 0

    ready = bool(st.session_state.portfolios and st.session_state.model)

    if st.button("Run rebalance", type="primary", disabled=not ready):
        config = TradeConfig(
            drift_threshold=drift_threshold,
            min_trade_value=min_trade_value,
            whole_shares=not fractional_shares,
            managed_fund_dp=decimal_places,
        )
        d_reports = []
        t_results = []
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

    if not ready:
        if not st.session_state.portfolios:
            st.caption("Load a portfolio first.")
        if not st.session_state.model:
            st.caption("Activate a model in Model Portfolios first.")

    drift_reports = st.session_state.drift_reports
    trade_results = st.session_state.trade_results
    all_trades = [t for tr in trade_results for t in tr.trades]

    if drift_reports:
        st.divider()

        model = st.session_state.model
        gross_buy  = sum(t.estimated_value for t in all_trades if t.action == "BUY")
        gross_sell = sum(t.estimated_value for t in all_trades if t.action == "SELL")

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("Accounts", len(drift_reports))
        sm2.metric("Need rebalance",
                   sum(1 for dr in drift_reports if dr.requires_rebalance))
        sm3.metric("Flagged holdings",
                   sum(dr.flagged_count for dr in drift_reports))
        sm4.metric("Total trades", len(all_trades))
        sm5.metric("Net cash flow", f"${gross_sell - gross_buy:+,.4f}")

        st.divider()

        acct_tabs = st.tabs([f"  {dr.account_id}  " for dr in drift_reports])

        for tab_a, dr in zip(acct_tabs, drift_reports):
            with tab_a:
                tr = next(
                    (x for x in trade_results if x.account_id == dr.account_id),
                    None,
                )

                ca, cb, cc, cd = st.columns(4)
                ca.metric("Portfolio value", f"${dr.total_portfolio_value:,.4f}")
                cb.metric("Holdings", len(dr.holdings))
                cc.metric("Flagged", dr.flagged_count)
                if tr:
                    cd.metric(
                        "Shortfall" if tr.has_funding_shortfall else "Closing cash",
                        f"${tr.closing_cash:,.4f}",
                    )

                st.divider()
                st.markdown("#### Drift analysis")

                drift_rows = [{
                    "Ticker":       h.ticker,
                    "Status":       h.status.value,
                    "Current (%)":  round(h.current_weight * 100, 4),
                    "Target (%)":   round(h.target_weight * 100, 4),
                    "Drift (pp)":   round(h.drift * 100, 4),
                    "Market Value": round(h.market_value, 4),
                    "Flag":         h.exceeds_threshold,
                } for h in dr.holdings]

                def _cd(val: float) -> str:
                    if val > 0:
                        return "background-color: #fff8e1; color: #5d4200"
                    if val < 0:
                        return "background-color: #e8f0fe; color: #1a237e"
                    return ""

                st.dataframe(
                    pd.DataFrame(drift_rows)
                    .style.map(_cd, subset=["Drift (pp)"])
                    .format({
                        "Current (%)":  "{:.4f}%",
                        "Target (%)":   "{:.4f}%",
                        "Drift (pp)":   "{:+.4f}pp",
                        "Market Value": "${:,.4f}",
                    })
                    .hide(axis="index"),
                    use_container_width=True,
                    height=280,
                )

                if dr.requires_rebalance:
                    st.caption(
                        f"{dr.flagged_count} holding(s) exceed the "
                        f"{drift_threshold*100:.1f}pp threshold."
                    )
                else:
                    st.caption("All holdings within drift band.")

                if tr and (tr.trades or tr.suppressed_trades):
                    st.divider()
                    st.markdown("#### Trade instructions")

                    tf1, tf2, tf3, tf4 = st.columns(4)
                    tf1.metric("Opening cash",  f"${tr.opening_cash:,.4f}")
                    tf2.metric("Sell proceeds", f"${tr.sell_proceeds:,.4f}")
                    tf3.metric("Buy cost",      f"${tr.buy_cost:,.4f}")
                    tf4.metric(
                        "Shortfall" if tr.has_funding_shortfall else "Closing cash",
                        f"${tr.closing_cash:,.4f}",
                    )

                    if tr.has_funding_shortfall:
                        st.warning(
                            f"Funding shortfall of ${abs(tr.closing_cash):,.4f}. "
                            "Adviser review required."
                        )

                    if tr.trades:
                        trows = [{
                            "Action":     t.action,
                            "Ticker":     t.ticker,
                            "Quantity":   round(t.quantity, 4),
                            "Est. Value": round(t.estimated_value, 4),
                        } for t in tr.trades]

                        def _ca(val: str) -> str:
                            if val == "BUY":
                                return "color: #155724; font-weight: 600"
                            if val == "SELL":
                                return "color: #721c24; font-weight: 600"
                            return ""

                        st.dataframe(
                            pd.DataFrame(trows)
                            .style.map(_ca, subset=["Action"])
                            .format({
                                "Quantity":   "{:.4f}",
                                "Est. Value": "${:,.4f}",
                            })
                            .hide(axis="index"),
                            use_container_width=True,
                            height=220,
                        )

                    if tr.suppressed_trades:
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

        st.divider()
        st.markdown("#### Download")

        if all_trades and st.session_state.model:
            model = st.session_state.model
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
                    file_name=f"trades_{model.model_id}_{model.version}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                st.caption(
                    f"{len(all_trades)} trade(s)  |  "
                    f"Gross buy: ${gross_buy:,.4f}  |  "
                    f"Gross sell: ${gross_sell:,.4f}  |  "
                    f"Net: ${gross_sell - gross_buy:+,.4f}"
                )
        else:
            st.caption("No active trades to download.")


# ===========================================================================
# TAB 4 — SETTINGS
# ===========================================================================

with tab_settings:
    st.markdown("### Settings - Questionnaire Weighting")
    st.caption(
        "Adjust the relative importance of each section in the risk score calculation. "
        "All changes are logged with a timestamp for compliance purposes."
    )

    locked = st.session_state.weights_locked

    if locked:
        st.error("Weightings are locked. Unlock below to make changes.")

    st.divider()
    st.markdown("#### Section weights")
    st.caption(
        f"Min: {WEIGHT_MIN}x  |  Max: {WEIGHT_MAX}x  |  "
        "1.0x = equal weight (default)"
    )

    new_weights = {}
    wc1, wc2 = st.columns(2)
    for i, (sec, label) in enumerate(SECTION_LABELS.items()):
        col = wc1 if i % 2 == 0 else wc2
        with col:
            new_weights[sec] = st.slider(
                f"Section {sec} - {label}",
                min_value=WEIGHT_MIN,
                max_value=WEIGHT_MAX,
                value=float(st.session_state.weights.get(sec, 1.0)),
                step=0.1,
                disabled=locked,
                key=f"weight_slider_{sec}",
            )

    total_w = sum(new_weights.values())
    st.markdown("**Contribution to final score**")
    st.dataframe(
        pd.DataFrame([{
            "Section": f"{s} - {SECTION_LABELS[s]}",
            "Weight": new_weights[s],
            "Contribution (%)": f"{new_weights[s]/total_w*100:.1f}%",
        } for s in SECTION_LABELS]),
        use_container_width=False,
        hide_index=True,
    )

    col_note, col_btns = st.columns([3, 2])
    with col_note:
        change_note = st.text_input(
            "Reason for change (required to save)",
            placeholder="e.g. Practice policy - increase time horizon weighting",
            disabled=locked,
        )
    with col_btns:
        st.write("")
        st.write("")
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button(
                "Save weightings",
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
                st.success("Saved.")
        with bc2:
            if st.button(
                "Restore defaults",
                disabled=locked,
                use_container_width=True,
            ):
                st.session_state.weights = dict(DEFAULT_WEIGHTS)
                st.session_state.weight_log.append({
                    "timestamp": _utc_now_str(),
                    "weights":   dict(DEFAULT_WEIGHTS),
                    "note":      "Restored to default equal weighting",
                })
                st.success("Restored.")
                st.rerun()

    st.divider()
    st.markdown("#### Principal controls")

    if locked:
        if st.button("Unlock weightings"):
            st.session_state.weights_locked = False
            st.rerun()
    else:
        if st.button("Lock weightings"):
            st.session_state.weights_locked = True
            st.rerun()
    st.caption(
        "When locked, individual advisers cannot modify section weightings."
    )

    st.divider()
    st.markdown("#### Weighting change log")

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
