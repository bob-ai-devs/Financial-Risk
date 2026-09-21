"""
================================================================================
 BOB AI FINANCIAL RISK ANALYZER  —  Streamlit Edition
================================================================================

A from-scratch Streamlit port of the original Flask "BOB AI Financial
Analyst" app. Same core idea — type in company names, let Gemini resolve
the Yahoo Finance ticker, pull the Balance Sheet / P&L / Cash Flow, and get
an AI-written risk report — but re-built around Streamlit's model, with a
Bank of Baroda colour scheme and several new analyst tools bolted on.

WHAT CHANGED FROM THE FLASK VERSION
------------------------------------------------------------------------------
1. No hardcoded Gemini key. The key is read from `st.secrets["GEMINI_API_KEY"]`
   (see the sidebar / secrets.toml note below).
2. No manual thread/session-state juggling, no polling `/thread-output_ai`
   endpoint, no Jinja templates — Streamlit's own rerun model + st.session_state
   replaces all of that.
3. The original silently ran AI analysis on ONE combined string even when the
   user typed several comma-separated companies (`max_time` was computed from
   the count but never used to loop). This version actually loops over every
   company you enter and analyzes each one independently.
4. The custom regex markdown->HTML parser (text_to_html / process_table /
   bold_inner_words etc.) is gone — Gemini is asked to reply in plain
   Markdown and Streamlit's native `st.markdown` renders headings, bold text
   and tables for you, which is both simpler and safer.
5. New: ticker confirmation step (you can correct a wrong AI ticker guess
   before spending an API call on the report), live price chart, key-stats
   cards, an auto-extracted Risk Rating gauge, a multi-company comparison
   view, and one-click report downloads.

SECRETS SETUP
------------------------------------------------------------------------------
Create `.streamlit/secrets.toml` next to this file:

    GEMINI_API_KEY = "your-key-here"

Or set it in Streamlit Community Cloud's "Secrets" settings panel.

RUN
------------------------------------------------------------------------------
    pip install -r requirements.txt
    streamlit run app.py
================================================================================
"""

import json
import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

# ==============================================================================
# BANK OF BARODA BRAND PALETTE
# ==============================================================================
BOB_ORANGE = "#F7941D"       # primary — "Baroda Sun"
BOB_ORANGE_DEEP = "#E8531B"  # sun-ray gradient end
BOB_MAROON = "#8E1B3A"       # sun-ray gradient end / accents
BOB_NAVY = "#12284C"         # wordmark / headings
BOB_NAVY_LIGHT = "#1E3E73"
BOB_CREAM = "#FFF8F1"        # page background
BOB_GREY = "#5B6675"

RISK_COLORS = {1: "#2E9E4E", 2: "#8FC93A", 3: "#F7C700", 4: "#F0862C", 5: "#D62839"}

# ==============================================================================
# PAGE CONFIG + THEME
# ==============================================================================
st.set_page_config(
    page_title="BOB AI Financial Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {BOB_CREAM};
    }}
    #bob-banner {{
        background: radial-gradient(circle at 15% 50%, {BOB_ORANGE} 0%, {BOB_ORANGE_DEEP} 45%, {BOB_MAROON} 100%);
        padding: 22px 30px;
        border-radius: 12px;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }}
    #bob-banner h1 {{
        color: white;
        margin: 0;
        font-size: 1.9em;
        font-weight: 800;
        letter-spacing: 0.3px;
    }}
    #bob-banner p {{
        color: #FFEFE0;
        margin: 4px 0 0 0;
        font-size: 0.95em;
    }}
    h1, h2, h3 {{ color: {BOB_NAVY}; }}
    div.stButton > button {{
        background-color: {BOB_ORANGE};
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }}
    div.stButton > button:hover {{
        background-color: {BOB_MAROON};
        color: white;
        transform: scale(1.02);
    }}
    div.stDownloadButton > button {{
        background-color: {BOB_NAVY};
        color: white;
        border-radius: 6px;
        font-weight: 600;
    }}
    /* selectbox closed state (sits inside the sidebar) */
   section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
       background-color: white !important;
       border: 1px solid #F7941D;
       border-radius: 6px;
   }}
   section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
       color: #12284C !important;
   }}
   /* dropdown option list renders in a portal OUTSIDE the sidebar, so it
      needs its own (unscoped) rule rather than the sidebar selector above */
   div[data-baseweb="popover"] ul[role="listbox"] {{
       background-color: white !important;
   }}
   div[data-baseweb="popover"] ul[role="listbox"] li {{
       color: #12284C !important;
   }}
   div[data-baseweb="popover"] ul[role="listbox"] li:hover {{
       background-color: #F7941D22 !important;
   }}
    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 14px 16px;
        border-left: 5px solid {BOB_ORANGE};
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================
MODEL_OPTIONS = [
    "models/gemini-flash-lite-latest",
    "models/gemini-flash-latest",
]

STATUS_MESSAGES = [
    "AI reading Balance Sheet …",
    "AI analyzing Balance Sheet …",
    "AI reading Income Statement …",
    "AI analyzing Income Statement …",
    "AI reading Cash Flow Statement …",
    "AI analyzing Cash Flow Statement …",
    "AI calculating key financial ratios …",
    "AI running industry / peer context …",
    "AI drafting risk opinion …",
    "AI finalizing the report …",
]

# ==============================================================================
# TICKER CACHE — persistence on Streamlit Community Cloud
# ==============================================================================
# Streamlit Cloud containers are writable while running, but that filesystem
# is NOT persistent: it's wiped on every reboot (a git push, a manual reboot,
# or the container waking back up after sleeping from inactivity), and it
# isn't shared across replicas. A plain ticker_cache.json on disk (the
# original Flask approach) would silently lose its contents on the next
# redeploy, so this app uses two tiers instead:
#
#   Tier 1 — in-memory (st.cache_resource): a dict that lives for as long as
#            the container is running, shared by every user hitting it. Zero
#            setup, but resets on reboot. This is always active.
#   Tier 2 — OPTIONAL persistent cache via a GitHub Gist: survives reboots
#            and redeploys because it reads/writes JSON through the GitHub
#            API instead of local disk. Turns on automatically if you add
#            GITHUB_TOKEN and GIST_ID to st.secrets.
#
# To enable Tier 2:
#   1. Create a (secret) Gist at gist.github.com containing one file named
#      ticker_cache.json with the content: {}
#   2. Copy its Gist ID from the URL (gist.github.com/<user>/<GIST_ID>).
#   3. Create a GitHub fine-grained personal access token with only the
#      "gist" scope.
#   4. Add both to st.secrets:
#        GITHUB_TOKEN = "ghp_..."
#        GIST_ID = "your-gist-id"
#
# Other solid options if you outgrow this (larger shared state, multi-table
# data, etc.): Google Sheets via gspread, or a free-tier hosted database
# such as Supabase/Neon (Postgres) or Upstash (Redis) through st.connection.
# ==============================================================================

GIST_FILENAME = "ticker_cache.json"


@st.cache_resource(show_spinner=False)
def _memory_cache() -> dict:
    """Lives for as long as this container is running; shared by every
    session connected to it. Baseline cache tier — always active."""
    return {}


def _gist_configured() -> bool:
    return bool(st.secrets.get("GITHUB_TOKEN")) and bool(st.secrets.get("GIST_ID"))


def _gist_headers() -> dict:
    return {
        "Authorization": f"token {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }


def _gist_load() -> dict:
    try:
        url = f"https://api.github.com/gists/{st.secrets['GIST_ID']}"
        resp = requests.get(url, headers=_gist_headers(), timeout=10)
        resp.raise_for_status()
        files = resp.json().get("files", {})
        content = files.get(GIST_FILENAME, {}).get("content", "{}")
        return json.loads(content)
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"[Gist load] {e}")
        return {}


def _gist_save(cache: dict) -> None:
    try:
        url = f"https://api.github.com/gists/{st.secrets['GIST_ID']}"
        payload = {"files": {GIST_FILENAME: {"content": json.dumps(cache, indent=2)}}}
        resp = requests.patch(url, headers=_gist_headers(), json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"[Gist save] {e}")


def load_ticker_cache() -> dict:
    mem = _memory_cache()
    if not mem and _gist_configured():
        mem.update(_gist_load())
    return mem


def save_ticker_cache(cache: dict) -> None:
    _memory_cache().update(cache)  # keep tier-1 in sync
    if _gist_configured():
        _gist_save(cache)


if "ticker_cache" not in st.session_state:
    st.session_state.ticker_cache = load_ticker_cache()

# ==============================================================================
# GEMINI HELPERS
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_model(model_name: str, api_key: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def search_yfinance_ticker(company: str, model) -> str | None:
    """Ask Gemini for the Yahoo Finance ticker of a company."""
    prompt = f"""Return ONLY the Yahoo Finance ticker symbol for the following company.

Rules:
- For NSE listed Indian companies, append ".NS".
- For BSE-only companies, append ".BO".
- For US companies, return only the ticker.
- Return ONLY the ticker, no explanations, no markdown, no quotes.

Company: {company}"""
    try:
        response = model.generate_content(prompt)
        ticker = re.sub(r"[`\"']", "", response.text.strip()).strip()
        match = re.search(r"\b[A-Z0-9.-]+\b", ticker)
        return match.group(0) if match else None
    except Exception as e:
        st.session_state.setdefault("errors", []).append(f"[Ticker lookup] {company}: {e}")
        return None


def get_ticker(company: str, model) -> tuple[str | None, str]:
    """Returns (ticker, source) — source is 'cache' or 'ai'."""
    key = company.strip().lower()
    cache = st.session_state.ticker_cache
    if key in cache:
        return cache[key], "cache"
    ticker = search_yfinance_ticker(company, model)
    if ticker:
        cache[key] = ticker
        save_ticker_cache(cache)
        return ticker, "ai"
    return None, "not found"

# ==============================================================================
# FINANCIAL DATA
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_data(ticker: str):
    stock = yf.Ticker(ticker)
    try:
        bs = stock.balance_sheet
    except Exception:
        bs = pd.DataFrame()
    try:
        pl = stock.financials
    except Exception:
        pl = pd.DataFrame()
    try:
        cf = stock.cashflow
    except Exception:
        cf = pd.DataFrame()
    try:
        info = stock.info
    except Exception:
        info = {}
    try:
        hist = stock.history(period="1y")
    except Exception:
        hist = pd.DataFrame()
    return bs, pl, cf, info, hist


def financials_to_text(bs, pl, cf) -> str:
    def df_text(df):
        if df is None or df.empty:
            return "No data available."
        return df.fillna("").to_string()

    parts = []
    for title, df in (("BALANCE SHEET", bs), ("PROFIT & LOSS", pl), ("CASH FLOW", cf)):
        parts.append(f"\n\n{title}\n{df_text(df)}")
    return "".join(parts)

# ==============================================================================
# AI REPORT GENERATION
# ==============================================================================
def build_prompt(financial_text: str) -> str:
    question = (
        "Please read the following Balance Sheet, Profit & Loss, and Cash Flow "
        "statement data as tables. Based on this, write a report in clean "
        "Markdown (use ## headings, **bold**, and | pipe | tables |) with a "
        "first-person-plural voice ('we', 'us') describing our own analysis of "
        "this third-party company (e.g. 'we have analyzed their filings'). "
        "Do not literally write the words 'first person plural' anywhere. "
        "Structure the report under these four headings, phrased as natural "
        "report headings rather than restating the instructions:\n\n"
        "1. Total Assets & Revenue for the 3 most recent years, as a table "
        "with columns Year | Total Assets | Total Revenue.\n"
        "2. Five key financial ratios appropriate to this company's industry, "
        "for the 3 most recent years, as a table with columns Ratio | Formula "
        "| <Year 1> | <Year 2> | <Year 3>, followed by the detailed year-wise "
        "calculation for each ratio outside the table.\n"
        "3. A detailed opinion on the financial health of the company.\n"
        "4. A Risk Rating from 1 to 5 (1 = low risk, 5 = high risk), written "
        "exactly in the form 'Risk Rating: X/5' on its own line, followed by "
        "bullet-point justification."
    )
    return f"{question}\n\n{financial_text}"


def generate_report(model, financial_text: str) -> str:
    prompt = build_prompt(financial_text)
    response = model.generate_content(prompt)
    return response.text


def extract_risk_rating(text: str) -> int | None:
    match = re.search(r"risk\s*rating\D{0,15}?([1-5])\s*(?:/\s*5)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# ==============================================================================
# UI PIECES
# ==============================================================================
def render_banner():
    st.markdown(
        """
        <div id="bob-banner">
            <h1>🏦 BOB AI Financial Risk Analyzer</h1>
            <p>AI-assisted fundamental &amp; risk analysis, powered by Gemini + Yahoo Finance</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_key_stats(info: dict):
    def fmt_money(v):
        if not v:
            return "—"
        for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if abs(v) >= div:
                return f"{v/div:,.2f}{unit}"
        return f"{v:,.0f}"

    cols = st.columns(5)
    stats = [
        ("Market Cap", fmt_money(info.get("marketCap"))),
        ("P/E (TTM)", f"{info.get('trailingPE'):.2f}" if info.get("trailingPE") else "—"),
        ("52W High", f"{info.get('fiftyTwoWeekHigh', '—')}"),
        ("52W Low", f"{info.get('fiftyTwoWeekLow', '—')}"),
        ("Dividend Yield", f"{info.get('dividendYield', 0):.2%}" if info.get("dividendYield") else "—"),
    ]
    for col, (label, value) in zip(cols, stats):
        with col:
            st.markdown(
                f"""<div class="metric-card"><div style="color:{BOB_GREY};font-size:0.8em;">{label}</div>
                <div style="color:{BOB_NAVY};font-size:1.3em;font-weight:700;">{value}</div></div>""",
                unsafe_allow_html=True,
            )


def render_price_chart(hist: pd.DataFrame, company: str):
    if hist is None or hist.empty:
        st.info("No price history available for this ticker.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["Close"],
            mode="lines",
            line=dict(color=BOB_ORANGE, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(247,148,29,0.12)",
            name="Close",
        )
    )
    fig.update_layout(
        title=f"{company} — 1 Year Price Trend",
        title_font_color=BOB_NAVY,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        yaxis_title="Price",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_risk_gauge(rating: int | None):
    if rating is None:
        st.warning("Could not automatically detect a Risk Rating in the AI report — check the full text below.")
        return
    color = RISK_COLORS.get(rating, BOB_GREY)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rating,
            number={"suffix": " / 5", "font": {"color": BOB_NAVY, "size": 40}},
            gauge={
                "axis": {"range": [0, 5], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 1], "color": RISK_COLORS[1]},
                    {"range": [1, 2], "color": RISK_COLORS[2]},
                    {"range": [2, 3], "color": RISK_COLORS[3]},
                    {"range": [3, 4], "color": RISK_COLORS[4]},
                    {"range": [4, 5], "color": RISK_COLORS[5]},
                ],
            },
            title={"text": "AI Risk Rating", "font": {"color": BOB_NAVY}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_comparison(results: dict):
    rows = []
    for company, r in results.items():
        info = r.get("info", {}) or {}
        rows.append(
            {
                "Company": company,
                "Ticker": r.get("ticker"),
                "Risk Rating": r.get("risk_rating"),
                "Market Cap": info.get("marketCap"),
                "P/E (TTM)": info.get("trailingPE"),
                "Sector": info.get("sector", "—"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    rated = df.dropna(subset=["Risk Rating"])
    if not rated.empty:
        fig = go.Figure(
            go.Bar(
                x=rated["Company"],
                y=rated["Risk Rating"],
                marker_color=[RISK_COLORS.get(int(r), BOB_GREY) for r in rated["Risk Rating"]],
            )
        )
        fig.update_layout(
            title="Risk Rating Comparison",
            title_font_color=BOB_NAVY,
            yaxis=dict(range=[0, 5], title="Risk (1 = low, 5 = high)"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# SIDEBAR
# ==============================================================================
def render_sidebar():
    st.sidebar.markdown("### ⚙️ Settings")

    api_key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    if api_key:
        st.sidebar.success("Gemini API key loaded from st.secrets ✅")
    else:
        st.sidebar.error(
            "No `GEMINI_API_KEY` found in st.secrets.\n\n"
            "Add it to `.streamlit/secrets.toml`:\n\n"
            '`GEMINI_API_KEY = "your-key-here"`'
        )

    model_name = st.sidebar.selectbox("Gemini model", MODEL_OPTIONS, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗂️ Ticker Cache")
    if _gist_configured():
        st.sidebar.success("Persistent cache: GitHub Gist ✅")
    else:
        st.sidebar.warning("In-memory cache only — resets on app reboot.\n\nAdd GITHUB_TOKEN + GIST_ID to st.secrets for persistence.")
    st.sidebar.caption(f"{len(st.session_state.ticker_cache)} companies cached right now.")
    if st.sidebar.button("Clear ticker cache"):
        st.session_state.ticker_cache.clear()
        save_ticker_cache({})
        st.sidebar.success("Cache cleared.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.caption(
        "Enter one or more company names (comma-separated). The app resolves "
        "each to a Yahoo Finance ticker, pulls its financial statements, and "
        "asks Gemini for a structured risk report."
    )

    return api_key, model_name

# ==============================================================================
# MAIN APP
# ==============================================================================
def main():
    render_banner()
    api_key, model_name = render_sidebar()

    if "resolved" not in st.session_state:
        st.session_state.resolved = None  # DataFrame of company -> ticker
    if "results" not in st.session_state:
        st.session_state.results = {}

    st.markdown("#### Step 1 — Enter company names")
    names_raw = st.text_input(
        "Company names (comma-separated)",
        placeholder="e.g. Tata Consultancy Services, Apple, Reliance Industries",
        label_visibility="collapsed",
    )

    resolve_col, _ = st.columns([1, 4])
    with resolve_col:
        resolve_clicked = st.button("🔎 Resolve Tickers", use_container_width=True)

    if resolve_clicked:
        if not names_raw.strip():
            st.warning("Please enter at least one company name.")
        elif not api_key:
            st.error("Add your Gemini API key to st.secrets first (see sidebar).")
        elif genai is None:
            st.error("`google-generativeai` is not installed. Run `pip install google-generativeai`.")
        else:
            model = get_model(model_name, api_key)
            companies = [c.strip() for c in names_raw.split(",") if c.strip()]
            rows = []
            with st.spinner("Resolving tickers via Gemini + local cache …"):
                for company in companies:
                    ticker, source = get_ticker(company, model)
                    rows.append({"Company": company, "Ticker": ticker or "", "Source": source})
            st.session_state.resolved = pd.DataFrame(rows)
            st.session_state.results = {}  # reset downstream results on re-resolve

    if st.session_state.resolved is not None:
        st.markdown("#### Step 2 — Confirm or correct tickers")
        st.caption("AI-guessed tickers can be wrong — edit the **Ticker** column below before running the analysis.")
        edited = st.data_editor(
            st.session_state.resolved,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            disabled=["Company", "Source"],
            key="ticker_editor",
        )
        st.session_state.resolved = edited

        st.markdown("#### Step 3 — Generate the AI risk analysis")
        run_clicked = st.button("📊 Generate Financial Risk Analysis", type="primary")

        if run_clicked:
            if not api_key:
                st.error("Add your Gemini API key to st.secrets first (see sidebar).")
            else:
                model = get_model(model_name, api_key)
                valid_rows = edited[edited["Ticker"].str.strip() != ""]
                if valid_rows.empty:
                    st.warning("No valid tickers to analyze.")
                for _, row in valid_rows.iterrows():
                    company, ticker = row["Company"], row["Ticker"].strip()
                    status_box = st.status(f"Analyzing {company} ({ticker}) …", expanded=False)
                    try:
                        for i, msg in enumerate(STATUS_MESSAGES):
                            status_box.update(label=msg)
                            if i in (1, 3, 5):  # only actually do work at meaningful points
                                pass
                        bs, pl, cf, info, hist = get_financial_data(ticker)
                        financial_text = financials_to_text(bs, pl, cf)
                        report_text = generate_report(model, financial_text)
                        risk_rating = extract_risk_rating(report_text)
                        st.session_state.results[company] = {
                            "ticker": ticker,
                            "info": info,
                            "hist": hist,
                            "bs": bs,
                            "pl": pl,
                            "cf": cf,
                            "report": report_text,
                            "risk_rating": risk_rating,
                            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        status_box.update(label=f"{company} done ✅", state="complete")
                    except Exception as e:
                        status_box.update(label=f"{company} failed: {e}", state="error")

    # --------------------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------------------
    if st.session_state.results:
        st.markdown("---")
        st.markdown("### 📁 Results")

        if len(st.session_state.results) > 1:
            with st.expander("📊 Multi-company comparison", expanded=True):
                render_comparison(st.session_state.results)

        tabs = st.tabs(list(st.session_state.results.keys()))
        for tab, (company, r) in zip(tabs, st.session_state.results.items()):
            with tab:
                st.caption(f"Ticker: **{r['ticker']}**  ·  Generated {r['generated_at']}")

                render_key_stats(r["info"] or {})
                st.write("")

                chart_col, gauge_col = st.columns([2, 1])
                with chart_col:
                    render_price_chart(r["hist"], company)
                with gauge_col:
                    render_risk_gauge(r["risk_rating"])

                st.markdown("#### AI Risk Report")
                st.markdown(r["report"])

                with st.expander("Raw financial statements"):
                    st.markdown("**Balance Sheet**")
                    st.dataframe(r["bs"], use_container_width=True)
                    st.markdown("**Profit & Loss**")
                    st.dataframe(r["pl"], use_container_width=True)
                    st.markdown("**Cash Flow**")
                    st.dataframe(r["cf"], use_container_width=True)

                st.download_button(
                    f"⬇️ Download {company} report (Markdown)",
                    data=r["report"],
                    file_name=f"{company.replace(' ', '_')}_risk_report.md",
                    mime="text/markdown",
                )

        if len(st.session_state.results) > 1:
            combined = "\n\n---\n\n".join(
                f"# {c}\n\n{r['report']}" for c, r in st.session_state.results.items()
            )
            st.download_button(
                "⬇️ Download combined report (all companies)",
                data=combined,
                file_name="BOB_AI_Financial_Risk_Report.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
