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

RISK_COLORS = {5: "#2E9E4E", 4: "#8FC93A", 3: "#F7C700", 2: "#F0862C", 1: "#D62839"}

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
    section[data-testid="stSidebar"] {{
        background-color: #fff1e0;
    }}
    section[data-testid="stSidebar"] * {{
        color: #12284C !important;
    }}
    /* selectbox closed state (sits inside the sidebar) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: #fff1e0 !important;
        border: 1px solid #F7941D;
        border-radius: 6px;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
        color: #12284C !important;
    }}
    /* dropdown option list renders in a portal OUTSIDE the sidebar, so it
       needs its own (unscoped) rule rather than the sidebar selector above */
    div[data-baseweb="popover"] ul[role="listbox"] {{
        background-color: #fff1e0 !important;
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
    """
    Lives for as long as this container is running; shared by every
    session connected to it. Baseline cache tier — always active.
    """
    return {}


def _gist_configured() -> bool:
    return (
        bool(st.secrets.get("GITHUB_TOKEN"))
        and bool(st.secrets.get("GIST_ID"))
    )


def _gist_headers() -> dict:
    return {
        "Authorization": f"token {st.secrets['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }


def _gist_load() -> dict:
    try:
        url = f"https://api.github.com/gists/{st.secrets['GIST_ID']}"

        resp = requests.get(
            url,
            headers=_gist_headers(),
            timeout=10,
        )
        resp.raise_for_status()

        files = resp.json().get("files", {})
        content = files.get(
            GIST_FILENAME,
            {}
        ).get("content", "{}")

        return json.loads(content)

    except Exception as e:
        st.session_state.setdefault("errors", []).append(
            f"[Gist load] {e}"
        )
        return {}


def _gist_save(updates: dict) -> None:
    """
    Update only the supplied company -> ticker pairs.

    All existing entries in ticker_cache.json remain intact.
    """
    try:
        # --------------------------------------------------------
        # 1. Read the current JSON from Gist
        # --------------------------------------------------------
        existing_cache = _gist_load()

        if not isinstance(existing_cache, dict):
            existing_cache = {}

        # --------------------------------------------------------
        # 2. Update ONLY the supplied entries
        # --------------------------------------------------------
        existing_cache.update(updates)

        # --------------------------------------------------------
        # 3. Save the complete merged cache
        # --------------------------------------------------------
        url = f"https://api.github.com/gists/{st.secrets['GIST_ID']}"

        payload = {
            "files": {
                GIST_FILENAME: {
                    "content": json.dumps(
                        existing_cache,
                        indent=2,
                        ensure_ascii=False,
                    )
                }
            }
        }

        resp = requests.patch(
            url,
            headers=_gist_headers(),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()

    except Exception as e:
        st.session_state.setdefault("errors", []).append(
            f"[Gist save] {e}"
        )


def load_ticker_cache() -> dict:
    mem = _memory_cache()

    if not mem and _gist_configured():
        mem.update(_gist_load())

    return mem


def save_ticker_cache(updates: dict) -> None:
    """
    Update only the supplied ticker mappings.

    Existing cache entries that are not included in `updates`
    remain untouched.
    """
    if not updates:
        return

    # Keep tier-1 memory cache in sync
    _memory_cache().update(updates)

    # Update Gist while preserving all existing rows
    if _gist_configured():
        _gist_save(updates)


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
    Sometime acronyms are given, verify popular acronyms like 'BOB' always stands for 'Bank of Baroda' etc.
    Some important index ticker has '^' in their ticker name, do not miss such '^' if available.

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
        return cache[key], "Yahoo Finance Cache"
    ticker = search_yfinance_ticker(company, model)
    if ticker:
        cache[key] = ticker
        save_ticker_cache(cache)
        return ticker, "Yahoo Finance"
    return None, "not found"

# ==============================================================================
# FINANCIAL DATA
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_data(ticker: str):
    """Fetches ONLY the three financial statements + price history. Headline
    market stats (market cap, P/E, 52-week range, dividend yield, etc.) are
    fetched separately by get_market_stats(), from lighter/more reliable
    yfinance endpoints rather than being bundled in here."""
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
        hist = stock.history(period="1y")
    except Exception:
        hist = pd.DataFrame()
    return bs, pl, cf, hist


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_stats(ticker: str) -> dict:
    """
    Pulls headline market stats from yfinance sources OTHER than the balance
    sheet / P&L / cash flow used above:

      - `Ticker.fast_info` — a lightweight snapshot endpoint (market cap,
        last price, 52-week high/low, moving averages) that's quicker and
        far less prone to throttling than the full `.info` scrape.
      - `Ticker.dividends` — the actual dividend-payment history, used to
        compute a trailing-12-month yield ourselves rather than trusting
        info["dividendYield"], whose scaling has changed across yfinance
        versions and is often stale or missing.
      - `.info` — used only as a last-resort fallback for the handful of
        fields fast_info doesn't carry (P/E, beta, sector, analyst target),
        so a slow/blocked info call can't take down the headline stats.
    """
    stock = yf.Ticker(ticker)
    stats: dict = {}

    try:
        fi = stock.fast_info
        stats["market_cap"] = fi.get("marketCap") or fi.get("market_cap")
        stats["last_price"] = fi.get("lastPrice") or fi.get("last_price")
        stats["year_high"] = fi.get("yearHigh") or fi.get("year_high")
        stats["year_low"] = fi.get("yearLow") or fi.get("year_low")
        stats["fifty_day_avg"] = fi.get("fiftyDayAverage") or fi.get("fifty_day_average")
        stats["two_hundred_day_avg"] = fi.get("twoHundredDayAverage") or fi.get("two_hundred_day_average")
        stats["currency"] = fi.get("currency")
        stats["shares_outstanding"] = fi.get("shares")
    except Exception:
        pass

    try:
        divs = stock.dividends
        if divs is not None and not divs.empty and stats.get("last_price"):
            cutoff = divs.index.max() - pd.Timedelta(days=365)
            ttm_dividends = divs[divs.index >= cutoff].sum()
            stats["dividend_yield"] = (ttm_dividends / stats["last_price"]) if stats["last_price"] else None
    except Exception:
        pass

    try:
        info = stock.info
        stats["trailing_pe"] = info.get("trailingPE")
        stats["forward_pe"] = info.get("forwardPE")
        stats["beta"] = info.get("beta")
        stats["sector"] = info.get("sector")
        stats["industry"] = info.get("industry")
        stats["target_mean_price"] = info.get("targetMeanPrice")
        stats["recommendation"] = info.get("recommendationKey")
    except Exception:
        pass

    return stats


def financials_to_text(bs, pl, cf) -> str:
    def df_text(df):
        if df is None or df.empty:
            return "No data available."
        return df.fillna("").to_string()

    parts = []
    for title, df in (("BALANCE SHEET", bs), ("PROFIT & LOSS", pl), ("CASH FLOW", cf)):
        parts.append(f"\n\n{title}\n{df_text(df)}")
    return "".join(parts)


def market_stats_to_text(stats: dict) -> str:
    """Renders the fast_info / dividend / info-derived stats as a text block
    to hand to Gemini alongside the financial statements."""
    if not stats:
        return "No market snapshot available."

    def pct(v):
        return f"{v:.2%}" if isinstance(v, (int, float)) else "N/A"

    def num(v):
        return f"{v:,}" if isinstance(v, (int, float)) else "N/A"

    lines = [
        f"Market Cap: {num(stats.get('market_cap'))}",
        f"Last Price: {num(stats.get('last_price'))} {stats.get('currency', '')}",
        f"52-Week High: {num(stats.get('year_high'))}",
        f"52-Week Low: {num(stats.get('year_low'))}",
        f"50-Day Average: {num(stats.get('fifty_day_avg'))}",
        f"200-Day Average: {num(stats.get('two_hundred_day_avg'))}",
        f"Trailing P/E: {num(stats.get('trailing_pe'))}",
        f"Forward P/E: {num(stats.get('forward_pe'))}",
        f"Trailing Dividend Yield (TTM, computed from payout history): {pct(stats.get('dividend_yield'))}",
        f"Beta: {num(stats.get('beta'))}",
        f"Sector: {stats.get('sector') or 'N/A'}",
        f"Industry: {stats.get('industry') or 'N/A'}",
        f"Analyst Mean Target Price: {num(stats.get('target_mean_price'))}",
        f"Analyst Recommendation: {stats.get('recommendation') or 'N/A'}",
    ]
    return "\n".join(lines)

# ==============================================================================
# AI REPORT GENERATION
# ==============================================================================
def build_prompt(financial_text: str, market_text: str) -> str:
    question = (
        "Please read the following Balance Sheet, Profit & Loss, and Cash Flow "
        "statement data as tables, along with the Market Snapshot section "
        "(valuation multiples, 52-week range, dividend yield, sector/industry, "
        "analyst target/recommendation). Based on all of this, write a report "
        "in clean Markdown (use ## headings, **bold** for emphasis, and "
        "| pipe | tables | for tabular data) with a first-person-plural voice "
        "('we', 'us') describing our own analysis of this third-party company "
        "(e.g. 'we have analyzed their filings'). Do not literally write the "
        "words 'first person plural' anywhere.\n\n"
        "Markdown formatting rules — follow these exactly so the report "
        "renders cleanly:\n"
        "- Use double asterisks only for **bold**; never a single asterisk.\n"
        "- Never use an asterisk for multiplication in a formula — write "
        "'x' or the word 'times' instead (e.g. 'Net Income x 100', not "
        "'Net Income*100').\n"
        "- Never use underscores inside words, ratio names, or numbers "
        "(write 'Debt to Equity', not 'Debt_to_Equity'; write '12,000', "
        "not '12_000').\n"
        "- Use '-' for bullet points, not '*'.\n\n"
        "Structure the report under these four headings, phrased as natural "
        "report headings rather than restating the instructions:\n\n"
        "1. Total Assets & Revenue for the 3 most recent years, as a table "
        "with columns Year | Total Assets | Total Revenue.\n"
        "2. Five key financial ratios appropriate to this company's industry, "
        "for the 3 most recent years, as a table with columns Ratio | Formula "
        "| <Year 1> | <Year 2> | <Year 3>, followed by the detailed year-wise "
        "calculation for each ratio outside the table. Where relevant, relate "
        "these to the valuation multiples and analyst context in the Market "
        "Snapshot (e.g. whether the P/E or dividend yield looks rich or cheap "
        "relative to the computed ratios).\n"
        "3. A detailed opinion on the financial health of the company, "
        "incorporating the Market Snapshot context (valuation, analyst "
        "sentiment, sector) alongside the statement analysis.\n"
        "4. A Risk Rating from 1 to 5 (1 = high risk, 5 = low risk), written "
        "exactly in the form 'Risk Rating: X/5' on its own line, followed by "
        "bullet-point justification."
    )
    return f"{question}\n\nMARKET SNAPSHOT\n{market_text}\n\n{financial_text}"


def generate_report(model, financial_text: str, market_text: str) -> str:
    prompt = build_prompt(financial_text, market_text)
    response = model.generate_content(prompt)
    return response.text


def extract_risk_rating(text: str) -> int | None:
    match = re.search(r"risk\s*rating\D{0,15}?([1-5])\s*(?:/\s*5)?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def sanitize_markdown(text: str) -> str:
    """
    Gemini's Markdown occasionally contains a stray single '*' (used for
    multiplication in a formula, e.g. 'Assets*Turnover') or an underscore
    inside a number/identifier (e.g. '12_000'). Streamlit's CommonMark
    renderer reads an unmatched '*' or a mid-word '_' as the START of
    *italics* with no visible closing partner, which is exactly the
    "random italics with no space" symptom. The prompt above now asks
    Gemini to avoid these, but this is a defensive second layer for
    whatever slips through — it leaves real **bold** text and | tables |
    alone.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if line.strip().startswith("|"):
            # Table rows are more fragile to touch — leave formatting as-is.
            cleaned.append(line)
            continue
        # Escape underscores sitting between two word characters — these
        # are almost never intended as emphasis in a financial report.
        line = re.sub(r"(?<=\w)_(?=\w)", r"\_", line)
        # If a line has an odd number of single '*' outside of any
        # '**bold**' pairs, it has an unmatched emphasis marker — escape
        # every remaining lone '*' so it can't swallow the rest of the line.
        without_bold = re.sub(r"\*\*.*?\*\*", "", line)
        if without_bold.count("*") % 2 == 1:
            line = re.sub(r"(?<!\*)\*(?!\*)", r"\*", line)
        cleaned.append(line)
    return "\n".join(cleaned)

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


def render_key_stats(stats: dict):
    def fmt_money(v):
        if not v:
            return "—"
        for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
            if abs(v) >= div:
                return f"{v/div:,.2f}{unit}"
        return f"{v:,.0f}"

    cols = st.columns(5)
    values = [
        ("Market Cap", fmt_money(stats.get("market_cap"))),
        ("P/E (TTM)", f"{stats['trailing_pe']:.2f}" if stats.get("trailing_pe") else "—"),
        ("52W High", f"{stats['year_high']:,.2f}" if stats.get("year_high") else "—"),
        ("52W Low", f"{stats['year_low']:,.2f}" if stats.get("year_low") else "—"),
        ("Dividend Yield", f"{stats['dividend_yield']:.2%}" if stats.get("dividend_yield") else "—"),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                f"""<div class="metric-card"><div style="color:{BOB_GREY};font-size:0.8em;">{label}</div>
                <div style="color:{BOB_NAVY};font-size:1.3em;font-weight:700;">{value}</div></div>""",
                unsafe_allow_html=True,
            )


def _safe_key(*parts: str) -> str:
    """Turn arbitrary text (company names, tickers) into a stable, unique
    Streamlit widget key. Needed because Streamlit auto-generates element
    IDs from an element's type + parameters — two companies that happen to
    render an IDENTICAL chart (e.g. same risk rating, or both show the
    'no price data' fallback) would otherwise collide and raise
    StreamlitDuplicateElementId."""
    raw = "_".join(str(p) for p in parts)
    return re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()


def render_price_chart(hist: pd.DataFrame, company: str):
    if hist is None or hist.empty:
        st.info("No price history available for this ticker.")
        return

    hist = hist.copy()
    hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
    hist = hist.dropna(subset=["Close"])

    if hist.empty:
        st.info("No valid price history available for this ticker.")
        return

    # Highest and lowest points
    max_idx = hist["Close"].idxmax()
    min_idx = hist["Close"].idxmin()

    max_value = hist.loc[max_idx, "Close"]
    min_value = hist.loc[min_idx, "Close"]

    # Format dates
    max_date = pd.to_datetime(max_idx).strftime("%d-%b-%Y")
    min_date = pd.to_datetime(min_idx).strftime("%d-%b-%Y")

    fig = go.Figure()

    # Price line
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

    # Highest value — green dotted line
    fig.add_vline(
        x=max_idx,
        line=dict(
            color="green",
            width=1,
            dash="dot",
        ),
    )

    # Lowest value — red dotted line
    fig.add_vline(
        x=min_idx,
        line=dict(
            color="red",
            width=1,
            dash="dot",
        ),
    )

    # Highest date annotation
    fig.add_annotation(
        x=max_idx,
        y=1.02,
        xref="x",
        yref="paper",
        text=max_date,
        showarrow=False,
        textangle=-90,
        font=dict(
            color="green",
            size=10,
        ),
        xanchor="center",
        yanchor="bottom",
    )

    # Lowest date annotation
    fig.add_annotation(
        x=min_idx,
        y=1.02,
        xref="x",
        yref="paper",
        text=min_date,
        showarrow=False,
        textangle=-90,
        font=dict(
            color="red",
            size=10,
        ),
        xanchor="center",
        yanchor="bottom",
    )

    fig.update_layout(
        title=f"{company} — 1 Year Price Trend",
        title_font_color=BOB_NAVY,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=65, b=10),
        height=320,
        yaxis_title="Price",
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=_safe_key("price_chart", company),
    )


def render_risk_gauge(rating: int | None, company: str):
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
                "bar": {"color": "blue"},
                "steps": [
                    {"range": [0, 1], "color": RISK_COLORS[1]},
                    {"range": [1, 2], "color": RISK_COLORS[2]},
                    {"range": [2, 3], "color": RISK_COLORS[3]},
                    {"range": [3, 4], "color": RISK_COLORS[4]},
                    {"range": [4, 5], "color": RISK_COLORS[5]},
                ],
            },
            title={"text": f"{company} — AI Risk Rating", "font": {"color": BOB_NAVY}},
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, key=_safe_key("risk_gauge", company))


def render_comparison(results: dict):
    rows = []
    for company, r in results.items():
        stats = r.get("stats", {}) or {}
        rows.append(
            {
                "Company": company,
                "Ticker": r.get("ticker"),
                "Risk Rating": r.get("risk_rating"),
                "Market Cap": stats.get("market_cap"),
                "P/E (TTM)": stats.get("trailing_pe"),
                "Sector": stats.get("sector", "—"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, key="comparison_table")

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
        st.plotly_chart(fig, use_container_width=True, key="comparison_bar_chart")

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
    
    # Check if the cache exists and is not empty before rendering
    if 'ticker_cache' in st.session_state and st.session_state.ticker_cache:
        
        # 1. Setup the sidebar section
        st.sidebar.write("### Ticker Cache Data")
        
        # 2. Wrap the table inside an expander in the sidebar
        with st.sidebar.expander("Ticker Data", expanded=False):
            
            # Convert JSON dict to DataFrame: Key becomes Row Index, Value becomes Column 0
            df = pd.DataFrame.from_dict(st.session_state.ticker_cache, orient='index')
            
            # Reset index so "Company Name" becomes a regular, named column
            df = df.reset_index()
            
            # Now there are exactly 2 columns, so we can rename them safely
            df.columns = ["Company Name", "Ticker"]
            
            # Display the clean interactive table
            st.dataframe(df, use_container_width=True, hide_index=True) 
            
    else:
        st.sidebar.warning("The ticker cache is currently empty or not initialized.")

     
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

        # ------------------------------------------------------------
        # Update ticker cache with edited values
        # ------------------------------------------------------------
        edited_rows = st.session_state.ticker_editor.get("edited_rows", {})
        
        ticker_updates = {}
        
        for row_idx, changes in edited_rows.items():
        
            if "Ticker" not in changes:
                continue
        
            company = str(
                edited.iloc[row_idx]["Company"]
            ).strip()
        
            ticker = str(
                changes["Ticker"]
            ).strip()
        
            if company and ticker and ticker.lower() != "nan":
                ticker_updates[company] = ticker
        
        
        if ticker_updates:
            save_ticker_cache(ticker_updates)

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
                        bs, pl, cf, hist = get_financial_data(ticker)
                        market_stats = get_market_stats(ticker)
                        financial_text = financials_to_text(bs, pl, cf)
                        market_text = market_stats_to_text(market_stats)
                        report_text = generate_report(model, financial_text, market_text)
                        risk_rating = extract_risk_rating(report_text)
                        st.session_state.results[company] = {
                            "ticker": ticker,
                            "stats": market_stats,
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

                render_key_stats(r["stats"] or {})
                st.write("")

                chart_col, gauge_col = st.columns([2, 1])
                with chart_col:
                    render_price_chart(r["hist"], company)
                with gauge_col:
                    render_risk_gauge(r["risk_rating"], company)

                st.markdown("#### AI Risk Report")
                st.markdown(
                    f"""
                    <div style="
                        background-color: #fff8ef;
                        border: 1px solid #f7941d;
                        border-radius: 8px;
                        padding: 10px 14px;
                        color: #002e6e;
                    ">
                        {sanitize_markdown(r["report"])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                # st.markdown(sanitize_markdown(r["report"]))

                st.html("<br>")

                with st.expander("Raw financial statements"):
                    st.markdown("**Balance Sheet**")
                    st.dataframe(r["bs"], use_container_width=True, key=_safe_key("bs", company))
                    st.markdown("**Profit & Loss**")
                    st.dataframe(r["pl"], use_container_width=True, key=_safe_key("pl", company))
                    st.markdown("**Cash Flow**")
                    st.dataframe(r["cf"], use_container_width=True, key=_safe_key("cf", company))

                st.download_button(
                    f"⬇️ Download {company} report (Markdown)",
                    data=r["report"],
                    file_name=f"{company.replace(' ', '_')}_risk_report.md",
                    mime="text/markdown",
                    key=_safe_key("download", company),
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
                key="download_combined",
            )


if __name__ == "__main__":
    main()
