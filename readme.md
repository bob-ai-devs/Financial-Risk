# 🏦 BOB AI Financial Risk Analyzer

An AI-assisted fundamental and risk analysis dashboard built with **Streamlit**, **yfinance**, and **Google Gemini**. Type in one or more company names, confirm the resolved ticker, and get a structured AI-written risk report backed by real financial statements and market data — styled in a Bank of Baroda colour scheme.

This is a Streamlit rebuild of an earlier Flask prototype, with the ticker-resolution/report-generation flow re-architected around Streamlit's rerun model instead of manual threads and polling endpoints.

---

## ✨ Features

- **Multi-company analysis** — enter several comma-separated company names and each is analyzed independently (not concatenated into one query).
- **AI ticker resolution** — Gemini maps a company name to its Yahoo Finance ticker; results are cached so you don't spend an API call twice on the same company.
- **Editable ticker confirmation step** — review and correct AI-guessed tickers *before* the (more expensive) financial report is generated.
- **Financial statement analysis** — pulls Balance Sheet, Profit & Loss, and Cash Flow via yfinance and feeds them to Gemini for ratio analysis and a written opinion.
- **Market snapshot, sourced independently of the statements** — market cap, 52-week range, moving averages, and a *computed* trailing dividend yield come from `yfinance`'s lightweight `fast_info` and dividend-history endpoints rather than the slower, less reliable `.info` scrape; the AI report references this context too.
- **Auto-extracted Risk Rating gauge** — the AI's "Risk Rating: X/5" is parsed out and shown as a colour-coded gauge.
- **1-year price chart** per company.
- **Multi-company comparison view** — table + bar chart of risk ratings, market caps, and P/E across everything you've analyzed in the session.
- **Markdown sanitization** — defends against stray `*`/`_` characters in AI output being misread as unintended italics.
- **Downloadable reports** — per-company or combined Markdown export.
- **Bank of Baroda theming** — orange/maroon "sun" banner, navy sidebar, branded buttons.

---

## 🗂 Project Structure

```
.
├── app.py                          # the Streamlit app
├── requirements.txt
└── .streamlit/
    └── secrets.toml.example        # template — copy to secrets.toml, fill in, never commit the real one
```

---

## 🔑 Secrets

The app needs a Gemini API key, and optionally a GitHub token + Gist ID for persistent ticker caching (see [Ticker cache persistence](#-ticker-cache-persistence) below).

**Locally:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in your values:

```toml
GEMINI_API_KEY = "your-gemini-api-key"

# Optional — only needed for the ticker cache to survive app reboots
GITHUB_TOKEN = "github_pat_..."
GIST_ID = "your-gist-id"
```

Add `.streamlit/secrets.toml` to `.gitignore` — never commit real keys.

**On Streamlit Community Cloud:** open your deployed app → **⋮ menu → Settings → Secrets**, and paste the same content in there instead.

Get a Gemini API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at your repo, branch, and `app.py`.
4. Before or after the first deploy, add your secrets under **App settings → Secrets** (same TOML content as above).
5. Deploy.

---

## 🗄 Ticker Cache Persistence

Streamlit Cloud's filesystem is writable while the app is running but is **wiped on every reboot** (a `git push`, a manual restart, or waking from inactivity-sleep). So the ticker cache uses two tiers:

- **Tier 1 — in-memory (always on, zero setup).** Shared by every user of the currently-running container; resets on reboot.
- **Tier 2 — optional, persistent via a GitHub Gist.** Set `GITHUB_TOKEN` (a fine-grained PAT with the **Gists: Read and write** account permission) and `GIST_ID` (of a secret Gist containing a `ticker_cache.json` file with `{}` as its content) in secrets, and the cache survives reboots and redeploys.

The sidebar shows which tier is currently active.

---

## 🧠 How It Works

1. **Resolve tickers** — for each company name, check the local cache; on a miss, ask Gemini for the Yahoo Finance ticker.
2. **Confirm tickers** — an editable table lets you fix any wrong guesses before analysis runs.
3. **Fetch data** — `yfinance` pulls the three financial statements (`balance_sheet`, `financials`, `cashflow`) plus a market snapshot from `fast_info` and dividend history.
4. **Generate report** — the statements and market snapshot are sent to Gemini with a prompt asking for a structured Markdown report: assets/revenue trend, five key ratios, a financial-health opinion, and a 1–5 Risk Rating.
5. **Render** — Streamlit renders the report's Markdown natively (headings, bold, tables), with a defensive pass to fix any stray formatting characters, alongside a price chart, key-stat cards, and a risk gauge.

---

## ⚠️ Known Limitations

- AI-generated analysis is not financial advice — always sanity-check figures against the raw statements shown in the "Raw financial statements" expander.
- `yfinance` pulls from Yahoo Finance, which can rate-limit or occasionally miss data for smaller/less-liquid tickers.
- The in-memory ticker cache (Tier 1) resets on every app reboot unless the optional Gist persistence (Tier 2) is configured.
- Gemini's ticker guesses can be wrong, especially for companies with common names — always check the confirmation step before running the full analysis.

---

## 📄 License

Internal / demo project — add a license here if you plan to distribute this.
