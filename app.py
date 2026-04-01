from __future__ import annotations

import html
import os
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import time

from logic import (
    RELIABILITY_BUFFER_DAYS,
    apply_kinetic_hormuz_maximum_override,
    evaluate_strait_status_from_live_entries,
    fetch_live_google_news_multiquery,
    fetch_liveuamap_mideast_kinetic,
    fetch_official_tehran_narrative,
    fetch_hormuz_stats,
    fetch_realtime_shipping_stats,
)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_portwatch_snapshot():
    return fetch_realtime_shipping_stats()


@st.cache_data(ttl=300, show_spinner=False)
def _cached_hormuz_stats():
    return fetch_hormuz_stats()


# Sent with RSS fetches so cloud egress IPs are less likely to be blocked by Google News.
NEWS_RSS_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}

DEADLINE_UTC = datetime(2026, 4, 1, 16, 30, tzinfo=UTC)  # 16:30 GMT == 20:00 Tehran


def deadline_window_active(now_utc: datetime) -> bool:
    return now_utc <= (DEADLINE_UTC + timedelta(hours=24))


def format_countdown(now_utc: datetime) -> tuple[str, bool]:
    delta = DEADLINE_UTC - now_utc
    if delta.total_seconds() >= 0:
        total = int(delta.total_seconds())
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"T-{hh:02d}:{mm:02d}:{ss:02d}", True

    since = now_utc - DEADLINE_UTC
    total = int(since.total_seconds())
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"T+{hh:02d}:{mm:02d}:{ss:02d}", False


def risk_level_from_score(score: int) -> str:
    if score >= 75:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def _newsdata_api_key() -> str | None:
    v = (os.environ.get("NEWSDATA_API_KEY") or "").strip()
    if v:
        return v
    try:
        return str(st.secrets["NEWSDATA_API_KEY"]).strip()
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def _cached_liveuamap_bundle():
    return fetch_liveuamap_mideast_kinetic(request_headers=NEWS_RSS_REQUEST_HEADERS)


@st.cache_data(ttl=90, show_spinner=False)
def _cached_tehran_narrative():
    return fetch_official_tehran_narrative(
        api_key=_newsdata_api_key(),
        request_headers=NEWS_RSS_REQUEST_HEADERS,
    )


def _prepare_intel_dataframe(
    entries: list[dict], *, include_time_column: bool = True
) -> pd.DataFrame | None:
    if not entries:
        return None
    df_news = pd.DataFrame(entries)
    if df_news.empty:
        return None
    for col in ["Date/Time (UTC)", "Source", "Title", "Link"]:
        if col not in df_news.columns:
            df_news[col] = "" if col == "Date/Time (UTC)" else "N/A"
    raw_dt = df_news["Date/Time (UTC)"].astype(str)
    cleaned = raw_dt.str.replace(" GMT", "", regex=False).str.strip()
    _dt = pd.to_datetime(
        cleaned,
        errors="coerce",
        utc=True,
        format="%a, %d %b %Y %H:%M:%S",
    )
    if _dt.isna().all():
        _dt = pd.to_datetime(raw_dt, errors="coerce", utc=True)
    df_news = df_news.assign(_dt=_dt).sort_values(by="_dt", ascending=False).reset_index(drop=True)
    if include_time_column:
        df_news["Date/Time (UTC)"] = df_news["_dt"].dt.strftime("%H:%M GMT").fillna("")
        df_news = df_news.drop(columns=["_dt"])
    else:
        df_news = df_news.drop(columns=["_dt", "Date/Time (UTC)"])
    df_news.insert(0, "No.", range(1, len(df_news) + 1))
    return df_news


def _intel_highlight_row(row):
    title = str(row.get("Title", "")).lower()
    if ("target" in title) or ("strike" in title):
        return ["background-color: rgba(255, 215, 0, 0.22)"] * len(row)
    return [""] * len(row)


def _trade_drop_split_pcts(raw_val) -> tuple[str, str]:
    """Split regional trade shock into retail vs maritime deficit (%), both shown negative."""
    if raw_val is None:
        return "—", "—"
    if isinstance(raw_val, str) and raw_val.strip() in ("", "—", "N/A"):
        return "—", "—"
    try:
        mag = abs(float(raw_val))
        retail_mag = round(max(0.6, min(4.5, mag * 0.072 + 0.9)), 1)
        inc_mag = round(mag, 1)
        return f"-{retail_mag}%", f"-{inc_mag}%"
    except (TypeError, ValueError):
        return "—", "—"


def _region_trade_column_markdown(
    region_label: str,
    retail_pct: str,
    incoming_pct: str,
    retail_tooltip: str,
    supply_tooltip: str,
) -> str:
    return (
        f'<div class="eu-metric-slot">'
        f'<div class="region-trade-heading">{html.escape(region_label)}</div>'
        f'<div class="eu-metric-block" title="{html.escape(retail_tooltip)}">'
        f'<div class="eu-metric-label">Final Sales (Retail Index)</div>'
        f'<div class="eu-final-sales-value">{html.escape(retail_pct)}</div>'
        f"</div>"
        f'<div class="eu-metric-block" title="{html.escape(supply_tooltip)}">'
        f'<div class="eu-metric-label">Incoming Supply Chain (Maritime Inbound)</div>'
        f'<div class="eu-supply-chain-value">{html.escape(incoming_pct)}</div>'
        f"</div>"
        f"</div>"
    )


def shipping_impact_table(now_utc: datetime) -> pd.DataFrame:
    """
    'Suez vs Cape' transit impact for the three EU ports.

    Transit Delay (Days) is computed as:
      (Current ETA) - (Normal Suez ETA)

    Baseline (2026) Cape deltas:
      - Rotterdam: +20 days via Cape
      - Trieste:   +17 days via Cape
    """
    baseline_cape_delta_days: dict[str, int] = {
        "Rotterdam": 20,
        "Trieste": 17,
        # Not provided; keep conservative midpoint for display until you specify a baseline.
        "Fos-sur-Mer": 18,
    }

    baseline_normal_suez_transit_days: dict[str, int] = {
        "Rotterdam": 12,
        "Trieste": 10,
        "Fos-sur-Mer": 11,
    }

    forced = deadline_window_active(now_utc)
    pw = _cached_portwatch_snapshot()
    cape_mode = forced or pw.cape_mode
    ports = ["Trieste", "Rotterdam", "Fos-sur-Mer"]
    rows: list[dict] = []
    for p in ports:
        normal_days = baseline_normal_suez_transit_days.get(p, 11)
        normal_suez_eta = now_utc + timedelta(days=normal_days)

        cape_delta = baseline_cape_delta_days.get(p, 18)
        current_eta = normal_suez_eta + (timedelta(days=cape_delta) if cape_mode else timedelta(days=0))

        transit_delay_days = (current_eta - normal_suez_eta).total_seconds() / 86400.0
        transit_delay_days_with_buffer = transit_delay_days + float(RELIABILITY_BUFFER_DAYS)
        rows.append(
            {
                "Port": p,
                "Normal Suez ETA (UTC)": normal_suez_eta.strftime("%Y-%m-%d"),
                "Current ETA (UTC)": current_eta.strftime("%Y-%m-%d"),
                "Transit Delay (Days)": round(transit_delay_days_with_buffer, 1),
                "Reliability Buffer (Days)": float(RELIABILITY_BUFFER_DAYS),
                "Cape Mode": "ON" if cape_mode else "OFF",
                "Suez Calls/Day": pw.suez_transits_per_day if pw.suez_transits_per_day is not None else "—",
            }
        )
    return pd.DataFrame(rows)


def render_tactical_alert_banner(now_utc: datetime) -> None:
    t_str, _is_before = format_countdown(now_utc)
    st.markdown(
        f"""
        <style>
          @keyframes threatPulse {{
            0%   {{ box-shadow: 0 0 0px rgba(255, 59, 59, 0.0); transform: scale(1.000); }}
            50%  {{ box-shadow: 0 0 26px rgba(255, 59, 59, 0.45); transform: scale(1.005); }}
            100% {{ box-shadow: 0 0 0px rgba(255, 59, 59, 0.0); transform: scale(1.000); }}
          }}
          .tacticalBanner {{
            border: 2px solid #ff3b3b;
            background: rgba(255, 0, 0, 0.16);
            padding: 14px 16px;
            border-radius: 12px;
            margin-bottom: 12px;
            animation: threatPulse 0.95s ease-in-out infinite;
          }}
          .tacticalTitle {{
            font-size: 14px;
            letter-spacing: 0.10em;
            color: #ffd0d0;
            font-weight: 900;
          }}
          .tacticalTimer {{
            font-size: 42px;
            line-height: 1.05;
            color: #FFFF00;
            font-weight: 1000;
            margin-top: 6px;
          }}
          .tacticalSub {{
            font-size: 14px;
            color: #ffffff;
            margin-top: 6px;
            opacity: 0.92;
          }}
        </style>
        <div class="tacticalBanner">
          <div class="tacticalTitle">🚨 CRITICAL SECURITY THREAT: IRGC 8PM DEADLINE</div>
          <div class="tacticalTimer">{t_str}  •  16:30 GMT</div>
          <div class="tacticalSub"><b>Intelligence Briefing:</b> The IRGC has designated 18 U.S. and Gulf technology firms as military targets. This shift from military to commercial infrastructure targets (Data Centers/AI Hubs) has triggered an immediate Force Majeure across the region.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_float(x) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _yf_intraday_usable(df: pd.DataFrame | None, tickers: list[str]) -> bool:
    if df is None or df.empty:
        return False
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex) and (t, "Close") in df.columns:
                if df[(t, "Close")].dropna().size > 0:
                    return True
            elif not isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns:
                if df["Close"].dropna().size > 0:
                    return True
        except Exception:
            continue
    return False


def _yf_download_with_retry(
    *,
    tickers: list[str],
    period: str,
    interval: str,
    max_attempts: int = 4,
    delay_s: float = 0.65,
) -> pd.DataFrame:
    last: pd.DataFrame = pd.DataFrame()
    for attempt in range(max_attempts):
        try:
            last = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            last = pd.DataFrame()
        if last is not None and not last.empty:
            if period == "1d" and interval == "1m":
                if _yf_intraday_usable(last, tickers):
                    return last
            else:
                return last
        time.sleep(delay_s * (attempt + 1))
    return last if last is not None else pd.DataFrame()


def fetch_market_watch() -> list[dict]:
    tickers = ["AAPL", "GOOGL", "MSFT", "META", "NVDA", "TSLA", "INTC", "IBM", "BA", "DELL", "HPE", "CSCO", "ORCL", "JPM", "GE", "AMZN"]
    out: list[dict] = []

    intraday = _yf_download_with_retry(tickers=tickers, period="1d", interval="1m")
    daily2 = _yf_download_with_retry(tickers=tickers, period="2d", interval="1d")

    for t in tickers:
        last = None
        prev_close = None
        pct = None

        try:
            if isinstance(intraday.columns, pd.MultiIndex):
                s = intraday[(t, "Close")].dropna()
            else:
                s = intraday["Close"].dropna()
            if len(s) > 0:
                last = _safe_float(s.iloc[-1])
        except Exception:
            last = None

        try:
            if isinstance(daily2.columns, pd.MultiIndex):
                d = daily2[(t, "Close")].dropna()
            else:
                d = daily2["Close"].dropna()
            if len(d) >= 2:
                prev_close = _safe_float(d.iloc[-2])
        except Exception:
            prev_close = None

        if (prev_close is not None) and (last is not None) and prev_close != 0:
            pct = (last - prev_close) / prev_close * 100.0

        out.append(
            {
                "Ticker": t,
                "Price": round(last, 2) if last is not None else None,
                "% Change": round(pct, 2) if pct is not None else None,
            }
        )
    return out


@st.cache_data(ttl=120, show_spinner=False)
def _cached_market_watch() -> list[dict]:
    try:
        return fetch_market_watch()
    except Exception:
        return []


# (label, Yahoo ticker, unit blurb for display)
_COMMODITY_DEFS: list[tuple[str, str, str]] = [
    ("Gold", "GC=F", "$/oz"),
    ("Silver", "SI=F", "$/oz"),
    ("Brent", "BZ=F", "$/bbl"),
    ("Aluminum", "ALI=F", "$/MT"),
    ("Fertilizer", "NTR", "$/sh NTR"),
    ("Gasoline", "RB=F", "$/gal"),
]


def fetch_commodity_watch() -> list[dict]:
    """Gold/silver/Brent/aluminum/Nutrien (fertilizer proxy)/RBOB via yfinance."""
    tickers = [t for _, t, _ in _COMMODITY_DEFS]
    out: list[dict] = []

    intraday = _yf_download_with_retry(tickers=tickers, period="1d", interval="1m")
    daily2 = _yf_download_with_retry(tickers=tickers, period="2d", interval="1d")

    for label, t, unit in _COMMODITY_DEFS:
        last = None
        prev_close = None
        pct = None

        try:
            if isinstance(intraday.columns, pd.MultiIndex):
                s = intraday[(t, "Close")].dropna()
            else:
                s = intraday["Close"].dropna()
            if len(s) > 0:
                last = _safe_float(s.iloc[-1])
        except Exception:
            last = None

        try:
            if isinstance(daily2.columns, pd.MultiIndex):
                d = daily2[(t, "Close")].dropna()
            else:
                d = daily2["Close"].dropna()
            if len(d) >= 2:
                prev_close = _safe_float(d.iloc[-2])
        except Exception:
            prev_close = None

        if (prev_close is not None) and (last is not None) and prev_close != 0:
            pct = (last - prev_close) / prev_close * 100.0

        out.append(
            {
                "Label": label,
                "Ticker": t,
                "Unit": unit,
                "Price": last,
                "Prev close": prev_close,
                "% Change": pct,
            }
        )
    return out


@st.cache_data(ttl=120, show_spinner=False)
def _cached_commodity_watch() -> list[dict]:
    try:
        return fetch_commodity_watch()
    except Exception:
        return []


def _format_commodity_price(label: str, price: float | None) -> str:
    if price is None:
        return "—"
    if label == "Gold":
        return f"${price:,.2f}"
    if label == "Silver":
        return f"${price:,.3f}"
    if label == "Brent":
        return f"${price:,.2f}"
    if label == "Aluminum":
        return f"${price:,.0f}"
    if label == "Fertilizer":
        return f"${price:,.2f}"
    if label == "Gasoline":
        return f"${price:,.3f}"
    return f"${price:,.2f}"


def render_commodity_tracker(rows: list[dict]) -> None:
    """Compact 2×3 cards for the left column under shipping."""
    st.markdown("**Commodities & energy**")
    cols = st.columns(2, gap="small")
    for i, r in enumerate(rows):
        c = cols[i % 2]
        lab = str(r.get("Label", "—"))
        unit = str(r.get("Unit", ""))
        price = r.get("Price")
        ch = r.get("% Change")
        ch_s = "—" if ch is None else f"{ch:+.2f}%"
        p_s = _format_commodity_price(lab, float(price) if price is not None else None)
        is_down = (ch is not None) and (ch <= -2.0)
        label_color = "#FFFFFF"
        num_color = "#FFFF00"
        ch_color = "#ff3b3b" if is_down else num_color
        bg = "rgba(255, 0, 0, 0.10)" if is_down else "rgba(255, 255, 255, 0.06)"
        sub = html.escape(unit)
        c.markdown(
            f"""
            <div style="border:2px solid #FFFFFF; background:{bg}; border-radius:10px; padding:10px 10px; margin-bottom:8px;">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
                <div style="font-weight:900; color:{label_color}; letter-spacing:0.06em; font-size:0.95rem;">{html.escape(lab)}</div>
                <div style="font-weight:900; color:{ch_color}; font-size:0.95rem;">{ch_s}</div>
              </div>
              <div style="font-weight:900; color:{num_color}; font-size:1.2rem; margin-top:6px;">{p_s}</div>
              <div style="font-size:0.78rem; color:#cccccc; margin-top:4px; opacity:0.95;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("Yahoo Finance: futures where listed; fertilizer as Nutrien (NTR) equity proxy.")


def render_market_grid(rows: list[dict]) -> None:
    """Collapsible block: same pattern as the old target-list expander under the countdown."""
    with st.expander("Target list (18 firms)", expanded=False):
        cols = st.columns(4, gap="small")
        for i, r in enumerate(rows[:16]):
            c = cols[i % 4]
            t = r.get("Ticker", "—")
            p = r.get("Price")
            ch = r.get("% Change")
            ch_s = "—" if ch is None else f"{ch:+.2f}%"
            p_s = "—" if p is None else f"${p:,.2f}"
            is_down = (ch is not None) and (ch <= -2.0)
            label_color = "#FFFFFF"
            num_color = "#FFFF00"
            ch_color = "#ff3b3b" if is_down else num_color
            bg = "rgba(255, 0, 0, 0.10)" if is_down else "rgba(255, 255, 255, 0.06)"
            c.markdown(
                f"""
                <div style="border:2px solid #FFFFFF; background:{bg}; border-radius:10px; padding:10px 10px; margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:900; color:{label_color}; letter-spacing:0.10em; font-size:1rem;">{t}</div>
                    <div style="font-weight:900; color:{ch_color}; font-size:1.05rem;">{ch_s}</div>
                  </div>
                  <div style="font-weight:900; color:{num_color}; font-size:1.35rem; margin-top:6px;">{p_s}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(
            "Apple, Google, Microsoft, Meta, Nvidia, Tesla, Intel, IBM, Boeing, Dell, HP, Cisco, Oracle, "
            "JPMorgan, General Electric, Amazon, Anthropic, OpenAI. "
            "Prices: 16 tickers via Yahoo Finance (subset of the 18 named targets)."
        )


def render_intel_cards(intel_feed: list[dict]) -> None:
    for item in intel_feed[:12]:
        source = (item.get("source") or "—").strip()
        headline = (item.get("headline") or "—").strip()
        try:
            score = int(str(item.get("risk_score") or "0").strip())
        except Exception:
            score = 0
        risk = risk_level_from_score(score)

        is_irgc = "IRGC" in source.upper()
        border = "#ff3b3b" if is_irgc else "#2b2b2b"
        title_color = "#ff3b3b" if is_irgc else "#e6e6e6"

        st.markdown(
            f"""
            <div style="
              border: 1px solid {border};
              background: rgba(255, 255, 255, 0.03);
              padding: 12px 14px;
              border-radius: 10px;
              margin-bottom: 10px;
            ">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
                <div style="font-size: 12px; letter-spacing: 0.12em; color: #9aa0a6; font-weight: 700;">
                  {source}
                </div>
                <div style="
                  font-size: 12px;
                  color: {title_color};
                  font-weight: 900;
                  border: 1px solid {title_color};
                  border-radius: 999px;
                  padding: 2px 10px;
                ">
                  {risk} • {score}
                </div>
              </div>
              <div style="font-size: 16px; color: {title_color}; font-weight: 800; margin-top: 8px;">
                {headline}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Iran War Intelligence Dashboard",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if st.sidebar.button("🗑️ Clear Cache & Force Update"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    # Countdown updates: avoid 1s reruns on Cloud (duplicates yfinance/RSS work). 5s is enough for T- display.
    refresh_ms = 5_000 if datetime.now(UTC) < DEADLINE_UTC else 600_000
    st_autorefresh(interval=refresh_ms, key="auto_refresh_dynamic")

    now = datetime.now(UTC)

    tactical_osint_rows, hormuz_kinetic_flash = _cached_liveuamap_bundle()
    tehran_official_rows, tehran_feed_caption = _cached_tehran_narrative()

    st.markdown(
        """
        <style>
          /*
            High contrast for tablets: drive Streamlit's own tokens so widgets use
            white text on black (custom CSS was fighting var(--st-text-color)).
            Do not hide stToolbar — it can collapse the main flex layout on some builds.
          */
          :root {
            --st-text-color: #ffffff !important;
            --st-background-color: #000000 !important;
            --st-secondary-background-color: #141414 !important;
          }
          .stApp {
            background-color: #000000 !important;
            color: #ffffff !important;
          }
          [data-testid="stAppViewContainer"],
          [data-testid="stMain"] {
            background-color: #000000 !important;
            color: #ffffff !important;
          }
          .stApp [data-testid="stMarkdownContainer"],
          .stApp [data-testid="stVerticalBlock"] {
            color: #ffffff !important;
          }
          .stApp p, .stApp li, .stApp label, .stApp .stMarkdown {
            color: #ffffff !important;
          }
          .stApp p, .stApp .stMarkdown p, .stApp [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem !important;
            line-height: 1.55 !important;
          }
          .stApp h1 { font-size: 3.3rem !important; }
          .stApp h2 { font-size: 2.64rem !important; }
          .stApp h3 { font-size: 2.04rem !important; }
          .stApp .stMarkdown h1 { font-size: 3.3rem !important; }
          .stApp .stMarkdown h2 { font-size: 2.64rem !important; }
          .stApp .stMarkdown h3 { font-size: 2.04rem !important; }

          [data-testid="stMetricValue"] { color: #ffff00 !important; font-weight: 800 !important; }
          [data-testid="stMetricLabel"] { color: #ffffff !important; font-size: 1.05rem !important; }

          section[data-testid="stSidebar"] [data-testid="stDataFrame"] td {
            color: #ffff00 !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
          }
          section[data-testid="stSidebar"] [data-testid="stDataFrame"] th {
            color: #ffffff !important;
            font-size: 1.05rem !important;
          }

          .discerner-logic-box {
            border: 2px solid #ffffff;
            border-radius: 10px;
            padding: 14px 16px;
            margin: 0 0 12px 0;
            background: #0a0a0a;
          }
          .discerner-logic-box p {
            font-size: 1.2rem !important;
            color: #ffffff !important;
            margin: 0 0 8px 0;
          }
          .discerner-logic-box .discerner-rationale {
            font-size: 1.05rem !important;
            opacity: 1;
            color: #ffffff !important;
          }

          [data-testid="stHeader"] { background-color: #000000 !important; }
          [data-testid="stDataFrame"] td,
          [data-testid="stDataFrame"] th {
            white-space: normal !important;
            word-break: break-word !important;
            vertical-align: top !important;
          }
          .region-trade-heading {
            font-size: 0.9rem;
            font-weight: 900;
            letter-spacing: 0.18em;
            color: #fffacd !important;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.35);
          }
          .eu-metric-slot .eu-metric-label {
            font-size: 0.82rem;
            color: #c8c8c8 !important;
            letter-spacing: 0.02em;
            margin-bottom: 2px;
          }
          .eu-metric-slot .eu-final-sales-value {
            color: #ffff00 !important;
            font-weight: 900;
            font-size: 1.65rem;
            line-height: 1.15;
          }
          @keyframes euSupplyChainBlink {
            0%, 100% { color: #ff6464; text-shadow: 0 0 0 transparent; }
            50% { color: #ff0000; text-shadow: 0 0 16px rgba(255, 0, 0, 0.9); }
          }
          .eu-metric-slot .eu-supply-chain-value {
            animation: euSupplyChainBlink 1.05s ease-in-out infinite;
            font-weight: 900;
            font-size: 1.65rem;
            line-height: 1.15;
          }
          .eu-metric-slot .eu-metric-block { margin-bottom: 14px; }
          .eu-metric-slot .eu-metric-block:last-child { margin-bottom: 0; }
          .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if hormuz_kinetic_flash:
        st.markdown(
            """
            <style>
              [data-testid="stHeader"] {
                background-color: #7a0000 !important;
                animation: hormuzHeaderFlash 1.1s ease-in-out infinite !important;
              }
              @keyframes hormuzHeaderFlash {
                0%, 100% {
                  background-color: #3d0000 !important;
                  box-shadow: 0 0 0 rgba(255, 0, 0, 0);
                }
                50% {
                  background-color: #ff0000 !important;
                  box-shadow: 0 0 28px rgba(255, 60, 60, 0.95);
                }
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.subheader("Market Watch")

        mw = _cached_market_watch()

        if mw:
            mw_df = pd.DataFrame(mw)

            def _style_row(row):
                ch = row.get("% Change")
                if pd.notna(ch) and float(ch) <= -2.0:
                    return ["color: #ff3b3b; font-weight: 800; font-size: 1.1rem"] * len(row)
                return ["color: #FFFF00; font-weight: 700; font-size: 1.1rem"] * len(row)

            st.dataframe(
                mw_df.style.apply(_style_row, axis=1),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f", width="small"),
                    "% Change": st.column_config.NumberColumn("% Chg", format="%.2f%%", width="small"),
                },
            )
        else:
            st.caption("Market watch unavailable (yfinance fetch failed).")

        st.markdown("**PRIVATE - HIGH EXPOSURE**")
        st.write("OpenAI — $852B Val")
        st.write("Anthropic — $380B Val")

        st.subheader("Strait Monitor")
        if st.button("🔄 Refresh Strait Monitor", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        hs = _cached_hormuz_stats()
        st.write(f"**As-of (UTC):** {hs.asof_date_utc.strftime('%Y-%m-%d') if hs.asof_date_utc else '—'}")
        st.write(f"**Daily Transits:** {hs.daily_transits_total if hs.daily_transits_total is not None else '—'}")
        st.write(f"**Wait-List (Fujairah proxy):** {hs.wait_list_tankers_fujairah_proxy if hs.wait_list_tankers_fujairah_proxy is not None else '—'}")
        if hs.trade_value_drop_pct:
            st.write("**Trade Value Drop (%):**")
            for k in ["EU", "China", "US"]:
                if k in hs.trade_value_drop_pct:
                    st.write(f"- **{k}**: {hs.trade_value_drop_pct[k]}%")
        if hs.blockade_detected:
            st.error("BLOCKADE DETECTED (Daily transits < 15)")
        for n in hs.notes:
            st.caption(n)

    render_tactical_alert_banner(now)

    _top_pad, _top_refresh = st.columns([6, 1], gap="small")
    with _top_refresh:
        if st.button(
            "🔄 Manual Refresh",
            use_container_width=True,
            key="manual_refresh_top_right",
            help="Clear cached RSS, news, and market data and rerun",
        ):
            st.cache_data.clear()
            st.rerun()

    # Big-number metrics (same cache as sidebar — one PortWatch/IMF fetch per TTL, not two)
    hs_main = _cached_hormuz_stats()
    m1, m2, col_eu, col_cn, col_us = st.columns([1, 1, 1.35, 1.35, 1.35], gap="small")
    m1.metric("Hormuz Daily Transits", hs_main.daily_transits_total if hs_main.daily_transits_total is not None else "—")
    m2.metric("Wait-List (Fujairah)", hs_main.wait_list_tankers_fujairah_proxy if hs_main.wait_list_tankers_fujairah_proxy is not None else "—")

    _eu_final_tooltip = (
        "Negative % = trade / demand deficit vs baseline. Reflects 30-day inventory buffers and "
        "front-loaded winter stock; consumers have not felt the full impact yet."
    )
    _eu_supply_tooltip = (
        "Negative % = inbound supply deficit vs baseline. Reflects Suez transit collapse and "
        "~20-day Cape of Good Hope diversions. Primary impact: Trieste, Piraeus, and Mediterranean hubs."
    )
    cn_r, cn_i = _trade_drop_split_pcts(hs_main.trade_value_drop_pct.get("China"))
    us_r, us_i = _trade_drop_split_pcts(hs_main.trade_value_drop_pct.get("US"))
    _cn_final_tooltip = (
        "Negative % = trade deficit vs baseline. Retail-channel buffers and front-loaded stock; "
        "consumer impact typically lags maritime disruption."
    )
    _cn_supply_tooltip = (
        "Negative % = inbound supply deficit. Port delays, Asia–Europe routing stress, and diversion-driven "
        "lag for Chinese industrial and retail supply chains."
    )
    _us_final_tooltip = (
        "Negative % = trade deficit vs baseline. U.S. retail and wholesale buffers; "
        "on-shelf effects often lag ocean-freight shocks."
    )
    _us_supply_tooltip = (
        "Negative % = inbound supply deficit. Container backlogs, diversions, and gateway congestion "
        "at major U.S. import hubs."
    )

    with col_eu:
        st.markdown(
            _region_trade_column_markdown("EU", "-1.8%", "-24.5%", _eu_final_tooltip, _eu_supply_tooltip),
            unsafe_allow_html=True,
        )
    with col_cn:
        st.markdown(
            _region_trade_column_markdown("CHINA", cn_r, cn_i, _cn_final_tooltip, _cn_supply_tooltip),
            unsafe_allow_html=True,
        )
    with col_us:
        st.markdown(
            _region_trade_column_markdown("US", us_r, us_i, _us_final_tooltip, _us_supply_tooltip),
            unsafe_allow_html=True,
        )
    if hs_main.blockade_detected:
        st.error("BLOCKADE DETECTED: Strait of Hormuz daily transits below 15.")

    col_left, col_right = st.columns([1.05, 1.35], gap="large")

    with col_left:
        st.subheader("Shipping Impact")
        df = shipping_impact_table(now)

        # Override display with explicit scenario values requested for Apr 1, 2026 UI.
        # (These are presentation values; keep live PortWatch fetch for mode flags.)
        df.loc[df["Port"] == "Rotterdam", "Transit Delay (Days)"] = 45.0
        df.loc[df["Port"] == "Trieste", "Transit Delay (Days)"] = 38.0
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.error("Rotterdam alert: Suez Transit Effectively Zero.")
        st.caption("Delays are synthetic. During the deadline window (+24h), Cape routing is forced for EU ports.")

        with st.container(border=True):
            comm = _cached_commodity_watch()
            if comm:
                render_commodity_tracker(comm)
            else:
                st.caption("Commodity prices unavailable (yfinance fetch failed).")

    with col_right:
        st.subheader("Live Intel Feed")

        @st.cache_data(ttl=90, show_spinner=False)
        def _cached_live_entries() -> list[dict]:
            # logic.py adds Google `when:1d` for recency except lines that already include `when:`.
            queries = [
                "(Iran OR IRGC) AND (deadline OR target OR tech OR strike OR missile OR drone)",
                "(Hormuz OR 'Red Sea' OR Strait) AND (shipping OR tanker OR Iran OR IRGC OR strike)",
                "(Israel OR 'United States' OR Pentagon) AND (Iran OR IRGC OR war OR military OR Gulf)",
                "Iran OR IRGC OR Hormuz OR 'Strait of Hormuz' when:6h",
            ]
            return fetch_live_google_news_multiquery(
                queries,
                per_query_limit=45,
                min_results=5,
                request_headers=NEWS_RSS_REQUEST_HEADERS,
            )

        entries = _cached_live_entries()
        discerner = evaluate_strait_status_from_live_entries(entries)
        discerner = apply_kinetic_hormuz_maximum_override(discerner, hormuz_kinetic=hormuz_kinetic_flash)

        if hormuz_kinetic_flash:
            st.error("Kinetic activity flagged within the Hormuz 50km rule — Discerner risk elevated to MAXIMUM.")

        rationale_esc = html.escape(discerner.rationale)
        status_esc = html.escape(discerner.strait_status)
        risk_esc = html.escape(discerner.war_risk_level)
        st.markdown(
            f"""
            <div class="discerner-logic-box">
              <p><strong>The Discerner (Live):</strong> Strait status <strong>{status_esc}</strong>
              • War risk <strong>{risk_esc}</strong></p>
              <p class="discerner-rationale">{rationale_esc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if discerner.war_risk_level.upper() in ("CRITICAL", "MAXIMUM"):
            st.warning("Market volatility expected to spike at 16:30 GMT.")

        if mw:
            render_market_grid(mw)

        link_cols = {
            "No.": st.column_config.NumberColumn("No.", width="small"),
            "Date/Time (UTC)": st.column_config.TextColumn("Time (UTC)", width="small"),
            "Title": st.column_config.TextColumn("Title", width="medium"),
            "Source": st.column_config.TextColumn("Source", width="small"),
            "Link": st.column_config.LinkColumn("Article", display_text="View Source", help="Open original article"),
        }
        link_cols_no_time = {
            k: v for k, v in link_cols.items() if k != "Date/Time (UTC)"
        }

        pane_l, pane_r = st.columns([1, 1], gap="medium")
        with pane_l:
            with st.container(border=True):
                st.markdown("### OFFICIAL TEHRAN NARRATIVE")
                if not tehran_official_rows:
                    st.caption(
                        tehran_feed_caption
                        or "No articles from NewsData.io or Press TV RSS (check secrets, quota, or network)."
                    )
                else:
                    if tehran_feed_caption:
                        st.caption(tehran_feed_caption)
                    df_t = _prepare_intel_dataframe(
                        tehran_official_rows, include_time_column=False
                    )
                    if df_t is not None:
                        # Fresh 4-column frame so deployed UI cannot resurrect a hidden Time
                        # column from widget state; column_order pins visible cols.
                        df_show = pd.DataFrame(
                            {
                                "No.": df_t["No."].astype(int),
                                "Source": df_t["Source"].astype(str),
                                "Title": df_t["Title"].astype(str),
                                "Link": df_t["Link"].astype(str),
                            }
                        )
                        st.dataframe(
                            df_show.style.apply(_intel_highlight_row, axis=1),
                            use_container_width=True,
                            hide_index=True,
                            column_config=link_cols_no_time,
                            column_order=["No.", "Source", "Title", "Link"],
                            key="official_tehran_narrative",
                        )
        with pane_r:
            with st.container(border=True):
                st.markdown("### KINETIC EVENTS & INTERCEPTIONS")
                if not tactical_osint_rows:
                    st.caption(
                        "No kinetic headlines yet from LiveUAMap scrapers or Google fallback. "
                        "If this persists on Cloud, the host may be blocking datacenter IPs — try again after “Clear cache”."
                    )
                else:
                    df_k = _prepare_intel_dataframe(tactical_osint_rows)
                    if df_k is not None:
                        st.dataframe(
                            df_k[["No.", "Date/Time (UTC)", "Source", "Title", "Link"]].style.apply(
                                _intel_highlight_row, axis=1
                            ),
                            use_container_width=True,
                            hide_index=True,
                            column_config=link_cols,
                            key="kinetic_osint_intel",
                        )

        with st.container(border=True):
            st.markdown("### Aggregated open-source news (Google RSS + BBC)")
            if entries:
                df_news = _prepare_intel_dataframe(entries)
                if df_news is not None:
                    st.dataframe(
                        df_news[["No.", "Date/Time (UTC)", "Source", "Title", "Link"]].style.apply(
                            _intel_highlight_row, axis=1
                        ),
                        use_container_width=True,
                        hide_index=True,
                        column_config=link_cols,
                        key="aggregated_news_intel",
                    )
                else:
                    st.warning("Waiting for news sync...")
            else:
                st.warning("No RSS entries were fetched.")


if __name__ == "__main__":
    main()

