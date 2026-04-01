from __future__ import annotations

import html
from datetime import UTC, datetime, timedelta

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import time

from logic import (
    RELIABILITY_BUFFER_DAYS,
    evaluate_strait_status_from_live_entries,
    fetch_live_google_news_multiquery,
    fetch_hormuz_stats,
    fetch_realtime_shipping_stats,
)

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
    pw = fetch_realtime_shipping_stats()
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

    with st.expander("Target list (18 firms)", expanded=False):
        st.write(
            "Apple, Google, Microsoft, Meta, Nvidia, Tesla, Intel, IBM, Boeing, Dell, HP, Cisco, Oracle, JPMorgan, General Electric, Amazon, Anthropic, OpenAI."
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


def render_market_grid(rows: list[dict]) -> None:
    # Compact grid above the news feed.
    st.markdown("**Targeted Firm Market Monitor**")
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

    # Active countdown needs frequent reruns; caches prevent heavy re-fetching.
    refresh_ms = 1_000 if datetime.now(UTC) < DEADLINE_UTC else 600_000
    st_autorefresh(interval=refresh_ms, key="auto_refresh_dynamic")

    now = datetime.now(UTC)

    st.markdown(
        """
        <style>
          /* iPad / accessibility: maximum contrast */
          .stApp {
            background: #000000 !important;
            color: #FFFFFF !important;
          }
          /* Scoped surfaces only — never `*` (breaks Streamlit layers/widgets). */
          .stApp [data-testid="stAppViewContainer"] {
            background-color: transparent !important;
            color: #FFFFFF !important;
          }
          .stApp [data-testid="stMarkdownContainer"],
          .stApp [data-testid="stVerticalBlock"] {
            color: #FFFFFF !important;
          }
          .stApp p, .stApp li, .stApp label, .stApp .stMarkdown {
            color: #FFFFFF !important;
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

          [data-testid="stMetricValue"] { color: #FFFF00 !important; font-weight: 800 !important; }
          [data-testid="stMetricLabel"] { color: #FFFFFF !important; font-size: 1.05rem !important; }

          section[data-testid="stSidebar"] [data-testid="stDataFrame"] td {
            color: #FFFF00 !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
          }
          section[data-testid="stSidebar"] [data-testid="stDataFrame"] th {
            color: #FFFFFF !important;
            font-size: 1.05rem !important;
          }

          .discerner-logic-box {
            border: 2px solid #FFFFFF;
            border-radius: 10px;
            padding: 14px 16px;
            margin: 0 0 12px 0;
            background: #0a0a0a;
          }
          .discerner-logic-box p { font-size: 1.2rem !important; color: #FFFFFF !important; margin: 0 0 8px 0; }
          .discerner-logic-box .discerner-rationale { font-size: 1.05rem !important; opacity: 1; color: #FFFFFF !important; }

          [data-testid="stHeader"] { background: #000000 !important; }
          [data-testid="stToolbar"] { visibility: hidden; height: 0px; }
          .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Market Watch")

        try:
            mw = fetch_market_watch()
        except Exception:
            mw = []

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

        @st.cache_data(ttl=300, show_spinner=False)
        def _cached_hormuz_stats():
            return fetch_hormuz_stats()

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

    # Big-number metrics
    @st.cache_data(ttl=300, show_spinner=False)
    def _cached_hormuz_stats_main():
        return fetch_hormuz_stats()

    hs_main = _cached_hormuz_stats_main()
    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    m1.metric("Hormuz Daily Transits", hs_main.daily_transits_total if hs_main.daily_transits_total is not None else "—")
    m2.metric("Wait-List (Fujairah)", hs_main.wait_list_tankers_fujairah_proxy if hs_main.wait_list_tankers_fujairah_proxy is not None else "—")
    m3.metric("Trade Drop EU (%)", hs_main.trade_value_drop_pct.get("EU", "—"))
    m4.metric("Trade Drop China (%)", hs_main.trade_value_drop_pct.get("China", "—"))
    m5.metric("Trade Drop US (%)", hs_main.trade_value_drop_pct.get("US", "—"))
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

    with col_right:
        st.subheader("Live Intel Feed")

        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("🔄 Manual Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        @st.cache_data(ttl=600, show_spinner=False)
        def _cached_live_entries() -> list[dict]:
            queries = [
                "(Iran OR IRGC) AND (deadline OR target OR tech)",
                "(Hormuz OR 'Red Sea') AND (shipping OR tanker OR strike)",
                "(Israel OR 'United States') AND Iran AND war AND live",
            ]
            return fetch_live_google_news_multiquery(
                queries,
                per_query_limit=15,
                min_results=5,
                request_headers=NEWS_RSS_REQUEST_HEADERS,
            )

        entries = _cached_live_entries()
        discerner = evaluate_strait_status_from_live_entries(entries)
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
        if discerner.war_risk_level.upper() == "CRITICAL":
            st.warning("Market volatility expected to spike at 16:30 GMT.")

        # Compact grid above the news feed (scrolling marquee alternative).
        try:
            mw_main = fetch_market_watch()
        except Exception:
            mw_main = []
        if mw_main:
            render_market_grid(mw_main)

        if entries:
            df_news = pd.DataFrame(entries)
            # Safety Net: force ensure all columns exist even if scraper missed one
            if not df_news.empty:
                for col in ["Date/Time (UTC)", "Source", "Title", "Link"]:
                    if col not in df_news.columns:
                        df_news[col] = "N/A"

                # Convert to datetime for correct ordering, then format back for display.
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
                df_news["Date/Time (UTC)"] = df_news["_dt"].dt.strftime("%H:%M GMT").fillna("N/A")
                df_news = df_news.drop(columns=["_dt"])

                # 1-based row numbering at the top.
                df_news.insert(0, "No.", range(1, len(df_news) + 1))

                def _highlight_row(row):
                    title = str(row.get("Title", "")).lower()
                    if ("target" in title) or ("strike" in title):
                        return ["background-color: rgba(255, 215, 0, 0.22)"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df_news[["No.", "Date/Time (UTC)", "Source", "Title", "Link"]].style.apply(_highlight_row, axis=1),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "No.": st.column_config.NumberColumn("No.", width="small"),
                        "Date/Time (UTC)": st.column_config.TextColumn("Date/Time (UTC)", width="small"),
                        "Link": st.column_config.LinkColumn(
                            "Link",
                            display_text="open",
                            help="Open the original reports",
                        ),
                    },
                )
            else:
                st.warning("Waiting for news sync...")
        else:
            st.warning("No RSS entries were fetched.")


if __name__ == "__main__":
    main()

