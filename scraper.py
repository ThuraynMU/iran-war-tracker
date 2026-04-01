from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from random import Random
from typing import Iterable

import feedparser
import requests
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass(frozen=True)
class SearchHit:
    timestamp_utc: datetime
    source: str
    keyword: str
    title: str
    snippet: str
    url: str


class MockMfaSearch:
    """
    Deterministic mock of searching mfa.ir for specific keywords.
    This does not fetch the network; it generates plausible "hits".
    """

    def __init__(self, seed: int = 7) -> None:
        self._rng = Random(seed)

    def search(self, keywords: Iterable[str], now_utc: datetime | None = None, limit: int = 12) -> list[SearchHit]:
        now_utc = now_utc or datetime.now(UTC)
        kws = [k.strip() for k in keywords if k and k.strip()]
        if not kws:
            return []

        templates = [
            (
                "Foreign Ministry Spokesperson briefing: {kw}",
                "Spokesperson addressed developments related to {kw}, emphasizing de-escalation and maritime security.",
            ),
            (
                "Diplomatic consultations on regional transit and {kw}",
                "Officials discussed shipping lanes, humanitarian corridors, and risk mitigation measures linked to {kw}.",
            ),
            (
                "Statement on navigation safety and {kw}",
                "The statement called for respect of international law and the protection of commercial vessels in areas associated with {kw}.",
            ),
            (
                "Readout: calls with partners regarding {kw}",
                "The readout referenced recent incidents, urging restraint and coordinated maritime monitoring around {kw}.",
            ),
        ]

        def mk_url(i: int) -> str:
            # Keep URLs consistent and clearly mocked.
            return f"https://mfa.ir/en/news/{now_utc:%Y%m%d}/{i:04d}"

        hits: list[SearchHit] = []
        for i in range(limit):
            kw = kws[i % len(kws)]
            title_t, snippet_t = templates[self._rng.randrange(0, len(templates))]
            minutes_ago = 9 + i * self._rng.choice([7, 11, 13])
            ts = now_utc - timedelta(minutes=minutes_ago)
            hits.append(
                SearchHit(
                    timestamp_utc=ts,
                    source="mfa.ir (mock)",
                    keyword=kw,
                    title=title_t.format(kw=kw),
                    snippet=snippet_t.format(kw=kw),
                    url=mk_url(i + 1),
                )
            )

        hits.sort(key=lambda h: h.timestamp_utc, reverse=True)
        return hits


DEFAULT_NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q="
    "Iran%20Military"
    "&hl=en-US&gl=US&ceid=US:en"
)

RISK_KEYWORDS = ("IRGC", "Hormuz", "Missile", "Blockade", "Tanker", "Red Sea")

SIMULATED_TODAY_EVENTS: list[dict] = [
    {
        "source": "IRGC (State Media)",
        "headline": "IRGC: Deadline set for 8PM Tehran time to destroy units of 18 tech companies including Microsoft, Apple, and Google.",
        "risk_score": "90",
    },
    {
        "source": "Al Jazeera",
        "headline": "Missile launched from Iran hits oil tanker off Qatar coast; Strait of Hormuz effectively shut.",
        "risk_score": "95",
    },
    {
        "source": "Reuters",
        "headline": "Iran FM Araghchi confirms direct contact with US Envoy Witkoff but denies negotiations.",
        "risk_score": "55",
    },
]


def _extract_source_from_google_news_title(title: str) -> tuple[str, str]:
    # Common format: "Headline text - Publisher"
    if " - " in title:
        head, publisher = title.rsplit(" - ", 1)
        head = head.strip()
        publisher = publisher.strip()
        if head and publisher:
            return publisher, head
    return "Google News", title.strip()


def _headline_risk_score(headline: str, *, keywords: Iterable[str] = RISK_KEYWORDS) -> int:
    h = (headline or "").lower()
    hits = sum(1 for k in keywords if k.lower() in h)
    if hits >= 3:
        return 90
    if hits == 2:
        return 75
    if hits == 1:
        return 55
    return 25


def get_live_intelligence_feed(
    *,
    feed_url: str = DEFAULT_NEWS_RSS_URL,
    limit: int = 8,
    include_simulated_today: bool = True,
) -> list[dict]:
    """
    Returns a list of dict entries:
      - source
      - headline
      - risk_score (string)
    """
    out: list[dict] = []

    # NOTE: These are simulated test items provided by the user for UI/testing.
    if include_simulated_today:
        out.extend(SIMULATED_TODAY_EVENTS)

    try:
        headlines = fetch_rss_headlines(feed_url, limit=limit)
    except Exception:
        return out

    for t in headlines:
        src, headline = _extract_source_from_google_news_title(t)
        out.append(
            {
                "source": src,
                "headline": headline,
                "risk_score": str(_headline_risk_score(headline)),
            }
        )

    return out


def display_news(intel_feed: list[dict], *, width: int = 120) -> Panel:
    """
    Render a 'News Ticker' style strip suitable for placing above the shipping tables.
    """
    items: list[str] = []
    for entry in intel_feed[:10]:
        src = str(entry.get("source", "")).strip() or "Source"
        head = str(entry.get("headline", "")).strip() or "Headline"
        score = str(entry.get("risk_score", "")).strip() or "?"
        items.append(f"[{score}] {src}: {head}")

    tape = "  •  ".join(items) if items else "No intel items available."
    if width < 20:
        width = 20
    window = tape if len(tape) <= width else (tape[: max(0, width - 1)] + "…")

    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_row(Text("NEWS TICKER", style="bold bright_white"))
    table.add_row(Text(window, style="bright_white"))

    return Panel(
        table,
        title=None,
        border_style="bright_black",
        box=box.SQUARE,
        padding=(0, 1),
    )


def fetch_rss_headlines(feed_url: str = DEFAULT_NEWS_RSS_URL, *, limit: int = 5, timeout_s: float = 8.0) -> list[str]:
    """
    Fetch headlines from a live RSS feed.
    Uses requests for a predictable timeout and feedparser for parsing.
    """
    headers = {
        "User-Agent": "iran-war-tracker/0.1 (+local; rss)",
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
    }
    r = requests.get(feed_url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    parsed = feedparser.parse(r.content)
    titles: list[str] = []
    for entry in getattr(parsed, "entries", [])[: max(0, limit)]:
        t = (getattr(entry, "title", "") or "").strip()
        if t:
            titles.append(t)
    return titles


def get_latest_risk_level(
    *,
    feed_url: str = DEFAULT_NEWS_RSS_URL,
    keywords: Iterable[str] = RISK_KEYWORDS,
    headlines_limit: int = 5,
) -> tuple[str, list[str], list[str]]:
    """
    Scan the latest headlines and derive a war_risk_level.

    Logic:
    - If 2+ keywords are found across the last 5 headlines => HIGH
    - If 1 keyword is found => MEDIUM
    - Otherwise => LOW
    """
    kws = [k.strip() for k in keywords if k and k.strip()]
    try:
        headlines = fetch_rss_headlines(feed_url, limit=headlines_limit)
    except Exception:
        return "LOW", [], []

    haystack = " | ".join(headlines).lower()
    matched = sorted({k for k in kws if k.lower() in haystack}, key=str.upper)
    if len(matched) >= 2:
        return "HIGH", matched, headlines
    if len(matched) == 1:
        return "MEDIUM", matched, headlines
    return "LOW", matched, headlines
