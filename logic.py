from __future__ import annotations

from calendar import timegm
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
import html
import math
import re
from typing import Any, Sequence

import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus


PORTWATCH_DATASET_ID = "42132aa4e2fc4d41bdaf9a445f688931_0"
PORTWATCH_SEARCH_API_BASE = "https://portwatch.imf.org/api/search/v1"

PORTWATCH_FEATURESERVER_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Chokepoints_Data/FeatureServer/0"
)

# Xeneta 2026 schedule reliability: ~27% on-time; use a fixed buffer for systemic delay.
RELIABILITY_BUFFER_DAYS = 4.2


@dataclass(frozen=True)
class PortWatchTransitSnapshot:
    fetched_at_utc: datetime
    date_utc: datetime | None
    suez_transits_per_day: int | None
    cape_transits_per_day: int | None
    danger_zone_threshold: int
    cape_mode: bool
    reliability_buffer_days: float
    source: str


@dataclass(frozen=True)
class StraitStatus:
    evaluated_at_utc: datetime
    war_risk_level: str  # LOW/MEDIUM/HIGH/CRITICAL
    strait_status: str  # UNRESTRICTED_SAFE_PASSAGE / RESTRICTED_OR_UNCONFIRMED
    rationale: str
    confirmations: list[str]


@dataclass(frozen=True)
class UkraineOilInfraScrapeRow:
    """LiveUAMap Ukraine feed: sidebar card time/title/link plus optional map coordinates from the event page."""

    time: str
    title: str
    link: str
    lat: float | None
    lon: float | None


GOOGLE_NEWS_HORMUZ_RSS_URL = (
    "https://news.google.com/rss/search?q=Iran+war+Strait+of+Hormuz+shipping&hl=en-US&gl=US&ceid=US:en"
)

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"

# Faster-moving than Google’s index for the same stories; merged after Google queries.
BBC_MIDDLE_EAST_RSS_URL = "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"

# Browser-like defaults reduce blocks from Google RSS on cloud IPs.
DEFAULT_RSS_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}


def _google_news_rss_url(query: str, *, hl: str = "en-US", gl: str = "US", ceid: str = "US:en") -> str:
    q = quote_plus(query)
    return f"{GOOGLE_NEWS_RSS_BASE}?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


GROUND_TRUTH_APR_1_2026: list[dict[str, str]] = [
    {"Source": "KUNA", "Title": "Iranian drone strike on Kuwait Airport fuel tanks; massive fire reported.", "Link": ""},
    {"Source": "WAM", "Title": "UAE Air Defense intercepts drone over Umm Al Thuoob industrial zone.", "Link": ""},
    {
        "Source": "Sepah News",
        "Title": "Final 15-minute warning issued to 18 tech firms; units designated for kinetic destruction.",
        "Link": "",
    },
]


def _normalized_entry(*, dt: str, source: str, title: str, link: str) -> dict[str, str]:
    return {
        "Date/Time (UTC)": (dt or "").strip(),
        "Source": (source or "").strip(),
        "Title": (title or "").strip(),
        "Link": (link or "").strip(),
    }


_RE_PRESSTV_DETAIL = re.compile(r"presstv\.ir/Detail/(\d{4})/(\d{2})/(\d{2})/", re.IGNORECASE)


def _utc_datetime_from_presstv_detail_link(link: str) -> datetime | None:
    """Press TV RSS often omits item pubDate; article URLs embed /Detail/YYYY/MM/DD/."""
    m = _RE_PRESSTV_DETAIL.search(link or "")
    if not m:
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, 12, 0, 0, tzinfo=UTC)
    except ValueError:
        return None


def _parse_header_date_ts(label: str) -> float:
    """RFC 2822 / RSS pubDate → sortable unix timestamp (UTC)."""
    s = (label or "").strip()
    if not s:
        return float("-inf")
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except (TypeError, ValueError, OverflowError):
        return float("-inf")


def _published_ts_from_feed_entry(e: Any, link: str = "") -> float:
    t = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
    if t and getattr(t, "tm_year", 0) > 1990:
        try:
            return float(timegm(t))
        except (TypeError, ValueError, OverflowError):
            pass
    lbl = (getattr(e, "published", "") or getattr(e, "updated", "") or "").strip()
    ts = _parse_header_date_ts(lbl)
    if ts > float("-inf"):
        return ts
    dt_link = _utc_datetime_from_presstv_detail_link(link)
    if dt_link is not None:
        return dt_link.timestamp()
    return float("-inf")


def _row_matches_intel_focus(row: dict[str, str]) -> bool:
    blob = f"{row.get('Title', '')} {row.get('Source', '')}".lower()
    keys = (
        "iran",
        "irgc",
        "hormuz",
        "strait",
        "tehran",
        "israel",
        "missile",
        "drone",
        "strike",
        "military",
        "shipping",
        "tanker",
        "kuwait",
        "uae",
        "dubai",
        "pentagon",
        "u.s.",
        " us ",
        "navy",
        "deadline",
        "tech",
        "red sea",
        "oil",
        "houthis",
        "yemen",
        "gulf",
    )
    return any(k in blob for k in keys)


def _entry_title(it: dict[str, str]) -> str:
    return str(it.get("Title") or it.get("title") or "")


def _entry_source(it: dict[str, str]) -> str:
    return str(it.get("Source") or it.get("source") or "")


def fetch_live_rss_entries(
    feed_url: str = GOOGLE_NEWS_HORMUZ_RSS_URL,
    *,
    limit: int = 10,
    timeout_s: float = 12.0,
    request_headers: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """
    Fetch a live RSS feed and normalize entries as:
      { "Date/Time (UTC)": "...", "Source": "...", "Title": "...", "Link": "..." }

    For Google News RSS, `title` often looks like: "Headline text - Publisher".

    Fetches a larger pool from the wire, sorts by real publish time (struct_time),
    then returns the newest `limit` rows — Google often returns a mixed order, and
    taking entries[:limit] without sorting hid fresher items past position N.
    """
    headers = {
        **DEFAULT_RSS_REQUEST_HEADERS,
        **(request_headers or {}),
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    last_err: Exception | None = None
    for _ in range(2):
        try:
            r = requests.get(
                feed_url,
                headers=headers,
                timeout=timeout_s,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as err:
            last_err = err
    else:
        raise RuntimeError(f"RSS fetch failed for {feed_url}: {last_err!s}") from last_err

    parsed = feedparser.parse(r.content)
    entries = getattr(parsed, "entries", []) or []
    pool = entries[:150]
    scored: list[tuple[float, dict[str, str]]] = []
    for e in pool:
        raw_title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()

        # Prefer RSS/Atom text dates; many feeds (e.g. some Press TV items) omit the string
        # but feedparser still fills published_parsed / updated_parsed for sorting/UI.
        dt_label = (getattr(e, "published", "") or "").strip()
        if not dt_label:
            dt_label = (getattr(e, "updated", "") or "").strip()
        if not dt_label:
            t_struct = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
            if t_struct and getattr(t_struct, "tm_year", 0) > 1990:
                try:
                    dt_label = datetime.fromtimestamp(timegm(t_struct), tz=UTC).strftime(
                        "%a, %d %b %Y %H:%M:%S GMT"
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        if not dt_label:
            dt_utc = _utc_datetime_from_presstv_detail_link(link)
            if dt_utc is not None:
                dt_label = dt_utc.strftime("%a, %d %b %Y %H:%M:%S GMT")

        source = "Google News"
        title = raw_title
        if " - " in raw_title:
            head, publisher = raw_title.rsplit(" - ", 1)
            head = head.strip()
            publisher = publisher.strip()
            if head:
                title = head
            if publisher:
                source = publisher
        if title or source or link or dt_label:
            ts = _published_ts_from_feed_entry(e, link)
            scored.append((ts, _normalized_entry(dt=dt_label, source=source, title=title, link=link)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[: max(0, limit)]]


def fetch_live_google_news_multiquery(
    queries: list[str],
    *,
    per_query_limit: int = 10,
    timeout_s: float = 12.0,
    min_results: int = 5,
    request_headers: dict[str, str] | None = None,
    include_bbc_middle_east: bool = True,
) -> list[dict[str, str]]:
    """
    Fetch multiple Google News RSS queries and merge results.
    If fewer than `min_results` total results, inject Ground Truth (Apr 1, 2026).

    Appends ` when:1d` (Google News operator) when the query has no `when:` yet,
    skewing RSS toward recent items (Google’s feed order alone is unreliable).
    """
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    live_query_successes = 0

    for q in queries:
        q2 = q.strip()
        if " when:" not in q2.lower():
            q2 = f"{q2} when:1d"
        url = _google_news_rss_url(q2)
        try:
            items = fetch_live_rss_entries(
                url, limit=per_query_limit, timeout_s=timeout_s, request_headers=request_headers
            )
            if items:
                live_query_successes += 1
        except Exception:
            items = [
                _normalized_entry(
                    dt=datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT"),
                    source="Feed status",
                    title=f"News temporarily unavailable ({q.strip()[:60]})",
                    link="",
                )
            ]
        for it in items:
            key = (it.get("Source", ""), it.get("Title", ""), it.get("Link", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(it)

    if include_bbc_middle_east:
        try:
            bbc_items = fetch_live_rss_entries(
                BBC_MIDDLE_EAST_RSS_URL,
                limit=max(per_query_limit, 30),
                timeout_s=timeout_s,
                request_headers=request_headers,
            )
            for it in bbc_items:
                if not _row_matches_intel_focus(it):
                    continue
                key = (it.get("Source", ""), it.get("Title", ""), it.get("Link", ""))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(it)
        except Exception:
            pass

    # If every Google query failed, don't flood UI with placeholder lines.
    # Keep one explicit status row and let ground-truth fill the rest.
    if live_query_successes == 0:
        merged = [
            _normalized_entry(
                dt=datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT"),
                source="Feed status",
                title="All open-source news providers temporarily unavailable on current network path.",
                link="",
            )
        ]

    if len(merged) < min_results:
        now_label = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
        for gt in GROUND_TRUTH_APR_1_2026:
            merged.append(
                _normalized_entry(
                    dt=now_label,
                    source=gt.get("Source", ""),
                    title=gt.get("Title", ""),
                    link=gt.get("Link", ""),
                )
            )

    merged.sort(key=lambda x: _parse_header_date_ts(x.get("Date/Time (UTC)", "")), reverse=True)
    return merged


# --- NewsData.io (Iran / English) + LiveUAMap Middle East scrape -----------------

NEWSDATA_API_LATEST = "https://newsdata.io/api/1/latest"

# Approx. centre Strait of Hormuz for haversine checks when URLs embed coordinates.
HORMUZ_LAT = 26.75
HORMUZ_LON = 56.25

# Public /rss on liveuamap.com now redirects to the paid API; home-page HTML still lists incident cards.
LIVEUAMAP_SCRAPE_HOME_PAGES: tuple[str, ...] = (
    "https://mideast.liveuamap.com/",
    "https://iran.liveuamap.com/",
    "https://israelpalestine.liveuamap.com/",
    "https://yemen.liveuamap.com/",
    "https://syria.liveuamap.com/",
)

# Ukraine "oil" path is often a shell/404; `/en` still ships the live sidebar (`div.event`) HTML.
LIVEUAMAP_UKRAINE_OIL_TARGET_URL = "https://ukraine.liveuamap.com/en/event/oil"
LIVEUAMAP_UKRAINE_HOME_EN_URL = "https://ukraine.liveuamap.com/en"

_UKRAINE_OIL_INFRA_KEYWORDS: tuple[str, ...] = ("depot", "refinery", "terminal")

_RE_LU_EVENT_PAGE_LATLNG = re.compile(
    r"lat\s*=\s*([-0-9.]+)\s*;\s*lng\s*=\s*([-0-9.]+)\s*;",
    re.I,
)

# Attribute order varies; Cloudflare/mobile shells sometimes differ slightly.
_RE_LU_RECD = re.compile(
    r'class="recd_descr"\s+href="(https://[^"]+)"\s+title="([^"]*)"',
    re.I,
)
_RE_LU_RECD_ALT = re.compile(
    r'class="recd_descr"\s+title="([^"]*)"\s+href="(https://[^"]+)"',
    re.I,
)
_RE_COORD_QUERY = re.compile(
    r"[?&#](?:lat|latitude)=([-0-9.]+)(?:&[^#]*)?[?&#](?:lng|lon|longitude)=([-0-9.]+)",
    re.I,
)

KINETIC_TERMS: tuple[str, ...] = (
    "explosion",
    "explosions",
    "strike",
    "strikes",
    "airstrike",
    "missile",
    "rocket",
    "intercept",
    "air defense",
    "air-defence",
    "drone",
    "kinetic",
    "blast",
    "bombed",
    "shelling",
    "attack",
    "wounded",
    "killed",
    "airstrikes",
    "sortie",
)

HORMUZ_PROXIMITY_TERMS: tuple[str, ...] = (
    "hormuz",
    "strait of hormuz",
    "bandar abbas",
    "bandar-e abbas",
    "qeshm",
    "jask",
    "chabahar",
    "fujairah",
    "musandam",
    "khasab",
    "gulf of oman",
    "oman coast",
)

# Rotate UAs — some datacenter IPs get a shell page until a desktop UA is used.
_LU_FETCH_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
)


def _lu_recommend_link_title_pairs(html_text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in _RE_LU_RECD.finditer(html_text):
        pairs.append((html.unescape(m.group(1).strip()), html.unescape(m.group(2).strip())))
    for m in _RE_LU_RECD_ALT.finditer(html_text):
        pairs.append((html.unescape(m.group(2).strip()), html.unescape(m.group(1).strip())))
    return pairs


def _fetch_liveuamap_home_html(url: str, *, timeout_s: float, base_headers: dict[str, str]) -> str | None:
    for ua in _LU_FETCH_USER_AGENTS:
        h = {
            **base_headers,
            "User-Agent": ua,
            "Referer": url,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
        }
        try:
            r = requests.get(
                url, headers=h, timeout=timeout_s, proxies={"http": None, "https": None}
            )
            r.raise_for_status()
            if "recd_descr" in r.text:
                return r.text
        except Exception:
            continue
    return None


def _fetch_liveuamap_ukraine_sidebar_html(
    url: str, *, timeout_s: float, base_headers: dict[str, str]
) -> str | None:
    """First page that returns HTTP 200 and includes sidebar `div.event` markup."""
    for ua in _LU_FETCH_USER_AGENTS:
        h = {
            **base_headers,
            "User-Agent": ua,
            "Referer": url,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
        }
        try:
            r = requests.get(
                url, headers=h, timeout=timeout_s, proxies={"http": None, "https": None}
            )
            r.raise_for_status()
            if 'class="event' in r.text or "class='event" in r.text:
                return r.text
        except Exception:
            continue
    return None


def _liveuamap_abs_url(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("//"):
        return f"https:{u}"
    return u


def _liveuamap_event_page_latlng(
    page_url: str, *, timeout_s: float, base_headers: dict[str, str]
) -> tuple[float, float] | None:
    """
    Event detail pages embed the incident map centre as:
        lat=…; lng=…; inside $(document).ready
    """
    dest = _liveuamap_abs_url(page_url)
    if not dest:
        return None
    for ua in _LU_FETCH_USER_AGENTS:
        h = {
            **base_headers,
            "User-Agent": ua,
            "Referer": dest,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
        }
        try:
            r = requests.get(
                dest, headers=h, timeout=timeout_s, proxies={"http": None, "https": None}
            )
            r.raise_for_status()
            m = _RE_LU_EVENT_PAGE_LATLNG.search(r.text)
            if not m:
                continue
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except Exception:
            continue
    return None


def fetch_ukraine_liveuamap_oil_infra_rows(
    *,
    timeout_s: float = 22.0,
    request_headers: dict[str, str] | None = None,
    top_n: int = 5,
    max_scan: int = 40,
) -> list[UkraineOilInfraScrapeRow]:
    """
    Scrape https://ukraine.liveuamap.com/en/event/oil when it serves sidebar cards;
    otherwise fall back to https://ukraine.liveuamap.com/en.

    Walks sidebar ``div.event`` nodes in order (most recent first). Keeps up to
    ``top_n`` rows whose titles mention depot, refinery, or terminal
    (case-insensitive), scanning at most ``max_scan`` cards. Resolves coordinates
    from each LiveUAMap detail page when possible (incident map centre).
    """
    headers = {
        **DEFAULT_RSS_REQUEST_HEADERS,
        **(request_headers or {}),
        "Cache-Control": "no-cache",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    html_doc: str | None = None
    for home in (LIVEUAMAP_UKRAINE_OIL_TARGET_URL, LIVEUAMAP_UKRAINE_HOME_EN_URL):
        html_doc = _fetch_liveuamap_ukraine_sidebar_html(
            home, timeout_s=timeout_s, base_headers=headers
        )
        if html_doc:
            break
    if not html_doc:
        return []

    soup = BeautifulSoup(html_doc, "html.parser")
    events = soup.select("div.event")[:max_scan]
    rows: list[UkraineOilInfraScrapeRow] = []
    for ev in events:
        if len(rows) >= top_n:
            break
        title_el = ev.select_one("div.title")
        raw_title = (title_el.get_text(" ", strip=True) if title_el else "").strip()
        if not raw_title:
            continue
        tl = raw_title.lower()
        if not any(k in tl for k in _UKRAINE_OIL_INFRA_KEYWORDS):
            continue
        time_el = ev.select_one("span.date_add")
        time_s = (time_el.get_text(" ", strip=True) if time_el else "").strip() or "—"
        link = _liveuamap_abs_url((ev.get("data-link") or "").strip())
        if not link:
            continue
        coords = _liveuamap_event_page_latlng(link, timeout_s=timeout_s, base_headers=headers)
        lat, lon = (coords[0], coords[1]) if coords else (None, None)
        rows.append(UkraineOilInfraScrapeRow(time=time_s, title=raw_title, link=link, lat=lat, lon=lon))

    return rows


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r_km * math.asin(min(1.0, math.sqrt(a)))


def _is_kinetic_title(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in KINETIC_TERMS)


def _title_implies_hormuz_corridor(title: str) -> bool:
    t = (title or "").lower()
    return any(k in t for k in HORMUZ_PROXIMITY_TERMS)


def _coords_from_text(blob: str) -> tuple[float, float] | None:
    m = _RE_COORD_QUERY.search(blob or "")
    if not m:
        return None
    try:
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except ValueError:
        pass
    return None


def kinetic_event_within_hormuz_zone(title: str, link: str) -> bool:
    """
    True if the item describes kinetic activity and is within ~50km of Hormuz
    (coordinates in URL when present, otherwise Hormuz maritime keyword heuristic).
    """
    if not _is_kinetic_title(title):
        return False
    blob = f"{link} {title}"
    coords = _coords_from_text(blob)
    if coords is not None:
        return haversine_km(coords[0], coords[1], HORMUZ_LAT, HORMUZ_LON) <= 50.0
    return _title_implies_hormuz_corridor(title)


def _newsdata_rows_from_results(results: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not isinstance(results, list):
        return out
    for row in results:
        if not isinstance(row, dict):
            continue
        title = (row.get("title") or "").strip()
        link = (row.get("link") or "").strip()
        raw_dt = str(row.get("pubDate") or "").strip()
        if raw_dt:
            try:
                dtn = datetime.strptime(raw_dt[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
                dt_label = dtn.strftime("%a, %d %b %Y %H:%M:%S GMT")
            except ValueError:
                dt_label = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
        else:
            dt_label = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
        creators = row.get("creator")
        if isinstance(creators, list) and creators:
            source = ", ".join(str(x) for x in creators[:3] if x)
        else:
            source = str(row.get("source_id") or row.get("source_name") or "newsdata.io").strip()
        if title or link:
            lang = str(row.get("language") or "").strip()
            if lang:
                source = f"{source} [{lang}]"
            out.append(_normalized_entry(dt=dt_label, source=source, title=title, link=link))
    out.sort(key=lambda x: _parse_header_date_ts(x.get("Date/Time (UTC)", "")), reverse=True)
    return out


def _newsdata_error_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "NewsData.io returned an unexpected payload."
    if payload.get("status") == "success":
        return None
    res = payload.get("results")
    if isinstance(res, dict):
        for k in ("message", "msg", "error"):
            if res.get(k):
                return str(res[k])
    for k in ("message", "msg"):
        if payload.get(k):
            return str(payload[k])
    return "NewsData.io status was not success (check API key, plan limits, and parameters)."


def fetch_newsdata_iran_feed(
    *,
    api_key: str | None,
    timeout_s: float = 25.0,
    size: int = 28,
    request_headers: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    """
    NewsData.io: Iran-based publishers. Tries English first; many wires file in other
    languages, so we fall back to country=ir without a language filter.

    Returns (rows, user_hint). user_hint is None on success with rows, or an error / empty explanation.
    """
    if not (api_key or "").strip():
        return [], None

    key = api_key.strip()
    headers = {
        **DEFAULT_RSS_REQUEST_HEADERS,
        **(request_headers or {}),
        "Accept": "application/json",
    }
    def call(params: dict[str, str | int]) -> tuple[list[dict[str, str]], str | None]:
        try:
            r = requests.get(
                NEWSDATA_API_LATEST,
                params=params,
                headers=headers,
                timeout=timeout_s,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            payload = r.json()
        except requests.exceptions.HTTPError as e:
            txt = ""
            try:
                txt = (e.response.text or "")[:300]
            except Exception:
                pass
            return [], f"NewsData.io HTTP {e.response.status_code}. {txt}".strip()
        except Exception as e:
            return [], f"NewsData.io request failed: {e!s}"

        err = _newsdata_error_from_payload(payload)
        if err:
            return [], err
        rows = _newsdata_rows_from_results(payload.get("results"))
        return rows, None

    en_params: dict[str, str | int] = {
        "apikey": key,
        "country": "ir",
        "language": "en",
        "size": size,
    }
    rows_en, err_en = call(en_params)
    if rows_en:
        return rows_en, None

    ir_params: dict[str, str | int] = {"apikey": key, "country": "ir", "size": size}
    rows_ir, err_ir = call(ir_params)
    if rows_ir:
        return rows_ir, "No Iran/English-only hits; showing latest Iran-country items (any language, tag in Source)."

    if err_en and err_ir:
        return [], f"{err_en} | {err_ir}"
    if err_en:
        return [], err_en
    if err_ir:
        return [], err_ir
    return [], "NewsData.io returned no Iran articles for English or any-language filters."


def fetch_newsdata_iran_english(
    *,
    api_key: str | None,
    timeout_s: float = 25.0,
    size: int = 28,
    request_headers: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Backward-compatible wrapper (rows only). Prefer fetch_newsdata_iran_feed for diagnostics."""
    rows, _ = fetch_newsdata_iran_feed(
        api_key=api_key, timeout_s=timeout_s, size=size, request_headers=request_headers
    )
    return rows


# English Iran narrative RSS (state-affiliated) when NewsData.io has no rows / no API key.
PRESSTV_IRAN_RSS_URL = "https://www.presstv.ir/rss/rss-101.xml"


def fetch_official_tehran_narrative(
    *,
    api_key: str | None,
    timeout_s: float = 25.0,
    request_headers: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], str | None]:
    """
    Prefer NewsData.io (Iran). If it returns no articles, use Press TV Iran RSS (English).
    Second return value is an optional short caption for the UI (source / limitations).
    """
    rows_nd, hint_nd = fetch_newsdata_iran_feed(
        api_key=api_key, timeout_s=timeout_s, request_headers=request_headers
    )
    if rows_nd:
        return rows_nd, None

    try:
        pv = fetch_live_rss_entries(
            PRESSTV_IRAN_RSS_URL,
            limit=22,
            timeout_s=timeout_s,
            request_headers=request_headers,
        )
    except Exception:
        pv = []

    if pv:
        for it in pv:
            it["Source"] = "Press TV (Iran RSS)"
        # No nag caption when RSS fills the panel; NewsData remains preferred when it returns rows.
        return pv, None

    tail = hint_nd
    if not tail:
        tail = "No articles from NewsData.io or Press TV RSS (check network or feeds)."
    now_label = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
    fallback_rows = [
        _normalized_entry(
            dt=now_label,
            source="Fallback brief",
            title="Tehran narrative feed temporarily unavailable (network/provider block).",
            link="",
        ),
        _normalized_entry(
            dt=now_label,
            source="Fallback brief",
            title="Showing continuity mode until live Iran-source feeds recover.",
            link="",
        ),
    ]
    return fallback_rows, tail


def fetch_liveuamap_mideast_kinetic(
    *,
    timeout_s: float = 20.0,
    max_items: int = 45,
    request_headers: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], bool]:
    """
    Scrape regional LiveUAMap landing pages for kinetic-style cards (RSS is paywalled).
    Second return value: True if any kinetic headline falls in the Hormuz ~50km rule.
    """
    headers = {
        **DEFAULT_RSS_REQUEST_HEADERS,
        **(request_headers or {}),
        "Cache-Control": "no-cache",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    seen_links: set[str] = set()
    ordered: list[dict[str, str]] = []
    hormuz_kinetic = False
    now_stamp = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")

    for home in LIVEUAMAP_SCRAPE_HOME_PAGES:
        html_doc = _fetch_liveuamap_home_html(home, timeout_s=timeout_s, base_headers=headers)
        if not html_doc:
            continue
        host_seg = (home.split("//", 1)[1].split(".")[0] if "//" in home else "liveuamap").lower()
        for link, title in _lu_recommend_link_title_pairs(html_doc):
            if not title or link in seen_links:
                continue
            if not _is_kinetic_title(title):
                continue
            seen_links.add(link)
            if kinetic_event_within_hormuz_zone(title, link):
                hormuz_kinetic = True
            src = f"LiveUAMap ({host_seg})"
            ordered.append(_normalized_entry(dt=now_stamp, source=src, title=title, link=link))
            if len(ordered) >= max_items:
                return ordered, hormuz_kinetic

    if not ordered:
        try:
            gq = (
                "(explosion OR missile OR drone OR intercept OR airstrike OR strike OR blast) "
                "AND (Iran OR Iraq OR Israel OR Syria OR Lebanon OR Yemen OR Gulf OR Hormuz OR UAE OR Qatar) "
                "when:1d"
            )
            gurl = _google_news_rss_url(gq)
            for row in fetch_live_rss_entries(
                gurl, limit=28, timeout_s=timeout_s, request_headers=request_headers
            ):
                t = row.get("Title", "") or ""
                lk = row.get("Link", "") or ""
                if not _is_kinetic_title(t) or not lk or lk in seen_links:
                    continue
                seen_links.add(lk)
                if kinetic_event_within_hormuz_zone(t, lk):
                    hormuz_kinetic = True
                dt = row.get("Date/Time (UTC)") or now_stamp
                ordered.append(
                    _normalized_entry(
                        dt=dt,
                        source="OSINT fallback (Google News)",
                        title=t.strip(),
                        link=lk.strip(),
                    )
                )
                if len(ordered) >= max_items:
                    break
        except Exception:
            pass

    if not ordered:
        now_stamp = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")
        ordered.append(
            _normalized_entry(
                dt=now_stamp,
                source="Kinetic monitor status",
                title="No live kinetic feed available (provider/network blocked).",
                link="",
            )
        )

    return ordered, hormuz_kinetic


def apply_kinetic_hormuz_maximum_override(status: StraitStatus, *, hormuz_kinetic: bool) -> StraitStatus:
    if not hormuz_kinetic:
        return status
    prefix = "Kinetic event within 50km of the Strait of Hormuz (LiveUAMap / Middle East scrape). "
    return StraitStatus(
        evaluated_at_utc=status.evaluated_at_utc,
        war_risk_level="MAXIMUM",
        strait_status=status.strait_status,
        rationale=prefix + status.rationale,
        confirmations=list(status.confirmations),
    )


def evaluate_strait_status(
    headlines: list[str],
    *,
    llm_callable: Any | None = None,
) -> StraitStatus:
    """
    "The Discerner".

    If `llm_callable` is provided, it should accept a list[str] headlines and return
    a dict-like response with keys:
      - strait_status
      - confirmations (list[str])
      - rationale

    Rule (hard):
      Unless there is official confirmation of *Unrestricted Safe Passage* from
      the IMO or US Fifth Fleet, war_risk_level must remain CRITICAL.

    This function is safe-by-default: if LLM evaluation fails or is absent, it
    falls back to deterministic matching.
    """
    now = datetime.now(UTC)
    h = " | ".join([x or "" for x in (headlines or [])]).lower()

    def has_imo() -> bool:
        return "imo" in h or "international maritime organization" in h

    def has_us_fifth_fleet() -> bool:
        return "us fifth fleet" in h or "u.s. fifth fleet" in h or "c5f" in h

    def has_unrestricted_safe_passage_phrase() -> bool:
        return "unrestricted safe passage" in h

    confirmations: list[str] = []

    # Optional LLM path (not required for correctness).
    if llm_callable is not None:
        try:
            resp = llm_callable(headlines)
            status = str((resp or {}).get("strait_status", "")).strip().upper()
            conf = (resp or {}).get("confirmations") or []
            conf_list = [str(x).strip() for x in conf if str(x).strip()]
            rationale = str((resp or {}).get("rationale", "")).strip() or "LLM evaluation."
            if status == "UNRESTRICTED_SAFE_PASSAGE" and conf_list:
                confirmations = conf_list
                return StraitStatus(
                    evaluated_at_utc=now,
                    war_risk_level="LOW",
                    strait_status="UNRESTRICTED_SAFE_PASSAGE",
                    rationale=rationale,
                    confirmations=confirmations,
                )
        except Exception:
            # Fall through to deterministic logic.
            pass

    # Deterministic safeguard: only downgrade if we see BOTH (source) and (phrase).
    if has_unrestricted_safe_passage_phrase() and (has_imo() or has_us_fifth_fleet()):
        if has_imo():
            confirmations.append("IMO")
        if has_us_fifth_fleet():
            confirmations.append("US Fifth Fleet")
        return StraitStatus(
            evaluated_at_utc=now,
            war_risk_level="LOW",
            strait_status="UNRESTRICTED_SAFE_PASSAGE",
            rationale="Detected explicit 'Unrestricted Safe Passage' from an official authority.",
            confirmations=confirmations,
        )

    return StraitStatus(
        evaluated_at_utc=now,
        war_risk_level="CRITICAL",
        strait_status="RESTRICTED_OR_UNCONFIRMED",
        rationale="No official confirmation of 'Unrestricted Safe Passage' from IMO or US Fifth Fleet present in the latest headlines.",
        confirmations=[],
    )


def evaluate_strait_status_from_live_entries(entries: list[dict[str, str]]) -> StraitStatus:
    """
    Discerner v2 (live OSINT mode):

    - If the last 10 headlines include keywords like Strike/Missile/Hormuz Closed/IRGC/8PM => CRITICAL
    - If headlines confirm 'Hormuz Open' or 'Safe Transit' from sources like US Fifth Fleet or IMO => LOW
    - Otherwise => HIGH (risk remains elevated absent clear all-clear).
    """
    now = datetime.now(UTC)
    recent = (entries or [])[:10]

    def norm(s: str) -> str:
        return (s or "").lower()

    threat_keywords = ["strike", "missile", "hormuz closed", "irgc", "8pm"]
    all_clear_keywords = ["hormuz open", "safe transit", "unrestricted safe passage"]
    authority_sources = ["us fifth fleet", "u.s. fifth fleet", "imo", "international maritime organization"]

    # Check for official all-clear first (it can override general threat chatter).
    confirmations: list[str] = []
    for it in recent:
        title = norm(_entry_title(it))
        source = norm(_entry_source(it))
        if any(k in title for k in all_clear_keywords) and any(a in source or a in title for a in authority_sources):
            if "imo" in source or "international maritime organization" in source or "imo" in title:
                confirmations.append("IMO")
            if "fifth fleet" in source or "fifth fleet" in title:
                confirmations.append("US Fifth Fleet")
            return StraitStatus(
                evaluated_at_utc=now,
                war_risk_level="LOW",
                strait_status="UNRESTRICTED_SAFE_PASSAGE",
                rationale="Live headlines indicate Hormuz is open / safe transit, with attribution to IMO or US Fifth Fleet.",
                confirmations=sorted(set(confirmations)),
            )

    # Threat scan.
    for it in recent:
        title = norm(_entry_title(it))
        if any(k in title for k in threat_keywords):
            return StraitStatus(
                evaluated_at_utc=now,
                war_risk_level="CRITICAL",
                strait_status="RESTRICTED_OR_UNCONFIRMED",
                rationale="Threat keywords detected in the last 10 live headlines (Strike/Missile/Hormuz Closed/IRGC/8PM).",
                confirmations=[],
            )

    return StraitStatus(
        evaluated_at_utc=now,
        war_risk_level="HIGH",
        strait_status="RESTRICTED_OR_UNCONFIRMED",
        rationale="No official IMO/US Fifth Fleet all-clear detected; maintaining elevated risk.",
        confirmations=[],
    )


DAILY_PORTS_FEATURESERVER_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Ports_Data/FeatureServer/0"
)

MONTHLY_TRADENOW_FEATURESERVER_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Monthly_TradeNow/FeatureServer/0"
)


@dataclass(frozen=True)
class HormuzStats:
    fetched_at_utc: datetime
    asof_date_utc: datetime | None
    daily_transits_total: int | None
    wait_list_tankers_fujairah_proxy: int | None
    trade_value_drop_pct: dict[str, float]  # keys: EU/China/US
    blockade_detected: bool
    notes: list[str]


def _query_arcgis_feature_layer(url: str, *, params: dict[str, Any], timeout_s: float = 12.0) -> dict[str, Any]:
    last_err: Exception | None = None
    for _ in range(2):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout_s,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as err:
            last_err = err
    raise RuntimeError(f"ArcGIS request failed for {url}: {last_err!s}") from last_err


def _latest_value_for_where(
    layer_url: str,
    *,
    where: str,
    out_fields: str,
    order_by: str,
    timeout_s: float,
) -> dict[str, Any] | None:
    js = _query_arcgis_feature_layer(
        f"{layer_url}/query",
        params={
            "f": "json",
            "where": where,
            "outFields": out_fields,
            "orderByFields": order_by,
            "resultRecordCount": 1,
            "returnGeometry": "false",
        },
        timeout_s=timeout_s,
    )
    feats = js.get("features") or []
    if not feats:
        return None
    return (feats[0] or {}).get("attributes") or None


def _latest_two_trade_values_for_region(region: str, *, timeout_s: float) -> tuple[float | None, float | None]:
    js = _query_arcgis_feature_layer(
        f"{MONTHLY_TRADENOW_FEATURESERVER_URL}/query",
        params={
            "f": "json",
            "where": f"region='{region}'",
            "outFields": "date,trade_value",
            "orderByFields": "date DESC",
            "resultRecordCount": 2,
            "returnGeometry": "false",
        },
        timeout_s=timeout_s,
    )
    feats = js.get("features") or []
    vals: list[float] = []
    for f in feats:
        a = (f or {}).get("attributes") or {}
        v = a.get("trade_value")
        try:
            vals.append(float(v))
        except Exception:
            pass
    latest = vals[0] if len(vals) > 0 else None
    prev = vals[1] if len(vals) > 1 else None
    return latest, prev


def fetch_hormuz_stats(
    *,
    blockade_threshold_daily_transits: int = 15,
    timeout_s: float = 12.0,
) -> HormuzStats:
    """
    Strait of Hormuz monitor:
    - Daily Transits: n_total from PortWatch Daily Chokepoints dataset (last observed date).
    - Wait-List: proxy from Fujairah 'portcalls_tanker' (PortWatch Daily Ports dataset).
      (PortWatch does not expose a clean 'loitering' count in the public catalog; this is the closest public proxy.)
    - Export/Import Deficit: trade value % drop from Monthly TradeNow (EU/China/US), computed as drop vs prior month.
    """
    fetched_at = datetime.now(UTC)
    notes: list[str] = []

    # Daily transits through Hormuz.
    try:
        attrs = _latest_value_for_where(
            PORTWATCH_FEATURESERVER_URL,
            where="portname='Strait of Hormuz'",
            out_fields="date,n_total",
            order_by="date DESC",
            timeout_s=timeout_s,
        )
    except Exception as e:
        attrs = None
        notes.append(f"Hormuz transit feed temporarily unavailable: {e!s}")
    asof_dt = None
    daily_transits = None
    if attrs:
        raw_date = attrs.get("date")
        if isinstance(raw_date, (int, float)):
            try:
                asof_dt = datetime.fromtimestamp(raw_date / 1000.0, tz=UTC)
            except Exception:
                asof_dt = None
        try:
            daily_transits = int(attrs.get("n_total"))
        except Exception:
            daily_transits = None

    # Wait-list proxy: Fujairah tanker port calls (daily).
    try:
        fuj = _latest_value_for_where(
            DAILY_PORTS_FEATURESERVER_URL,
            where="portname='Fujairah'",
            out_fields="date,portcalls_tanker",
            order_by="date DESC",
            timeout_s=timeout_s,
        )
    except Exception as e:
        fuj = None
        notes.append(f"Fujairah wait-list feed temporarily unavailable: {e!s}")
    wait_list_proxy = None
    if fuj:
        try:
            wait_list_proxy = int(fuj.get("portcalls_tanker"))
        except Exception:
            wait_list_proxy = None
        notes.append("Wait-List uses Fujairah tanker port calls as a public proxy for loitering/queueing.")

    # Trade Nowcast drop (Monthly TradeNow): EU/China/US.
    regions = {"EU": "European Union", "China": "China", "US": "United States"}
    trade_drop: dict[str, float] = {}
    for k, region in regions.items():
        try:
            latest, prev = _latest_two_trade_values_for_region(region, timeout_s=timeout_s)
        except Exception as e:
            notes.append(f"Trade nowcast feed unavailable for {k}: {e!s}")
            continue
        if latest is None or prev is None or prev == 0:
            continue
        pct_change = (latest - prev) / prev * 100.0
        drop = max(0.0, -pct_change)
        trade_drop[k] = round(drop, 1)

    blockade = bool(daily_transits is not None and daily_transits < blockade_threshold_daily_transits)

    return HormuzStats(
        fetched_at_utc=fetched_at,
        asof_date_utc=asof_dt,
        daily_transits_total=daily_transits,
        wait_list_tankers_fujairah_proxy=wait_list_proxy,
        trade_value_drop_pct=trade_drop,
        blockade_detected=blockade,
        notes=notes,
    )


def _arcgis_query(
    *,
    where: str,
    out_fields: str,
    order_by: str,
    record_count: int,
    timeout_s: float,
) -> dict[str, Any]:
    url = f"{PORTWATCH_FEATURESERVER_URL}/query"
    params = {
        "f": "json",
        "where": where,
        "outFields": out_fields,
        "orderByFields": order_by,
        "resultRecordCount": record_count,
        "returnGeometry": "false",
    }
    last_err: Exception | None = None
    for _ in range(2):
        try:
            r = requests.get(
                url,
                params=params,
                timeout=timeout_s,
                proxies={"http": None, "https": None},
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as err:
            last_err = err
    raise RuntimeError(f"ArcGIS transit query failed for {url}: {last_err!s}") from last_err


def _latest_transits_for_portname(portname: str, *, timeout_s: float = 12.0) -> tuple[datetime | None, int | None]:
    js = _arcgis_query(
        where=f"portname='{portname}'",
        out_fields="date,n_total",
        order_by="date DESC",
        record_count=1,
        timeout_s=timeout_s,
    )
    feats = js.get("features") or []
    if not feats:
        return None, None
    attrs = (feats[0] or {}).get("attributes") or {}

    # ArcGIS dates are usually ms since epoch.
    dt = None
    raw_date = attrs.get("date")
    if isinstance(raw_date, (int, float)):
        try:
            dt = datetime.fromtimestamp(raw_date / 1000.0, tz=UTC)
        except Exception:
            dt = None

    n_total = attrs.get("n_total")
    try:
        n_total_i = int(n_total)
    except Exception:
        n_total_i = None

    return dt, n_total_i


def fetch_realtime_shipping_stats(
    *,
    danger_zone_suez_transits_per_day: int = 40,
    timeout_s: float = 12.0,
) -> PortWatchTransitSnapshot:
    """
    Fetch daily transit calls from IMF PortWatch for:
      - Suez Canal
      - Cape of Good Hope

    Logic:
      - If Suez transits drop below 40/day => Cape Mode
    Also returns a fixed Xeneta-derived reliability buffer for systemic delays.
    """
    fetched_at = datetime.now(UTC)
    source = "IMF PortWatch (Daily Chokepoint Transit Calls)"

    try:
        suez_dt, suez_n = _latest_transits_for_portname("Suez Canal", timeout_s=timeout_s)
        cape_dt, cape_n = _latest_transits_for_portname("Cape of Good Hope", timeout_s=timeout_s)
        date_utc = suez_dt or cape_dt
    except Exception:
        return PortWatchTransitSnapshot(
            fetched_at_utc=fetched_at,
            date_utc=None,
            suez_transits_per_day=None,
            cape_transits_per_day=None,
            danger_zone_threshold=danger_zone_suez_transits_per_day,
            cape_mode=False,
            reliability_buffer_days=RELIABILITY_BUFFER_DAYS,
            source=source,
        )

    cape_mode = bool(suez_n is not None and suez_n < danger_zone_suez_transits_per_day)

    return PortWatchTransitSnapshot(
        fetched_at_utc=fetched_at,
        date_utc=date_utc,
        suez_transits_per_day=suez_n,
        cape_transits_per_day=cape_n,
        danger_zone_threshold=danger_zone_suez_transits_per_day,
        cape_mode=cape_mode,
        reliability_buffer_days=RELIABILITY_BUFFER_DAYS,
        source=source,
    )


# --- DeepState UA — Telegram web preview (t.me/s/…) energy-keyword monitor ---

DEEPSTATE_UA_TELEGRAM_WEB = "https://t.me/s/DeepStateUA"

# Oil depot / plant / incoming hit (strike arrival) — UA Cyrillic filters
DEEPSTATE_ENERGY_KEYWORDS: tuple[str, ...] = (
    "Нафтобаза",
    "нафтобаза",
    "Завод",
    "завод",
    "Приліт",
    "приліт",
)

# Map substring hits to Kinetic Events-style labels (longer keys first)
KINETIC_EVENT_CITY_ALIASES: tuple[tuple[str, str], ...] = (
    ("Кривий Ріг", "Kryvyi Rih — kinetic sector"),
    ("Запоріжжя", "Zaporizhzhia — kinetic sector"),
    ("Миколаїв", "Mykolaiv — kinetic sector"),
    ("Маріуполь", "Mariupol — kinetic sector"),
    ("Покровськ", "Pokrovsk — kinetic sector"),
    ("Краматорськ", "Kramatorsk — kinetic sector"),
    ("Слов'янськ", "Sloviansk — kinetic sector"),
    ("Словянськ", "Sloviansk — kinetic sector"),
    ("Бахмут", "Bakhmut — kinetic sector"),
    ("Часів Яр", "Chasiv Yar — kinetic sector"),
    ("Куп'янськ", "Kupiansk — kinetic sector"),
    ("Купянськ", "Kupiansk — kinetic sector"),
    ("Харків", "Kharkiv — kinetic sector"),
    ("Одеса", "Odesa — kinetic sector"),
    ("Дніпро", "Dnipro — kinetic sector"),
    ("Київ", "Kyiv — kinetic sector"),
    ("Львів", "Lviv — kinetic sector"),
    ("Суми", "Sumy — kinetic sector"),
    ("Чернігів", "Chernihiv — kinetic sector"),
    ("Полтава", "Poltava — kinetic sector"),
    ("Рівне", "Rivne — kinetic sector"),
    ("Вінниця", "Vinnytsia — kinetic sector"),
)


@dataclass(frozen=True)
class DeepStateEnergyHit:
    """Single DeepState post matching energy / kinetic keywords."""

    summary: str
    matched_keyword: str
    latitude: float | None
    longitude: float | None
    kinetic_label: str
    source_url: str | None


def _deepstate_match_energy_keyword(text: str) -> str | None:
    for kw in DEEPSTATE_ENERGY_KEYWORDS:
        if kw in text:
            return kw
    return None


def _deepstate_parse_coordinates(text: str) -> tuple[float, float] | None:
    """Extract first plausible WGS84 pair in or near Ukraine."""
    patterns = (
        re.compile(r"(\d{2}\.\d{3,8})\s*[,;]\s*(\d{2}\.\d{3,8})"),
        re.compile(r"(\d{2}\.\d{3,8})\s*/\s*(\d{2}\.\d{3,8})"),
    )
    for pat in patterns:
        for m in pat.finditer(text):
            try:
                a, b = float(m.group(1)), float(m.group(2))
            except ValueError:
                continue
            if 44.0 <= a <= 53.5 and 22.0 <= b <= 42.0:
                return a, b
            if 44.0 <= b <= 53.5 and 22.0 <= a <= 42.0:
                return b, a
    return None


def _deepstate_kinetic_city_label(text: str) -> str:
    for ua_fragment, label in KINETIC_EVENT_CITY_ALIASES:
        if ua_fragment in text:
            return label
    return "Energy hit — location pending (see post)"


def get_deepstate_updates(
    *,
    channel_web_url: str = DEEPSTATE_UA_TELEGRAM_WEB,
    request_headers: dict[str, str] | None = None,
    max_messages_scan: int = 100,
    timeout_s: float = 28.0,
) -> list[DeepStateEnergyHit]:
    """
    Scrape the public Telegram channel preview (HTML) for DeepState UA posts that mention
    oil depots, plants/factories, or incoming strikes («Приліт»).

    Coordinate pairs in the message (decimal °) are preferred; otherwise a city is mapped
    to the kinetic-sector list when its Ukrainian name appears in the text.
    """
    headers = request_headers or dict(DEFAULT_RSS_REQUEST_HEADERS)
    out: list[DeepStateEnergyHit] = []
    try:
        r = requests.get(
            channel_web_url,
            headers=headers,
            timeout=timeout_s,
            proxies={"http": None, "https": None},
        )
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    blocks = soup.select(".tgme_widget_message")[-max_messages_scan:]
    # Telegram lists oldest → newest; process newest first
    for block in reversed(blocks):
        text_el = block.select_one(".tgme_widget_message_text")
        if not text_el:
            continue
        text = text_el.get_text("\n", strip=True)
        if not text:
            continue
        kw_hit = _deepstate_match_energy_keyword(text)
        if not kw_hit:
            continue

        coords = _deepstate_parse_coordinates(text)
        if coords:
            lat, lon = coords
            k_label = f"{lat:.4f}, {lon:.4f} (WGS84)"
        else:
            lat, lon = None, None
            k_label = _deepstate_kinetic_city_label(text)

        link_el = block.select_one("a.tgme_widget_message_date")
        href = str(link_el.get("href")).strip() if link_el and link_el.get("href") else None

        preview = text.replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:217] + "…"

        out.append(
            DeepStateEnergyHit(
                summary=preview,
                matched_keyword=kw_hit,
                latitude=lat,
                longitude=lon,
                kinetic_label=k_label,
                source_url=href,
            )
        )

    return out


# --- Shadow fleet intelligence (Baltic / AIS–STS heuristics; wire to live AIS when available) ---

DEFAULT_RUSSIAN_TANKER_LIST: frozenset[str] = frozenset(
    {
        "IMO9783431",
        "IMO9454567",
        "IMO9881234",
        "IMO9732108",
        "RU-TNK-SHADOW-01",
    }
)

GOTLAND_TRANSFER_ZONE_LAT_MIN = 57.0
GOTLAND_TRANSFER_ZONE_LON_MIN = 18.0
GOTLAND_TRANSFER_ZONE_LAT_MAX = 59.0
GOTLAND_TRANSFER_ZONE_LON_MAX = 20.0


@dataclass(frozen=True)
class ShadowVesselState:
    """One vessel snapshot for the rules engine."""

    vessel_id: str
    name: str
    position_updated_at: datetime | None
    latitude: float | None
    longitude: float | None
    current_draft_m: float | None
    previous_draft_m: float | None
    inside_port: bool
    path_start_lat: float | None
    path_start_lon: float | None
    path_end_lat: float | None
    path_end_lon: float | None
    segment_had_port_call: bool
    on_russian_tanker_list: bool


@dataclass(frozen=True)
class ShadowVesselAssessment:
    """Outputs for map styling, tooltips, and sidebar."""

    vessel_id: str
    name: str
    latitude: float
    longitude: float
    gray_ghost: bool
    probable_sts: bool
    cargo_discharge_alert: bool
    reasons: tuple[str, ...]


class ShadowFleetIntelligence:
    """
    - Gray Ghost: Russian-listed tanker with no AIS position for > ``ais_stale_hours`` (default 4).
    - Probable STS: track starts or ends inside Gotland box without a port call on segment.
    - Cargo discharge alert: draft drops > ``draft_drop_alert_m`` outside a port.
    """

    ais_stale_hours: float = 4.0
    draft_drop_alert_m: float = 2.0

    def __init__(
        self,
        *,
        russian_tanker_list: frozenset[str] | None = None,
        ais_stale_hours: float | None = None,
        draft_drop_alert_m: float | None = None,
    ) -> None:
        self._russian_list = (
            russian_tanker_list if russian_tanker_list is not None else DEFAULT_RUSSIAN_TANKER_LIST
        )
        if ais_stale_hours is not None:
            self.ais_stale_hours = float(ais_stale_hours)
        if draft_drop_alert_m is not None:
            self.draft_drop_alert_m = float(draft_drop_alert_m)

    @staticmethod
    def gotland_transfer_zone_contains(lat: float, lon: float) -> bool:
        """Gotland transfer zone bounding box [57.0,18.0] → [59.0,20.0]."""
        return (
            GOTLAND_TRANSFER_ZONE_LAT_MIN <= lat <= GOTLAND_TRANSFER_ZONE_LAT_MAX
            and GOTLAND_TRANSFER_ZONE_LON_MIN <= lon <= GOTLAND_TRANSFER_ZONE_LON_MAX
        )

    def is_listed_russian_tanker(self, vessel_id: str) -> bool:
        vid = (vessel_id or "").strip().upper()
        return vid in {x.strip().upper() for x in self._russian_list}

    def _gray_ghost(self, state: ShadowVesselState, *, now: datetime) -> bool:
        if not state.on_russian_tanker_list:
            return False
        if state.position_updated_at is None:
            return True
        pu = state.position_updated_at
        if pu.tzinfo is None:
            pu = pu.replace(tzinfo=UTC)
        age_h = (now.astimezone(UTC) - pu.astimezone(UTC)).total_seconds() / 3600.0
        return age_h > self.ais_stale_hours

    def _probable_sts(self, state: ShadowVesselState) -> bool:
        if state.segment_had_port_call:
            return False
        inside_start = inside_end = False
        if state.path_start_lat is not None and state.path_start_lon is not None:
            inside_start = self.gotland_transfer_zone_contains(state.path_start_lat, state.path_start_lon)
        if state.path_end_lat is not None and state.path_end_lon is not None:
            inside_end = self.gotland_transfer_zone_contains(state.path_end_lat, state.path_end_lon)
        return inside_start or inside_end

    def _cargo_discharge_alert(self, state: ShadowVesselState) -> bool:
        if state.inside_port:
            return False
        if state.previous_draft_m is None or state.current_draft_m is None:
            return False
        return (float(state.previous_draft_m) - float(state.current_draft_m)) > self.draft_drop_alert_m

    def assess(self, state: ShadowVesselState, *, now: datetime | None = None) -> ShadowVesselAssessment:
        now_utc = now if now is not None else datetime.now(UTC)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=UTC)

        gg = self._gray_ghost(state, now=now_utc)
        sts = self._probable_sts(state)
        discharge = self._cargo_discharge_alert(state)

        reasons: list[str] = []
        if gg:
            reasons.append(
                f"Gray Ghost: Russian-listed tanker AIS silent >{self.ais_stale_hours:.0f}h or no fix"
            )
        if sts:
            reasons.append(
                "Probable STS: segment start/end inside Gotland transfer zone without port call"
            )
        if discharge:
            reasons.append(
                f"Cargo discharge alert: draft −Δ>{self.draft_drop_alert_m:g} m outside port"
            )

        lat = float(state.latitude) if state.latitude is not None else 57.8
        lon = float(state.longitude) if state.longitude is not None else 19.2

        return ShadowVesselAssessment(
            vessel_id=state.vessel_id,
            name=state.name,
            latitude=lat,
            longitude=lon,
            gray_ghost=gg,
            probable_sts=sts,
            cargo_discharge_alert=discharge,
            reasons=tuple(reasons),
        )

    @staticmethod
    def fleet_confidence_percent(assessments: Sequence[ShadowVesselAssessment]) -> int:
        """
        0–100%: equal weight to Gray Ghost rate, Probable STS rate, discharge-alert rate.
        """
        if not assessments:
            return 0
        n = float(len(assessments))
        ghost_rate = sum(1 for a in assessments if a.gray_ghost) / n
        sts_rate = sum(1 for a in assessments if a.probable_sts) / n
        discharge_rate = sum(1 for a in assessments if a.cargo_discharge_alert) / n
        raw = 100.0 * (ghost_rate + sts_rate + discharge_rate) / 3.0
        return int(max(0, min(100, round(raw))))


def demo_shadow_fleet_assessments(*, now: datetime | None = None) -> list[ShadowVesselAssessment]:
    """Synthetic fleet rows for dashboard wiring until AIS ingestion exists."""
    now_utc = now if now is not None else datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    intel = ShadowFleetIntelligence()
    stale = now_utc - timedelta(hours=5)

    states: list[ShadowVesselState] = [
        ShadowVesselState(
            vessel_id="IMO9783431",
            name="NORDIC RELIANCE (demo)",
            position_updated_at=stale,
            latitude=57.9,
            longitude=19.1,
            current_draft_m=10.5,
            previous_draft_m=10.4,
            inside_port=False,
            path_start_lat=59.2,
            path_start_lon=21.0,
            path_end_lat=57.9,
            path_end_lon=19.1,
            segment_had_port_call=False,
            on_russian_tanker_list=True,
        ),
        ShadowVesselState(
            vessel_id="IMO9454567",
            name="BALTIC FRONTIER (demo)",
            position_updated_at=now_utc,
            latitude=58.2,
            longitude=18.6,
            current_draft_m=8.0,
            previous_draft_m=11.2,
            inside_port=False,
            path_start_lat=58.2,
            path_start_lon=18.6,
            path_end_lat=58.2,
            path_end_lon=18.6,
            segment_had_port_call=False,
            on_russian_tanker_list=True,
        ),
        ShadowVesselState(
            vessel_id="IMO9881234",
            name="SAFE LEGAL TANKER (demo)",
            position_updated_at=now_utc,
            latitude=55.0,
            longitude=12.0,
            current_draft_m=12.0,
            previous_draft_m=12.1,
            inside_port=False,
            path_start_lat=55.0,
            path_start_lon=12.0,
            path_end_lat=55.1,
            path_end_lon=12.1,
            segment_had_port_call=True,
            on_russian_tanker_list=False,
        ),
    ]

    return [intel.assess(s, now=now_utc) for s in states]

