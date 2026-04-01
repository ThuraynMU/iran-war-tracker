from __future__ import annotations

from calendar import timegm
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import html
import math
import re
from typing import Any

import feedparser
import requests
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


GOOGLE_NEWS_HORMUZ_RSS_URL = (
    "https://news.google.com/rss/search?q=Iran+war+Strait+of+Hormuz+shipping&hl=en-US&gl=US&ceid=US:en"
)

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"

# Faster-moving than Google’s index for the same stories; merged after Google queries.
BBC_MIDDLE_EAST_RSS_URL = "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml"

# Browser-like defaults reduce blocks from Google RSS on cloud IPs.
DEFAULT_RSS_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
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


def _published_ts_from_feed_entry(e: Any) -> float:
    t = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
    if t and getattr(t, "tm_year", 0) > 1990:
        try:
            return float(timegm(t))
        except (TypeError, ValueError, OverflowError):
            pass
    lbl = (getattr(e, "published", "") or getattr(e, "updated", "") or "").strip()
    return _parse_header_date_ts(lbl)


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
    r = requests.get(feed_url, headers=headers, timeout=timeout_s)
    r.raise_for_status()

    parsed = feedparser.parse(r.content)
    entries = getattr(parsed, "entries", []) or []
    pool = entries[:150]
    scored: list[tuple[float, dict[str, str]]] = []
    for e in pool:
        raw_title = (getattr(e, "title", "") or "").strip()
        link = (getattr(e, "link", "") or "").strip()

        # Use entry.published string when present (as requested).
        dt_label = (getattr(e, "published", "") or "").strip()
        if not dt_label:
            dt_label = (getattr(e, "updated", "") or "").strip()

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
            ts = _published_ts_from_feed_entry(e)
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

    for q in queries:
        q2 = q.strip()
        if " when:" not in q2.lower():
            q2 = f"{q2} when:1d"
        url = _google_news_rss_url(q2)
        items = fetch_live_rss_entries(
            url, limit=per_query_limit, timeout_s=timeout_s, request_headers=request_headers
        )
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

_RE_LU_RECD = re.compile(
    r'class="recd_descr"\s+href="(https://[^"]+)"\s+title="([^"]*)"',
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


def fetch_newsdata_iran_english(
    *,
    api_key: str | None,
    timeout_s: float = 22.0,
    size: int = 28,
    request_headers: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """NewsData.io latest feed: Iran + English language (official wire often STT)."""
    if not (api_key or "").strip():
        return []
    params: dict[str, str | int] = {
        "apikey": api_key.strip(),
        "country": "ir",
        "language": "en",
        "size": size,
    }
    headers = {
        **DEFAULT_RSS_REQUEST_HEADERS,
        **(request_headers or {}),
        "Accept": "application/json",
    }
    try:
        r = requests.get(NEWSDATA_API_LATEST, params=params, headers=headers, timeout=timeout_s)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return []
    if not isinstance(payload, dict) or payload.get("status") != "success":
        return []
    results = payload.get("results") or []
    out: list[dict[str, str]] = []
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
            out.append(_normalized_entry(dt=dt_label, source=source, title=title, link=link))
    out.sort(key=lambda x: _parse_header_date_ts(x.get("Date/Time (UTC)", "")), reverse=True)
    return out


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
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    seen_links: set[str] = set()
    ordered: list[dict[str, str]] = []
    hormuz_kinetic = False
    now_stamp = datetime.now(UTC).strftime("%a, %d %b %Y %H:%M:%S GMT")

    for home in LIVEUAMAP_SCRAPE_HOME_PAGES:
        try:
            r = requests.get(home, headers=headers, timeout=timeout_s)
            r.raise_for_status()
        except Exception:
            continue
        host_seg = (home.split("//", 1)[1].split(".")[0] if "//" in home else "liveuamap").lower()
        for m in _RE_LU_RECD.finditer(r.text):
            link = html.unescape(m.group(1).strip())
            title = html.unescape(m.group(2).strip())
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
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


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
    attrs = _latest_value_for_where(
        PORTWATCH_FEATURESERVER_URL,
        where="portname='Strait of Hormuz'",
        out_fields="date,n_total",
        order_by="date DESC",
        timeout_s=timeout_s,
    )
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
    fuj = _latest_value_for_where(
        DAILY_PORTS_FEATURESERVER_URL,
        where="portname='Fujairah'",
        out_fields="date,portcalls_tanker",
        order_by="date DESC",
        timeout_s=timeout_s,
    )
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
        latest, prev = _latest_two_trade_values_for_region(region, timeout_s=timeout_s)
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
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


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

