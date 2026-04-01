from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import sleep

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scraper import get_latest_risk_level, get_live_intelligence_feed
from shipping_data import fos_sur_mer_table, rotterdam_table, trieste_table


DEFAULT_WAR_RISK_LEVEL = "HIGH"
EUROPE_ROUTING_ASSUMPTION = "CAPE OF GOOD HOPE"
DEADLINE_UTC = datetime(2026, 4, 1, 16, 30, tzinfo=UTC)  # 16:30 GMT == 20:00 Tehran


def _deadline_window_active(now_utc: datetime) -> bool:
    return now_utc <= (DEADLINE_UTC + timedelta(hours=24))


def _countdown_panel(now_utc: datetime) -> Panel:
    delta = DEADLINE_UTC - now_utc
    if delta.total_seconds() >= 0:
        total = int(delta.total_seconds())
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        msg = f"COUNTDOWN TO 20:00 TEHRAN (16:30 GMT) • T-{hh:02d}:{mm:02d}:{ss:02d} • APR 01 2026"
    else:
        since = now_utc - DEADLINE_UTC
        total = int(since.total_seconds())
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        window = "WINDOW ACTIVE (+24H)" if _deadline_window_active(now_utc) else "WINDOW ENDED"
        msg = f"DEADLINE PASSED • T+{hh:02d}:{mm:02d}:{ss:02d} • {window} • APR 01 2026"

    return Panel(
        Align.center(Text(msg, style="bold red")),
        box=box.SQUARE,
        border_style="red",
        padding=(0, 1),
    )


def _risk_level_style(level: str) -> str:
    lvl = level.strip().upper()
    if lvl == "EXTREME":
        return "bold white on red"
    if lvl == "HIGH":
        return "bold red"
    if lvl in {"ELEVATED", "MEDIUM"}:
        return "bold dark_orange"
    if lvl in {"WATCH", "LOW"}:
        return "yellow"
    return "green"


def _header(now_utc: datetime, war_risk_level: str) -> Panel:
    left = Text("IRAN WAR TRACKER", style="bold bright_white")
    mid = Text("SHIPPING + DIPLOMATIC SIGNALS", style="bright_black")
    right = Text(f"UTC {now_utc:%Y-%m-%d %H:%M:%S}", style="bold bright_white")

    bar = Table.grid(expand=True)
    bar.add_column(justify="left")
    bar.add_column(justify="center")
    bar.add_column(justify="right")
    bar.add_row(left, mid, right)

    risk_line = Table.grid(expand=True)
    risk_line.add_column(justify="left")
    risk_line.add_column(justify="right")
    risk_line.add_row(
        Text("WAR RISK LEVEL", style="bright_black"),
        Text(war_risk_level.strip().upper(), style=_risk_level_style(war_risk_level)),
    )

    return Panel(
        Group(bar, risk_line),
        box=box.SQUARE,
        border_style="bright_black",
        padding=(0, 1),
        style="on black",
    )


def _intel_feed_table(intel_feed: list[dict]) -> Table:
    table = Table(
        title="LIVE INTELLIGENCE FEED",
        title_style="bold bright_white",
        header_style="bold bright_white",
        border_style="bright_black",
        show_lines=False,
        expand=True,
    )
    table.add_column("Risk", style="bright_black", no_wrap=True, width=5, justify="right")
    table.add_column("SOURCE", style="bold cyan", no_wrap=True, width=16)
    table.add_column("HEADLINE", style="bright_white")

    for item in (intel_feed or [])[:10]:
        source = str(item.get("source", "")).strip()
        headline = str(item.get("headline", "")).strip()
        risk_score = str(item.get("risk_score", "")).strip()

        headline_text: Text | str
        if "IRGC" in source.upper():
            headline_text = Text(headline, style="bold red")
        else:
            headline_text = headline

        table.add_row(risk_score or "—", source or "—", headline_text or "—")

    return table


def _risk_tape(now_utc: datetime) -> Panel:
    items = [
        ("Strait of Hormuz", "ELEVATED", "dark_orange"),
        ("Red Sea", "HIGH", "red"),
        ("Suez", "WATCH", "yellow"),
        ("East Med", "MIXED", "cyan"),
        ("North Sea", "NORMAL", "green"),
    ]

    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")

    for name, level, color in items:
        grid.add_row(
            Text(name, style="bright_white"),
            Text(level, style=f"bold {color}"),
        )

    return Panel(
        Group(
            Text(f"RISK TAPE  |  {now_utc:%b %d}", style="bold bright_white"),
            Text(""),
            grid,
        ),
        border_style="bright_black",
        box=box.SQUARE,
        padding=(0, 1),
    )


def _status_intelligence_briefing_table(intel_feed: list[dict]) -> Table:
    table = Table(
        title="STATUS & INTELLIGENCE BRIEFING (LIVE)",
        title_style="bold bright_white",
        header_style="bold bright_white",
        border_style="bright_black",
        box=box.SQUARE,
        show_lines=True,
        expand=True,
    )
    table.add_column("Source", style="bold cyan", width=14, no_wrap=False, overflow="fold")
    table.add_column("Intelligence / Headline", style="bright_white", no_wrap=False, overflow="fold")
    table.add_column("Risk Level", style="bright_white", width=9, no_wrap=True, justify="center")

    for item in (intel_feed or [])[:10]:
        source = str(item.get("source", "")).strip() or "—"
        headline = str(item.get("headline", "")).strip() or "—"
        risk_score_s = str(item.get("risk_score", "")).strip()

        try:
            risk_score = int(risk_score_s)
        except Exception:
            risk_score = 0

        if risk_score >= 75:
            risk_level = Text("HIGH", style="bold red")
        elif risk_score >= 50:
            risk_level = Text("MEDIUM", style="bold dark_orange")
        else:
            risk_level = Text("LOW", style="green")

        headline_text: Text | str
        if "IRGC" in source.upper():
            headline_text = Text(headline, style="bold red")
        else:
            headline_text = headline

        table.add_row(source, headline_text, risk_level)

    return table


def _ports_panel(
    now_utc: datetime,
    *,
    europe_routing_assumption: str,
    intel_feed: list[dict] | None,
    console_width: int,
) -> Panel:
    banner = Table.grid(expand=True)
    banner.add_column(justify="left")
    banner.add_column(justify="right")
    banner.add_row(
        Text("EUROPE ROUTES", style="bold bright_white"),
        Text(f"ROUTING: {europe_routing_assumption}", style="bold cyan"),
    )

    ports = Table.grid(expand=True)
    ports.add_column(ratio=1)
    ports.add_column(ratio=1)
    ports.add_column(ratio=1)
    ports.add_row(
        trieste_table(now_utc, routing_mode=europe_routing_assumption),
        fos_sur_mer_table(now_utc, routing_mode=europe_routing_assumption),
        rotterdam_table(now_utc, routing_mode=europe_routing_assumption),
    )

    return Panel(
        Group(banner, Text(""), ports),
        border_style="bright_black",
        box=box.SQUARE,
        padding=(0, 0),
    )


def _routing_assumption_for_risk(war_risk_level: str) -> str:
    lvl = war_risk_level.strip().upper()
    if lvl == "HIGH":
        return "CAPE OF GOOD HOPE"
    if lvl == "MEDIUM":
        return "SUEZ (CONGESTION/SECURITY PRICED)"
    return "STANDARD (SUEZ/DIRECT)"


def build_dashboard(
    now_utc: datetime,
    keywords: list[str],
    *,
    war_risk_level: str = DEFAULT_WAR_RISK_LEVEL,
    console_width: int = 120,
    intel_feed: list[dict] | None = None,
) -> Layout:
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="countdown", size=3),
        Layout(name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=1),
    )

    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )

    layout["left"].split_column(
        Layout(name="ports", ratio=2),
        Layout(name="mfa", ratio=1),
    )

    if _deadline_window_active(now_utc):
        europe_routing_assumption = "CAPE OF GOOD HOPE"
    else:
        europe_routing_assumption = _routing_assumption_for_risk(war_risk_level)

    layout["countdown"].update(_countdown_panel(now_utc))
    layout["header"].update(_header(now_utc, war_risk_level=war_risk_level))
    layout["left"]["ports"].update(
        _ports_panel(
            now_utc,
            europe_routing_assumption=europe_routing_assumption,
            intel_feed=intel_feed,
            console_width=console_width,
        )
    )
    layout["left"]["mfa"].update(_status_intelligence_briefing_table(intel_feed or []))
    layout["right"].update(_risk_tape(now_utc))

    layout["footer"].update(Align.center(Text("Press Ctrl+C to exit", style="bright_black")))
    return layout


def main() -> None:
    console = Console()
    keywords = ["Hormuz", "Red Sea", "Suez", "navigation", "sanctions"]
    war_risk_level, matched, _headlines = get_latest_risk_level()
    # Use explicit 'Live' briefing items for readability/testing.
    intel_feed = [
        {
            "source": "IRGC",
            "headline": "8PM Tehran Deadline: 18 Tech Firms (Apple/Google/MSFT) targeted for destruction. Employees advised to evacuate.",
            "risk_score": "90",
        },
        {
            "source": "QatarEnergy",
            "headline": "Missile hits tanker 'Aqua 1' off Qatar coast; transit dropping 90%.",
            "risk_score": "95",
        },
        {
            "source": "Kuwait Gov",
            "headline": "Iranian strike reported at Kuwait International Airport; regional flight disruptions.",
            "risk_score": "80",
        },
    ]

    console.clear()
    try:
        with Live(
            build_dashboard(
                datetime.now(UTC),
                keywords,
                war_risk_level=war_risk_level,
                console_width=console.size.width,
                intel_feed=intel_feed,
            ),
            console=console,
            refresh_per_second=4,
            screen=True,
        ) as live:
            while True:
                now = datetime.now(UTC)
                live.update(
                    build_dashboard(
                        now,
                        keywords,
                        war_risk_level=war_risk_level,
                        console_width=console.size.width,
                        intel_feed=intel_feed,
                    )
                )
                sleep(0.75)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
