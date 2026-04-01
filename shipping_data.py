from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable

from rich.table import Table
from rich.text import Text


@dataclass(frozen=True)
class ShipmentRow:
    vessel: str
    flag: str
    cargo: str
    destination: str
    eta_utc: datetime
    status: str
    risk: int  # 0-100


def _risk_style(risk: int) -> str:
    if risk >= 80:
        return "bold red"
    if risk >= 60:
        return "bold dark_orange"
    if risk >= 40:
        return "yellow"
    return "green"


def _status_style(status: str) -> str:
    s = status.lower()
    if "delayed" in s or "holding" in s:
        return "bold dark_orange"
    if "divert" in s:
        return "bold red"
    if "loading" in s:
        return "cyan"
    return "bright_white"


def _shipments_for_port(port: str, now_utc: datetime, *, routing_mode: str) -> list[ShipmentRow]:
    # Synthetic data for a "terminal" dashboard; swap with real feeds later.
    base = [
        ShipmentRow(
            vessel="ARCADIA STAR",
            flag="MT",
            cargo="Refined products",
            destination="East Med",
            eta_utc=now_utc + timedelta(hours=7),
            status="Underway",
            risk=42,
        ),
        ShipmentRow(
            vessel="NORTHWIND",
            flag="GR",
            cargo="Crude",
            destination="Suez",
            eta_utc=now_utc + timedelta(hours=15),
            status="Delayed (routing)",
            risk=71,
        ),
        ShipmentRow(
            vessel="HELIX TRADER",
            flag="NL",
            cargo="LNG",
            destination="Red Sea",
            eta_utc=now_utc + timedelta(hours=28),
            status="Holding offshore",
            risk=84,
        ),
        ShipmentRow(
            vessel="BALTIC RUNNER",
            flag="PA",
            cargo="Container",
            destination="Gulf",
            eta_utc=now_utc + timedelta(hours=11),
            status="Loading",
            risk=33,
        ),
    ]

    port_bias = {
        "Trieste": (0, 0, 0, 1),
        "Fos-sur-Mer": (1, 0, 0, 0),
        "Rotterdam": (0, 1, 1, 0),
    }.get(port, (0, 0, 0, 0))

    adjusted: list[ShipmentRow] = []
    for i, row in enumerate(base):
        add_h = port_bias[i % len(port_bias)] * 2
        add_r = port_bias[i % len(port_bias)] * 6
        route_h = 0
        route_r = 0
        rm = routing_mode.strip().upper()
        if "CAPE" in rm:
            # Longer routing + elevated operational risk.
            route_h = 96
            route_r = 10
        elif "SUEZ" in rm:
            route_h = 18
            route_r = 4
        adjusted.append(
            ShipmentRow(
                vessel=row.vessel,
                flag=row.flag,
                cargo=row.cargo,
                destination=row.destination,
                eta_utc=row.eta_utc + timedelta(hours=add_h + route_h),
                status=row.status,
                risk=min(100, row.risk + add_r + route_r),
            )
        )
    return adjusted


def build_port_table(port: str, now_utc: datetime | None = None, *, routing_mode: str = "STANDARD") -> Table:
    now_utc = now_utc or datetime.now(UTC)
    rows = _shipments_for_port(port, now_utc, routing_mode=routing_mode)

    table = Table(
        title=f"{port.upper()}  |  SHIPPING BLotter",
        title_style="bold bright_white",
        header_style="bold bright_white",
        border_style="bright_black",
        show_lines=False,
        expand=True,
    )
    table.add_column("Vessel", style="bright_white", no_wrap=True)
    table.add_column("Flag", style="bright_black", width=4, justify="center")
    table.add_column("Cargo", style="white")
    table.add_column("Dest.", style="white", no_wrap=True)
    table.add_column("ETA (UTC)", style="bright_black", no_wrap=True)
    table.add_column("Status", style="white", no_wrap=True)
    table.add_column("Risk", style="white", justify="right", no_wrap=True, width=6)

    for r in rows:
        eta = r.eta_utc.strftime("%b %d %H:%M")
        risk_text = Text(str(r.risk).rjust(3), style=_risk_style(r.risk))
        status_text = Text(r.status, style=_status_style(r.status))
        table.add_row(r.vessel, r.flag, r.cargo, r.destination, eta, status_text, risk_text)

    return table


def trieste_table(now_utc: datetime | None = None, *, routing_mode: str = "STANDARD") -> Table:
    return build_port_table("Trieste", now_utc=now_utc, routing_mode=routing_mode)


def fos_sur_mer_table(now_utc: datetime | None = None, *, routing_mode: str = "STANDARD") -> Table:
    return build_port_table("Fos-sur-Mer", now_utc=now_utc, routing_mode=routing_mode)


def rotterdam_table(now_utc: datetime | None = None, *, routing_mode: str = "STANDARD") -> Table:
    return build_port_table("Rotterdam", now_utc=now_utc, routing_mode=routing_mode)


def all_ports(now_utc: datetime | None = None) -> Iterable[Table]:
    now_utc = now_utc or datetime.now(UTC)
    yield trieste_table(now_utc)
    yield fos_sur_mer_table(now_utc)
    yield rotterdam_table(now_utc)
