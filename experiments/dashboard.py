"""Live `rich.live.Live` terminal dashboard for the runner."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .forecast import Forecast


def _fmt_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "??:??:??"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _fmt_money(amount: float) -> str:
    return f"${amount:,.4f}"


@dataclass
class ModelProgress:
    name: str
    short: str
    n_total: int = 0
    n_done: int = 0
    n_correct: int = 0
    n_parsed: int = 0
    cost_usd: float = 0.0
    cost_cap_usd: float = 0.0
    in_flight_cap: int = 0
    state: str = "running"  # "running" | "cost_cap_reached" | "model_unavailable"


@dataclass
class DashboardState:
    run_id: str
    started_at_iso: str
    started_at_monotonic: float
    dataset_filter: Optional[str]
    model_filter: Optional[str]
    n_pairs_total: int
    n_pairs_done: int = 0
    n_http_errors: int = 0
    n_parse_failures: int = 0
    current_concurrency: int = 0
    per_model: dict[str, ModelProgress] = field(default_factory=dict)


def _render_header(state: DashboardState, fc: Forecast) -> Panel:
    n_models = len(state.per_model)
    line1 = Text(
        f"BRACIS rerun · run {state.run_id} · {n_models} models · "
        f"{state.n_pairs_total} pairs",
        style="bold",
    )
    rate = (fc.pairs_done / fc.elapsed_s) if fc.elapsed_s > 0 else 0.0

    line2 = Text(
        f"Pending: {fc.pairs_pending}   "
        f"Done this run: {fc.pairs_done}   "
        f"Failures: {state.n_http_errors} (HTTP) / "
        f"{state.n_parse_failures} (parse)"
    )
    line3 = Text(
        f"Total cost so far: {_fmt_money(fc.total_cost_so_far)}    "
        f"Projected total: {_fmt_money(fc.projected_total_cost)}    "
        f"Mean/call: {_fmt_money(fc.mean_cost_per_pair)}"
    )
    line4 = Text(
        f"Elapsed: {_fmt_seconds(fc.elapsed_s)}    "
        f"ETA: {_fmt_seconds(fc.eta_s)}    "
        f"Rate: {rate:.2f} calls/s    "
        f"Concurrent: {state.current_concurrency}"
    )
    body = Group(line1, line2, line3, line4)
    return Panel(body, title="Run", border_style="cyan")


def _render_models_table(state: DashboardState) -> Panel:
    table = Table(show_lines=False, expand=True)
    table.add_column("model")
    table.add_column("done", justify="right")
    table.add_column("ok", justify="right")
    table.add_column("acc", justify="right")
    table.add_column("cost", justify="right")
    table.add_column("cap", justify="right")
    table.add_column("inflight", justify="right")
    table.add_column("state")
    for mp in state.per_model.values():
        acc = (mp.n_correct / mp.n_parsed * 100.0) if mp.n_parsed else 0.0
        progress = f"{mp.n_done}/{mp.n_total}"
        cap_str = (
            f"{mp.cost_cap_usd:.0f}" if mp.cost_cap_usd > 0 else "-"
        )
        table.add_row(
            mp.short,
            progress,
            str(mp.n_parsed),
            f"{acc:.1f}%",
            _fmt_money(mp.cost_usd),
            cap_str,
            str(mp.in_flight_cap),
            mp.state,
        )
    return Panel(table, title="Per-model progress", border_style="green")


class LiveDashboard:
    """Wraps `rich.live.Live` and falls back to one-line-per-N when not a TTY."""

    def __init__(self, console: Console, refresh_per_second: float = 1.0) -> None:
        self._console = console
        self._is_tty = console.is_terminal
        self._live: Optional[Live] = None
        self._refresh = refresh_per_second
        self._fallback_n = 0
        self._fallback_step = 100

    def __enter__(self):
        if self._is_tty:
            self._live = Live(
                "",
                console=self._console,
                refresh_per_second=self._refresh,
                transient=False,
            )
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)
        return False

    def update(self, state: DashboardState, fc: Forecast) -> None:
        if self._is_tty and self._live is not None:
            renderable = Group(_render_header(state, fc), _render_models_table(state))
            self._live.update(renderable)
            return
        if fc.pairs_done >= self._fallback_n + self._fallback_step or fc.pairs_done == fc.pairs_total:
            self._fallback_n = fc.pairs_done
            self._console.print(
                f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] "
                f"{fc.pairs_done}/{fc.pairs_total}  "
                f"cost={_fmt_money(fc.total_cost_so_far)} "
                f"projected={_fmt_money(fc.projected_total_cost)} "
                f"eta={_fmt_seconds(fc.eta_s)}"
            )

    def print_summary(self, state: DashboardState, fc: Forecast) -> None:
        self._console.print(_render_header(state, fc))
        self._console.print(_render_models_table(state))


def make_console() -> Console:
    """Console that auto-detects TTY for the runner."""
    return Console(file=sys.stdout)
