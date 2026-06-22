"""The full-run async driver (plan 3).

Replaces the plan-1 stub. Drives `(question, model)` pairs in parallel via
`anyio` task groups, with:

- Adaptive concurrency warm-up (4 → max_concurrent).
- Per-model in-flight cap that reacts to 429 storms.
- Per-model cost cap (kill switch).
- Resumable: rerun picks up at `db.pending_pairs(...)`.
- Live `rich` dashboard with running cost forecast and per-model progress.

CLI:

    python -m experiments.runner            # run
    python -m experiments.runner --plan     # dry head-count + cost forecast
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import anyio
from rich.console import Console
from rich.table import Table

from . import prompt as prompt_module
from .client import ChatRequest, ChatResponse, DeepInfraClient
from .config import (
    DEEPINFRA_BASE_URL,
    SQLITE_PATH,
    ModelSpec,
    get_deepinfra_api_key,
    load_models,
)
from .cost import CostTracker, compute_cost_usd
from .dashboard import DashboardState, LiveDashboard, ModelProgress, make_console
from .datasets import Question, load_all_questions, questions_by_id
from .db import (
    ResultRow,
    finish_run,
    open_db,
    pending_pairs,
    start_run,
    summary_counts,
    upsert_result,
)
from .forecast import model_weighted_forecast, update_forecast


DISABLED_MODEL_NAMES: frozenset[str] = frozenset({
    # Tested for format compatibility, but disabled to keep the SOTA
    # comparison near the roughly $10 budget.
    "deepseek-ai/DeepSeek-V4-Pro",
    "moonshotai/Kimi-K2.6",
})


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    max_concurrent: int = 32
    max_attempts_per_pair: int = 3
    per_request_max_tokens: int = 10_000
    temperature: float = 0.0
    dataset_filter: tuple[str, ...] | None = None
    model_filter: tuple[str, ...] | None = None
    limit: int | None = None
    per_model_cost_cap_usd: float = 15.0
    warmup_pairs: int = 0
    no_warmup: bool = True
    plan_only: bool = False


@dataclass
class RunReport:
    run_id: str
    started_at: str
    finished_at: str
    n_pairs_total: int
    n_pairs_done_this_run: int
    total_cost_usd_this_run: float
    cumulative_cost_usd: float
    per_model: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal: per-model concurrency / 429 tracking
# ---------------------------------------------------------------------------


@dataclass
class _ModelGate:
    """Per-model semaphore + 429 history for adaptive throttling."""

    name: str
    sem: anyio.Semaphore
    cap: int
    base_cap: int
    last_throttled_at: float = 0.0
    rate429_window: deque = field(default_factory=lambda: deque(maxlen=8))
    cost_capped: bool = False
    unavailable: bool = False


class _RunState:
    def __init__(
        self,
        cfg: RunConfig,
        models: list[ModelSpec],
        run_id: str,
        n_pairs_total: int,
    ) -> None:
        self.cfg = cfg
        self.run_id = run_id
        self.models_by_name: dict[str, ModelSpec] = {m.name: m for m in models}

        # Effective concurrency. Skip the warm-up ramp by default — the
        # smoke runs already validated `max_concurrent=32` without 429s,
        # and ramping wastes the first ~5 minutes on slow reasoning
        # models. Set --warmup-pairs > 0 to re-enable the ramp.
        if cfg.no_warmup or cfg.warmup_pairs <= 0:
            self.effective_concurrency = max(1, cfg.max_concurrent)
        else:
            self.effective_concurrency = max(1, min(8, cfg.max_concurrent))
        self.global_sem = anyio.Semaphore(self.effective_concurrency)

        # Per-model gates. Each model gets the full concurrency budget;
        # the adaptive throttle below shrinks individual gates to 4 only
        # in response to a 429 storm. With many models in the fleet this
        # would let them collectively oversubscribe the global cap, but
        # `state.global_sem` already enforces the ceiling, so the gates
        # only matter as a per-model brake under throttling.
        self.gates: dict[str, _ModelGate] = {}
        for m in models:
            cap = cfg.max_concurrent
            self.gates[m.name] = _ModelGate(
                name=m.name,
                sem=anyio.Semaphore(cap),
                cap=cap,
                base_cap=cfg.max_concurrent,
            )

        self.tracker = CostTracker()
        self.per_model_cost: dict[str, float] = defaultdict(float)
        self.dashboard_state = DashboardState(
            run_id=run_id,
            started_at_iso=datetime.now(timezone.utc).isoformat(),
            started_at_monotonic=time.monotonic(),
            dataset_filter=",".join(cfg.dataset_filter) if cfg.dataset_filter else None,
            model_filter=",".join(cfg.model_filter) if cfg.model_filter else None,
            n_pairs_total=n_pairs_total,
            current_concurrency=self.effective_concurrency,
        )
        self.sum_latency_ms: float = 0.0
        self.lock = anyio.Lock()
        self.warmup_done = False
        self.warmup_last_step_at = time.monotonic()
        self.cancelled = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_full(config: RunConfig) -> RunReport:
    console = make_console()
    started_iso = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()

    console.print("[bold]Plan-3 full run[/bold]")
    console.print("Loading models from models.json...")
    all_models = [m for m in load_models() if m.name not in DISABLED_MODEL_NAMES]
    if config.model_filter:
        keep = set(config.model_filter)
        models = [m for m in all_models if m.name in keep]
        missing = keep - {m.name for m in all_models}
        if missing:
            raise RuntimeError(
                f"--model-filter referenced unknown models: {sorted(missing)}"
            )
    else:
        models = list(all_models)
    console.print(f"  {len(models)} models in scope.")

    console.print("Loading questions (ENEM + BLUEX)...")
    all_questions = load_all_questions()
    if config.dataset_filter:
        keep_ds = set(config.dataset_filter)
        questions = [q for q in all_questions if q.dataset in keep_ds]
    else:
        questions = list(all_questions)
    console.print(f"  {len(questions)} questions in scope.")

    api_key = get_deepinfra_api_key()
    client = DeepInfraClient(
        api_key=api_key,
        base_url=DEEPINFRA_BASE_URL,
        max_concurrent=max(config.max_concurrent, 4),
        timeout_s=300.0,
    )

    try:
        await _check_catalog(client, models, console)

        conn = open_db(SQLITE_PATH)
        try:
            pending = pending_pairs(conn, questions, [m.name for m in models])
            if config.limit is not None:
                pending = pending[: config.limit]

            if not pending:
                console.print("[bold green]Nothing to do — pending list is empty.[/bold green]")
                _print_summary(conn, console)
                return _empty_report(started_iso, n_total=0)

            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            console.print(f"Run ID: [bold]{run_id}[/bold]")
            console.print(f"Pending pairs: {len(pending)}")

            if config.plan_only:
                _print_plan(console, conn, pending, models, config)
                return _empty_report(started_iso, n_total=len(pending))

            start_run(
                conn,
                run_id,
                dataset_filter=",".join(config.dataset_filter) if config.dataset_filter else None,
                model_filter=",".join(config.model_filter) if config.model_filter else None,
                n_pairs_total=len(pending),
            )

            qs_by_id = questions_by_id(questions)
            state = _RunState(config, models, run_id, n_pairs_total=len(pending))

            # Per-model totals for the dashboard.
            per_model_totals: dict[str, int] = defaultdict(int)
            for _qid, mname in pending:
                per_model_totals[mname] += 1
            for m in models:
                state.dashboard_state.per_model[m.name] = ModelProgress(
                    name=m.name,
                    short=m.short,
                    n_total=per_model_totals.get(m.name, 0),
                    cost_cap_usd=config.per_model_cost_cap_usd,
                    in_flight_cap=state.gates[m.name].cap,
                )

            # Order pairs to balance per-model cadence: round-robin by model
            # within each dataset block (keeps the dashboard moving evenly).
            ordered_pairs = _round_robin_by_model(pending)

            cancelled_exc_class = anyio.get_cancelled_exc_class()
            with LiveDashboard(console) as dash:
                try:
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_warmup_and_dashboard_loop, state, dash, conn)
                        for qid, mname in ordered_pairs:
                            q = qs_by_id.get(qid)
                            if q is None:
                                continue
                            mspec = state.models_by_name.get(mname)
                            if mspec is None:
                                continue
                            tg.start_soon(
                                _process_pair,
                                state,
                                client,
                                conn,
                                q,
                                mspec,
                                dash,
                            )
                except (KeyboardInterrupt, cancelled_exc_class):
                    state.cancelled = True
                    console.print("[yellow]Cancelled by user; results so far are committed.[/yellow]")
                    raise
                finally:
                    snap = state.tracker.snapshot()
                    finish_run(
                        conn,
                        run_id,
                        n_pairs_done=state.dashboard_state.n_pairs_done,
                        total_cost_usd=float(snap["total_cost_usd"]),
                    )

                fc = update_forecast(
                    state.tracker,
                    state.dashboard_state.n_pairs_done,
                    state.dashboard_state.n_pairs_total,
                    state.dashboard_state.started_at_monotonic,
                    sum_latency_ms=state.sum_latency_ms,
                )
                dash.print_summary(state.dashboard_state, fc)

            cumulative = _cumulative_cost(conn)
            return RunReport(
                run_id=run_id,
                started_at=started_iso,
                finished_at=datetime.now(timezone.utc).isoformat(),
                n_pairs_total=len(pending),
                n_pairs_done_this_run=state.dashboard_state.n_pairs_done,
                total_cost_usd_this_run=float(state.tracker.snapshot()["total_cost_usd"]),
                cumulative_cost_usd=cumulative,
                per_model={
                    name: {
                        "n_total": mp.n_total,
                        "n_done": mp.n_done,
                        "n_correct": mp.n_correct,
                        "n_parsed": mp.n_parsed,
                        "cost_usd": mp.cost_usd,
                        "state": mp.state,
                    }
                    for name, mp in state.dashboard_state.per_model.items()
                },
            )
        finally:
            conn.close()
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round_robin_by_model(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Interleave pairs by model so per-model progress moves in lock-step."""
    by_model: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for qid, m in pairs:
        by_model[m].append((qid, m))
    out: list[tuple[str, str]] = []
    queues = [iter(v) for v in by_model.values()]
    while queues:
        next_round: list = []
        for it in queues:
            try:
                out.append(next(it))
                next_round.append(it)
            except StopIteration:
                continue
        queues = next_round
    return out


async def _check_catalog(
    client: DeepInfraClient,
    models: list[ModelSpec],
    console: Console,
) -> None:
    console.print("Checking DeepInfra catalog...")
    catalog = await client.list_models()
    if not catalog:
        console.print(
            "[yellow]WARN[/yellow]: could not fetch /models from DeepInfra. "
            "Continuing; per-model 4xx responses will be handled at runtime."
        )
        return
    catalog_set = set(catalog)
    absent = [m.name for m in models if m.name not in catalog_set]
    if absent:
        console.print("[red]ERROR[/red]: the following slugs are not served by DeepInfra:")
        for name in absent:
            console.print(f"  - {name}")
        console.print(
            "Edit models.json to remove these entries (or correct the slug) "
            "before re-running."
        )
        raise SystemExit(2)
    console.print(f"  catalog OK: {len(models)} models present.")


async def _warmup_and_dashboard_loop(
    state: _RunState,
    dash: LiveDashboard,
    conn,
) -> None:
    """Background loop: refresh the dashboard ~1Hz and bump warm-up tier."""
    try:
        while True:
            fc = update_forecast(
                state.tracker,
                state.dashboard_state.n_pairs_done,
                state.dashboard_state.n_pairs_total,
                state.dashboard_state.started_at_monotonic,
                sum_latency_ms=state.sum_latency_ms,
            )
            dash.update(state.dashboard_state, fc)

            # Concurrency warm-up: bump every 30s with no recent 429.
            await _maybe_bump_concurrency(state)

            # Per-model adaptive cap restoration: 2-min quiet -> restore.
            now = time.monotonic()
            for gate in state.gates.values():
                if gate.cost_capped or gate.unavailable:
                    continue
                if (
                    gate.cap < state.effective_concurrency
                    and now - gate.last_throttled_at > 120.0
                ):
                    gate.cap = state.effective_concurrency
                    gate.sem = anyio.Semaphore(gate.cap)
                    state.dashboard_state.per_model[gate.name].in_flight_cap = gate.cap

            if state.dashboard_state.n_pairs_done >= state.dashboard_state.n_pairs_total:
                return
            await anyio.sleep(1.0)
    except anyio.get_cancelled_exc_class():
        return


async def _maybe_bump_concurrency(state: _RunState) -> None:
    cfg = state.cfg
    if cfg.no_warmup or cfg.warmup_pairs <= 0:
        state.warmup_done = True
        return
    if state.effective_concurrency >= cfg.max_concurrent:
        state.warmup_done = True
        return
    if state.dashboard_state.n_pairs_done < cfg.warmup_pairs:
        return
    now = time.monotonic()
    if now - state.warmup_last_step_at < 10.0:
        return
    # Don't bump during a recent 429 storm (any model throttled in last 60s).
    recent_throttle = any(
        now - g.last_throttled_at < 60.0 for g in state.gates.values()
    )
    if recent_throttle:
        return
    new_eff = min(state.effective_concurrency * 2, cfg.max_concurrent)
    if new_eff == state.effective_concurrency:
        return
    state.effective_concurrency = new_eff
    state.warmup_last_step_at = now
    state.global_sem = anyio.Semaphore(new_eff)
    state.dashboard_state.current_concurrency = new_eff
    for gate in state.gates.values():
        if gate.cost_capped or gate.unavailable:
            continue
        if gate.cap < new_eff:
            gate.cap = new_eff
            gate.sem = anyio.Semaphore(gate.cap)
            state.dashboard_state.per_model[gate.name].in_flight_cap = gate.cap


def _is_provider_rejection_error(err: str) -> bool:
    return ('"code":"InvalidParameter' in err) or (
        '"code":"SensitiveContentDetected' in err
    )


def _is_unavailable_error(http_status: int, err: str) -> bool:
    if http_status in (400, 401, 404):
        return True
    return False


async def _process_pair(
    state: _RunState,
    client: DeepInfraClient,
    conn,
    question: Question,
    model: ModelSpec,
    dash: LiveDashboard,
) -> None:
    gate = state.gates[model.name]
    if gate.cost_capped:
        # Mark this pair as "cost_cap_reached" once and increment dashboard.
        _mark_terminal(
            conn,
            state,
            question,
            model,
            parse_status="cost_cap_reached",
            raw_response="per-model cost cap reached before submission",
        )
        return
    if gate.unavailable:
        _mark_terminal(
            conn,
            state,
            question,
            model,
            parse_status="model_unavailable",
            raw_response="model marked unavailable earlier in this run",
        )
        return

    cfg = state.cfg
    messages = prompt_module.build_messages(question)
    allowed = prompt_module.allowed_values(question)

    last_resp: Optional[ChatResponse] = None
    last_parsed: Optional[prompt_module.ParseResult] = None
    attempts = 0
    terminal_status: Optional[str] = None

    for attempt_i in range(1, cfg.max_attempts_per_pair + 1):
        attempts = attempt_i
        async with state.global_sem, gate.sem:
            req = ChatRequest(
                model=model.name,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.per_request_max_tokens,
            )
            resp = await client.chat(req)
        last_resp = resp

        # 429 bookkeeping for adaptive throttling.
        if resp.http_status == 429 or (
            resp.error and "http_429" in (resp.error or "")
        ):
            now = time.monotonic()
            gate.rate429_window.append(now)
            recent = [t for t in gate.rate429_window if now - t < 30.0]
            if len(recent) >= 3 and gate.cap > 4:
                gate.cap = 4
                gate.sem = anyio.Semaphore(4)
                gate.last_throttled_at = now
                state.dashboard_state.per_model[model.name].in_flight_cap = 4

        # Detect terminal failure modes from a single response.
        if resp.error:
            if _is_provider_rejection_error(resp.error):
                terminal_status = "provider_rejected"
                break
            if _is_unavailable_error(resp.http_status, resp.error):
                terminal_status = "model_unavailable"
                gate.unavailable = True
                state.dashboard_state.per_model[model.name].state = "model_unavailable"
                break

        parsed = prompt_module.parse_response(resp.raw_text or "", allowed)
        last_parsed = parsed
        if (
            parsed.status in {"ok", "truncated_but_answered"}
            and parsed.answer is not None
        ):
            break
        if attempt_i < cfg.max_attempts_per_pair:
            await anyio.sleep(1.5 * attempt_i)

    assert last_resp is not None
    pair_cost = compute_cost_usd(
        model,
        last_resp.prompt_tokens or 0,
        last_resp.completion_tokens or 0,
    )
    state.tracker.record(last_resp, model)

    # Per-model cost cap kill switch.
    state.per_model_cost[model.name] += pair_cost
    if (
        not gate.cost_capped
        and state.per_model_cost[model.name] >= cfg.per_model_cost_cap_usd
    ):
        gate.cost_capped = True
        state.dashboard_state.per_model[model.name].state = "cost_cap_reached"

    is_correct: Optional[bool] = None
    if terminal_status is not None:
        parse_status = terminal_status
        parsed_answer = None
        raw_response = (
            (last_resp.error or "") if last_resp.error else (last_resp.raw_text or "")
        )
    else:
        assert last_parsed is not None
        parse_status = last_parsed.status
        parsed_answer = last_parsed.answer
        if last_resp.text_source == "reasoning_content":
            parse_status = f"{parse_status}_from_reasoning_content"
        raw_response = last_resp.raw_text or ""
        if parsed_answer is not None:
            is_correct = parsed_answer == question.correct_answer

    row = ResultRow(
        dataset=question.dataset,
        question_id=question.question_id,
        model=model.name,
        year=question.year,
        subject=question.subject,
        alternatives_type=question.alternatives_type,
        has_images=question.has_images,
        images_in_alt=question.images_in_alternatives,
        correct_answer=question.correct_answer,
        parsed_answer=parsed_answer,
        is_correct=is_correct,
        parse_status=parse_status,
        raw_response=raw_response,
        prompt_tokens=last_resp.prompt_tokens,
        completion_tokens=last_resp.completion_tokens,
        cost_usd=pair_cost,
        latency_ms=last_resp.latency_ms,
        attempts=attempts,
        finish_reason=last_resp.finish_reason,
        run_id=state.run_id,
        max_tokens=cfg.per_request_max_tokens,
    )

    async with state.lock:
        upsert_result(conn, row)
        state.sum_latency_ms += last_resp.latency_ms or 0
        mp = state.dashboard_state.per_model[model.name]
        mp.n_done += 1
        mp.cost_usd = state.per_model_cost[model.name]
        if parsed_answer is not None:
            mp.n_parsed += 1
        if is_correct:
            mp.n_correct += 1
        state.dashboard_state.n_pairs_done += 1
        if last_resp.error:
            state.dashboard_state.n_http_errors += 1
        if parsed_answer is None and terminal_status is None:
            state.dashboard_state.n_parse_failures += 1


def _mark_terminal(
    conn,
    state: _RunState,
    question: Question,
    model: ModelSpec,
    *,
    parse_status: str,
    raw_response: str,
) -> None:
    row = ResultRow(
        dataset=question.dataset,
        question_id=question.question_id,
        model=model.name,
        year=question.year,
        subject=question.subject,
        alternatives_type=question.alternatives_type,
        has_images=question.has_images,
        images_in_alt=question.images_in_alternatives,
        correct_answer=question.correct_answer,
        parsed_answer=None,
        is_correct=None,
        parse_status=parse_status,
        raw_response=raw_response,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=0.0,
        latency_ms=None,
        attempts=0,
        finish_reason=None,
        run_id=state.run_id,
        max_tokens=state.cfg.per_request_max_tokens,
    )
    upsert_result(conn, row)
    mp = state.dashboard_state.per_model[model.name]
    mp.n_done += 1
    mp.state = parse_status if parse_status in {"cost_cap_reached", "model_unavailable"} else mp.state
    state.dashboard_state.n_pairs_done += 1


def _print_plan(
    console: Console,
    conn,
    pending: list[tuple[str, str]],
    models: list[ModelSpec],
    config: RunConfig,
) -> None:
    """Compute and print the model-weighted cost forecast (no API calls)."""
    pending_per_model: dict[str, int] = defaultdict(int)
    for _qid, mname in pending:
        pending_per_model[mname] += 1

    # Smoke-test mean tokens (read from existing results rows).
    cur = conn.execute(
        """
        SELECT AVG(prompt_tokens), AVG(completion_tokens), COUNT(*)
        FROM results
        WHERE prompt_tokens IS NOT NULL AND completion_tokens IS NOT NULL
        """
    )
    mean_prompt, mean_completion, n_smoke = cur.fetchone()
    if not n_smoke or mean_prompt is None:
        # Fallback to plan-3 §2 envelope.
        mean_prompt, mean_completion = 800.0, 400.0
        console.print(
            "[yellow]No smoke rows found; using fallback (prompt=800, "
            "completion=400) for the forecast.[/yellow]"
        )
    else:
        console.print(
            f"Forecast basis (from {int(n_smoke)} smoke rows): "
            f"mean_prompt={mean_prompt:.0f}  mean_completion={mean_completion:.0f}"
        )

    prices = {
        m.name: (m.cost_per_m_input_usd, m.cost_per_m_output_usd) for m in models
    }
    forecast = model_weighted_forecast(
        pending_per_model=pending_per_model,
        model_prices=prices,
        mean_prompt_tokens=float(mean_prompt),
        mean_completion_tokens=float(mean_completion),
    )

    table = Table(title="Cost forecast per model (model-weighted)")
    table.add_column("model")
    table.add_column("pending", justify="right")
    table.add_column("est. cost (USD)", justify="right")
    table.add_column("cost cap (USD)", justify="right")
    table.add_column("warning")
    by_name = {m.name: m for m in models}
    over_cap_warnings: list[str] = []
    for name in sorted(pending_per_model.keys()):
        n = pending_per_model[name]
        est = forecast.get(name, 0.0)
        cap = config.per_model_cost_cap_usd
        warn = ""
        if est > 0.8 * cap:
            warn = "[red]>80% of cap[/red]"
            over_cap_warnings.append(name)
        table.add_row(
            by_name[name].short if name in by_name else name,
            str(n),
            f"${est:.4f}",
            f"${cap:.2f}",
            warn,
        )
    console.print(table)
    console.print(
        f"\n[bold]Total estimated cost: ${forecast['__total__']:.4f}[/bold]   "
        f"Pairs pending: {len(pending)}   Models: {len(pending_per_model)}"
    )
    if over_cap_warnings:
        console.print(
            "[yellow]Warning:[/yellow] forecast exceeds 80% of the per-model cost cap "
            f"for: {', '.join(over_cap_warnings)}. Bump --per-model-cost-cap or accept "
            "that those models will be cost-capped mid-run."
        )
    console.print(
        "\nRe-run without --plan to actually start the run.",
    )


def _print_summary(conn, console: Console) -> None:
    summary = summary_counts(conn)
    if not summary:
        return
    table = Table(title="DB summary (results table)")
    table.add_column("dataset")
    table.add_column("model")
    table.add_column("n_done", justify="right")
    table.add_column("n_parsed", justify="right")
    table.add_column("n_correct", justify="right")
    table.add_column("cost_usd", justify="right")
    for (dataset, model), s in sorted(summary.items()):
        table.add_row(
            dataset,
            model,
            str(s["n_done"]),
            str(s["n_parsed"]),
            str(s["n_correct"]),
            f"${s['sum_cost_usd']:.4f}",
        )
    console.print(table)


def _cumulative_cost(conn) -> float:
    cur = conn.execute("SELECT COALESCE(SUM(cost_usd), 0.0) FROM results")
    return float(cur.fetchone()[0])


def _empty_report(started_iso: str, *, n_total: int) -> RunReport:
    now = datetime.now(timezone.utc).isoformat()
    return RunReport(
        run_id="",
        started_at=started_iso,
        finished_at=now,
        n_pairs_total=n_total,
        n_pairs_done_this_run=0,
        total_cost_usd_this_run=0.0,
        cumulative_cost_usd=0.0,
        per_model={},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(prog="experiments.runner")
    p.add_argument("--plan", action="store_true", help="dry-run head count + cost forecast")
    p.add_argument("--max-concurrent", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=10_000)
    p.add_argument("--per-model-cost-cap", type=float, default=15.0)
    p.add_argument(
        "--warmup-pairs",
        type=int,
        default=0,
        help="if >0, ramp concurrency 8 -> max_concurrent after this many "
        "pairs (10 s steps, doubling); default 0 disables the ramp",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        default=True,
        help="(default) start at max_concurrent immediately",
    )
    p.add_argument(
        "--warmup",
        dest="no_warmup",
        action="store_false",
        help="opt back into the warm-up ramp; pair with --warmup-pairs",
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="restrict to one dataset (ENEM | FUVEST | UNICAMP); repeatable",
    )
    p.add_argument(
        "--model",
        action="append",
        default=None,
        help="restrict to one model name (full DeepInfra slug); repeatable",
    )
    args = p.parse_args(argv)
    return RunConfig(
        max_concurrent=args.max_concurrent,
        per_request_max_tokens=args.max_tokens,
        dataset_filter=tuple(args.dataset) if args.dataset else None,
        model_filter=tuple(args.model) if args.model else None,
        limit=args.limit,
        per_model_cost_cap_usd=args.per_model_cost_cap,
        warmup_pairs=args.warmup_pairs,
        no_warmup=args.no_warmup,
        plan_only=args.plan,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv)
    try:
        report = asyncio.run(run_full(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted; partial results are committed.", file=sys.stderr)
        return 130
    if cfg.plan_only:
        return 0
    print(
        f"\nRun {report.run_id} done: "
        f"{report.n_pairs_done_this_run}/{report.n_pairs_total} pairs, "
        f"${report.total_cost_usd_this_run:.4f} this run, "
        f"${report.cumulative_cost_usd:.4f} cumulative."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
