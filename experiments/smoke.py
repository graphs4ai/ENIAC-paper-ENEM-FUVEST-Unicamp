"""10-question smoke test driver for plan 2.

Run with:

    python -m experiments.smoke

It selects 10 hand-picked questions (deterministic) and 3 models, hits
DeepInfra sequentially, prints full prompt/response artifacts, writes
results to the shared SQLite DB, and appends a summary table.
"""

from __future__ import annotations

import asyncio
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anyio
from rich.console import Console
from rich.table import Table

from . import prompt as prompt_module
from ._samples import pick_smoke_questions
from .client import ChatRequest, ChatResponse, DeepInfraClient
from .config import (
    DEEPINFRA_BASE_URL,
    REPO_ROOT,
    SQLITE_PATH,
    ModelSpec,
    get_deepinfra_api_key,
    load_models,
)
from .cost import CostTracker, compute_cost_usd
from .datasets import Question, load_all_questions
from .db import ResultRow, open_db, upsert_result


SMOKE_MODEL_NAMES: tuple[str, ...] = (
    "google/gemma-3-4b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3.5-4B",
)


class Tee(io.TextIOBase):
    """Duplicate stdout writes into a log file."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log = log_file

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._log.write(s)
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        self._stream.flush()
        self._log.flush()


def _select_models(all_models: list[ModelSpec]) -> list[ModelSpec]:
    by_name = {m.name: m for m in all_models}
    chosen: list[ModelSpec] = []
    missing: list[str] = []
    for name in SMOKE_MODEL_NAMES:
        if name in by_name:
            chosen.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Smoke test requires these models in models.json but they are "
            f"missing: {missing}"
        )
    return chosen


async def _check_catalog(
    client: DeepInfraClient,
    all_models: list[ModelSpec],
    smoke_models: list[ModelSpec],
    console: Console,
) -> None:
    console.print("\n[bold]Checking DeepInfra catalog...[/bold]")
    catalog = await client.list_models()
    if not catalog:
        console.print(
            "[yellow]WARN:[/yellow] could not fetch /models from DeepInfra. "
            "Skipping catalog check (proceeding with the smoke models anyway)."
        )
        return
    catalog_set = set(catalog)
    present: list[str] = []
    absent: list[str] = []
    for m in all_models:
        (present if m.name in catalog_set else absent).append(m.name)
    console.print(f"  present in catalog: {len(present)} / {len(all_models)}")
    if absent:
        console.print("  [yellow]absent from catalog (slug mismatch or unreleased):[/yellow]")
        for name in absent:
            console.print(f"    - {name}")
    smoke_absent = [m.name for m in smoke_models if m.name not in catalog_set]
    if smoke_absent:
        console.print(
            f"[red]ERROR:[/red] smoke models missing from DeepInfra: {smoke_absent}"
        )
        raise SystemExit(2)


def _format_attempts(status: str, attempts: int, max_attempts: int) -> str:
    return f"{attempts} / {max_attempts}   (status={status})"


async def _run_one_pair(
    client: DeepInfraClient,
    question: Question,
    model: ModelSpec,
    *,
    max_attempts: int = 3,
    max_tokens: int = 10000,
) -> tuple[ChatResponse, prompt_module.ParseResult, int]:
    messages = prompt_module.build_messages(question)
    allowed = prompt_module.allowed_values(question)
    last_resp: Optional[ChatResponse] = None
    last_parsed: Optional[prompt_module.ParseResult] = None
    attempts = 0
    for attempt_i in range(1, max_attempts + 1):
        attempts = attempt_i
        req = ChatRequest(
            model=model.name,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        resp = await client.chat(req)
        last_resp = resp
        parsed = prompt_module.parse_response(resp.raw_text, allowed)
        last_parsed = parsed
        if (
            parsed.status in {"ok", "truncated_but_answered"}
            and parsed.answer is not None
        ):
            break
        if attempt_i < max_attempts:
            await anyio.sleep(1.5 * attempt_i)

    assert last_resp is not None and last_parsed is not None
    return last_resp, last_parsed, attempts


def _print_pair(
    console: Console,
    idx: int,
    total: int,
    question: Question,
    model: ModelSpec,
    resp: ChatResponse,
    parsed: prompt_module.ParseResult,
    attempts: int,
    max_attempts: int,
    pair_cost: float,
    running_cost: float,
    messages: list[dict],
) -> None:
    console.print("\n───────────────────────────────────────────────")
    is_correct: Optional[bool] = None
    if parsed.answer is not None:
        is_correct = parsed.answer == question.correct_answer
    console.print(
        f"[{idx}/{total}]  question_id = {question.question_id}\n"
        f"        model       = {model.name}\n"
        f"        has_images  = {question.has_images}\n"
        f"        alt. images = {question.images_in_alternatives}\n"
        f"        attempts    = {_format_attempts(parsed.status, attempts, max_attempts)}\n"
        f"        finish      = {resp.finish_reason}   "
        f"text_source={resp.text_source}   "
        f"prompt_tok={resp.prompt_tokens}  completion_tok={resp.completion_tokens}\n"
        f"        cost        = ${pair_cost:.6f}   (running: ${running_cost:.6f})"
    )
    user_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"), ""
    )
    console.print("\n--- PROMPT (user) ---")
    console.print(user_content, highlight=False, soft_wrap=True)
    console.print("\n--- RAW RESPONSE ---")
    console.print("```")
    console.print(resp.raw_text or "<empty>", highlight=False, soft_wrap=True)
    console.print("```")
    if resp.error:
        console.print(f"[red]HTTP error:[/red] {resp.error} (status={resp.http_status})")
    console.print("\n--- PARSED ---")
    console.print(f"chosen alternative : {parsed.answer}")
    console.print(f"correct alternative: {question.correct_answer}")
    console.print(
        "is_correct         : "
        + ("TRUE" if is_correct else "FALSE" if is_correct is False else "N/A")
    )


def _render_summary(
    console: Console,
    per_model_stats: dict[str, dict],
) -> None:
    table = Table(title="Smoke-test summary", show_lines=False)
    table.add_column("model")
    table.add_column("n", justify="right")
    table.add_column("n_ok", justify="right")
    table.add_column("n_truncated", justify="right")
    table.add_column("n_json_error", justify="right")
    table.add_column("n_correct", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("total_cost_usd", justify="right")
    table.add_column("mean_latency_ms", justify="right")
    for model_name, s in per_model_stats.items():
        n = s["n"]
        mean_lat = (s["sum_latency_ms"] / n) if n else 0.0
        acc = (s["n_correct"] / n) if n else 0.0
        table.add_row(
            model_name,
            str(n),
            str(s["n_ok"]),
            str(s["n_truncated"]),
            str(s["n_json_error"]),
            str(s["n_correct"]),
            f"{acc:.1%}",
            f"${s['total_cost_usd']:.6f}",
            f"{mean_lat:.0f}",
        )
    console.print("\n")
    console.print(table)


async def _async_main(console: Console) -> int:
    console.print("[bold]Plan-2 smoke test[/bold]: 10 questions × 3 models")

    console.print("Loading models from models.json...")
    all_models = load_models()
    smoke_models = _select_models(all_models)
    console.print(f"  {len(all_models)} total models; {len(smoke_models)} will run.")
    for m in smoke_models:
        console.print(f"    - {m.name}")

    console.print("\nLoading questions (ENEM + BLUEX)...")
    all_questions = load_all_questions()
    console.print(f"  loaded {len(all_questions)} questions total.")

    console.print("\nPicking 10 smoke questions (seed=42)...")
    smoke_questions = pick_smoke_questions(all_questions)
    for i, q in enumerate(smoke_questions, start=1):
        console.print(
            f"  [{i}] {q.question_id}  dataset={q.dataset}  year={q.year}  "
            f"subject={q.subject}  has_images={q.has_images}  "
            f"alt_images={q.images_in_alternatives}  type={q.alternatives_type}"
        )

    api_key = get_deepinfra_api_key()
    client = DeepInfraClient(
        api_key=api_key,
        base_url=DEEPINFRA_BASE_URL,
        max_concurrent=4,
        timeout_s=120.0,
    )

    try:
        await _check_catalog(client, all_models, smoke_models, console)

        conn = open_db(SQLITE_PATH)
        tracker = CostTracker()
        per_model_stats: dict[str, dict] = {
            m.name: {
                "n": 0,
                "n_ok": 0,
                "n_truncated": 0,
                "n_json_error": 0,
                "n_correct": 0,
                "total_cost_usd": 0.0,
                "sum_latency_ms": 0.0,
            }
            for m in smoke_models
        }

        pairs: list[tuple[Question, ModelSpec]] = [
            (q, m) for q in smoke_questions for m in smoke_models
        ]
        total_pairs = len(pairs)
        max_attempts = 3
        max_tokens = 10000
        all_parsed_ok = True

        for idx, (q, m) in enumerate(pairs, start=1):
            messages = prompt_module.build_messages(q)
            resp, parsed, attempts = await _run_one_pair(
                client, q, m, max_attempts=max_attempts, max_tokens=max_tokens
            )
            pair_cost = compute_cost_usd(
                m,
                resp.prompt_tokens or 0,
                resp.completion_tokens or 0,
            )
            tracker.record(resp, m)
            running_cost = tracker.snapshot()["total_cost_usd"]

            _print_pair(
                console,
                idx,
                total_pairs,
                q,
                m,
                resp,
                parsed,
                attempts,
                max_attempts,
                pair_cost,
                running_cost,
                messages,
            )

            is_correct: Optional[bool] = None
            if parsed.answer is not None:
                is_correct = parsed.answer == q.correct_answer
            parse_status = parsed.status
            if resp.text_source == "reasoning_content":
                parse_status = f"{parse_status}_from_reasoning_content"

            row = ResultRow(
                dataset=q.dataset,
                question_id=q.question_id,
                model=m.name,
                year=q.year,
                subject=q.subject,
                alternatives_type=q.alternatives_type,
                has_images=q.has_images,
                images_in_alt=q.images_in_alternatives,
                correct_answer=q.correct_answer,
                parsed_answer=parsed.answer,
                is_correct=is_correct,
                parse_status=parse_status,
                raw_response=resp.raw_text or "",
                prompt_tokens=resp.prompt_tokens,
                completion_tokens=resp.completion_tokens,
                cost_usd=pair_cost,
                latency_ms=resp.latency_ms,
                attempts=attempts,
            )
            upsert_result(conn, row)

            stats = per_model_stats[m.name]
            stats["n"] += 1
            stats["total_cost_usd"] += pair_cost
            stats["sum_latency_ms"] += resp.latency_ms
            if parsed.status == "ok":
                stats["n_ok"] += 1
            elif parsed.status == "truncated_but_answered":
                stats["n_truncated"] += 1
            elif parsed.status in {"json_error", "empty", "missing_key", "disallowed_value"}:
                stats["n_json_error"] += 1
            if is_correct:
                stats["n_correct"] += 1
            if parsed.answer is None:
                all_parsed_ok = False

        _render_summary(console, per_model_stats)
        snap = tracker.snapshot()
        console.print(
            f"\nTotal calls: {snap['n_calls']}  "
            f"Total cost: ${snap['total_cost_usd']:.6f}  "
            f"prompt_tokens={snap['total_prompt_tokens']}  "
            f"completion_tokens={snap['total_completion_tokens']}"
        )

        conn.close()
        return 0 if all_parsed_ok else 1
    finally:
        await client.aclose()


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir: Path = REPO_ROOT / "experiments" / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"smoke_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")
    try:
        tee = Tee(sys.stdout, log_file)
        console = Console(file=tee, force_terminal=False, width=120)
        console.print(f"Logging to {log_path}")
        return asyncio.run(_async_main(console))
    finally:
        log_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
