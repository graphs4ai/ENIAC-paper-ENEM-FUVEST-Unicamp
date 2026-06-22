"""SQLite cache for per-(question, model) results."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .datasets import Question


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS results (
    dataset              TEXT    NOT NULL,
    question_id          TEXT    NOT NULL,
    model                TEXT    NOT NULL,
    year                 INTEGER NOT NULL,
    subject              TEXT    NOT NULL,
    alternatives_type    TEXT    NOT NULL,
    has_images           INTEGER NOT NULL,
    images_in_alt        INTEGER NOT NULL,
    correct_answer       TEXT    NOT NULL,
    parsed_answer        TEXT,
    is_correct           INTEGER,
    parse_status         TEXT    NOT NULL,
    raw_response         TEXT    NOT NULL,
    prompt_tokens        INTEGER,
    completion_tokens    INTEGER,
    cost_usd             REAL    NOT NULL,
    latency_ms           INTEGER,
    attempts             INTEGER NOT NULL,
    finished_at          TEXT    NOT NULL,
    finish_reason        TEXT,
    run_id               TEXT,
    max_tokens           INTEGER,
    PRIMARY KEY (question_id, model)
);

CREATE INDEX IF NOT EXISTS idx_results_model   ON results(model);
CREATE INDEX IF NOT EXISTS idx_results_dataset ON results(dataset);
CREATE INDEX IF NOT EXISTS idx_results_unparsed
    ON results(parsed_answer) WHERE parsed_answer IS NULL;

CREATE TABLE IF NOT EXISTS run_log (
    run_id         TEXT PRIMARY KEY,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    dataset_filter TEXT,
    model_filter   TEXT,
    n_pairs_total  INTEGER,
    n_pairs_done   INTEGER,
    total_cost_usd REAL,
    notes          TEXT
);
"""


@dataclass
class ResultRow:
    dataset: str
    question_id: str
    model: str
    year: int
    subject: tuple[str, ...]
    alternatives_type: str
    has_images: bool
    images_in_alt: bool
    correct_answer: str
    parsed_answer: Optional[str]
    is_correct: Optional[bool]
    parse_status: str
    raw_response: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_usd: float
    latency_ms: Optional[int]
    attempts: int
    finished_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finish_reason: Optional[str] = None
    run_id: Optional[str] = None
    max_tokens: Optional[int] = None


def open_db(path: Path) -> sqlite3.Connection:
    """Open (and if needed create) the SQLite database with WAL + schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA_SQL)
    _migrate_results_columns(conn)
    conn.commit()
    return conn


def _migrate_results_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add the plan-3 columns to the `results` table.

    Old plan-1/plan-2 databases only have the original 19-column schema;
    `CREATE TABLE IF NOT EXISTS` is a no-op for them, so we add the new
    columns explicitly via `ALTER TABLE` if absent.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(results)")}
    additions = [
        ("finish_reason", "TEXT"),
        ("run_id", "TEXT"),
        ("max_tokens", "INTEGER"),
    ]
    for col, ddl_type in additions:
        if col not in existing:
            conn.execute(f"ALTER TABLE results ADD COLUMN {col} {ddl_type}")


def upsert_result(conn: sqlite3.Connection, row: ResultRow) -> None:
    data = asdict(row)
    data["subject"] = json.dumps(list(row.subject), ensure_ascii=False)
    data["has_images"] = 1 if row.has_images else 0
    data["images_in_alt"] = 1 if row.images_in_alt else 0
    data["is_correct"] = (
        None if row.is_correct is None else (1 if row.is_correct else 0)
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO results (
            dataset, question_id, model, year, subject, alternatives_type,
            has_images, images_in_alt, correct_answer, parsed_answer,
            is_correct, parse_status, raw_response, prompt_tokens,
            completion_tokens, cost_usd, latency_ms, attempts, finished_at,
            finish_reason, run_id, max_tokens
        ) VALUES (
            :dataset, :question_id, :model, :year, :subject, :alternatives_type,
            :has_images, :images_in_alt, :correct_answer, :parsed_answer,
            :is_correct, :parse_status, :raw_response, :prompt_tokens,
            :completion_tokens, :cost_usd, :latency_ms, :attempts, :finished_at,
            :finish_reason, :run_id, :max_tokens
        )
        """,
        data,
    )
    conn.commit()


TERMINAL_FAILURE_STATUSES: frozenset[str] = frozenset({
    "model_unavailable",
    "cost_cap_reached",
    "provider_rejected",
})


def pending_pairs(
    conn: sqlite3.Connection,
    questions: list[Question],
    models: list[str],
    batch_size: int = 500,
) -> list[tuple[str, str]]:
    """Return (question_id, model) pairs that need to be run.

    A pair is pending if either no row exists or its `parsed_answer IS NULL`
    AND its `parse_status` is not one of the terminal failure markers
    (``model_unavailable``, ``cost_cap_reached``, ``provider_rejected``)
    that the runner uses to permanently shelve a pair.
    """
    all_pairs = [(q.question_id, m) for q in questions for m in models]
    if not all_pairs:
        return []

    terminal_list = ",".join(["?"] * len(TERMINAL_FAILURE_STATUSES))
    terminal_values = list(TERMINAL_FAILURE_STATUSES)

    pending: list[tuple[str, str]] = []
    for start in range(0, len(all_pairs), batch_size):
        chunk = all_pairs[start : start + batch_size]
        placeholders = ",".join(["(?, ?)"] * len(chunk))
        flat: list[str] = []
        for qid, m in chunk:
            flat.extend([qid, m])
        sql = f"""
            WITH wanted(question_id, model) AS (VALUES {placeholders})
            SELECT w.question_id, w.model
            FROM wanted w
            LEFT JOIN results r
              ON r.question_id = w.question_id AND r.model = w.model
            WHERE r.question_id IS NULL
               OR (r.parsed_answer IS NULL
                   AND r.parse_status NOT IN ({terminal_list}))
        """
        cur = conn.execute(sql, flat + terminal_values)
        pending.extend((qid, m) for qid, m in cur.fetchall())
    return pending


def summary_counts(conn: sqlite3.Connection) -> dict[tuple[str, str], dict]:
    cur = conn.execute(
        """
        SELECT dataset, model,
               COUNT(*)                                        AS n_done,
               SUM(CASE WHEN parsed_answer IS NOT NULL THEN 1 ELSE 0 END) AS n_parsed,
               SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS n_correct,
               COALESCE(SUM(cost_usd), 0.0)                    AS sum_cost_usd
        FROM results
        GROUP BY dataset, model
        """
    )
    out: dict[tuple[str, str], dict] = {}
    for dataset, model, n_done, n_parsed, n_correct, sum_cost in cur.fetchall():
        out[(dataset, model)] = {
            "n_done": int(n_done or 0),
            "n_parsed": int(n_parsed or 0),
            "n_correct": int(n_correct or 0),
            "sum_cost_usd": float(sum_cost or 0.0),
        }
    return out


def start_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    dataset_filter: str | None = None,
    model_filter: str | None = None,
    n_pairs_total: int | None = None,
    notes: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO run_log (
            run_id, started_at, dataset_filter, model_filter,
            n_pairs_total, n_pairs_done, total_cost_usd, notes
        ) VALUES (?, ?, ?, ?, ?, 0, 0.0, ?)
        """,
        (
            run_id,
            datetime.now(timezone.utc).isoformat(),
            dataset_filter,
            model_filter,
            n_pairs_total,
            notes,
        ),
    )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    n_pairs_done: int,
    total_cost_usd: float,
) -> None:
    conn.execute(
        """
        UPDATE run_log
           SET finished_at = ?,
               n_pairs_done = ?,
               total_cost_usd = ?
         WHERE run_id = ?
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            n_pairs_done,
            total_cost_usd,
            run_id,
        ),
    )
    conn.commit()


def table_schema(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return list(conn.execute(f"PRAGMA table_info({table})").fetchall())


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "results": int(
            conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        ),
        "run_log": int(
            conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0]
        ),
    }
