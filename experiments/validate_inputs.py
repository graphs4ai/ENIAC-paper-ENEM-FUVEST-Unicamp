"""Offline validation of the experiment infrastructure (no network calls).

Run with::

    python -m experiments.validate_inputs

It does NOT require DEEPINFRA_API_KEY.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .config import REPO_ROOT, SQLITE_PATH, load_models
from .datasets import (
    Question,
    load_bluex_questions,
    load_enem_questions,
)
from .db import open_db, row_counts, table_schema
from .prompt import build_messages


_FUTURE_HEURISTICS = ("3.5", "3.6", "V3.2", "V4", "gemma-4")


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _validate_models() -> list[str]:
    _section("1) models.json")
    specs = load_models()
    n_slm = sum(1 for s in specs if s.is_slm)
    n_non_slm = len(specs) - n_slm
    print(f"Loaded {len(specs)} models ({n_slm} SLM, {n_non_slm} non-SLM).")

    warnings: list[str] = []
    for s in specs:
        if any(tok in s.name for tok in _FUTURE_HEURISTICS):
            warnings.append(
                f"  WARN: {s.name!r} looks unreleased / future-versioned "
                "(may not be served by DeepInfra under this slug)."
            )
    if warnings:
        print("Model name warnings:")
        for w in warnings:
            print(w)
    else:
        print("No model name warnings.")
    return warnings


def _validate_enem(missing_sink: list[str]) -> list[Question]:
    _section("2) ENEM loader")
    qs = load_enem_questions(strict=False, missing_sink=missing_sink)
    n_total = len(qs)
    n_with_images = sum(1 for q in qs if q.has_images)
    n_with_img_in_alt = sum(1 for q in qs if q.images_in_alternatives)
    print(f"ENEM: total={n_total}, with_images={n_with_images}, "
          f"with_img_in_alt={n_with_img_in_alt}")
    assert n_total > 0, "ENEM loader returned 0 questions"

    # Show the set of observed `correct_answer` values (should be A..E).
    letters = Counter(q.correct_answer for q in qs)
    print(f"ENEM answer distribution: {dict(sorted(letters.items()))}")
    return qs


def _validate_bluex(missing_sink: list[str]) -> list[Question]:
    _section("3) BLUEX loader (FUVEST + UNICAMP)")
    qs = load_bluex_questions(strict=False, missing_sink=missing_sink)
    assert len(qs) > 0, "BLUEX loader returned 0 questions"

    per_dataset: dict[str, list[Question]] = defaultdict(list)
    for q in qs:
        per_dataset[q.dataset].append(q)

    for dataset, group in sorted(per_dataset.items()):
        n_total = len(group)
        n_img = sum(1 for q in group if q.has_images)
        n_img_alt = sum(1 for q in group if q.images_in_alternatives)
        print(
            f"{dataset}: total={n_total}, with_images={n_img}, "
            f"with_img_in_alt={n_img_alt}"
        )

    alt_types = Counter(q.alternatives_type for q in qs)
    print(f"BLUEX alternatives_type counts: {dict(alt_types)}")
    for at, _count in alt_types.items():
        if at == "string":
            continue
        example = next(q for q in qs if q.alternatives_type == at)
        print(f"  non-string example ({at}): {example.question_id}")
    return qs


def _report_missing_descriptions(missing: list[str]) -> int:
    _section("4) Seed-1.8 description file coverage")
    out_path = REPO_ROOT / "experiments" / "data" / "missing_descriptions.txt"
    if missing:
        out_path.write_text("\n".join(missing), encoding="utf-8")
        print(
            f"Missing descriptions: {len(missing)} "
            f"-- written to {out_path}"
        )
        print("First 10:")
        for line in missing[:10]:
            print(f"  {line}")
    else:
        if out_path.is_file():
            out_path.unlink()
        print("All referenced Seed-1.8 description files were loaded successfully.")
    return len(missing)


def _print_sample_prompts(all_questions: list[Question]) -> None:
    _section("5) Sample built prompts (for visual inspection)")

    def _pick(predicate) -> Question | None:
        for q in all_questions:
            if predicate(q):
                return q
        return None

    enem_sample = _pick(lambda q: q.dataset == "ENEM" and q.has_images)
    fuvest_sample = _pick(lambda q: q.dataset == "FUVEST" and q.has_images)
    unicamp_sample = _pick(
        lambda q: q.dataset == "UNICAMP" and q.images_in_alternatives
    ) or _pick(lambda q: q.dataset == "UNICAMP" and q.has_images)

    for label, q in (
        ("ENEM (with image in context)", enem_sample),
        ("FUVEST (with image)", fuvest_sample),
        ("UNICAMP (image in alt if any)", unicamp_sample),
    ):
        print()
        print(f"--- Sample: {label} ---")
        if q is None:
            print("(no matching question found)")
            continue
        print(f"question_id={q.question_id}  "
              f"has_images={q.has_images}  "
              f"images_in_alt={q.images_in_alternatives}  "
              f"correct={q.correct_answer}")
        msgs = build_messages(q)
        for m in msgs:
            print()
            print(f"[{m['role']}]")
            print(m["content"])


def _validate_db() -> None:
    _section("6) SQLite database")
    conn = open_db(SQLITE_PATH)
    try:
        print(f"DB path: {SQLITE_PATH}")
        for table in ("results", "run_log"):
            cols = table_schema(conn, table)
            print(f"  {table}: {len(cols)} columns")
        counts = row_counts(conn)
        print(f"  row_counts: {counts}")
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    failures: list[str] = []
    try:
        _validate_models()
    except Exception as e:  # noqa: BLE001
        failures.append(f"models: {e!r}")

    enem_qs: list[Question] = []
    bluex_qs: list[Question] = []
    missing_descriptions: list[str] = []
    try:
        enem_qs = _validate_enem(missing_descriptions)
    except Exception as e:  # noqa: BLE001
        failures.append(f"enem: {e!r}")
    try:
        bluex_qs = _validate_bluex(missing_descriptions)
    except Exception as e:  # noqa: BLE001
        failures.append(f"bluex: {e!r}")

    all_qs = enem_qs + bluex_qs

    try:
        _report_missing_descriptions(missing_descriptions)
    except Exception as e:  # noqa: BLE001
        failures.append(f"descriptions: {e!r}")

    try:
        _print_sample_prompts(all_qs)
    except Exception as e:  # noqa: BLE001
        failures.append(f"sample_prompts: {e!r}")

    try:
        _validate_db()
    except Exception as e:  # noqa: BLE001
        failures.append(f"db: {e!r}")

    _section("Summary")
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
