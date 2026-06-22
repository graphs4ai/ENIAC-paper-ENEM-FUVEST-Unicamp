"""Deterministic selection of 10 smoke-test questions."""

from __future__ import annotations

import random
from typing import Callable, Iterable

from .datasets import Question


SEED = 42


def _filter(questions: Iterable[Question], pred: Callable[[Question], bool]) -> list[Question]:
    out = [q for q in questions if pred(q)]
    out.sort(key=lambda q: (q.dataset, q.year, q.question_id))
    return out


def _pick_one(
    rng: random.Random,
    pool: list[Question],
    *,
    exclude_ids: set[str],
    description: str,
) -> Question:
    candidates = [q for q in pool if q.question_id not in exclude_ids]
    if not candidates:
        raise AssertionError(
            f"pick_smoke_questions: no candidate for slot: {description}"
        )
    return rng.choice(candidates)


def pick_smoke_questions(all_questions: list[Question]) -> list[Question]:
    """Return exactly 10 deterministically-selected questions for the smoke run.

    See plan_2.md §4 for the slot definitions.
    """
    rng = random.Random(SEED)
    picked: list[Question] = []
    chosen_ids: set[str] = set()

    def pick(pool: list[Question], description: str) -> None:
        q = _pick_one(rng, pool, exclude_ids=chosen_ids, description=description)
        picked.append(q)
        chosen_ids.add(q.question_id)

    # Slot 1: ENEM matematica, 2018, no images.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "ENEM"
        and q.year == 2018
        and q.subject == ("matematica",)
        and not q.has_images,
    )
    pick(pool, "ENEM 2018 matematica no-images")

    # Slot 2: ENEM ciencias-humanas, 2020, no images.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "ENEM"
        and q.year == 2020
        and q.subject == ("ciencias-humanas",)
        and not q.has_images,
    )
    pick(pool, "ENEM 2020 ciencias-humanas no-images")

    # Slot 3: FUVEST (USP) portuguese, 2019, no images.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "FUVEST"
        and q.year == 2019
        and "portuguese" in q.subject
        and not q.has_images,
    )
    pick(pool, "FUVEST 2019 portuguese no-images")

    # Slot 4: UNICAMP history, 2020, no images.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "UNICAMP"
        and q.year == 2020
        and "history" in q.subject
        and not q.has_images,
    )
    pick(pool, "UNICAMP 2020 history no-images")

    # Slot 5: UNICAMP true_false (any year), no images; fallback to string.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "UNICAMP"
        and q.alternatives_type == "true_false"
        and not q.has_images,
    )
    if not [q for q in pool if q.question_id not in chosen_ids]:
        pool = _filter(
            all_questions,
            lambda q: q.dataset == "UNICAMP"
            and q.alternatives_type == "string"
            and not q.has_images,
        )
    pick(pool, "UNICAMP true_false (or fallback string) no-images")

    # Slot 6: ENEM with images_in_alternatives=True. Required.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "ENEM" and q.images_in_alternatives,
    )
    if not [q for q in pool if q.question_id not in chosen_ids]:
        raise AssertionError(
            "pick_smoke_questions: no ENEM question with images_in_alternatives; "
            "cannot satisfy plan 2's required image-in-alt slot."
        )
    pick(pool, "ENEM images_in_alternatives")

    # Slot 7: ENEM matematica 2019, image in context (not in alternatives).
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "ENEM"
        and q.year == 2019
        and q.subject == ("matematica",)
        and q.has_images
        and not q.images_in_alternatives,
    )
    pick(pool, "ENEM 2019 matematica image-in-context")

    # Slot 8: FUVEST with image in context, 2019.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "FUVEST"
        and q.year == 2019
        and q.has_images
        and not q.images_in_alternatives,
    )
    pick(pool, "FUVEST 2019 image-in-context")

    # Slot 9: UNICAMP with image in context, 2020 (fallback 2019).
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "UNICAMP"
        and q.year == 2020
        and q.has_images
        and not q.images_in_alternatives,
    )
    if not [q for q in pool if q.question_id not in chosen_ids]:
        pool = _filter(
            all_questions,
            lambda q: q.dataset == "UNICAMP"
            and q.year == 2019
            and q.has_images
            and not q.images_in_alternatives,
        )
    pick(pool, "UNICAMP 2020/2019 image-in-context")

    # Slot 10: UNICAMP images_in_alternatives; fallback FUVEST images_in_alternatives.
    pool = _filter(
        all_questions,
        lambda q: q.dataset == "UNICAMP" and q.images_in_alternatives,
    )
    if not [q for q in pool if q.question_id not in chosen_ids]:
        pool = _filter(
            all_questions,
            lambda q: q.dataset == "FUVEST" and q.images_in_alternatives,
        )
    pick(pool, "UNICAMP (or fallback FUVEST) images_in_alternatives")

    assert len(picked) == 10, f"expected 10 picks, got {len(picked)}"
    assert any(q.images_in_alternatives for q in picked), (
        "pick_smoke_questions: no picked question has images in alternatives; "
        "this violates the plan 2 acceptance criteria."
    )
    return picked
