"""Unified Question representation and loaders for ENEM + BLUEX."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    BLUEX_DIR,
    BLUEX_EXCLUDED_QUESTIONS,
    BLUEX_UNIVERSITIES,
    ENEM_DIR,
    ENEM_EXCLUDED_QUESTIONS,
    ENEM_SUBJECTS,
    ENEM_YEARS,
)
from .descriptions import (
    bluex_description_for_image,
    enem_description,
    format_description_block,
)


log = logging.getLogger(__name__)


_BLANK_RUN_RE = re.compile(r"\n{3,}")
_IMAGE_TOKEN_RE = re.compile(r"\[IMAGE\s+(\d+)\]")
_CONTEXT_IMAGES_SEP_RE = re.compile(r"[;,]")

_ENEM_SUBJECT_ABBR: dict[str, str] = {
    "ciencias-humanas": "cih",
    "ciencias-natureza": "cin",
    "linguagens": "lin",
    "matematica": "mat",
}


@dataclass(frozen=True)
class Question:
    dataset: str
    question_id: str
    year: int
    subject: tuple[str, ...]
    question_text: str
    alternatives: tuple[str, ...]
    alternatives_type: str
    correct_answer: str
    has_images: bool
    images_in_alternatives: bool
    # BLUEX BNCC capability tags. Always present so ENEM rows are
    # trivially compatible (they keep the default `False`). `CI`
    # ("Com Imagem") mirrors `has_images` for BLUEX rows.
    cap_BK: bool = False
    cap_TU: bool = False
    cap_MR: bool = False
    cap_IU: bool = False
    cap_ML: bool = False
    cap_PRK: bool = False
    cap_CI: bool = False


def _clean_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    s = _BLANK_RUN_RE.sub("\n\n", s)
    return s.strip()


# ---------------------------------------------------------------------------
# ENEM loader
# ---------------------------------------------------------------------------


def _enem_question_id(year: int, subject: str, number: int) -> str:
    abbr = _ENEM_SUBJECT_ABBR[subject]
    return f"ENEM_{year}_{abbr}_{number}"


def load_enem_questions(
    *, strict: bool = True, missing_sink: list[str] | None = None
) -> list[Question]:
    """Load all ENEM questions with Seed-1.8 image descriptions spliced in.

    If `strict` is True (default), a missing description file raises
    FileNotFoundError. If False, the question is skipped and the missing
    path is appended to `missing_sink` (which must then be supplied).
    """
    if not strict and missing_sink is None:
        raise ValueError("missing_sink must be provided when strict=False")
    questions: list[Question] = []
    invalid_answer_count = 0
    invalid_answer_samples: list[str] = []

    for year in ENEM_YEARS:
        for subject in ENEM_SUBJECTS:
            csv_path = ENEM_DIR / "enem-data" / f"enem-{year}" / f"{subject}.csv"
            if not csv_path.is_file():
                log.warning("ENEM CSV missing: %s", csv_path)
                continue

            df = pd.read_csv(csv_path)
            # ENEM linguagens 2010+ has Spanish + English variants of
            # questions 91-95 (the foreign-language slot). Both rows
            # share the same `number`; we keep the first occurrence
            # under the canonical id and suffix later occurrences with
            # `_v2`, `_v3`, ... to keep ids unique without invalidating
            # any row already in `results.sqlite`.
            occurrence: dict[int, int] = {}
            for _, row in df.iterrows():
                number = int(row["number"])
                if (year, number) in ENEM_EXCLUDED_QUESTIONS:
                    continue
                occ = occurrence.get(number, 0)
                occurrence[number] = occ + 1
                context = _clean_text(row.get("context"))
                question = _clean_text(row.get("question"))

                context_images_raw = row.get("context-images")
                context_images: list[str] = []
                if isinstance(context_images_raw, str) and context_images_raw.strip():
                    context_images = [
                        p.strip()
                        for p in _CONTEXT_IMAGES_SEP_RE.split(context_images_raw)
                        if p.strip()
                    ]

                skip_this_row = False
                if context_images:
                    blocks: list[str] = []
                    for i, rel_path in enumerate(context_images):
                        try:
                            desc = enem_description(rel_path)
                        except (FileNotFoundError, ValueError) as e:
                            if strict:
                                raise
                            missing_sink.append(str(e))  # type: ignore[union-attr]
                            skip_this_row = True
                            break
                        blocks.append(
                            format_description_block(
                                desc,
                                index=(i if len(context_images) > 1 else None),
                            )
                        )
                    if skip_this_row:
                        continue
                    joined = "\n\n".join(blocks)
                    context = (context + "\n\n" + joined).strip() if context else joined

                full_question_text = f"{context}\n\n{question}".strip()

                alternatives_list: list[str] = []
                images_in_alternatives = False
                image_folder = (
                    ENEM_DIR / "enem-data" / f"enem-{year}" / f"{number}-images"
                )
                for letter in "ABCDE":
                    alt_raw = row.get(letter)
                    alt_text = _clean_text(alt_raw)
                    alt_line = f"{letter}) {alt_text}" if alt_text else f"{letter})"

                    alt_img_rel = (
                        f"enem-data/enem-{year}/{number}-images/"
                        f"alt_img_{ord(letter) - ord('A')}.png"
                    )
                    alt_img_abs = image_folder / f"alt_img_{ord(letter) - ord('A')}.png"
                    if alt_img_abs.is_file():
                        try:
                            desc = enem_description(alt_img_rel)
                        except (FileNotFoundError, ValueError) as e:
                            if strict:
                                raise
                            missing_sink.append(str(e))  # type: ignore[union-attr]
                            skip_this_row = True
                            break
                        alt_line = (
                            alt_line + "\n" + format_description_block(desc)
                        )
                        images_in_alternatives = True
                    alternatives_list.append(alt_line)

                if skip_this_row:
                    continue

                correct = str(row.get("answer", "")).strip().upper()
                if correct not in set("ABCDE"):
                    invalid_answer_count += 1
                    if len(invalid_answer_samples) < 5:
                        invalid_answer_samples.append(
                            f"{year}/{subject}/{number}: answer={correct!r}"
                        )
                    continue

                qid = _enem_question_id(year, subject, number)
                if occ > 0:
                    qid = f"{qid}_v{occ + 1}"
                questions.append(
                    Question(
                        dataset="ENEM",
                        question_id=qid,
                        year=year,
                        subject=(subject,),
                        question_text=full_question_text,
                        alternatives=tuple(alternatives_list),
                        alternatives_type="string",
                        correct_answer=correct,
                        has_images=bool(context_images) or images_in_alternatives,
                        images_in_alternatives=images_in_alternatives,
                    )
                )

    if invalid_answer_count:
        log.warning(
            "ENEM: skipped %d rows with invalid answer (samples: %s)",
            invalid_answer_count,
            invalid_answer_samples,
        )

    questions.sort(key=lambda q: (q.year, q.subject, q.question_id))
    return questions


# ---------------------------------------------------------------------------
# BLUEX loader
# ---------------------------------------------------------------------------


_BLUEX_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _bluex_positional_image_rel(
    university: str, year: int, number: int, idx: int
) -> str | None:
    """Fallback when `associated_images` is too short for `[IMAGE idx]`.

    Returns the `imgs/<uni>/<year>/<num>/<idx>.<ext>` relative path if a
    source image with stem `<idx>` exists, otherwise None.
    """
    folder = BLUEX_DIR / "bluex_dataset" / "imgs" / university / str(year) / str(number)
    if not folder.is_dir():
        return None
    for ext in _BLUEX_IMG_EXTENSIONS:
        candidate = folder / f"{idx}{ext}"
        if candidate.is_file():
            return f"imgs/{university}/{year}/{number}/{idx}{ext}"
    return None


def _splice_bluex_images(
    text: str,
    associated_images: list[str],
    total_images: int,
    university: str,
    year: int,
    number: int,
    *,
    strict: bool = True,
    missing_sink: list[str] | None = None,
) -> tuple[str, bool, bool]:
    """Replace [IMAGE i] tokens in `text` with Seed-1.8 description blocks.

    `[IMAGE i]` usually refers to `associated_images[i]` (same convention
    as BLUEX's `run_descriptions.ipynb`). When `associated_images` is
    shorter than the referenced index (known-malformed questions, e.g.
    `USP_2018_19/21/25`), falls back to the positional file
    `imgs/<uni>/<year>/<num>/<i>.<ext>`.

    Returns ``(new_text, any_replacement_happened, had_missing_description)``.
    """
    changed = False
    had_missing = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed, had_missing
        idx = int(match.group(1))

        image_rel: str | None
        if 0 <= idx < len(associated_images):
            image_rel = associated_images[idx]
        else:
            image_rel = _bluex_positional_image_rel(university, year, number, idx)
            if image_rel is None:
                if strict:
                    raise IndexError(
                        f"[IMAGE {idx}] in {university}/{year}/{number}: "
                        f"out of range for associated_images "
                        f"(len={len(associated_images)}) and no positional "
                        f"fallback file <{idx}>.jpg/.png found"
                    )
                had_missing = True
                if missing_sink is not None:
                    missing_sink.append(
                        f"{university}/{year}/{number}: [IMAGE {idx}] "
                        f"unresolvable (associated_images has {len(associated_images)} "
                        f"entries and no positional <{idx}>.* file)"
                    )
                return match.group(0)

        try:
            desc = bluex_description_for_image(image_rel)
        except (FileNotFoundError, ValueError) as e:
            if strict:
                raise
            had_missing = True
            if missing_sink is not None:
                missing_sink.append(str(e))
            return match.group(0)
        changed = True
        return format_description_block(
            desc, index=idx if total_images > 1 else None
        )

    new_text = _IMAGE_TOKEN_RE.sub(repl, text)
    return new_text, changed, had_missing


def load_bluex_questions(
    *, strict: bool = True, missing_sink: list[str] | None = None
) -> list[Question]:
    """Load all BLUEX (FUVEST + UNICAMP) questions with images spliced in.

    If `strict` is True (default), a missing Seed-1.8 description raises.
    If False, the offending question is skipped and the missing path is
    appended to `missing_sink`.
    """
    if not strict and missing_sink is None:
        raise ValueError("missing_sink must be provided when strict=False")
    questions: list[Question] = []

    for university in BLUEX_UNIVERSITIES:
        dataset_name = "FUVEST" if university == "USP" else "UNICAMP"
        uni_dir = BLUEX_DIR / "bluex_dataset" / "questions" / university
        if not uni_dir.is_dir():
            log.warning("BLUEX university folder missing: %s", uni_dir)
            continue

        for year_dir in sorted(p for p in uni_dir.iterdir() if p.is_dir()):
            try:
                year = int(year_dir.name)
            except ValueError:
                log.warning("Skipping non-year folder: %s", year_dir)
                continue

            for json_path in sorted(year_dir.glob("*.json")):
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                number = int(data["number"])
                qid = str(data["id"])
                if qid in BLUEX_EXCLUDED_QUESTIONS:
                    continue
                raw_question = _clean_text(data.get("question", ""))
                raw_alternatives = list(data.get("alternatives", []))
                answer = str(data.get("answer", "")).strip().upper()
                subject = tuple(data.get("subject", []))
                alternatives_type = data.get("alternatives_type", "string")
                associated_images = list(data.get("associated_images", []))

                total_images = len(
                    set(_IMAGE_TOKEN_RE.findall(raw_question))
                    | {
                        m
                        for alt in raw_alternatives
                        for m in _IMAGE_TOKEN_RE.findall(str(alt))
                    }
                )
                # `total_images` is a count of distinct [IMAGE i] indices
                # across question + alternatives.

                question_text, q_changed, q_missing = _splice_bluex_images(
                    raw_question,
                    associated_images,
                    total_images,
                    university,
                    year,
                    number,
                    strict=strict,
                    missing_sink=missing_sink,
                )
                any_missing = q_missing

                alternatives_list: list[str] = []
                images_in_alternatives = False
                for alt in raw_alternatives:
                    alt_text = _clean_text(str(alt))
                    alt_text, a_changed, a_missing = _splice_bluex_images(
                        alt_text,
                        associated_images,
                        total_images,
                        university,
                        year,
                        number,
                        strict=strict,
                        missing_sink=missing_sink,
                    )
                    if a_missing:
                        any_missing = True
                    if a_changed:
                        images_in_alternatives = True

                    # BLUEX alts start with "a)", "b)", ... -> uppercase.
                    m = re.match(r"^([a-eA-E])\)\s*(.*)", alt_text, flags=re.DOTALL)
                    if m:
                        letter = m.group(1).upper()
                        rest = m.group(2)
                        alt_text = f"{letter}) {rest}"
                    alternatives_list.append(alt_text)

                # Normalize true/false answers.
                if alternatives_type == "true_false":
                    mapping = {
                        "TRUE": "V",
                        "T": "V",
                        "V": "V",
                        "VERDADEIRO": "V",
                        "FALSE": "F",
                        "F": "F",
                        "FALSO": "F",
                    }
                    answer = mapping.get(answer, answer)

                if any_missing and not strict:
                    continue

                has_images = bool(data.get("has_associated_images")) or q_changed or images_in_alternatives

                def _flag(name: str) -> bool:
                    return bool(data.get(name, False))

                questions.append(
                    Question(
                        dataset=dataset_name,
                        question_id=qid,
                        year=year,
                        subject=subject,
                        question_text=question_text,
                        alternatives=tuple(alternatives_list),
                        alternatives_type=alternatives_type,
                        correct_answer=answer,
                        has_images=has_images,
                        images_in_alternatives=images_in_alternatives,
                        cap_BK=_flag("BK"),
                        cap_TU=_flag("TU"),
                        cap_MR=_flag("MR"),
                        cap_IU=_flag("IU"),
                        cap_ML=_flag("ML"),
                        cap_PRK=_flag("PRK"),
                        cap_CI=bool(data.get("has_associated_images")),
                    )
                )

    questions.sort(key=lambda q: (q.dataset, q.year, q.question_id))
    return questions


# ---------------------------------------------------------------------------
# Combined helpers
# ---------------------------------------------------------------------------


def load_all_questions() -> list[Question]:
    enem = load_enem_questions()
    bluex = load_bluex_questions()
    combined = enem + bluex
    combined.sort(key=lambda q: (q.dataset, q.year, q.question_id))
    return combined


def questions_by_id(questions: list[Question]) -> dict[str, Question]:
    out: dict[str, Question] = {}
    for q in questions:
        if q.question_id in out:
            raise ValueError(f"Duplicate question_id: {q.question_id}")
        out[q.question_id] = q
    return out
