"""Unified prompt construction and tolerant JSON response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .datasets import Question


SYSTEM_PROMPT = (
    "Você é um assistente especialista em questões de vestibulares e do ENEM.\n"
    "Sempre responda em PORTUGUÊS. Leia a questão com atenção e escolha uma\n"
    "alternativa. Sua saída DEVE ser um único objeto JSON no seguinte formato,\n"
    "sem texto antes nem depois:\n"
    "\n"
    '{"reasoning_summary":"<até 150 palavras>", "predicted_answer":"<LETRA_OU_VALOR>"}\n'
    "\n"
    "Regras:\n"
    "- predicted_answer deve ser EXATAMENTE um dos valores permitidos para esta\n"
    "  questão (mostrados abaixo).\n"
    "- reasoning_summary deve ter no máximo 150 palavras, em português, e deve\n"
    "  explicar como você chegou a sua resposta.\n"
    "- Não use crases, não use markdown, não envolva o JSON em ``` ou similares.\n"
    "- Se houver descrições de imagens entre [início da descrição da imagem ...]\n"
    "  e [fim da descrição da imagem ...], trate-as como a imagem real.\n"
)


USER_TEMPLATE = (
    "# Questão ({dataset}, {year}, {subject_str})\n"
    "\n"
    "{question_text}\n"
    "\n"
    "# Alternativas\n"
    "\n"
    "{alternatives_block}\n"
    "\n"
    "# Valores permitidos para predicted_answer\n"
    "\n"
    "{allowed_values_str}\n"
    "\n"
    "Responda APENAS com o JSON."
)


def allowed_values(question: Question) -> tuple[str, ...]:
    if question.alternatives_type == "true_false":
        return ("V", "F")
    if question.alternatives_type == "number":
        values: list[str] = []
        for alt in question.alternatives:
            m = re.match(r"^\s*([A-Za-z])\)\s*(.+)", alt, flags=re.DOTALL)
            if m:
                values.append(m.group(2).strip())
            else:
                values.append(alt.strip())
        return tuple(values)
    # "string" (default) -> letters derived from alt count.
    letters = "ABCDE"[: len(question.alternatives)]
    return tuple(letters)


def _allowed_values_str(question: Question) -> str:
    return ", ".join(allowed_values(question))


def build_user_prompt(question: Question) -> str:
    subject_str = ", ".join(question.subject) if question.subject else "-"
    alternatives_block = "\n".join(question.alternatives)
    return USER_TEMPLATE.format(
        dataset=question.dataset,
        year=question.year,
        subject_str=subject_str,
        question_text=question.question_text,
        alternatives_block=alternatives_block,
        allowed_values_str=_allowed_values_str(question),
    )


def build_messages(question: Question) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question)},
    ]


# ---------------------------------------------------------------------------
# Tolerant response parser
# ---------------------------------------------------------------------------


@dataclass
class ParseResult:
    answer: str | None
    reasoning: str | None
    status: str  # "ok" | "truncated_but_answered" | "json_error" | "missing_key" | "disallowed_value" | "empty"


_ANSWER_REGEX = re.compile(
    r'"predicted_answer"\s*:\s*"([^"]+)"'
)


def _try_json_prefixes(raw: str, first_brace: int) -> tuple[dict | None, bool]:
    """Try parsing progressively longer prefixes that end in `}`.

    Returns (parsed_dict_or_None, had_closing_brace_in_raw).
    """
    close_indices = [i for i, ch in enumerate(raw) if ch == "}" and i >= first_brace]
    if not close_indices:
        return None, False
    for end in close_indices:
        candidate = raw[first_brace : end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, True
    return None, True


def parse_response(raw: str, allowed: tuple[str, ...]) -> ParseResult:
    if raw is None or raw.strip() == "":
        return ParseResult(answer=None, reasoning=None, status="empty")

    text = raw.strip()
    first_brace = text.find("{")

    if first_brace != -1:
        obj, had_close = _try_json_prefixes(text, first_brace)
        if obj is not None:
            if "predicted_answer" not in obj:
                return ParseResult(
                    answer=None,
                    reasoning=obj.get("reasoning_summary"),
                    status="missing_key",
                )
            answer = str(obj.get("predicted_answer", "")).strip()
            reasoning = obj.get("reasoning_summary")
            if reasoning is not None:
                reasoning = str(reasoning)
            status = "ok" if had_close else "truncated_but_answered"
            if answer not in allowed:
                # Try a case-insensitive / letter-only relaxation.
                normalized = answer.upper()
                if normalized in allowed:
                    answer = normalized
                else:
                    return ParseResult(
                        answer=answer,
                        reasoning=reasoning,
                        status="disallowed_value",
                    )
            return ParseResult(answer=answer, reasoning=reasoning, status=status)

    m = _ANSWER_REGEX.search(text)
    if m:
        answer = m.group(1).strip()
        if answer not in allowed:
            normalized = answer.upper()
            if normalized in allowed:
                answer = normalized
            else:
                return ParseResult(
                    answer=answer, reasoning=None, status="disallowed_value"
                )
        return ParseResult(
            answer=answer, reasoning=None, status="truncated_but_answered"
        )

    return ParseResult(answer=None, reasoning=None, status="json_error")
