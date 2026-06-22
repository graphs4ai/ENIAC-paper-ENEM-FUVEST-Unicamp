"""Generate Seed-1.8 descriptions for every missing `.txt` file.

Reads `experiments/data/missing_descriptions.txt` (written by
`experiments.validate_inputs`), figures out the source image for each
missing description, and calls DeepInfra's `ByteDance/Seed-1.8` VLM with
the exact prompt used by both sibling repos
(``"Descreva a imagem e seus detalhes em portugues."``, ``max_tokens=300``).

The generator is idempotent: any `.txt` that already exists on disk is
skipped. Safe to rerun after partial failures.

Usage::

    python -m experiments.fill_missing_descriptions
    python -m experiments.fill_missing_descriptions --dry-run
    python -m experiments.fill_missing_descriptions --limit 5
    python -m experiments.fill_missing_descriptions --workers 8
"""

from __future__ import annotations

import argparse
import base64
import logging
import mimetypes
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import (
    BLUEX_DIR,
    DEEPINFRA_BASE_URL,
    ENEM_DIR,
    REPO_ROOT,
    VLM_DESCRIPTOR_ENEM_REL,
    get_deepinfra_api_key,
)


log = logging.getLogger(__name__)


DESCRIPTOR_MODEL_ID = "ByteDance/Seed-1.8"
DESCRIPTION_PROMPT_PT = "Descreva a imagem e seus detalhes em portugues."
MAX_TOKENS = 300
# DeepInfra pricing for Seed-1.8 (USD per 1M tokens), from the ENEM repo.
PRICE_INPUT_PER_M = 0.25
PRICE_OUTPUT_PER_M = 2.00


MISSING_LIST_PATH = REPO_ROOT / "experiments" / "data" / "missing_descriptions.txt"


_MISSING_LINE_RE = re.compile(
    r"Seed-1\.8 description not found:\s*(.+)$"
)


@dataclass
class MissingEntry:
    desc_txt_path: Path  # absolute path where the .txt must be written
    image_path: Path  # absolute path to the source image
    source: str  # "ENEM" | "BLUEX"


def _resolve_enem_image(desc_txt_path: Path) -> Path:
    """Given `.../texts/ByteDance/Seed-1.8/<rel>.txt`, return `.../enem-data/<rel>`.

    The ENEM descriptions layout is:
        ENEM_DIR / outputs/multimodal/runs/.../texts/ByteDance/Seed-1.8 / <image_rel> + ".txt"
    where `<image_rel>` already includes the `enem-data/enem-YYYY/...`
    prefix (the image actually lives at `ENEM_DIR / <image_rel>`).
    """
    anchor = ENEM_DIR / VLM_DESCRIPTOR_ENEM_REL
    rel_with_suffix = desc_txt_path.relative_to(anchor)
    rel_no_txt = rel_with_suffix.with_suffix("")
    # `rel_no_txt` is e.g. `enem-data/enem-2023/1-images/context_img_0.png`.
    # The image itself lives at `ENEM_DIR / rel_no_txt`.
    return ENEM_DIR / rel_no_txt


def _resolve_bluex_image(desc_txt_path: Path) -> Path:
    """Given `.../imgs/<UNI>/<year>/<num>/ByteDance__Seed-1.8/<idx>.txt`,
    return `.../imgs/<UNI>/<year>/<num>/<idx>.<ext>`.
    """
    parent = desc_txt_path.parent.parent  # strip the ByteDance__Seed-1.8 dir
    stem = desc_txt_path.stem  # "0", "1", ...
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = parent / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find source image for {desc_txt_path} "
        f"(searched {parent}/{stem}.{{jpg,jpeg,png,webp}})"
    )


def parse_missing_list(path: Path = MISSING_LIST_PATH) -> list[MissingEntry]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing-descriptions list not found at {path}. "
            "Run `python -m experiments.validate_inputs` first to produce it."
        )

    entries: list[MissingEntry] = []
    seen: set[Path] = set()
    enem_anchor = str(ENEM_DIR / VLM_DESCRIPTOR_ENEM_REL)
    bluex_anchor = str(BLUEX_DIR / "bluex_dataset" / "imgs")

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _MISSING_LINE_RE.search(line)
        if not m:
            continue
        desc_txt_path = Path(m.group(1).strip())
        if desc_txt_path in seen:
            continue
        seen.add(desc_txt_path)

        s = str(desc_txt_path)
        if s.startswith(enem_anchor):
            source = "ENEM"
            image_path = _resolve_enem_image(desc_txt_path)
        elif s.startswith(bluex_anchor):
            source = "BLUEX"
            image_path = _resolve_bluex_image(desc_txt_path)
        else:
            raise ValueError(f"Unrecognized missing-description path: {desc_txt_path}")

        entries.append(
            MissingEntry(
                desc_txt_path=desc_txt_path,
                image_path=image_path,
                source=source,
            )
        )
    return entries


def _encode_image_to_data_url(image_abs_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_abs_path))[0] or "image/jpeg"
    data = image_abs_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _build_description_messages(image_abs_path: Path) -> list[dict]:
    data_url = _encode_image_to_data_url(image_abs_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": DESCRIPTION_PROMPT_PT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]


class CallResult:
    __slots__ = ("text", "input_tokens", "output_tokens", "error", "latency_ms")

    def __init__(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        error: str | None,
        latency_ms: int,
    ) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.error = error
        self.latency_ms = latency_ms


def _call_seed_18(
    client: httpx.Client,
    api_key: str,
    messages: list[dict],
    *,
    max_retries: int = 3,
) -> CallResult:
    payload = {
        "model": DESCRIPTOR_MODEL_ID,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{DEEPINFRA_BASE_URL}/chat/completions"

    last_error: str | None = None
    start = time.monotonic()
    for attempt in range(max_retries):
        try:
            resp = client.post(url, json=payload, headers=headers, timeout=120.0)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise httpx.HTTPStatusError(
                    f"status={resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            body = resp.json()
            text = body["choices"][0]["message"]["content"] or ""
            usage = body.get("usage") or {}
            latency_ms = int((time.monotonic() - start) * 1000)
            return CallResult(
                text=text,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                error=None,
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                break
    latency_ms = int((time.monotonic() - start) * 1000)
    return CallResult(
        text="",
        input_tokens=0,
        output_tokens=0,
        error=last_error or "unknown error",
        latency_ms=latency_ms,
    )


def _process_one(
    entry: MissingEntry,
    client: httpx.Client,
    api_key: str,
) -> tuple[MissingEntry, CallResult, str]:
    """Returns (entry, call_result, status) where status is one of
    'written' | 'cached' | 'missing_image' | 'empty_response' | 'error'.
    """
    if entry.desc_txt_path.is_file() and entry.desc_txt_path.stat().st_size > 0:
        return entry, CallResult("", 0, 0, None, 0), "cached"
    if not entry.image_path.is_file():
        return (
            entry,
            CallResult("", 0, 0, f"image not found: {entry.image_path}", 0),
            "missing_image",
        )

    messages = _build_description_messages(entry.image_path)
    result = _call_seed_18(client, api_key, messages)
    if result.error:
        return entry, result, "error"
    text = result.text.strip()
    if not text:
        return entry, result, "empty_response"

    entry.desc_txt_path.parent.mkdir(parents=True, exist_ok=True)
    entry.desc_txt_path.write_text(text + "\n", encoding="utf-8")
    return entry, result, "written"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--missing-list",
        type=Path,
        default=MISSING_LIST_PATH,
        help="Path to missing_descriptions.txt (default: %(default)s).",
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="Concurrent API calls."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N entries (useful for a dry run on real API).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be done; make no API calls.",
    )
    parser.add_argument(
        "--only",
        choices=("ENEM", "BLUEX"),
        default=None,
        help="Restrict to one source.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    entries = parse_missing_list(args.missing_list)
    if args.only:
        entries = [e for e in entries if e.source == args.only]

    n_total = len(entries)
    enem_n = sum(1 for e in entries if e.source == "ENEM")
    bluex_n = sum(1 for e in entries if e.source == "BLUEX")
    print(f"Missing descriptions: {n_total} "
          f"(ENEM: {enem_n}, BLUEX: {bluex_n})")

    # Show the "already cached" count (covers entries that were filled
    # by a partial prior run).
    to_do = [
        e
        for e in entries
        if not (e.desc_txt_path.is_file() and e.desc_txt_path.stat().st_size > 0)
    ]
    print(f"Already on disk: {n_total - len(to_do)}")
    print(f"Would call Seed-1.8 for: {len(to_do)}")

    if args.limit is not None:
        to_do = to_do[: args.limit]
        print(f"  (limited to first {len(to_do)})")

    if args.dry_run:
        for e in to_do[:20]:
            print(f"  [{e.source}] {e.image_path.name}  ->  {e.desc_txt_path}")
        if len(to_do) > 20:
            print(f"  ... and {len(to_do) - 20} more")
        return 0

    missing_source = [e for e in to_do if not e.image_path.is_file()]
    if missing_source:
        print(f"WARNING: {len(missing_source)} source images are missing on disk:")
        for e in missing_source[:10]:
            print(f"  {e.image_path}")
        if len(missing_source) > 10:
            print(f"  ... and {len(missing_source) - 10} more")

    if not to_do:
        print("Nothing to do.")
        return 0

    api_key = get_deepinfra_api_key()

    counts = {
        "written": 0,
        "cached": 0,
        "missing_image": 0,
        "empty_response": 0,
        "error": 0,
    }
    total_in = 0
    total_out = 0
    errors: list[str] = []

    with httpx.Client(http2=True) as client:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, e, client, api_key): e for e in to_do
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                entry, result, status = fut.result()
                counts[status] += 1
                total_in += result.input_tokens
                total_out += result.output_tokens
                if status == "error":
                    errors.append(f"{entry.desc_txt_path}: {result.error}")
                if i % 10 == 0 or i == len(to_do):
                    cost = (
                        total_in * PRICE_INPUT_PER_M / 1_000_000
                        + total_out * PRICE_OUTPUT_PER_M / 1_000_000
                    )
                    print(
                        f"[{i}/{len(to_do)}] "
                        f"written={counts['written']} "
                        f"errors={counts['error']} "
                        f"empty={counts['empty_response']} "
                        f"missing_img={counts['missing_image']} "
                        f"tokens(in/out)={total_in}/{total_out} "
                        f"~cost=${cost:.4f}"
                    )

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    for k, v in counts.items():
        print(f"  {k}: {v}")
    cost = (
        total_in * PRICE_INPUT_PER_M / 1_000_000
        + total_out * PRICE_OUTPUT_PER_M / 1_000_000
    )
    print(f"  total prompt_tokens:    {total_in}")
    print(f"  total completion_tokens:{total_out}")
    print(f"  estimated cost:         ${cost:.4f}")
    if errors:
        print()
        print(f"First 10 errors of {len(errors)}:")
        for line in errors[:10]:
            print(f"  - {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
