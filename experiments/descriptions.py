"""Seed-1.8 image -> text splicer.

All access to the pre-computed VLM description `.txt` files in the sibling
repositories must go through this module so every dataset sees the same
formatting and cache.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import BLUEX_DIR, ENEM_DIR, VLM_DESCRIPTOR, VLM_DESCRIPTOR_ENEM_REL


_BLANK_RUN_RE = re.compile(r"\n{3,}")


@lru_cache(maxsize=4096)
def _read_description_file(abs_path: str) -> str:
    path = Path(abs_path)
    if not path.is_file():
        raise FileNotFoundError(f"Seed-1.8 description not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if len(raw) == 0 or path.stat().st_size == 0:
        raise ValueError(f"Seed-1.8 description file is empty: {path}")
    cleaned = raw.rstrip()
    cleaned = _BLANK_RUN_RE.sub("\n\n", cleaned)
    if cleaned.strip() == "":
        raise ValueError(f"Seed-1.8 description file has no content: {path}")
    return cleaned


def enem_description(image_rel_path: str) -> str:
    """Return the Seed-1.8 description for an ENEM image reference.

    `image_rel_path` is the value found in the CSV `context-images` (or
    equivalent) column, e.g.
    ``enem-data/enem-2009/136-images/context_img_0.png``.
    """
    rel = image_rel_path.strip()
    if not rel:
        raise ValueError("Empty image_rel_path")
    target = ENEM_DIR / VLM_DESCRIPTOR_ENEM_REL / (rel + ".txt")
    return _read_description_file(str(target))


def bluex_description_for_image(image_rel_path: str) -> str:
    """Return the Seed-1.8 description for a BLUEX image.

    `image_rel_path` is a value from a question JSON's `associated_images`
    field, e.g. ``"imgs/USP/2019/12/2.jpg"``. The Seed-1.8 `.txt` lives at

        BLUEX_DIR / bluex_dataset / <image_dir> / <VLM_DESCRIPTOR> / <stem>.txt

    which mirrors the layout used by BLUEX's `run_descriptions.ipynb`.
    """
    rel = image_rel_path.strip().lstrip("/")
    if not rel:
        raise ValueError("Empty image_rel_path")
    abs_image = BLUEX_DIR / "bluex_dataset" / rel
    target_dir = abs_image.parent / VLM_DESCRIPTOR
    target = target_dir / f"{abs_image.stem}.txt"
    if not target.is_file():
        # Fallback: the VLM description may have been saved with a different
        # filename than the image stem (e.g. image "1.jpg" but txt "5.txt").
        # Any .txt in the descriptor directory is the description we need.
        txt_files = sorted(target_dir.glob("*.txt")) if target_dir.is_dir() else []
        if txt_files:
            target = txt_files[0]
    return _read_description_file(str(target))


def bluex_description(university: str, year: int, number: int, idx: int) -> str:
    """Deprecated: assumes BLUEX descriptions are named `<idx>.txt`.

    BLUEX's `[IMAGE i]` token actually refers to the i-th entry in a
    question's `associated_images` list (which may have any stem), not to
    a file named `<i>.txt`. Use :func:`bluex_description_for_image` with
    the actual relative image path instead.
    """
    target = (
        BLUEX_DIR
        / "bluex_dataset"
        / "imgs"
        / university
        / str(year)
        / str(number)
        / VLM_DESCRIPTOR
        / f"{idx}.txt"
    )
    return _read_description_file(str(target))


def format_description_block(text: str, index: int | None = None) -> str:
    """Wrap a raw Seed-1.8 description with clear delimiters.

    The delimiters let the answerer LLM distinguish VLM-produced description
    text from the original question body.
    """
    suffix = f" {index}" if index is not None else ""
    body = text.rstrip()
    return (
        f"[início da descrição da imagem{suffix}]\n"
        f"{body}\n"
        f"[fim da descrição da imagem{suffix}]"
    )
