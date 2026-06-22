"""Constants, environment loading, and the model specification registry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PARENT_DIR: Path = REPO_ROOT.parent
BLUEX_DIR: Path = PARENT_DIR / "BLUEX"
ENEM_DIR: Path = PARENT_DIR / "ENEM-question-answering"

if not BLUEX_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected BLUEX repository at {BLUEX_DIR}. "
        "Clone BLUEX as a sibling of BRACIS-paper-ENEM-FUVEST-Unicamp."
    )
if not ENEM_DIR.is_dir():
    raise FileNotFoundError(
        f"Expected ENEM-question-answering repository at {ENEM_DIR}. "
        "Clone it as a sibling of BRACIS-paper-ENEM-FUVEST-Unicamp."
    )

load_dotenv(REPO_ROOT / ".env")

DEEPINFRA_BASE_URL: str = "https://api.deepinfra.com/v1/openai"
VLM_DESCRIPTOR: str = "ByteDance__Seed-1.8"
VLM_DESCRIPTOR_ENEM_REL: Path = Path(
    "outputs/multimodal/runs/20260322T200017Z/texts/ByteDance/Seed-1.8"
)

ENEM_YEARS = range(2009, 2024)
ENEM_SUBJECTS: tuple[str, ...] = (
    "ciencias-humanas",
    "ciencias-natureza",
    "linguagens",
    "matematica",
)
BLUEX_UNIVERSITIES: tuple[str, ...] = ("USP", "UNICAMP")

# ENEM questions whose scraped images in ENEM-question-answering are
# unrecoverable (Descomplica error pages saved as `.png`, zero-byte files,
# or otherwise unreadable). These are excluded from the dataset up front
# so they never reach the runner and never produce missing-description
# entries. Keyed by `(year, number)`.
# BLUEX questions whose Seed-1.8 description cannot be produced because
# the model's safety filter persistently refuses the image (returns
# `SensitiveContentDetected`). Keyed by `question_id`.
BLUEX_EXCLUDED_QUESTIONS: frozenset[str] = frozenset({
    # Tarsila do Amaral's "A negra" (1923), a canonical Brazilian
    # modernist painting; flagged as sensitive content by Seed-1.8.
    "UNICAMP_2020_81",
})

ENEM_EXCLUDED_QUESTIONS: frozenset[tuple[int, int]] = frozenset({
    (2022, 117),  # real JPEG misnamed .png; ignored per user instruction
    (2023, 1),    # context_img_0.png is a Descomplica HTML error page
    (2023, 5),    # context_img_0.png is a Descomplica HTML error page
    (2023, 27),   # context_img_0.png is a Descomplica HTML error page
    (2023, 77),   # context_img_0.png is a Descomplica HTML error page
    (2023, 86),   # context_img_0.png is a Descomplica HTML error page
    (2023, 116),  # context_img_0.png is a zero-byte XML file
    (2023, 119),  # alt_img_0.png is a Descomplica HTML error page
})

SQLITE_PATH: Path = REPO_ROOT / "experiments" / "data" / "results.sqlite"


def get_deepinfra_api_key() -> str:
    """Return the DeepInfra API key from the environment.

    Kept as a function so import-time validation (and thus offline
    tooling) does not require the key to be set.
    """
    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY is not set. Add it to "
            f"{REPO_ROOT / '.env'} or export it in your shell."
        )
    return key


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"Cannot coerce {value!r} to bool")


class ModelSpec(BaseModel):
    name: str
    short: str
    params_b: float
    is_slm: bool
    moe: bool
    active_params_b: Optional[float] = None
    cost_per_m_input_usd: float
    cost_per_m_output_usd: float
    cost_per_m_cached_usd: Optional[float] = Field(default=None)

    @field_validator("is_slm", "moe", mode="before")
    @classmethod
    def _validate_bools(cls, v: object) -> bool:
        return _coerce_bool(v)


def load_models() -> list[ModelSpec]:
    """Load the list of models from `models.json` at the repo root."""
    models_path = REPO_ROOT / "models.json"
    with models_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    specs: list[ModelSpec] = []
    for name, entry in raw.items():
        data = dict(entry)
        data["name"] = name
        specs.append(ModelSpec(**data))
    return specs
