"""Deterministic token -> USD cost accounting."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .client import ChatResponse
from .config import ModelSpec


def compute_cost_usd(
    model: ModelSpec,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Compute the USD cost for a single chat call.

    `cost_per_m_*_usd` fields are priced per 1_000_000 tokens.
    If `cached_tokens > 0` and the model exposes `cost_per_m_cached_usd`,
    the cached portion of the prompt uses the cached price.
    """
    prompt_tokens = max(int(prompt_tokens or 0), 0)
    completion_tokens = max(int(completion_tokens or 0), 0)
    cached_tokens = max(int(cached_tokens or 0), 0)
    cached_tokens = min(cached_tokens, prompt_tokens)

    input_cost: float
    if cached_tokens > 0 and model.cost_per_m_cached_usd is not None:
        regular_prompt = prompt_tokens - cached_tokens
        input_cost = (
            regular_prompt * model.cost_per_m_input_usd
            + cached_tokens * model.cost_per_m_cached_usd
        ) / 1_000_000.0
    else:
        input_cost = prompt_tokens * model.cost_per_m_input_usd / 1_000_000.0

    output_cost = completion_tokens * model.cost_per_m_output_usd / 1_000_000.0
    return input_cost + output_cost


@dataclass
class CostTracker:
    """Thread/async-safe running totals across calls."""

    n_calls: int = 0
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, resp: ChatResponse, model: ModelSpec) -> float:
        prompt_tokens = resp.prompt_tokens or 0
        completion_tokens = resp.completion_tokens or 0
        cost = compute_cost_usd(model, prompt_tokens, completion_tokens)
        with self._lock:
            self.n_calls += 1
            self.total_cost_usd += cost
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
        return cost

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "n_calls": self.n_calls,
                "total_cost_usd": self.total_cost_usd,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
            }
