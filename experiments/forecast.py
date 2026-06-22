"""Running cost / ETA forecast for the full-run driver.

Pure functions; called from the dashboard once per second.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .cost import CostTracker


@dataclass
class Forecast:
    pairs_done: int
    pairs_total: int
    pairs_pending: int
    elapsed_s: float
    eta_s: float
    total_cost_so_far: float
    projected_total_cost: float
    mean_cost_per_pair: float
    mean_latency_ms: float


def update_forecast(
    tracker: CostTracker,
    pairs_done: int,
    pairs_total: int,
    started_at: float,
    *,
    sum_latency_ms: float = 0.0,
) -> Forecast:
    """Compute a snapshot forecast.

    `started_at` is a `time.monotonic()` timestamp captured when the run
    began; `sum_latency_ms` is the sum of per-call latencies observed so
    far (used only to display the mean call latency).
    """
    snap = tracker.snapshot()
    total_cost = float(snap["total_cost_usd"])
    pairs_pending = max(pairs_total - pairs_done, 0)

    elapsed_s = max(time.monotonic() - started_at, 0.0)
    rate = (pairs_done / elapsed_s) if elapsed_s > 0 else 0.0
    eta_s = (pairs_pending / rate) if rate > 0 else math.inf

    mean_cost = (total_cost / pairs_done) if pairs_done > 0 else 0.0
    projected = total_cost + mean_cost * pairs_pending
    mean_latency = (sum_latency_ms / pairs_done) if pairs_done > 0 else 0.0

    return Forecast(
        pairs_done=pairs_done,
        pairs_total=pairs_total,
        pairs_pending=pairs_pending,
        elapsed_s=elapsed_s,
        eta_s=eta_s,
        total_cost_so_far=total_cost,
        projected_total_cost=projected,
        mean_cost_per_pair=mean_cost,
        mean_latency_ms=mean_latency,
    )


def model_weighted_forecast(
    *,
    pending_per_model: dict[str, int],
    model_prices: dict[str, tuple[float, float]],
    mean_prompt_tokens: float,
    mean_completion_tokens: float,
) -> dict[str, float]:
    """Estimate per-model cost from smoke-test mean token counts.

    `model_prices[name] = (cost_per_m_input_usd, cost_per_m_output_usd)`.
    Returns a dict `{name: usd_estimate}` plus a `__total__` entry.
    """
    out: dict[str, float] = {}
    total = 0.0
    for name, n_pending in pending_per_model.items():
        prices = model_prices.get(name)
        if prices is None or n_pending <= 0:
            out[name] = 0.0
            continue
        in_price, out_price = prices
        per_call = (
            mean_prompt_tokens * in_price
            + mean_completion_tokens * out_price
        ) / 1_000_000.0
        cost = per_call * n_pending
        out[name] = cost
        total += cost
    out["__total__"] = total
    return out
