"""Async DeepInfra (OpenAI-compatible) chat client."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import anyio
import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


@dataclass
class ChatRequest:
    model: str
    messages: list[dict]
    temperature: float = 0.0
    max_tokens: int = 10000
    top_p: float = 1.0


@dataclass
class ChatResponse:
    raw_text: str
    text_source: str
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_ms: int
    http_status: int
    error: Optional[str]


_NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404})

# DeepInfra surfaces a few client-visible errors over HTTP 500 with a JSON
# body whose `code` is one of these; treat them as deterministic refusals
# (no point burning two more retries on the same request body).
_NON_RETRYABLE_500_CODES: tuple[str, ...] = (
    "InvalidParameter",
    "SensitiveContentDetected",
)


def _is_provider_rejection_500(resp: httpx.Response) -> bool:
    if resp.status_code != 500:
        return False
    try:
        body = resp.text
    except Exception:
        return False
    return any(f'"code":"{code}' in body for code in _NON_RETRYABLE_500_CODES)


def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in _NON_RETRYABLE_STATUSES:
            return False
        if _is_provider_rejection_500(exc.response):
            return False
        return status == 429 or status >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


class DeepInfraClient:
    """Thin async wrapper around DeepInfra's OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        max_concurrent: int = 32,
        timeout_s: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            http2=True,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self._sem = anyio.Semaphore(max_concurrent)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post_once(self, payload: dict) -> httpx.Response:
        url = f"{self._base_url}/chat/completions"
        async with self._sem:
            resp = await self._client.post(url, json=payload)
        if resp.status_code >= 400:
            resp.raise_for_status()
        return resp

    async def chat(self, req: ChatRequest) -> ChatResponse:
        payload = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "top_p": req.top_p,
        }

        start = time.monotonic()
        last_status = 0
        last_error: Optional[str] = None

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_should_retry),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                stop=stop_after_attempt(3),
                reraise=True,
            ):
                with attempt:
                    resp = await self._post_once(payload)
                    last_status = resp.status_code
                    data = resp.json()
                    latency_ms = int((time.monotonic() - start) * 1000)
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    raw_text = message.get("content") or ""
                    text_source = "content"
                    if not raw_text and message.get("reasoning_content"):
                        raw_text = message.get("reasoning_content") or ""
                        text_source = "reasoning_content"
                    finish_reason = choice.get("finish_reason")
                    usage = data.get("usage") or {}
                    return ChatResponse(
                        raw_text=raw_text,
                        text_source=text_source,
                        finish_reason=finish_reason,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        latency_ms=latency_ms,
                        http_status=last_status,
                        error=None,
                    )
        except RetryError as e:
            last_error = f"retry_exhausted: {e.last_attempt.exception()!r}"
        except httpx.HTTPStatusError as e:
            last_status = e.response.status_code
            body_snippet = ""
            try:
                body_snippet = e.response.text[:500]
            except Exception:
                pass
            last_error = f"http_{last_status}: {body_snippet}"
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_error = f"transport_error: {e!r}"
        except Exception as e:  # noqa: BLE001
            last_error = f"unexpected: {e!r}"

        latency_ms = int((time.monotonic() - start) * 1000)
        return ChatResponse(
            raw_text="",
            text_source="error",
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            latency_ms=latency_ms,
            http_status=last_status,
            error=last_error or "unknown_error",
        )

    async def list_models(self) -> list[str]:
        """Return model ids from DeepInfra's /models endpoint (best effort).

        Used by the smoke driver to surface missing slugs before the run.
        """
        url = f"{self._base_url}/models"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return []
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        ids: list[str] = []
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                ids.append(entry["id"])
        return ids
